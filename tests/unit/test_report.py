from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from portcheck import tools

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
HEADINGS = {
    "environment": "## Environment",
    "audit": "## Audit findings",
    "parity": "## Parity",
    "cost": "## Performance and cost",
    "reproduce": "## How to reproduce",
}


def _numbers_in_json(value, out: set[str]) -> set[str]:
    """Every number in a JSON document: numeric values, and digits inside strings."""
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        out.add(json.dumps(value))
    elif isinstance(value, str):
        out.update(NUMBER_RE.findall(value))
    elif isinstance(value, list):
        for item in value:
            _numbers_in_json(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _numbers_in_json(item, out)
    return out


def _numbers_in_markdown(markdown: str) -> list[str]:
    """Numbers in the prose and tables. Code spans (ids, field names, commands) are dropped."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    return NUMBER_RE.findall(text)


@pytest.fixture
def inputs(fixture_repo, make_capture, base_logits):
    ref = make_capture("nvidia-ref", base_logits, tokens_per_second=128.0, vendor="nvidia")
    rep = make_capture("nvidia-repeat", base_logits + 0.0001, vendor="nvidia")
    cand = make_capture("amd-cand", base_logits + 0.0004, tokens_per_second=117.3, vendor="amd")
    scan = tools.scan_repo(".")
    compare = tools.compare_captures(ref, cand, baseline_repeat=rep)
    cost_ref = tools.estimate_cost(ref, 2.5)
    cost_cand = tools.estimate_cost(cand, 1.85)
    assert all(e["ok"] for e in (scan, compare, cost_ref, cost_cand))
    return {
        "root": fixture_repo,
        "captures": [ref, rep, cand],
        "scan": scan,
        "compare": compare,
        "costs": [cost_ref, cost_cand],
    }


def _report(root: Path, **kwargs) -> tuple[dict, str]:
    env = tools.build_report(**kwargs)
    assert env["ok"], env
    return env, (root / env["data"]["report_file"]).read_text(encoding="utf-8")


def _present(markdown: str) -> set[str]:
    return {name for name, heading in HEADINGS.items() if heading in markdown.splitlines()}


def test_full_report_has_all_six_sections_in_order(inputs):
    env, md = _report(
        inputs["root"],
        scan_id=inputs["scan"]["run_id"],
        compare_id=inputs["compare"]["run_id"],
        cost_ids=[c["run_id"] for c in inputs["costs"]],
    )
    assert env["run_id"].startswith("report-") and env["data"]["report_file"].endswith("report.md")
    assert env["data"]["sections"] == [
        "verdict",
        "environment",
        "audit",
        "parity",
        "cost",
        "reproduce",
    ]
    positions = [md.index("**Verdict:**")] + [md.index(h) for h in HEADINGS.values()]
    assert positions == sorted(positions)
    verdict_line = next(line for line in md.splitlines() if line.startswith("**Verdict:**"))
    assert "Audit grade: **red**" in verdict_line and "Parity: **pass**" in verdict_line
    assert "Cost ratio" in verdict_line
    # Environment shows both fingerprints; parity labels the greedy metrics informational.
    assert "Test nvidia GPU" in md and "Test amd GPU" in md
    assert md.count("(informational)") == 2 and "Noise floor" in md
    assert md.index("| Check |") < md.index("Noise floor") < md.index("(informational)")
    # Audit: red and yellow findings in a table, green as a count only.
    assert "| red | PC001 |" in md and "| yellow | PC007 |" in md and "PC010" not in md
    assert "Green findings (no change needed): 1" in md
    # Reproduce: exact commands and the workload hash.
    assert "a" * 64 in md
    for command in ("portcheck scan .", "portcheck capture", "portcheck compare", "portcheck cost"):
        assert command in md
    assert f"--baseline-repeat {inputs['captures'][1]}" in md


@pytest.mark.parametrize(
    ("use", "expected"),
    [
        ({"scan"}, {"audit", "reproduce"}),
        ({"compare"}, {"environment", "parity", "reproduce"}),
        ({"costs"}, {"environment", "cost", "reproduce"}),
        ({"scan", "costs"}, {"environment", "audit", "cost", "reproduce"}),
        ({"scan", "compare", "costs"}, set(HEADINGS)),
    ],
)
def test_sections_appear_only_when_inputs_are_present(inputs, use, expected):
    kwargs = {}
    if "scan" in use:
        kwargs["scan_id"] = inputs["scan"]["run_id"]
    if "compare" in use:
        kwargs["compare_id"] = inputs["compare"]["run_id"]
    if "costs" in use:
        kwargs["cost_ids"] = [c["run_id"] for c in inputs["costs"]]
    env, md = _report(inputs["root"], **kwargs)
    assert _present(md) == expected
    assert set(env["data"]["sections"]) == expected | {"verdict"}
    verdict_line = next(line for line in md.splitlines() if line.startswith("**Verdict:**"))
    assert ("Audit grade" in verdict_line) == ("scan" in use)
    assert ("Parity:" in verdict_line) == ("compare" in use)
    assert ("Cost ratio" in verdict_line) == ("costs" in use)


def test_every_number_in_the_markdown_appears_in_an_input_json(inputs):
    root = inputs["root"]
    env, md = _report(
        root,
        scan_id=inputs["scan"]["run_id"],
        compare_id=inputs["compare"]["run_id"],
        cost_ids=[c["run_id"] for c in inputs["costs"]],
    )
    allowed: set[str] = set()
    for item in (inputs["scan"], inputs["compare"], *inputs["costs"]):
        _numbers_in_json(
            json.loads((root / item["result_path"] / "result.json").read_text()), allowed
        )
    for capture in inputs["captures"]:
        path = root / ".portcheck" / "runs" / capture / "capture.json"
        _numbers_in_json(json.loads(path.read_text()), allowed)

    # The one derived number: the cost ratio the spec requires in the verdict line
    # (candidate cost / reference cost at batch size 1). See docs/SPEC_DEVIATIONS.md.
    ref_cost = inputs["costs"][0]["data"]["rows"][0]["cost_per_million_tokens"]
    cand_cost = inputs["costs"][1]["data"]["rows"][0]["cost_per_million_tokens"]
    ratio = round(cand_cost / ref_cost, 4)
    assert env["data"]["verdict"]["cost_ratio"] == ratio
    assert f"**{json.dumps(ratio)}**" in md
    assert json.dumps(ratio) not in allowed  # so the check below is not vacuous for it

    found = _numbers_in_markdown(md)
    assert len(found) > 40
    unexpected = sorted({n for n in found if n not in allowed and n != json.dumps(ratio)})
    assert unexpected == []
    # Spot checks that the important values really are printed.
    for value in (ref_cost, cand_cost, 2.5, 1.85, 128.0, 117.3):
        assert json.dumps(value) in found
    for check in inputs["compare"]["data"]["checks"]:
        assert json.dumps(check["value"]) in found and json.dumps(check["threshold"]) in found


def test_cost_ratio_needs_two_costs_and_follows_the_compare_roles(inputs):
    root = inputs["root"]
    ids = [c["run_id"] for c in inputs["costs"]]
    one, md = _report(root, cost_ids=ids[:1])
    assert one["data"]["verdict"]["cost_ratio"] is None and "Cost ratio" not in md
    forward, _ = _report(root, cost_ids=ids)
    swapped, _ = _report(root, cost_ids=ids[::-1])
    assert forward["data"]["verdict"]["cost_ratio"] != swapped["data"]["verdict"]["cost_ratio"]
    # With a compare result the roles come from it, whatever the order of cost_ids.
    with_compare, _ = _report(root, compare_id=inputs["compare"]["run_id"], cost_ids=ids[::-1])
    assert with_compare["data"]["verdict"]["cost_ratio"] == forward["data"]["verdict"]["cost_ratio"]


def test_json_format(inputs):
    env = tools.build_report(
        scan_id=inputs["scan"]["run_id"],
        compare_id=inputs["compare"]["run_id"],
        cost_ids=[c["run_id"] for c in inputs["costs"]],
        format="json",
    )
    assert env["ok"] and env["data"]["report_file"].endswith("report.json")
    payload = json.loads((inputs["root"] / env["data"]["report_file"]).read_text())
    assert payload["scan"] == inputs["scan"]["data"]
    assert payload["compare"] == inputs["compare"]["data"]
    assert payload["costs"] == [c["data"] for c in inputs["costs"]]
    assert payload["verdict"]["audit_grade"] == "red"
    assert payload["verdict"]["parity_verdict"] == "pass"
    assert set(payload["captures"]) == set(inputs["captures"])


def test_fault_injected_capture_is_flagged(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits)
    bad = make_capture("bad", base_logits + 3.0, fault=True)
    compare = tools.compare_captures(ref, bad)
    _env, md = _report(root, compare_id=compare["run_id"])
    assert "Parity: **fail**" in md and "**Fault injected:**" in md and "| FAIL |" in md
    assert "--inject-fault" in md


def test_report_errors(inputs):
    assert tools.build_report()["error"]["code"] == "invalid_input"
    assert tools.build_report(cost_ids=[])["error"]["code"] == "invalid_input"
    scan_id = inputs["scan"]["run_id"]
    assert tools.build_report(scan_id=scan_id, format="pdf")["error"]["code"] == "invalid_input"
    assert (
        tools.build_report(scan_id="scan-20200101T000000Z-abcdef")["error"]["code"] == "not_found"
    )
    assert tools.build_report(compare_id=scan_id)["error"]["code"] == "invalid_input"
    cost_id = inputs["costs"][0]["run_id"]
    assert tools.build_report(cost_ids=[cost_id, cost_id])["error"]["code"] == "invalid_input"
