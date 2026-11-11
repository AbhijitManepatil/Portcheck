"""Simulated end-to-end flow on CPU through the CLI, in a temp copy of the fixture."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from conftest import TEST_ROCM_IMAGE, slim
from portcheck.cli import app

runner = CliRunner()


def cli(*args: str, expect_exit: int = 0) -> dict:
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == expect_exit, result.output
    return json.loads(result.stdout)


def test_full_flow(fixture_repo: Path, test_config, tiny_workload, golden):
    # 1. scan gives grade red and matches golden. (The tiny workload files add no findings.)
    scan = cli("scan", ".")
    assert scan["ok"] and scan["data"]["grade"] == "red"
    assert slim(scan["data"]["findings"]) == golden

    # 2. plan, then dry-run and apply all 3 auto steps with the test config.
    plan = cli("plan", scan["run_id"])
    steps = plan["data"]["steps"]
    auto = [s for s in steps if s["kind"] == "auto"]
    assert [s["rule_id"] for s in auto] == ["PC007", "PC008", "PC009"]
    assert [s["rule_id"] for s in steps if s["kind"] == "manual"] == ["PC001", "PC002"]
    for step in auto:
        dry = cli("apply", step["step_id"], plan["run_id"])
        assert dry["data"]["applied"] is False and dry["data"]["diff"]
        done = cli(
            "apply",
            step["step_id"],
            plan["run_id"],
            "--apply",
            "--diff-sha256",
            dry["data"]["diff_sha256"],
        )
        assert done["data"]["applied"] is True
    assert (fixture_repo / "Dockerfile").read_text().startswith(f"FROM {TEST_ROCM_IMAGE}\n")

    rescan = cli("scan", ".")
    remaining = {f["rule_id"] for f in rescan["data"]["findings"]}
    assert "PC007" not in remaining and "PC008" not in remaining and "PC009" not in remaining
    assert remaining == {"PC001", "PC002", "PC010"}

    # 3. capture three times with the tiny model.
    ref = cli("capture", tiny_workload, "--label", "ref")
    repeat = cli("capture", tiny_workload, "--label", "repeat")
    cand = cli("capture", tiny_workload, "--label", "cand")
    for capture in (ref, repeat, cand):
        directory = fixture_repo / capture["result_path"]
        assert (directory / "capture.json").is_file()
        assert (directory / "capture.safetensors").is_file()

    # 4. compare ref cand --baseline-repeat repeat gives verdict pass.
    good = cli("compare", ref["run_id"], cand["run_id"], "--baseline-repeat", repeat["run_id"])
    assert good["data"]["verdict"] == "pass"
    assert good["data"]["baseline_repeat"] == repeat["run_id"]
    assert all(c["passed"] for c in good["data"]["checks"])

    # 5. a fault-injected capture must fail (and the CLI exits 2).
    bad = cli("capture", tiny_workload, "--label", "bad", "--inject-fault", "0.5")
    assert bad["data"]["fault_injected"] is True
    failed = cli("compare", ref["run_id"], bad["run_id"], expect_exit=2)
    assert failed["data"]["verdict"] == "fail"
    assert "Fault injected" in failed["summary"]

    # 6. cost on ref and cand with different prices; report with all inputs.
    cost_ref = cli("cost", ref["run_id"], "--price-per-hour", "2.5")
    cost_cand = cli("cost", cand["run_id"], "--price-per-hour", "1.9")
    for cost, capture, price in ((cost_ref, ref, 2.5), (cost_cand, cand, 1.9)):
        for row, bench in zip(cost["data"]["rows"], capture["data"]["bench"], strict=True):
            expected = round(price / (bench["tokens_per_second"] * 3600) * 1_000_000, 4)
            assert row["cost_per_million_tokens"] == expected

    report = cli(
        "report",
        "--scan",
        scan["run_id"],
        "--compare",
        good["run_id"],
        "--cost",
        cost_ref["run_id"],
        "--cost",
        cost_cand["run_id"],
    )
    path = fixture_repo / report["data"]["report_file"]
    assert path.name == "report.md" and path.parent.name == report["run_id"]
    markdown = path.read_text(encoding="utf-8")
    assert "**Verdict:** Audit grade: **red**. Parity: **pass**. Cost ratio" in markdown
    for heading in (
        "## Environment",
        "## Audit findings",
        "## Parity",
        "## Performance and cost",
        "## How to reproduce",
    ):
        assert heading in markdown.splitlines()
    assert ref["data"]["workload_sha256"] in markdown
    assert report["data"]["sections"] == [
        "verdict",
        "environment",
        "audit",
        "parity",
        "cost",
        "reproduce",
    ]

    # Every run left its result in .portcheck/runs/<run_id>/.
    runs = {p.name.split("-")[0] for p in (fixture_repo / ".portcheck" / "runs").iterdir()}
    assert runs == {"scan", "plan", "capture", "compare", "cost", "report"}
