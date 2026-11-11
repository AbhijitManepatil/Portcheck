"""``build_report`` (spec section 8.8).

Every number printed in the markdown is copied from an input JSON file, formatted
with ``num`` (the JSON representation). The single exception is the cost ratio in
the verdict line, which the spec asks for and which is derived from two inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from portcheck.envelope import InvalidInputError, PortcheckError
from portcheck.paths import load_result
from portcheck.schemas import Capture, CompareResult, CostResult, Fingerprint, ScanResult
from portcheck.validate.capture import load_capture

SECTION_TITLES = {
    "verdict": "Verdict",
    "environment": "Environment",
    "audit": "Audit findings",
    "parity": "Parity",
    "cost": "Performance and cost",
    "reproduce": "How to reproduce",
}
_FINGERPRINT_FIELDS = tuple(Fingerprint.model_fields)


@dataclass
class ReportOutput:
    format: str
    filename: str
    content: str
    sections: list[str]
    verdict: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)


def num(value: Any) -> str:
    """A number exactly as it appears in the input JSON."""
    return json.dumps(value)


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    text = (
        num(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else str(value)
    )
    return text.replace("|", "\\|").replace("\n", " ")


def _code(value: Any) -> str:
    return "`" + str(value).replace("`", "'").replace("|", "\\|").replace("\n", " ") + "`"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _load_captures(names: list[str]) -> dict[str, Capture]:
    captures: dict[str, Capture] = {}
    for name in names:
        if name in captures:
            continue
        try:
            _resolved, capture, _directory = load_capture(name)
        except PortcheckError:
            continue  # the capture folder is not on this machine; report without it
        captures[name] = capture
    return captures


def _cost_roles(
    compare: CompareResult | None, costs: list[CostResult]
) -> tuple[CostResult | None, CostResult | None]:
    """Pick the reference and candidate cost results, if both can be identified."""
    if len(costs) != 2:
        return None, None
    if compare is not None:
        by_capture = {c.capture: c for c in costs}
        if compare.reference in by_capture and compare.candidate in by_capture:
            return by_capture[compare.reference], by_capture[compare.candidate]
    return costs[0], costs[1]


def cost_ratio(compare: CompareResult | None, costs: list[CostResult]) -> float | None:
    """Candidate cost divided by reference cost at batch size 1, or None."""
    reference, candidate = _cost_roles(compare, costs)
    if reference is None or candidate is None or reference.currency != candidate.currency:
        return None
    ref_row = next((r for r in reference.rows if r.batch_size == 1), None)
    cand_row = next((r for r in candidate.rows if r.batch_size == 1), None)
    if ref_row is None or cand_row is None or ref_row.cost_per_million_tokens <= 0:
        return None
    return round(cand_row.cost_per_million_tokens / ref_row.cost_per_million_tokens, 4)


def _machine(capture: Capture | None, name: str) -> str:
    if capture is None:
        return _code(name)
    fp = capture.fingerprint
    device = fp.device_name or fp.vendor
    return f"{_code(capture.label)} ({_cell(device)})"


def _markdown(
    scan_id: str | None,
    scan: ScanResult | None,
    compare_id: str | None,
    compare: CompareResult | None,
    cost_ids: list[str],
    costs: list[CostResult],
    captures: dict[str, Capture],
    ratio: float | None,
) -> tuple[str, list[str]]:
    sections = ["verdict"]
    out: list[str] = ["# GPU Port Check report", ""]

    # 1. Verdict line
    parts = []
    if scan is not None:
        parts.append(f"Audit grade: **{scan.grade}**")
    if compare is not None:
        parts.append(f"Parity: **{compare.verdict}**")
    if ratio is not None:
        parts.append(f"Cost ratio (candidate / reference, batch size {num(1)}): **{num(ratio)}**")
    if not parts:
        parts.append("Cost estimate only; no audit or parity result was given")
    out += ["**Verdict:** " + ". ".join(parts) + ".", ""]

    roles: dict[str, str] = {}
    if compare is not None:
        roles[compare.reference] = "reference"
        roles.setdefault(compare.candidate, "candidate")
        if compare.baseline_repeat:
            roles.setdefault(compare.baseline_repeat, "baseline repeat")

    # 2. Environment
    if captures:
        sections.append("environment")
        names = list(captures)
        headers = ["Field"] + [
            f"{_code(captures[n].label)}" + (f" ({roles[n]})" if n in roles else "") for n in names
        ]
        rows = [["capture"] + [_code(n) for n in names]]
        for fname in _FINGERPRINT_FIELDS:
            rows.append(
                [_code(fname)] + [_cell(getattr(captures[n].fingerprint, fname)) for n in names]
            )
        out += [f"## {SECTION_TITLES['environment']}", ""] + _table(headers, rows) + [""]
        faulty = [n for n in names if captures[n].fault_injected]
        if faulty:
            out += [
                "**Fault injected:** "
                + ", ".join(_code(n) for n in faulty)
                + " ran with deliberately corrupted weights (negative test).",
                "",
            ]

    # 3. Audit findings
    if scan is not None:
        sections.append("audit")
        out += [f"## {SECTION_TITLES['audit']}", ""]
        out += [
            f"Scan {_code(scan_id)} of {_code(scan.root)}: grade **{scan.grade}**, "
            f"{num(scan.files_scanned)} files scanned.",
            "",
        ]
        blocking = [f for f in scan.findings if f.severity in ("red", "yellow")]
        if blocking:
            rows = [
                [
                    f.severity,
                    f.rule_id,
                    _cell(f.title),
                    _code(f.file),
                    _cell(f.line),
                    f.fix,
                    _cell(f.action),
                ]
                for f in blocking
            ]
            out += _table(["Severity", "Rule", "Title", "File", "Line", "Fix", "Action"], rows)
        else:
            out.append("No red or yellow findings.")
        out += ["", f"Green findings (no change needed): {num(scan.counts.get('green', 0))}", ""]

    # 4. Parity
    if compare is not None:
        sections.append("parity")
        out += [f"## {SECTION_TITLES['parity']}", ""]
        out += [
            f"Compare {_code(compare_id)}: reference {_code(compare.reference)}, candidate "
            f"{_code(compare.candidate)}, dtype {_code(compare.dtype)}. Verdict: "
            f"**{compare.verdict}**.",
            "",
        ]
        rule = {"max_abs_diff": "at most", "min_cosine": "at least", "top1_agreement": "at least"}
        rows = [
            [
                _code(c.metric),
                num(c.value),
                f"{rule.get(c.metric, '')} {num(c.threshold)}".strip(),
                "pass" if c.passed else "FAIL",
            ]
            for c in compare.checks
        ]
        out += _table(["Check", "Value", "Threshold", "Result"], rows) + [""]
        source = (
            f"measured against the repeat capture {_code(compare.baseline_repeat)}"
            if compare.baseline_repeat
            else "no repeat capture was given, so it is zero"
        )
        out += [
            f"Noise floor ({source}): {_code('max_abs_diff')} {num(compare.noise_floor.max_abs_diff)}, "
            f"{_code('mean_rel_error')} {num(compare.noise_floor.mean_rel_error)}.",
            "",
            f"Mean relative error between reference and candidate: "
            f"{num(compare.metrics.mean_rel_error)}.",
            "",
            "Greedy decoding (informational, never gates the verdict):",
            "",
            f"- {_code('greedy_exact_match_rate')} (informational): "
            f"{num(compare.metrics.greedy_exact_match_rate)}",
            f"- {_code('mean_matching_prefix_tokens')} (informational): "
            f"{num(compare.metrics.mean_matching_prefix_tokens)}",
            "",
        ]

    # 5. Performance and cost
    if costs:
        sections.append("cost")
        out += [f"## {SECTION_TITLES['cost']}", ""]
        rows = []
        for cost in costs:
            cap = captures.get(cost.capture)
            for row in cost.rows:
                rows.append(
                    [
                        _machine(cap, cost.capture)
                        + (f" {roles[cost.capture]}" if cost.capture in roles else ""),
                        num(row.batch_size),
                        num(row.tokens_per_second),
                        num(row.latency_p50_s),
                        num(row.latency_p95_s),
                        f"{num(cost.price_per_hour)} {_cell(cost.currency)}",
                        f"{num(row.cost_per_million_tokens)} {_cell(cost.currency)}",
                    ]
                )
        headers = [
            "Machine",
            "Batch size",
            "Tokens per second",
            _code("latency_p50_s"),
            _code("latency_p95_s"),
            "Price per hour",
            "Cost per million tokens",
        ]
        out += _table(headers, rows) + [""]
        peaks = [
            f"{_code(captures[c.capture].label)}: {num(captures[c.capture].peak_memory_bytes)} bytes"
            for c in costs
            if c.capture in captures and captures[c.capture].peak_memory_bytes is not None
        ]
        if peaks:
            out += ["Peak GPU memory: " + "; ".join(peaks) + ".", ""]

    # 6. How to reproduce
    sections.append("reproduce")
    out += [f"## {SECTION_TITLES['reproduce']}", ""]
    hashes = sorted({c.workload_sha256 for c in captures.values()})
    if hashes:
        out += [
            "Workload hash (`workload_sha256`, covering the workload file and its prompts file): "
            + ", ".join(_code(h) for h in hashes),
            "",
        ]
    commands: list[str] = []
    if scan is not None:
        commands.append(f"portcheck scan {scan.root}")
    for name, cap in captures.items():
        line = f"portcheck capture <workload.yaml> --label {cap.label}"
        if cap.fault_injected:
            line += " --inject-fault <std>"
        device = cap.fingerprint.device_name or cap.fingerprint.vendor
        commands.append(f"{line}   # on {device}; produced {name}")
    if compare is not None:
        line = f"portcheck compare {compare.reference} {compare.candidate}"
        if compare.baseline_repeat:
            line += f" --baseline-repeat {compare.baseline_repeat}"
        commands.append(line)
    for cost in costs:
        commands.append(
            f"portcheck cost {cost.capture} --price-per-hour {num(cost.price_per_hour)} "
            f"--currency {cost.currency}"
        )
    report_cmd = "portcheck report"
    if scan_id:
        report_cmd += f" --scan {scan_id}"
    if compare_id:
        report_cmd += f" --compare {compare_id}"
    for cost_id in cost_ids:
        report_cmd += f" --cost {cost_id}"
    commands.append(report_cmd)
    out += ["```bash"] + commands + ["```", ""]
    if captures:
        out += [
            "`<workload.yaml>` is the workload file whose hash is shown above. Run ids are "
            "created fresh on each run, so substitute the ids your own commands print.",
            "",
        ]
    return "\n".join(out), sections


def build_report(
    scan_id: str | None = None,
    compare_id: str | None = None,
    cost_ids: list[str] | None = None,
    format: str = "markdown",  # noqa: A002 - argument name fixed by the spec
) -> ReportOutput:
    """Build one report from earlier results. Reads results; the caller writes the file."""
    if format not in ("markdown", "json"):
        raise InvalidInputError("format must be 'markdown' or 'json'.")
    if cost_ids is not None and (
        not isinstance(cost_ids, list) or not all(isinstance(c, str) for c in cost_ids)
    ):
        raise InvalidInputError("cost_ids must be a list of cost run ids.")
    cost_ids = list(cost_ids or [])
    if not scan_id and not compare_id and not cost_ids:
        raise InvalidInputError("Give at least one of scan_id, compare_id or cost_ids.")
    if len(set(cost_ids)) != len(cost_ids):
        raise InvalidInputError("cost_ids contains the same run id more than once.")

    scan = ScanResult.model_validate(load_result(scan_id, "scan")) if scan_id else None
    compare = (
        CompareResult.model_validate(load_result(compare_id, "compare")) if compare_id else None
    )
    costs = [CostResult.model_validate(load_result(c, "cost")) for c in cost_ids]

    names: list[str] = []
    if compare is not None:
        names += [compare.reference, compare.candidate]
        if compare.baseline_repeat:
            names.append(compare.baseline_repeat)
    names += [c.capture for c in costs]
    captures = _load_captures(names)
    ratio = cost_ratio(compare, costs)
    verdict = {
        "audit_grade": scan.grade if scan else None,
        "parity_verdict": compare.verdict if compare else None,
        "cost_ratio": ratio,
    }
    inputs = {"scan_id": scan_id or None, "compare_id": compare_id or None, "cost_ids": cost_ids}
    markdown, sections = _markdown(
        scan_id, scan, compare_id, compare, cost_ids, costs, captures, ratio
    )
    if format == "markdown":
        return ReportOutput("markdown", "report.md", markdown, sections, verdict, inputs)
    payload = {
        "verdict": verdict,
        "inputs": inputs,
        "sections": sections,
        "scan": scan.model_dump(mode="json") if scan else None,
        "compare": compare.model_dump(mode="json") if compare else None,
        "costs": [c.model_dump(mode="json") for c in costs],
        "captures": {n: c.model_dump(mode="json") for n, c in captures.items()},
    }
    content = json.dumps(payload, indent=2) + "\n"
    return ReportOutput("json", "report.json", content, sections, verdict, inputs)
