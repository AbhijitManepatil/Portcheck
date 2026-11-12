"""``plan_migration``: turn scan findings into ordered steps (sections 7.3 and 8.3)."""

from __future__ import annotations

from portcheck.audit.scanner import get_rule
from portcheck.envelope import InvalidInputError
from portcheck.migrate.fixes import pynvml_call_lines
from portcheck.paths import load_result, resolve_in_root
from portcheck.schemas import Finding, Plan, PlanStep, ScanResult


def load_scan(scan_id: str) -> ScanResult:
    return ScanResult.model_validate(load_result(scan_id, "scan"))


def _lines(findings: list[Finding]) -> str:
    numbers = sorted({f.line for f in findings if f.line is not None})
    if not numbers:
        return ""
    label = "line" if len(numbers) == 1 else "lines"
    return f"{label} {', '.join(str(n) for n in numbers)}"


def _read(scan_root: str, file: str) -> str | None:
    try:
        return resolve_in_root(f"{scan_root}/{file}").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _auto_detail(rule_id: str, scan_root: str, file: str, findings: list[Finding]) -> str:
    where = _lines(findings)
    if rule_id == "PC007":
        return (
            f"Replaces the NVIDIA image reference on the FROM line ({where}) with "
            "target.rocm_image from portcheck.yaml, keeping any 'AS <name>' suffix."
        )
    if rule_id == "PC008":
        return (
            f"Deletes nvidia-* requirement lines and replaces any download.pytorch.org/whl/cu<N> "
            f"URL with target.torch_index_url from portcheck.yaml ({where})."
        )
    if rule_id == "PC009":
        detail = (
            f"Wraps the top-level 'import pynvml' ({where}) in try/except ImportError so that "
            "pynvml is None on machines without NVML. This fix covers the import only."
        )
        text = _read(scan_root, file)
        calls = pynvml_call_lines(text) if text is not None else []
        if calls:
            label = "Line" if len(calls) == 1 else "Lines"
            verb = "calls" if len(calls) == 1 else "call"
            detail += (
                f" {label} {', '.join(str(n) for n in calls)} {verb} 'pynvml.'; each call site "
                "needs a manual 'if pynvml is not None' check. That check alone is not enough "
                "when the pynvml package is installed on a machine without NVML: nvmlInit() "
                "then raises pynvml.NVMLError, so catch that too (seen on an AMD GPU)."
            )
        else:
            detail += " No 'pynvml.' call sites were found in this file."
        return detail
    return f"Applies the automatic fix for {rule_id} ({where})."


def _manual_detail(findings: list[Finding]) -> str:
    where = _lines(findings)
    prefix = f"Manual work ({where}): " if where else "Manual work: "
    return prefix + findings[0].action


def plan_migration(scan_id: str) -> Plan:
    """Build the migration plan for an earlier scan. Reads the scan, writes nothing."""
    scan = load_scan(scan_id)
    groups: dict[tuple[str, str, str], list[Finding]] = {}
    for finding in scan.findings:
        if finding.fix not in ("auto", "manual"):
            continue  # green findings produce no step
        groups.setdefault((finding.fix, finding.rule_id, finding.file), []).append(finding)

    steps: list[PlanStep] = []
    ordered = sorted(groups.items(), key=lambda kv: (kv[0][0] != "auto", kv[0][1], kv[0][2]))
    for number, ((kind, rule_id, file), findings) in enumerate(ordered, start=1):
        rule = get_rule(rule_id)
        detail = (
            _auto_detail(rule_id, scan.root, file, findings)
            if kind == "auto"
            else _manual_detail(findings)
        )
        steps.append(
            PlanStep(
                step_id=f"S{number}",
                rule_id=rule_id,
                kind=kind,  # type: ignore[arg-type]
                title=rule.plan_title.format(file=file),
                files=[file],
                detail=detail,
            )
        )
    return Plan(scan_id=scan_id, steps=steps)


def load_plan(plan_id: str) -> Plan:
    return Plan.model_validate(load_result(plan_id, "plan"))


def find_step(plan: Plan, step_id: str) -> PlanStep:
    for step in plan.steps:
        if step.step_id == step_id:
            return step
    known = ", ".join(s.step_id for s in plan.steps) or "none"
    raise InvalidInputError(f"Plan has no step '{step_id}'. Steps in this plan: {known}.")
