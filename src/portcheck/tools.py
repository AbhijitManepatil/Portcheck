"""The eight operations as envelope-returning functions (spec section 6.3).

Both the CLI and the MCP server call these. Each one calls the core function,
stores the result under ``<root>/.portcheck/runs/<run_id>/`` and never raises.
"""

from __future__ import annotations

from typing import Any

from portcheck import fingerprint as _fingerprint
from portcheck.audit import scanner as _scanner
from portcheck.envelope import InvalidInputError, guarded, ok_envelope
from portcheck.migrate import apply as _apply
from portcheck.migrate import planner as _planner
from portcheck.paths import (
    RUN_ID_RE,
    RUN_KINDS,
    create_run_dir,
    read_json,
    rel_to_root,
    run_dir,
    runs_dir,
    save_result,
    write_json,
)


def env_fingerprint() -> dict[str, Any]:
    def run() -> dict[str, Any]:
        fp = _fingerprint.env_fingerprint()
        if fp.vendor == "cpu":
            torch = f"torch {fp.torch_version}" if fp.torch_version else "torch is not installed"
            summary = f"This machine has no GPU visible to PyTorch (vendor cpu; {torch})."
        else:
            stack = f"ROCm {fp.rocm_version}" if fp.vendor == "amd" else f"CUDA {fp.cuda_version}"
            summary = (
                f"This machine is {fp.vendor}: {fp.device_name} x{fp.device_count}, {stack}, "
                f"driver {fp.driver_version or 'unknown'}, torch {fp.torch_version}."
            )
        return ok_envelope(summary, fp.model_dump(mode="json"))

    return guarded(run)


def scan_repo(path: str = ".") -> dict[str, Any]:
    def run() -> dict[str, Any]:
        result = _scanner.scan_repo(path)
        data = result.model_dump(mode="json")
        run_id, result_path = save_result("scan", data)
        c = result.counts
        summary = (
            f"Grade {result.grade}: {c['red']} red, {c['yellow']} yellow, {c['green']} green "
            f"findings in {result.files_scanned} files."
        )
        return ok_envelope(summary, data, run_id, result_path)

    return guarded(run)


def plan_migration(scan_id: str) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        plan = _planner.plan_migration(scan_id)
        data = plan.model_dump(mode="json")
        run_id, result_path = save_result("plan", data)
        auto = sum(1 for s in plan.steps if s.kind == "auto")
        summary = (
            f"{len(plan.steps)} steps: {auto} auto (applied as diffs you approve), "
            f"{len(plan.steps) - auto} manual."
        )
        return ok_envelope(summary, data, run_id, result_path)

    return guarded(run)


def apply_step(
    step_id: str, plan_id: str, dry_run: bool = True, diff_sha256: str | None = None
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        if not isinstance(dry_run, bool):
            raise InvalidInputError("dry_run must be true or false.")
        data = _apply.apply_step(step_id, plan_id, dry_run=dry_run, diff_sha256=diff_sha256)
        plan_dir = run_dir(plan_id, "plan")
        files = ", ".join(data["files"]) or "no files"
        if dry_run:
            if data["diff"]:
                summary = (
                    f"Dry run of {step_id}: would change {files}. Nothing was written. To apply, "
                    "pass dry_run=false with this diff_sha256."
                )
            else:
                summary = f"Dry run of {step_id}: nothing to change; the fix is already in place."
        elif data["applied"]:
            write_json(plan_dir / f"applied-{step_id}.json", data)
            summary = f"Applied {step_id}: changed {files}."
        else:
            summary = f"{step_id}: nothing to change; no file was written."
        return ok_envelope(summary, data, plan_id, rel_to_root(plan_dir))

    return guarded(run)


def capture_run(workload: str, label: str, inject_fault: float | None = None) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        from portcheck.paths import runs_dir
        from portcheck.validate import capture as _capture

        run_id, cap, warnings = _capture.capture_run(workload, label, inject_fault)
        fp = cap.fingerprint
        device = fp.device_name or fp.vendor
        first = cap.bench[0]
        summary = (
            f"Captured '{label}' on {device}: {cap.num_prompts} prompts, "
            f"{first.tokens_per_second} tokens/s at batch size {first.batch_size}."
        )
        if cap.fault_injected:
            summary += " Fault injected: this capture is for negative tests only."
        for warning in warnings:
            summary += f" Warning: {warning}"
        return ok_envelope(
            summary, cap.model_dump(mode="json"), run_id, rel_to_root(runs_dir() / run_id)
        )

    return guarded(run)


def compare_captures(
    reference: str, candidate: str, baseline_repeat: str | None = None
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        from portcheck.validate import compare as _compare

        result, faulty = _compare.compare_captures(reference, candidate, baseline_repeat)
        data = result.model_dump(mode="json")
        run_id, result_path = save_result("compare", data)
        failed = [c.metric for c in result.checks if not c.passed]
        if result.verdict == "pass":
            summary = f"Parity pass: all {len(result.checks)} checks passed"
        else:
            summary = f"Parity fail: {', '.join(failed)} outside threshold"
        summary += (
            f" (max_abs_diff {result.metrics.max_abs_diff} against threshold "
            f"{result.thresholds.max_abs_diff})."
        )
        if result.baseline_repeat is None:
            summary += " No repeat capture was given, so the noise floor is zero."
        if faulty:
            summary += f" Fault injected in: {', '.join(faulty)}."
        return ok_envelope(summary, data, run_id, result_path)

    return guarded(run)


def estimate_cost(capture: str, price_per_hour: float, currency: str = "USD") -> dict[str, Any]:
    def run() -> dict[str, Any]:
        from portcheck import cost as _cost

        result = _cost.estimate_cost(capture, price_per_hour, currency)
        data = result.model_dump(mode="json")
        run_id, result_path = save_result("cost", data)
        if result.rows:
            best = min(result.rows, key=lambda r: r.cost_per_million_tokens)
            summary = (
                f"At {result.price_per_hour} {result.currency} per hour: "
                f"{best.cost_per_million_tokens} {result.currency} per million tokens at batch "
                f"size {best.batch_size} (the cheapest measured)."
            )
        else:
            summary = "The capture has no bench rows, so no cost could be computed."
        return ok_envelope(summary, data, run_id, result_path)

    return guarded(run)


def build_report(
    scan_id: str | None = None,
    compare_id: str | None = None,
    cost_ids: list[str] | None = None,
    format: str = "markdown",  # noqa: A002 - argument name fixed by the spec
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        from portcheck import report as _report

        output = _report.build_report(scan_id, compare_id, cost_ids, format)
        run_id, directory = create_run_dir("report")
        (directory / output.filename).write_text(output.content, encoding="utf-8")
        result_path = rel_to_root(directory)
        data = {
            "format": output.format,
            "report_file": f"{result_path}/{output.filename}",
            "sections": output.sections,
            "verdict": output.verdict,
            "inputs": output.inputs,
        }
        write_json(directory / "result.json", data)
        bits = []
        if output.verdict["audit_grade"]:
            bits.append(f"audit grade {output.verdict['audit_grade']}")
        if output.verdict["parity_verdict"]:
            bits.append(f"parity {output.verdict['parity_verdict']}")
        if output.verdict["cost_ratio"] is not None:
            bits.append(f"cost ratio {output.verdict['cost_ratio']}")
        summary = f"Report written to {data['report_file']}" + (
            f" ({', '.join(bits)})." if bits else "."
        )
        return ok_envelope(summary, data, run_id, result_path)

    return guarded(run)


def _describe_run(kind: str, directory: Any) -> str:
    """One line about a stored run, read from its result file. Never raises."""
    try:
        if kind == "capture":
            data = read_json(directory / "capture.json")
            fp = data.get("fingerprint", {})
            text = f"label {data.get('label')}, {fp.get('device_name') or fp.get('vendor')}"
            text += f", workload {str(data.get('workload_sha256'))[:12]}"
            return text + (", FAULT INJECTED" if data.get("fault_injected") else "")
        data = read_json(directory / "result.json")
        if kind == "scan":
            c = data["counts"]
            return f"{data['root']}: grade {data['grade']} ({c['red']} red, {c['yellow']} yellow, {c['green']} green)"
        if kind == "plan":
            auto = sum(1 for s in data["steps"] if s["kind"] == "auto")
            return f"{len(data['steps'])} steps ({auto} auto) for {data['scan_id']}"
        if kind == "compare":
            return f"verdict {data['verdict']}: {data['reference']} vs {data['candidate']}"
        if kind == "cost":
            return f"{data['capture']} at {data['price_per_hour']} {data['currency']} per hour"
        if kind == "report":
            return str(data.get("report_file", ""))
    except Exception:  # noqa: BLE001 - a damaged run folder must not break the listing
        return "unreadable result"
    return ""


def list_runs(kind: str | None = None) -> dict[str, Any]:
    """List the stored runs under ``.portcheck/runs`` (CLI only; not an MCP tool)."""

    def run() -> dict[str, Any]:
        if kind is not None and kind not in RUN_KINDS:
            raise InvalidInputError(f"kind must be one of: {', '.join(RUN_KINDS)}.")
        base = runs_dir()
        rows = []
        if base.is_dir():
            for directory in base.iterdir():
                match = RUN_ID_RE.match(directory.name)
                if not match or not directory.is_dir() or directory.is_symlink():
                    continue
                if kind is not None and match.group(1) != kind:
                    continue
                rows.append(
                    {
                        "run_id": directory.name,
                        "kind": match.group(1),
                        "result_path": rel_to_root(directory),
                        "detail": _describe_run(match.group(1), directory),
                    }
                )
        # Run ids embed a UTC timestamp after the kind, so sort on that part.
        rows.sort(key=lambda r: (r["run_id"].split("-", 1)[1], r["run_id"]))
        what = f"{kind} runs" if kind else "runs"
        return ok_envelope(f"{len(rows)} {what} under .portcheck/runs.", {"runs": rows})

    return guarded(run)
