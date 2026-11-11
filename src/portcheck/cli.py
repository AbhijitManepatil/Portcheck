"""The ``portcheck`` command line (spec section 8.9)."""

from __future__ import annotations

import json
from typing import Any, Optional

import typer

from portcheck import __version__, tools

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Check whether a CUDA inference workload can move to AMD ROCm, and prove it.",
)

JSON_OPTION = typer.Option(False, "--json", help="Print the full result envelope as JSON.")


def _table(headers: list[str], rows: list[list[Any]]) -> None:
    cells = [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)
    ]
    typer.echo("  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip())
    typer.echo("  ".join("-" * w for w in widths))
    for row in cells:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())


def _finish(envelope: dict[str, Any], as_json: bool, human: Any = None, fail_code: int = 0) -> None:
    if as_json:
        typer.echo(json.dumps(envelope, indent=2))
    elif not envelope["ok"]:
        err = envelope["error"]
        typer.echo(f"Error ({err['code']}): {err['message']}", err=True)
    else:
        typer.echo(envelope["summary"])
        if human is not None:
            human(envelope["data"])
        if envelope.get("run_id"):
            typer.echo(f"\nrun id: {envelope['run_id']}  ({envelope['result_path']})")
    if not envelope["ok"]:
        raise typer.Exit(1)
    if fail_code:
        raise typer.Exit(fail_code)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"portcheck {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """All paths are relative to PORTCHECK_ROOT, or the current directory if it is not set."""


@app.command()
def fingerprint(as_json: bool = JSON_OPTION) -> None:
    """Describe this machine: GPU vendor, driver and library versions."""
    _finish(
        tools.env_fingerprint(),
        as_json,
        lambda d: _table(["field", "value"], [[k, v] for k, v in d.items()]),
    )


@app.command()
def scan(
    path: str = typer.Argument(".", help="Directory to scan, relative to the root."),
    as_json: bool = JSON_OPTION,
) -> None:
    """Audit a repo for NVIDIA-only dependencies. Read-only."""

    def human(data: dict[str, Any]) -> None:
        if data["findings"]:
            _table(
                ["severity", "rule", "file", "line", "fix", "title"],
                [
                    [f["severity"], f["rule_id"], f["file"], f["line"], f["fix"], f["title"]]
                    for f in data["findings"]
                ],
            )

    _finish(tools.scan_repo(path), as_json, human)


@app.command()
def plan(
    scan_id: str = typer.Argument(..., help="Run id of an earlier scan."),
    as_json: bool = JSON_OPTION,
) -> None:
    """Turn a scan into ordered migration steps. Changes nothing in the repo."""

    def human(data: dict[str, Any]) -> None:
        _table(
            ["step", "kind", "rule", "title"],
            [[s["step_id"], s["kind"], s["rule_id"], s["title"]] for s in data["steps"]],
        )
        for step in data["steps"]:
            typer.echo(f"\n{step['step_id']}: {step['detail']}")

    _finish(tools.plan_migration(scan_id), as_json, human)


@app.command()
def apply(
    step_id: str = typer.Argument(..., help="Step id from the plan, for example S1."),
    plan_id: str = typer.Argument(..., help="Run id of the plan."),
    do_apply: bool = typer.Option(
        False, "--apply", help="Write the change. Needs --diff-sha256 from a dry run."
    ),
    diff_sha256: Optional[str] = typer.Option(  # noqa: UP045 - typer needs Optional on 3.10
        None, "--diff-sha256", help="Hash of the diff you reviewed in the dry run."
    ),
    as_json: bool = JSON_OPTION,
) -> None:
    """Show one auto step as a diff (default), or apply it with --apply."""

    def human(data: dict[str, Any]) -> None:
        if data["diff"]:
            typer.echo("")
            typer.echo(data["diff"], nl=False)
        typer.echo(f"\ndiff_sha256: {data['diff_sha256']}")

    _finish(
        tools.apply_step(step_id, plan_id, dry_run=not do_apply, diff_sha256=diff_sha256),
        as_json,
        human,
    )


@app.command()
def capture(
    workload: str = typer.Argument(..., help="Workload YAML file, relative to the root."),
    label: str = typer.Option(
        ..., "--label", help="Name for this capture, for example nvidia-ref."
    ),
    inject_fault: Optional[float] = typer.Option(  # noqa: UP045
        None, "--inject-fault", help="Negative tests only: noise std added to the embeddings."
    ),
    as_json: bool = JSON_OPTION,
) -> None:
    """Run the workload on this machine and save its outputs and timings."""

    def human(data: dict[str, Any]) -> None:
        _table(
            ["batch size", "tokens", "seconds", "tokens/s", "p50 s", "p95 s"],
            [
                [
                    b["batch_size"],
                    b["tokens_generated"],
                    b["seconds"],
                    b["tokens_per_second"],
                    b["latency_p50_s"],
                    b["latency_p95_s"],
                ]
                for b in data["bench"]
            ],
        )
        typer.echo(f"\nworkload sha256: {data['workload_sha256']}")

    _finish(tools.capture_run(workload, label, inject_fault), as_json, human)


@app.command()
def compare(
    reference: str = typer.Argument(..., help="Reference capture: run id or directory."),
    candidate: str = typer.Argument(..., help="Candidate capture: run id or directory."),
    baseline_repeat: Optional[str] = typer.Option(  # noqa: UP045
        None, "--baseline-repeat", help="A second capture from the reference machine."
    ),
    as_json: bool = JSON_OPTION,
) -> None:
    """Compare two captures offline. Exits 2 when the verdict is fail."""
    envelope = tools.compare_captures(reference, candidate, baseline_repeat)

    def human(data: dict[str, Any]) -> None:
        _table(
            ["check", "value", "threshold", "result"],
            [
                [c["metric"], c["value"], c["threshold"], "pass" if c["passed"] else "FAIL"]
                for c in data["checks"]
            ],
        )
        m = data["metrics"]
        typer.echo(
            f"\ninformational: greedy_exact_match_rate {m['greedy_exact_match_rate']}, "
            f"mean_matching_prefix_tokens {m['mean_matching_prefix_tokens']}"
        )
        typer.echo(f"verdict: {data['verdict']}")

    failed = envelope["ok"] and envelope["data"]["verdict"] == "fail"
    _finish(envelope, as_json, human, fail_code=2 if failed else 0)


@app.command()
def cost(
    capture_ref: str = typer.Argument(..., metavar="CAPTURE", help="Capture run id or directory."),
    price_per_hour: float = typer.Option(
        ..., "--price-per-hour", help="Hourly price of the machine."
    ),
    currency: str = typer.Option("USD", "--currency"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Convert a capture's throughput into cost per million tokens."""

    def human(data: dict[str, Any]) -> None:
        _table(
            ["batch size", "tokens/s", "p50 s", "p95 s", f"{data['currency']} per 1M tokens"],
            [
                [
                    r["batch_size"],
                    r["tokens_per_second"],
                    r["latency_p50_s"],
                    r["latency_p95_s"],
                    r["cost_per_million_tokens"],
                ]
                for r in data["rows"]
            ],
        )

    _finish(tools.estimate_cost(capture_ref, price_per_hour, currency), as_json, human)


@app.command()
def report(
    scan_id: Optional[str] = typer.Option(None, "--scan", help="Scan run id."),  # noqa: UP045
    compare_id: Optional[str] = typer.Option(None, "--compare", help="Compare run id."),  # noqa: UP045
    cost_ids: Optional[list[str]] = typer.Option(  # noqa: UP045
        None, "--cost", help="Cost run id. Repeat for each machine, reference first."
    ),
    fmt: str = typer.Option("markdown", "--format", help="markdown or json."),
    as_json: bool = JSON_OPTION,
) -> None:
    """Build one report from earlier results."""

    def human(data: dict[str, Any]) -> None:
        typer.echo(f"sections: {', '.join(data['sections'])}")

    _finish(tools.build_report(scan_id, compare_id, list(cost_ids or []), fmt), as_json, human)


if __name__ == "__main__":  # pragma: no cover
    app()
