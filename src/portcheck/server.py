"""The ``portcheck-mcp`` server: the eight tools over stdio (spec section 8.10)."""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Callable
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP; see docs/SPEC_DEVIATIONS.md
    from mcp.server.mcpserver import MCPServer as FastMCP

from portcheck import tools

mcp = FastMCP("portcheck")

TOOL_NAMES = (
    "env_fingerprint",
    "scan_repo",
    "plan_migration",
    "apply_step",
    "capture_run",
    "compare_captures",
    "estimate_cost",
    "build_report",
)


def _call(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a tool with stdout pointed at stderr: stdout carries only protocol messages."""
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


@mcp.tool()
def env_fingerprint() -> dict[str, Any]:
    """Describe this machine: GPU vendor (nvidia, amd or cpu), device, driver and
    PyTorch, CUDA or ROCm versions. Needs nothing first. Writes no files."""
    return _call(tools.env_fingerprint)


@mcp.tool()
def scan_repo(path: str = ".") -> dict[str, Any]:
    """Statically audit the repo at `path` for NVIDIA-only dependencies and grade it
    red, yellow or green. Needs nothing first and no GPU. Reads the repo only; saves
    the findings under .portcheck/runs and returns a scan run_id."""
    return _call(tools.scan_repo, path)


@mcp.tool()
def plan_migration(scan_id: str) -> dict[str, Any]:
    """Turn a scan into ordered migration steps: auto steps the tool can apply, then
    manual steps a person must do. Needs a scan run_id from scan_repo. Does not change
    the repo; saves the plan under .portcheck/runs and returns a plan run_id."""
    return _call(tools.plan_migration, scan_id)


@mcp.tool()
def apply_step(
    step_id: str, plan_id: str, dry_run: bool = True, diff_sha256: str | None = None
) -> dict[str, Any]:
    """Preview or apply one auto step of a plan. Needs a plan run_id from plan_migration.
    With dry_run=true (default) it returns a unified diff and its diff_sha256 and writes
    nothing; with dry_run=false and that diff_sha256 it edits the repo files. Show the
    user the diff and get approval before applying."""
    return _call(tools.apply_step, step_id, plan_id, dry_run, diff_sha256)


@mcp.tool()
def capture_run(workload: str, label: str, inject_fault: float | None = None) -> dict[str, Any]:
    """Run a workload file on this machine's GPU and save the model's logits, greedy
    tokens and timings as a capture. Needs a workload YAML file and the `run` extra
    (torch, transformers); may download the model. Writes a capture folder under
    .portcheck/runs. Leave inject_fault unset except for negative tests."""
    return _call(tools.capture_run, workload, label, inject_fault)


@mcp.tool()
def compare_captures(
    reference: str, candidate: str, baseline_repeat: str | None = None
) -> dict[str, Any]:
    """Compare two captures of the same workload and give a pass or fail parity verdict.
    Needs capture run_ids (or capture folder paths) from capture_run; pass a second
    reference-machine capture as baseline_repeat to measure the noise floor. Needs no
    GPU; saves the comparison under .portcheck/runs."""
    return _call(tools.compare_captures, reference, candidate, baseline_repeat)


@mcp.tool()
def estimate_cost(capture: str, price_per_hour: float, currency: str = "USD") -> dict[str, Any]:
    """Convert a capture's measured throughput into cost per million tokens at the
    given hourly machine price. Needs a capture run_id from capture_run and a price
    from the user. Saves the result under .portcheck/runs."""
    return _call(tools.estimate_cost, capture, price_per_hour, currency)


@mcp.tool()
def build_report(
    scan_id: str | None = None,
    compare_id: str | None = None,
    cost_ids: list[str] | None = None,
    format: str = "markdown",  # noqa: A002 - argument name fixed by the spec
) -> dict[str, Any]:
    """Build one report (markdown or json) from earlier results: audit, parity and cost.
    Needs at least one of a scan run_id, a compare run_id or cost run_ids (reference
    machine first). Writes report.md or report.json under .portcheck/runs."""
    return _call(tools.build_report, scan_id, compare_id, cost_ids, format)


def main() -> None:
    """Run the server over stdio. Logs go to stderr."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
