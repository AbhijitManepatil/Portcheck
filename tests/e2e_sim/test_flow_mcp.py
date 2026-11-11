"""The server path gives the same results as the CLI path (flow steps 1, 3 and 4)."""

from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import load_file
from typer.testing import CliRunner

from conftest import slim
from portcheck import server
from portcheck.cli import app


def cli(*args: str) -> dict:
    result = CliRunner().invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_mcp_tools_match_the_cli(fixture_repo, tiny_workload, golden):
    # Step 1: scan.
    fingerprint = server.env_fingerprint()
    assert fingerprint["ok"] and fingerprint["data"]["vendor"] == "cpu"
    scan = server.scan_repo(".")
    assert scan["ok"] and scan["data"]["grade"] == "red"
    assert slim(scan["data"]["findings"]) == golden
    assert scan["data"] == cli("scan", ".")["data"]

    # Step 3: three captures through the server tools.
    ref = server.capture_run(tiny_workload, "ref")
    repeat = server.capture_run(tiny_workload, "repeat")
    cand = server.capture_run(tiny_workload, "cand")
    assert ref["ok"] and repeat["ok"] and cand["ok"]

    # A capture made through the CLI holds exactly the same tensors.
    cli_ref = cli("capture", tiny_workload, "--label", "cli-ref")
    assert cli_ref["data"]["workload_sha256"] == ref["data"]["workload_sha256"]
    mcp_tensors = load_file(str(fixture_repo / ref["result_path"] / "capture.safetensors"))
    cli_tensors = load_file(str(fixture_repo / cli_ref["result_path"] / "capture.safetensors"))
    for key in ("last_logits", "generated_ids"):
        np.testing.assert_array_equal(mcp_tensors[key], cli_tensors[key])

    # Step 4: compare; the verdict and every metric match the CLI.
    compared = server.compare_captures(ref["run_id"], cand["run_id"], repeat["run_id"])
    assert compared["ok"] and compared["data"]["verdict"] == "pass"
    via_cli = cli("compare", ref["run_id"], cand["run_id"], "--baseline-repeat", repeat["run_id"])
    assert via_cli["data"] == compared["data"]

    # The remaining tools share the same code path; check they chain from server results.
    plan = server.plan_migration(scan["run_id"])
    assert plan["ok"] and plan["data"] == cli("plan", scan["run_id"])["data"]
    cost = server.estimate_cost(ref["run_id"], 2.5)
    report = server.build_report(scan["run_id"], compared["run_id"], [cost["run_id"]])
    assert cost["ok"] and report["ok"]
    assert (fixture_repo / report["data"]["report_file"]).is_file()
