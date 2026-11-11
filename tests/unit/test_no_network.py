"""With sockets disabled, everything except a hub model download still works."""

from __future__ import annotations

import socket

import pytest

from conftest import slim
from portcheck import tools


def test_the_fixture_really_blocks_sockets(no_network):
    with pytest.raises(AssertionError):
        socket.socket()
    with pytest.raises(AssertionError):
        socket.create_connection(("example.test", 443))


def test_scan_plan_apply_compare_cost_report_work_offline(
    no_network, fixture_repo, test_config, golden, make_capture, base_logits
):
    scan = tools.scan_repo(".")
    assert scan["ok"] and slim(scan["data"]["findings"]) == golden

    plan = tools.plan_migration(scan["run_id"])
    assert plan["ok"] and len(plan["data"]["steps"]) == 5

    for step in ("S1", "S2", "S3"):
        dry = tools.apply_step(step, plan["run_id"])
        assert dry["ok"]
        applied = tools.apply_step(
            step, plan["run_id"], dry_run=False, diff_sha256=dry["data"]["diff_sha256"]
        )
        assert applied["ok"] and applied["data"]["applied"] is True

    ref = make_capture("ref", base_logits)
    repeat = make_capture("repeat", base_logits)
    cand = make_capture("cand", base_logits + 0.0002)
    compare = tools.compare_captures(ref, cand, baseline_repeat=repeat)
    assert compare["ok"] and compare["data"]["verdict"] == "pass"

    costs = [tools.estimate_cost(ref, 2.5), tools.estimate_cost(cand, 1.9)]
    assert all(c["ok"] for c in costs)

    for fmt in ("markdown", "json"):
        report = tools.build_report(
            scan_id=scan["run_id"],
            compare_id=compare["run_id"],
            cost_ids=[c["run_id"] for c in costs],
            format=fmt,
        )
        assert report["ok"] and (fixture_repo / report["data"]["report_file"]).is_file()

    assert tools.env_fingerprint()["ok"]


def test_capture_with_a_local_model_works_offline(no_network, root, tiny_workload):
    env = tools.capture_run(tiny_workload, "offline")
    assert env["ok"], env
