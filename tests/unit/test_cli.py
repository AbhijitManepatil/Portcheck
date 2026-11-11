from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from portcheck.cli import app

runner = CliRunner()


def run(*args: str):
    return runner.invoke(app, list(args))


def run_json(*args: str) -> dict:
    result = run(*args, "--json")
    return json.loads(result.stdout)


@pytest.fixture
def captures(root, make_capture, base_logits):
    return {
        "ref": make_capture("ref", base_logits),
        "repeat": make_capture("repeat", base_logits),
        "cand": make_capture("cand", base_logits + 0.0003),
        "bad": make_capture("bad", base_logits + np.float32(4.0), fault=True),
    }


def test_help_lists_the_eight_commands():
    result = run("--help")
    assert result.exit_code == 0
    for command in ("fingerprint", "scan", "plan", "apply", "capture", "compare", "cost", "report"):
        assert command in result.stdout


def test_fingerprint(root):
    plain = run("fingerprint")
    assert plain.exit_code == 0 and "vendor" in plain.stdout
    assert run_json("fingerprint")["data"]["vendor"] == "cpu"


def test_scan_prints_grade_and_counts(fixture_repo):
    plain = run("scan")
    assert plain.exit_code == 0
    first = plain.stdout.splitlines()[0]
    assert first == "Grade red: 2 red, 3 yellow, 1 green findings in 9 files."
    assert "PC001" in plain.stdout and "run id: scan-" in plain.stdout
    env = run_json("scan", ".")
    assert env["ok"] and env["data"]["counts"] == {"red": 2, "yellow": 3, "green": 1}


def test_errors_exit_one_with_and_without_json(root):
    plain = run("scan", "../outside")
    assert plain.exit_code == 1 and "path_outside_root" in plain.output
    result = run("scan", "../outside", "--json")
    assert result.exit_code == 1
    env = json.loads(result.stdout)
    assert env["ok"] is False and env["error"]["code"] == "path_outside_root"


def test_plan_and_apply(fixture_repo, test_config):
    scan_id = run_json("scan")["run_id"]
    plain = run("plan", scan_id)
    assert plain.exit_code == 0 and "S5" in plain.stdout and "manual" in plain.stdout
    plan_id = run_json("plan", scan_id)["run_id"]

    before = (fixture_repo / "Dockerfile").read_text()
    dry = run("apply", "S1", plan_id)  # dry run is the default
    assert dry.exit_code == 0 and "--- a/Dockerfile" in dry.stdout and "diff_sha256:" in dry.stdout
    assert (fixture_repo / "Dockerfile").read_text() == before
    digest = run_json("apply", "S1", plan_id)["data"]["diff_sha256"]

    no_hash = run("apply", "S1", plan_id, "--apply")
    assert no_hash.exit_code == 1 and "invalid_input" in no_hash.output
    wrong = run("apply", "S1", plan_id, "--apply", "--diff-sha256", "f" * 64)
    assert wrong.exit_code == 1 and "diff_mismatch" in wrong.output
    assert (fixture_repo / "Dockerfile").read_text() == before

    done = run("apply", "S1", plan_id, "--apply", "--diff-sha256", digest)
    assert done.exit_code == 0 and done.stdout.startswith("Applied S1")
    assert (fixture_repo / "Dockerfile").read_text() != before
    assert run("apply", "S4", plan_id).exit_code == 1  # manual step


def test_compare_exit_codes(captures):
    ok = run("compare", captures["ref"], captures["cand"], "--baseline-repeat", captures["repeat"])
    assert ok.exit_code == 0 and "verdict: pass" in ok.stdout
    fail = run("compare", captures["ref"], captures["bad"])
    assert fail.exit_code == 2 and "verdict: fail" in fail.stdout and "FAIL" in fail.stdout
    fail_json = run("compare", captures["ref"], captures["bad"], "--json")
    assert fail_json.exit_code == 2
    assert json.loads(fail_json.stdout)["data"]["verdict"] == "fail"
    missing = run("compare", captures["ref"], "capture-20200101T000000Z-abcdef")
    assert missing.exit_code == 1


def test_cost_and_report(captures, root):
    plain = run("cost", captures["ref"], "--price-per-hour", "2.5")
    assert plain.exit_code == 0 and "5.4253" in plain.stdout
    assert run("cost", captures["ref"], "--price-per-hour", "0").exit_code == 1
    cost_ref = run_json("cost", captures["ref"], "--price-per-hour", "2.5")["run_id"]
    cost_cand = run_json("cost", captures["cand"], "--price-per-hour", "1.9", "--currency", "EUR")[
        "run_id"
    ]
    compare_id = run_json("compare", captures["ref"], captures["cand"])["run_id"]

    plain = run("report", "--compare", compare_id, "--cost", cost_ref, "--cost", cost_cand)
    assert plain.exit_code == 0 and "Report written to" in plain.stdout
    env = run_json("report", "--compare", compare_id, "--cost", cost_ref, "--format", "json")
    assert env["ok"] and (root / env["data"]["report_file"]).name == "report.json"
    assert run("report").exit_code == 1


def test_capture_command(root, tiny_workload):
    plain = run("capture", tiny_workload, "--label", "ref")
    assert plain.exit_code == 0 and "workload sha256:" in plain.stdout
    env = run_json("capture", tiny_workload, "--label", "bad", "--inject-fault", "0.5")
    assert env["ok"] and env["data"]["fault_injected"] is True
    assert run("capture", "missing.yaml", "--label", "x").exit_code == 1


def test_version():
    result = run("--version")
    assert result.exit_code == 0 and result.stdout.strip() == "portcheck 0.1.0"


def test_runs_lists_stored_results_oldest_first(captures, root):
    run_json("cost", captures["ref"], "--price-per-hour", "2.5")
    run_json("compare", captures["ref"], captures["cand"])
    (root / ".portcheck" / "runs" / "not-a-run").mkdir()
    (root / ".portcheck" / "runs" / "scan-20200101T000000Z-abcdef").mkdir()  # empty, damaged

    listed = run_json("runs")["data"]["runs"]
    assert [r["kind"] for r in listed].count("capture") == 4
    assert {r["kind"] for r in listed} == {"scan", "capture", "cost", "compare"}
    assert listed[0]["run_id"] == "scan-20200101T000000Z-abcdef"
    assert listed[0]["detail"] == "unreadable result"
    stamps = [r["run_id"].split("-", 1)[1][:16] for r in listed]
    assert stamps == sorted(stamps)

    only = run_json("runs", "--kind", "capture")["data"]["runs"]
    assert len(only) == 4 and all(r["kind"] == "capture" for r in only)
    details = " ".join(r["detail"] for r in only)
    assert "label ref" in details and "FAULT INJECTED" in details

    plain = run("runs", "--kind", "compare")
    assert plain.exit_code == 0 and "verdict pass" in plain.stdout
    assert run("runs", "--kind", "nonsense").exit_code == 1


def test_runs_on_an_empty_root(root):
    env = run_json("runs")
    assert env["ok"] and env["data"]["runs"] == [] and env["run_id"] is None
    assert not (root / ".portcheck").exists()  # listing creates nothing


def test_long_finding_tables_are_capped(root):
    (root / "many.py").write_text("".join(f"x{i} = 'nvidia-smi'\n" for i in range(80)))
    plain = run("scan")
    assert plain.exit_code == 0
    assert plain.stdout.count("PC009") == 50
    assert "... and 30 more rows; use --json for the full list." in plain.stdout
    assert len(run_json("scan")["data"]["findings"]) == 80
