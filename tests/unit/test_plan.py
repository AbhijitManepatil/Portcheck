from __future__ import annotations

import pytest

from portcheck import tools
from portcheck.envelope import InvalidInputError, NotFoundError
from portcheck.migrate import plan_migration


def _scan_id() -> str:
    return tools.scan_repo(".")["run_id"]


def test_fixture_plan_has_three_auto_then_two_manual(fixture_repo):
    plan = plan_migration(_scan_id())
    assert [(s.step_id, s.kind, s.rule_id, s.files) for s in plan.steps] == [
        ("S1", "auto", "PC007", ["Dockerfile"]),
        ("S2", "auto", "PC008", ["requirements.txt"]),
        ("S3", "auto", "PC009", ["gpu_stats.py"]),
        ("S4", "manual", "PC001", ["ext/fast_op.cu"]),
        ("S5", "manual", "PC002", ["trt_backend.py"]),
    ]
    assert plan.steps[0].title == "Swap base image in Dockerfile"
    assert all(s.title and s.detail for s in plan.steps)


def test_plan_records_its_scan_and_green_findings_make_no_step(fixture_repo):
    scan_id = _scan_id()
    plan = plan_migration(scan_id)
    assert plan.scan_id == scan_id
    assert "PC010" not in {s.rule_id for s in plan.steps}


def test_pc009_detail_lists_every_pynvml_call_site(fixture_repo):
    step = plan_migration(_scan_id()).steps[2]
    assert "if pynvml is not None" in step.detail
    assert "import only" in step.detail
    source = (fixture_repo / "gpu_stats.py").read_text().splitlines()
    call_lines = [n for n, line in enumerate(source, 1) if "pynvml." in line]
    assert call_lines == [10, 12, 13, 15]
    assert ", ".join(str(n) for n in call_lines) in step.detail


def test_manual_red_steps_say_the_work_is_manual(fixture_repo):
    steps = {s.rule_id: s for s in plan_migration(_scan_id()).steps}
    assert "HIPIFY" in steps["PC001"].detail and "SCALE" in steps["PC001"].detail
    assert steps["PC001"].detail.startswith("Manual work")
    assert steps["PC002"].detail.startswith("Manual work")


def test_pc009_import_is_auto_and_nvidia_smi_call_is_manual(root):
    (root / "gpu.py").write_text("import pynvml\nprint('run nvidia-smi')\n")
    (root / "other.py").write_text("from pynvml import nvmlInit\n")
    plan = plan_migration(_scan_id())
    assert [(s.kind, s.rule_id, s.files) for s in plan.steps] == [
        ("auto", "PC009", ["gpu.py"]),
        ("manual", "PC009", ["gpu.py"]),
        ("manual", "PC009", ["other.py"]),
    ]


def test_one_step_per_rule_and_file(root):
    (root / "requirements.txt").write_text("nvidia-cudnn-cu12\nnvidia-cublas-cu12\n")
    (root / "requirements-dev.txt").write_text("nvidia-nccl-cu12\n")
    plan = plan_migration(_scan_id())
    assert [(s.rule_id, s.files) for s in plan.steps] == [
        ("PC008", ["requirements-dev.txt"]),
        ("PC008", ["requirements.txt"]),
    ]


def test_plan_tool_envelope_and_errors(fixture_repo):
    env = tools.plan_migration(_scan_id())
    assert env["ok"] and env["run_id"].startswith("plan-")
    assert (fixture_repo / env["result_path"] / "result.json").is_file()
    assert "3 auto" in env["summary"] and "2 manual" in env["summary"]
    assert tools.plan_migration("scan-20200101T000000Z-abcdef")["error"]["code"] == "not_found"
    assert tools.plan_migration("../../etc/passwd")["error"]["code"] == "invalid_input"
    assert tools.plan_migration(env["run_id"])["error"]["code"] == "invalid_input"  # a plan id


def test_core_errors(root):
    with pytest.raises(NotFoundError):
        plan_migration("scan-20200101T000000Z-abcdef")
    with pytest.raises(InvalidInputError):
        plan_migration("nonsense")
