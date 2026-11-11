"""Robustness checks: hostile inputs, tampered result files and odd repositories."""

from __future__ import annotations

import json
import os
import stat

import pytest

from conftest import TEST_ROCM_IMAGE
from portcheck import tools
from portcheck.audit import parsers
from portcheck.audit.scanner import scan_file
from portcheck.migrate import fixes

# ------------------------------------------------------------------------------ scan


def test_scan_does_not_read_through_symlinks_that_leave_the_root(root, tmp_path):
    secret = tmp_path / "secret.py"
    secret.write_text("import tensorrt  # outside the root\n")
    (root / "link.py").symlink_to(secret)
    inside = root / "real.py"
    inside.write_text("import tensorrt\n")
    (root / "alias.py").symlink_to(inside)  # a link that stays inside is fine
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "x.py").write_text("import tensorrt\n")
    (root / "dirlink").symlink_to(outside_dir, target_is_directory=True)
    (root / "dangling.py").symlink_to(tmp_path / "nowhere.py")

    data = tools.scan_repo(".")["data"]
    assert sorted(f["file"] for f in data["findings"]) == ["alias.py", "real.py"]
    assert data["files_scanned"] == 2


def test_scan_skips_virtualenvs_whatever_their_name(root):
    for name in ("venv", "env-3.12", ".venv"):
        (root / name / "lib").mkdir(parents=True)
        (root / name / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (root / name / "lib" / "dep.py").write_text("import tensorrt\n")
    (root / "app.py").write_text("x = 1\n")
    data = tools.scan_repo(".")["data"]
    assert data["findings"] == [] and data["files_scanned"] == 1
    # A virtualenv named as the scan path itself is scanned: that was asked for.
    assert tools.scan_repo("venv")["data"]["counts"]["red"] == 1


def test_scan_survives_hostile_file_contents(root):
    (root / "deep.py").write_text("x = " + "(" * 5000 + ")" * 5000 + "\nimport tensorrt\n")
    (root / "nul.py").write_bytes(b"import tensorrt\n\x00\x00")
    (root / "long.py").write_text("y = '" + "a" * 500_000 + "'  # nvidia-smi\n")
    (root / "bom.py").write_bytes(b"\xef\xbb\xbfimport tensorrt\n")
    (root / "crlf.py").write_bytes(b"import os\r\nimport tensorrt\r\n")
    (root / "pyproject.toml").write_text("[project\nbroken")
    (root / "environment.yml").write_text("dependencies:\n  - [unclosed\n")
    (root / "requirements.txt").write_bytes(b"\xff\xfe\x00bad")
    (root / "Dockerfile").write_text("FROM\nFROM   \n# FROM nvidia/cuda:1\n")
    (root / "empty.py").write_text("")
    env = tools.scan_repo(".")
    assert env["ok"], env
    by_file = {}
    for finding in env["data"]["findings"]:
        by_file.setdefault(finding["file"], []).append((finding["rule_id"], finding["line"]))
    assert by_file["deep.py"] == [("PC002", 2)]  # found through the regex fallback
    assert by_file["nul.py"] == [("PC002", 1)]
    assert by_file["bom.py"] == [("PC002", 1)]
    assert by_file["crlf.py"] == [("PC002", 2)]
    assert by_file["long.py"] == [("PC009", 1)]
    assert len(env["data"]["findings"][-1]["evidence"]) <= 200
    assert "Dockerfile" not in by_file and "requirements.txt" not in by_file
    json.dumps(env, allow_nan=False)


def test_scan_is_deterministic(fixture_repo):
    first = tools.scan_repo(".")["data"]
    second = tools.scan_repo(".")["data"]
    assert first == second


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        ("torch", "torch"),
        ("Torch_Vision>=0.1 ; python_version > '3.8'", "torch-vision"),
        ("nvidia-cudnn-cu12==9.1.0.70  # pinned", "nvidia-cudnn-cu12"),
        ("tensorrt[extra] @ https://example.test/t.whl", "tensorrt"),
        ("nvidia::cudatoolkit=12", "cudatoolkit"),
        ("https://example.test/nvidia-thing.whl", None),
        ("git+https://example.test/nvidia/repo.git#egg=x", None),
        ("./local/nvidia-pkg", None),
        ("-e .", None),
        ("--extra-index-url https://download.pytorch.org/whl/cu121", None),
        ("# nvidia-cudnn-cu12", None),
        ("", None),
    ],
)
def test_requirement_names(spec, name):
    assert parsers.requirement_name(spec) == name


# ---------------------------------------------------------------------------- migrate


def test_scanner_and_fix_agree_on_what_is_auto_fixable(root):
    """Whenever the scanner says a PC009 import is auto, the fix changes the file."""
    from portcheck.config import load_config

    cfg = load_config()
    samples = {
        "import pynvml\n": True,
        "import pynvml  # gpu stats; NVML only\n": True,
        "import pynvml\r\nx = 1\r\n": True,
        "import pynvml": True,  # no trailing newline
        "import pynvml; import os\n": False,
        "import os; import pynvml\n": False,
        "import pynvml as nv\n": False,
        "import os, pynvml\n": False,
        "from pynvml import nvmlInit\n": False,
        "if True:\n    import pynvml\n": False,
        "def f():\n    import pynvml\n": False,
    }
    for text, auto in samples.items():
        findings = [f for f in scan_file("g.py", text) if f.rule_id == "PC009"]
        assert [f.fix for f in findings] == ["auto" if auto else "manual"], text
        changed = fixes.fix_pc009(text, "g.py", cfg) != text
        assert changed is auto, text


def test_pc009_fix_preserves_line_endings_and_compiles(root):
    from portcheck.config import load_config

    fixed = fixes.fix_pc009("import os\r\nimport pynvml\r\nx = 1\r\n", "g.py", load_config())
    assert "\n" not in fixed.replace("\r\n", "")  # still CRLF throughout
    compile(fixed, "g.py", "exec")
    fixed = fixes.fix_pc009("import pynvml", "g.py", load_config())
    compile(fixed, "g.py", "exec")
    namespace: dict = {}
    exec(fixed, namespace)  # noqa: S102 - the generated guard must actually run
    assert "pynvml" in namespace


def test_apply_handles_a_byte_order_mark(root, test_config):
    (root / "g.py").write_bytes(b"\xef\xbb\xbfimport pynvml\n")
    (root / "Dockerfile").write_bytes(b"\xef\xbb\xbfFROM nvidia/cuda:12 AS base\n")
    scan = tools.scan_repo(".")
    assert [(f["rule_id"], f["line"], f["fix"]) for f in scan["data"]["findings"]] == [
        ("PC007", 1, "auto"),
        ("PC009", 1, "auto"),
    ]
    plan = tools.plan_migration(scan["run_id"])["run_id"]
    for step in ("S1", "S2"):
        dry = tools.apply_step(step, plan)
        assert dry["ok"] and dry["data"]["diff"], dry
        assert tools.apply_step(step, plan, dry_run=False, diff_sha256=dry["data"]["diff_sha256"])[
            "ok"
        ]
    assert (root / "g.py").read_bytes().startswith(b"\xef\xbb\xbftry:\n    import pynvml\n")
    assert (root / "Dockerfile").read_bytes() == (
        b"\xef\xbb\xbfFROM " + TEST_ROCM_IMAGE.encode() + b" AS base\n"
    )
    assert tools.scan_repo(".")["data"]["findings"] == []


def test_apply_keeps_file_mode_and_leaves_no_temp_files(fixture_repo, test_config):
    target = fixture_repo / "gpu_stats.py"
    target.chmod(0o750)
    plan = tools.plan_migration(tools.scan_repo(".")["run_id"])["run_id"]
    dry = tools.apply_step("S3", plan)
    done = tools.apply_step("S3", plan, dry_run=False, diff_sha256=dry["data"]["diff_sha256"])
    assert done["ok"] and done["data"]["applied"] is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o750
    assert not [p.name for p in fixture_repo.iterdir() if "portcheck-tmp" in p.name]


def test_apply_reports_a_read_only_directory_without_corrupting_the_file(fixture_repo, test_config):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    plan = tools.plan_migration(tools.scan_repo(".")["run_id"])["run_id"]
    dry = tools.apply_step("S1", plan)
    before = (fixture_repo / "Dockerfile").read_text()
    fixture_repo.chmod(0o555)
    try:
        env = tools.apply_step("S1", plan, dry_run=False, diff_sha256=dry["data"]["diff_sha256"])
    finally:
        fixture_repo.chmod(0o755)
    assert env["ok"] is False and (fixture_repo / "Dockerfile").read_text() == before


def test_tampered_plan_cannot_write_outside_the_root(fixture_repo, test_config, tmp_path):
    outside = tmp_path / "Dockerfile"
    outside.write_text("FROM nvidia/cuda:12\n")
    plan = tools.plan_migration(tools.scan_repo(".")["run_id"])
    plan_file = fixture_repo / plan["result_path"] / "result.json"
    data = json.loads(plan_file.read_text())
    data["steps"][0]["files"] = ["../Dockerfile"]
    plan_file.write_text(json.dumps(data))
    env = tools.apply_step("S1", plan["run_id"])
    assert env["ok"] is False and env["error"]["code"] == "path_outside_root"

    # The same through a tampered scan root.
    scan_file_path = fixture_repo / ".portcheck" / "runs" / data["scan_id"] / "result.json"
    scan = json.loads(scan_file_path.read_text())
    scan["root"] = ".."
    scan_file_path.write_text(json.dumps(scan))
    data["steps"][0]["files"] = ["Dockerfile"]
    plan_file.write_text(json.dumps(data))
    env = tools.apply_step("S1", plan["run_id"])
    assert env["ok"] is False and env["error"]["code"] == "path_outside_root"
    assert outside.read_text() == "FROM nvidia/cuda:12\n"


def test_corrupt_result_files_give_envelopes_not_exceptions(
    fixture_repo, make_capture, base_logits
):
    scan = tools.scan_repo(".")
    result = fixture_repo / scan["result_path"] / "result.json"
    result.write_text("{not json")
    assert tools.plan_migration(scan["run_id"])["error"]["code"] == "invalid_input"
    result.write_text('{"root": "."}')  # valid JSON, wrong shape
    assert tools.plan_migration(scan["run_id"])["error"]["code"] == "invalid_input"
    assert tools.build_report(scan_id=scan["run_id"])["error"]["code"] == "invalid_input"

    cap = make_capture("ref", base_logits)
    directory = fixture_repo / ".portcheck" / "runs" / cap
    (directory / "capture.safetensors").write_bytes(b"garbage")
    env = tools.compare_captures(cap, cap)
    assert env["ok"] is False and env["error"]["code"] in ("invalid_input", "internal")
    (directory / "capture.safetensors").unlink()
    assert tools.compare_captures(cap, cap)["error"]["code"] == "not_found"
    (directory / "capture.json").write_text("[]")
    assert tools.estimate_cost(cap, 1.0)["error"]["code"] == "invalid_input"


def test_dockerfile_fix_handles_arg_and_digest_forms(root):
    from portcheck.config import load_config

    (root / "portcheck.yaml").write_text(f"target:\n  rocm_image: {TEST_ROCM_IMAGE}\n")
    cfg = load_config()
    text = (
        "ARG BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
        "FROM ${BASE} AS a\n"
        "FROM nvidia/cuda@sha256:abc AS b\n"
        "FROM nvcr.io/nvidia/pytorch:24.05-py3\n"
    )
    assert fixes.fix_pc007(text, "Dockerfile", cfg) == (
        "ARG BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
        "FROM ${BASE} AS a\n"
        f"FROM {TEST_ROCM_IMAGE} AS b\n"
        f"FROM {TEST_ROCM_IMAGE}\n"
    )


# ---------------------------------------------------------------------------- capture


def test_prompt_longer_than_the_model_context_is_rejected(root, tiny_workload):
    path = root / tiny_workload
    path.write_text(path.read_text().replace("max_new_tokens: 8", "max_new_tokens: 60"))
    env = tools.capture_run(tiny_workload, "long")
    assert env["ok"] is False and env["error"]["code"] == "invalid_input"
    assert "context length (64)" in env["error"]["message"]
    assert not (root / ".portcheck").exists()
