from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import TEST_INDEX_URL, TEST_ROCM_IMAGE
from portcheck import tools
from portcheck.config import load_config
from portcheck.migrate import fixes

GUARD = (
    "try:\n"
    "    import pynvml\n"
    "except ImportError:  # not available on non-NVIDIA machines\n"
    "    pynvml = None\n"
)


def _snapshot(repo: Path) -> dict[str, bytes]:
    return {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file() and ".portcheck" not in p.parts
    }


@pytest.fixture
def plan_id(fixture_repo) -> str:
    scan = tools.scan_repo(".")
    return tools.plan_migration(scan["run_id"])["run_id"]


def _changed_lines(before: bytes, after: bytes) -> tuple[list[str], list[str]]:
    old, new = before.decode().splitlines(), after.decode().splitlines()
    return [x for x in old if x not in new], [x for x in new if x not in old]


def test_dry_run_writes_nothing(fixture_repo, test_config, plan_id):
    before = _snapshot(fixture_repo)
    runs_before = sorted(p.name for p in (fixture_repo / ".portcheck" / "runs").iterdir())
    plan_files = sorted(p.name for p in (fixture_repo / ".portcheck" / "runs" / plan_id).iterdir())
    for step in ("S1", "S2", "S3"):
        env = tools.apply_step(step, plan_id)  # dry_run defaults to True
        assert env["ok"] and env["data"]["applied"] is False
        assert env["data"]["diff"].startswith("--- a/")
        assert (
            env["data"]["diff_sha256"] == hashlib.sha256(env["data"]["diff"].encode()).hexdigest()
        )
    assert _snapshot(fixture_repo) == before
    assert sorted(p.name for p in (fixture_repo / ".portcheck" / "runs").iterdir()) == runs_before
    assert (
        sorted(p.name for p in (fixture_repo / ".portcheck" / "runs" / plan_id).iterdir())
        == plan_files
    )


def test_apply_with_wrong_hash_is_diff_mismatch_and_writes_nothing(
    fixture_repo, test_config, plan_id
):
    before = _snapshot(fixture_repo)
    env = tools.apply_step("S1", plan_id, dry_run=False, diff_sha256="0" * 64)
    assert env["ok"] is False and env["error"]["code"] == "diff_mismatch"
    assert _snapshot(fixture_repo) == before


def test_apply_without_hash_is_invalid_input(fixture_repo, test_config, plan_id):
    before = _snapshot(fixture_repo)
    env = tools.apply_step("S1", plan_id, dry_run=False)
    assert env["error"]["code"] == "invalid_input"
    assert _snapshot(fixture_repo) == before


def test_apply_fails_if_file_changed_after_the_dry_run(fixture_repo, test_config, plan_id):
    digest = tools.apply_step("S2", plan_id)["data"]["diff_sha256"]
    req = fixture_repo / "requirements.txt"
    req.write_text(req.read_text() + "numpy\n")
    before = _snapshot(fixture_repo)
    env = tools.apply_step("S2", plan_id, dry_run=False, diff_sha256=digest)
    assert env["error"]["code"] == "diff_mismatch"
    assert _snapshot(fixture_repo) == before


def test_apply_with_right_hash_changes_only_the_expected_lines(fixture_repo, test_config, plan_id):
    before = _snapshot(fixture_repo)
    expected = {
        "S1": (
            "Dockerfile",
            ["FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04"],
            [f"FROM {TEST_ROCM_IMAGE}"],
        ),
        "S2": ("requirements.txt", ["nvidia-cudnn-cu12==9.1.0.70"], []),
        "S3": ("gpu_stats.py", ["import pynvml"], GUARD.splitlines()),
    }
    for step, (file, removed, added) in expected.items():
        dry = tools.apply_step(step, plan_id)
        env = tools.apply_step(step, plan_id, dry_run=False, diff_sha256=dry["data"]["diff_sha256"])
        assert env["ok"] and env["data"]["applied"] is True and env["data"]["files"] == [file]
        assert env["data"]["diff"] == dry["data"]["diff"]
        after = _snapshot(fixture_repo)
        assert {k for k in after if after[k] != before[k]} == {file}
        assert _changed_lines(before[file], after[file]) == (removed, added)
        before = after
    assert (fixture_repo / "requirements.txt").read_text() == "torch\ntransformers\npynvml\n"
    assert GUARD in (fixture_repo / "gpu_stats.py").read_text()
    compile((fixture_repo / "gpu_stats.py").read_text(), "gpu_stats.py", "exec")


def test_rescan_after_apply_drops_the_auto_findings(fixture_repo, test_config, plan_id):
    for step in ("S1", "S2", "S3"):
        digest = tools.apply_step(step, plan_id)["data"]["diff_sha256"]
        assert tools.apply_step(step, plan_id, dry_run=False, diff_sha256=digest)["ok"]
    data = tools.scan_repo(".")["data"]
    assert {f["rule_id"] for f in data["findings"]} == {"PC001", "PC002", "PC010"}
    assert data["counts"] == {"red": 2, "yellow": 0, "green": 1}
    # A second dry run now has nothing to do.
    again = tools.apply_step("S1", plan_id)
    assert again["ok"] and again["data"]["diff"] == "" and again["data"]["applied"] is False


def test_default_rocm_image_is_used_without_a_user_config(fixture_repo, plan_id):
    env = tools.apply_step("S1", plan_id)
    assert env["ok"] and f"+FROM {load_config().target.rocm_image}\n" in env["data"]["diff"]


def test_empty_config_returns_config_missing(fixture_repo, plan_id):
    (fixture_repo / "portcheck.yaml").write_text("target:\n  rocm_image: ''\n")
    before = _snapshot(fixture_repo)
    env = tools.apply_step("S1", plan_id)
    assert env["ok"] is False and env["error"]["code"] == "config_missing"
    assert "target.rocm_image" in env["error"]["message"]
    assert _snapshot(fixture_repo) == before
    # PC008 on the fixture has no URL to replace, so it needs no config.
    assert tools.apply_step("S2", plan_id)["ok"]


def test_pc008_url_needs_config(root):
    (root / "requirements.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cu121\ntorch\nnvidia-cudnn-cu12\n"
    )
    plan = tools.plan_migration(tools.scan_repo(".")["run_id"])["run_id"]
    default_url = load_config().target.torch_index_url
    assert f"+--extra-index-url {default_url}\n" in tools.apply_step("S1", plan)["data"]["diff"]
    (root / "portcheck.yaml").write_text("target:\n  torch_index_url: ''\n")
    env = tools.apply_step("S1", plan)
    assert env["error"]["code"] == "config_missing"
    assert "target.torch_index_url" in env["error"]["message"]
    (root / "portcheck.yaml").write_text(f"target:\n  torch_index_url: {TEST_INDEX_URL}\n")
    dry = tools.apply_step("S1", plan)
    assert dry["ok"]
    assert tools.apply_step("S1", plan, dry_run=False, diff_sha256=dry["data"]["diff_sha256"])["ok"]
    assert (root / "requirements.txt").read_text() == f"--extra-index-url {TEST_INDEX_URL}\ntorch\n"


def test_manual_and_unknown_steps_are_invalid_input(fixture_repo, test_config, plan_id):
    env = tools.apply_step("S4", plan_id)
    assert env["error"]["code"] == "invalid_input" and "manual" in env["error"]["message"]
    assert tools.apply_step("S99", plan_id)["error"]["code"] == "invalid_input"
    assert tools.apply_step("S1", "plan-20200101T000000Z-abcdef")["error"]["code"] == "not_found"


def test_apply_refuses_a_file_that_escapes_the_root(fixture_repo, test_config, plan_id, tmp_path):
    outside = tmp_path / "outside.Dockerfile"
    outside.write_text("FROM nvidia/cuda:12\n")
    (fixture_repo / "Dockerfile").unlink()
    (fixture_repo / "Dockerfile").symlink_to(outside)
    env = tools.apply_step("S1", plan_id)
    assert env["ok"] is False and env["error"]["code"] == "path_outside_root"
    assert outside.read_text() == "FROM nvidia/cuda:12\n"


# ---- the pure fix functions ----


def _cfg(root: Path, image: str = TEST_ROCM_IMAGE, url: str = TEST_INDEX_URL):
    (root / "portcheck.yaml").write_text(
        f"target:\n  rocm_image: '{image}'\n  torch_index_url: '{url}'\n"
    )
    return load_config()


def test_fix_pc007_keeps_alias_flags_and_other_stages(root):
    text = (
        "FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:24.05-py3 AS build\n"
        "RUN make\n"
        "FROM python:3.11-slim\n"
        "COPY --from=build /app /app\n"
    )
    assert fixes.fix_pc007(text, "Dockerfile", _cfg(root)) == (
        f"FROM --platform=linux/amd64 {TEST_ROCM_IMAGE} AS build\n"
        "RUN make\n"
        "FROM python:3.11-slim\n"
        "COPY --from=build /app /app\n"
    )


def test_fix_pc008_keeps_dali_and_handles_continuations(root):
    text = (
        "torch\n"
        "nvidia-cudnn-cu12==9.1.0.70 \\\n"
        "    --hash=sha256:abc\n"
        "nvidia-dali-cuda120\n"
        "Nvidia_Cublas_cu12\n"
        "pynvml\n"
    )
    assert fixes.fix_pc008(text, "requirements.txt", _cfg(root)) == (
        "torch\nnvidia-dali-cuda120\npynvml\n"
    )


def test_fix_pc008_in_dockerfile_replaces_only_the_url(root):
    text = "FROM x\nRUN pip install nvidia-cudnn-cu12 -f https://download.pytorch.org/whl/cu118\n"
    assert fixes.fix_pc008(text, "Dockerfile", _cfg(root)) == (
        f"FROM x\nRUN pip install nvidia-cudnn-cu12 -f {TEST_INDEX_URL}\n"
    )


def test_fix_pc009_only_touches_plain_top_level_imports(root):
    cfg = _cfg(root)
    text = "import os\nimport pynvml\n\n\ndef f():\n    import pynvml\n    return pynvml\n"
    assert fixes.fix_pc009(text, "g.py", cfg) == (
        "import os\n" + GUARD + "\n\ndef f():\n    import pynvml\n    return pynvml\n"
    )
    for untouched in (
        "from pynvml import nvmlInit\n",
        "import pynvml as nv\n",
        "import os, pynvml\n",
        "def broken(:\n",
        GUARD,
    ):
        assert fixes.fix_pc009(untouched, "g.py", cfg) == untouched


def test_fixes_are_idempotent(root):
    cfg = _cfg(root)
    once = fixes.fix_pc009("import pynvml\n", "g.py", cfg)
    assert fixes.fix_pc009(once, "g.py", cfg) == once
    docker = fixes.fix_pc007("FROM nvidia/cuda:12\n", "Dockerfile", cfg)
    assert fixes.fix_pc007(docker, "Dockerfile", cfg) == docker
