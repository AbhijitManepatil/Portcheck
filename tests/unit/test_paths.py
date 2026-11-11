from __future__ import annotations

import re

import pytest

from portcheck import tools
from portcheck.config import load_config
from portcheck.envelope import InvalidInputError, NotFoundError, PathOutsideRootError
from portcheck.paths import (
    RUN_ID_RE,
    RUN_KINDS,
    get_root,
    is_run_id,
    new_run_id,
    resolve_in_root,
    run_dir,
    save_result,
)


def test_root_comes_from_env(root):
    assert get_root() == root


def test_root_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("PORTCHECK_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_root() == tmp_path.resolve()


def test_path_inside_root_resolves(root):
    assert resolve_in_root(".") == root
    assert resolve_in_root("a/b.txt") == root / "a" / "b.txt"
    assert resolve_in_root("a/../b.txt") == root / "b.txt"


@pytest.mark.parametrize("bad", ["..", "../x", "a/../../x", "/etc/passwd", "/"])
def test_path_outside_root_raises(root, bad):
    with pytest.raises(PathOutsideRootError):
        resolve_in_root(bad)


def test_sibling_with_same_prefix_is_outside(root):
    (root.parent / (root.name + "-other")).mkdir()
    with pytest.raises(PathOutsideRootError):
        resolve_in_root(f"../{root.name}-other")


def test_symlink_escaping_root_raises(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathOutsideRootError):
        resolve_in_root("link/file.txt")


@pytest.mark.parametrize("kind", RUN_KINDS)
def test_run_id_format(kind):
    run_id = new_run_id(kind)
    assert re.fullmatch(rf"{kind}-\d{{8}}T\d{{6}}Z-[0-9a-f]{{6}}", run_id)
    assert RUN_ID_RE.match(run_id)
    assert is_run_id(run_id, kind)


def test_unknown_run_kind_rejected():
    with pytest.raises(InvalidInputError):
        new_run_id("apply")


def test_run_ids_are_unique():
    assert len({new_run_id("scan") for _ in range(200)}) == 200


def test_save_result_writes_under_runs(root):
    run_id, result_path = save_result("scan", {"x": 1})
    assert result_path == f".portcheck/runs/{run_id}"
    assert (root / result_path / "result.json").is_file()
    assert run_dir(run_id, "scan") == root / result_path


def test_run_dir_rejects_traversal_and_wrong_kind(root):
    run_id, _ = save_result("scan", {})
    with pytest.raises(InvalidInputError):
        run_dir("../../etc", "scan")
    with pytest.raises(InvalidInputError):
        run_dir(run_id, "plan")
    with pytest.raises(NotFoundError):
        run_dir("scan-20200101T000000Z-abcdef", "scan")


def test_envelope_shape_on_success_and_failure(fixture_repo):
    ok = tools.scan_repo(".")
    assert set(ok) == {"ok", "run_id", "result_path", "summary", "data"}
    assert ok["ok"] is True and RUN_ID_RE.match(ok["run_id"])
    bad = tools.scan_repo("../outside")
    assert bad["ok"] is False and "data" not in bad
    assert bad["error"]["code"] == "path_outside_root"
    assert bad["error"]["message"]


def test_config_defaults_and_override(root):
    cfg = load_config()
    assert (
        cfg.target.rocm_image.startswith("rocm/pytorch:rocm") and cfg.target.torch_index_url == ""
    )
    assert cfg.parity.noise_multiplier == 3.0
    assert cfg.parity.max_abs_diff_floor == {"float32": 0.001, "float16": 0.05, "bfloat16": 0.05}
    assert cfg.bench.warmup_batches == 2 and cfg.bench.measured_batches == 5
    assert ".git" in cfg.scan.exclude and cfg.scan.max_file_bytes == 1_000_000
    (root / "portcheck.yaml").write_text(
        "target:\n  rocm_image: img:1\nparity:\n  max_abs_diff_floor:\n    float32: 0.5\n"
    )
    cfg = load_config()
    assert cfg.target.rocm_image == "img:1" and cfg.target.torch_index_url == ""
    assert cfg.parity.max_abs_diff_floor["float32"] == 0.5
    assert cfg.parity.max_abs_diff_floor["float16"] == 0.05  # deep merge keeps other keys


def test_invalid_config_is_invalid_input(root):
    (root / "portcheck.yaml").write_text("parity:\n  no_such_key: 1\n")
    with pytest.raises(InvalidInputError):
        load_config()
    assert tools.scan_repo(".")["error"]["code"] == "invalid_input"
