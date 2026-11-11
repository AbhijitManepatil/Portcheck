from __future__ import annotations

import pytest

from conftest import FIXTURE_DIR, slim
from portcheck import tools
from portcheck.audit import scan_repo
from portcheck.envelope import InvalidInputError, NotFoundError, PathOutsideRootError


def test_fixture_has_exactly_the_specified_files():
    files = sorted(
        p.relative_to(FIXTURE_DIR).as_posix() for p in FIXTURE_DIR.rglob("*") if p.is_file()
    )
    assert files == [
        "Dockerfile",
        "README.md",
        "ext/fast_op.cu",
        "gpu_stats.py",
        "prompts.jsonl",
        "requirements.txt",
        "serve.py",
        "trt_backend.py",
        "workload.yaml",
    ]
    assert (FIXTURE_DIR / "Dockerfile").read_text().splitlines()[0] == (
        "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04"
    )
    assert (FIXTURE_DIR / "requirements.txt").read_text().split() == [
        "torch",
        "transformers",
        "nvidia-cudnn-cu12==9.1.0.70",
        "pynvml",
    ]
    assert len((FIXTURE_DIR / "prompts.jsonl").read_text().splitlines()) == 20
    assert len((FIXTURE_DIR / "ext" / "fast_op.cu").read_text().splitlines()) == 10


def test_scan_equals_golden_exactly(fixture_repo, golden):
    result = scan_repo(".")
    assert slim([f.model_dump() for f in result.findings]) == golden
    assert result.grade == "red"
    assert result.counts == {"red": 2, "yellow": 3, "green": 1}
    assert result.files_scanned == 9 and result.root == "."


def test_scan_of_subdirectory_reports_paths_relative_to_it(root, golden):
    import shutil

    shutil.copytree(FIXTURE_DIR, root / "services" / "e2e")
    result = scan_repo("services/e2e")
    assert result.root == "services/e2e"
    assert slim([f.model_dump() for f in result.findings]) == golden


def test_findings_carry_evidence_and_action(fixture_repo):
    by_rule = {f.rule_id: f for f in scan_repo(".").findings}
    assert by_rule["PC007"].evidence == "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04"
    assert by_rule["PC008"].evidence == "nvidia-cudnn-cu12==9.1.0.70"
    assert by_rule["PC009"].evidence == "import pynvml"
    assert by_rule["PC001"].line is None
    assert all(f.action and f.title for f in by_rule.values())


def test_excluded_dirs_large_and_binary_files_are_skipped(fixture_repo):
    for name in (".git", ".venv", "node_modules", "__pycache__", ".portcheck"):
        (fixture_repo / name).mkdir(exist_ok=True)
        (fixture_repo / name / "bad.py").write_text("import tensorrt\n")
    (fixture_repo / "big.py").write_text("import tensorrt\n" + "#" * 1_000_001)
    (fixture_repo / "binary.py").write_bytes(b"import tensorrt\n\xff\xfe\x00")
    (fixture_repo / "blob.cu").write_bytes(b"\xff\xfe\x00")  # filename rule still applies
    result = scan_repo(".")
    files = [f.file for f in result.findings]
    assert not any(f.startswith((".git", ".venv", "node_modules", "__pycache__")) for f in files)
    assert "big.py" not in files and "binary.py" not in files
    assert "blob.cu" in files


def test_clean_repo_is_green(root):
    (root / "app.py").write_text("print('hello')\n")
    result = scan_repo(".")
    assert result.grade == "green" and result.findings == []
    assert result.counts == {"red": 0, "yellow": 0, "green": 0}


def test_scan_errors(root):
    with pytest.raises(PathOutsideRootError):
        scan_repo("..")
    with pytest.raises(NotFoundError):
        scan_repo("missing")
    (root / "file.txt").write_text("x")
    with pytest.raises(InvalidInputError):
        scan_repo("file.txt")
    assert tools.scan_repo("missing")["error"]["code"] == "not_found"


def test_scan_does_not_modify_the_repo(fixture_repo):
    before = {p: p.read_bytes() for p in fixture_repo.rglob("*") if p.is_file()}
    tools.scan_repo(".")
    after = {
        p: p.read_bytes()
        for p in fixture_repo.rglob("*")
        if p.is_file() and ".portcheck" not in p.parts
    }
    assert before == after
