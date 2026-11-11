"""Each rule fires on a minimal positive sample and stays silent on a negative one."""

from __future__ import annotations

import pytest

from portcheck.audit.scanner import glob_match, load_rules, scan_file

POSITIVE = [
    ("PC001", "ext/op.cu", "__global__ void f() {}\n"),
    ("PC001", "include/op.cuh", "// header\n"),
    ("PC001", "setup.py", "from torch.utils.cpp_extension import CUDAExtension\n"),
    ("PC001", "CMakeLists.txt", "enable_language( CUDA)\n"),
    ("PC001", "cmake/deps.cmake", "find_package(CUDA REQUIRED)\n"),
    ("PC002", "a.py", "import tensorrt as trt\n"),
    ("PC002", "a.py", "from torch_tensorrt import compile\n"),
    ("PC002", "a.py", "import tensorrt_llm.runtime\n"),
    ("PC002", "requirements.txt", "tensorrt-cu12==10.0\n"),
    ("PC002", "requirements.txt", "torch_tensorrt\n"),
    ("PC003", "m.py", "from flash_attn import flash_attn_func\n"),
    ("PC003", "pyproject.toml", '[project]\ndependencies = ["flash-attn>=2"]\n'),
    ("PC004", "d.py", "from nvidia.dali import pipeline_def\n"),
    ("PC004", "d.py", "from nvidia import dali\n"),
    ("PC004", "d.py", "import cudf\n"),
    ("PC004", "d.py", "import cuml.cluster\n"),
    ("PC004", "d.py", "import cugraph\n"),
    ("PC004", "environment.yml", "dependencies:\n  - cudf=24.02\n"),
    ("PC004", "requirements.txt", "nvidia-dali-cuda120\n"),
    ("PC005", "t.py", "import torch.distributed as dist\n"),
    ("PC005", "t.py", "m = torch.nn.parallel.DistributedDataParallel(m)\n"),
    ("PC005", "run.sh", "export BACKEND=NCCL\n"),
    ("PC006", "q.py", "mode = 'FP8'\n"),
    ("PC006", "cfg/q.yaml", "quant: nvfp4\n"),
    ("PC006", "q.json", '{"dtype": "float8"}\n'),
    ("PC007", "Dockerfile", "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"),
    ("PC007", "docker/Dockerfile.gpu", "FROM nvcr.io/nvidia/pytorch:24.05-py3 AS base\n"),
    ("PC008", "requirements.txt", "nvidia-cudnn-cu12==9.1.0.70\n"),
    ("PC008", "requirements-gpu.txt", "--index-url https://download.pytorch.org/whl/cu121\n"),
    ("PC008", "Dockerfile", "RUN pip install torch -f https://download.pytorch.org/whl/cu118\n"),
    ("PC008", "pyproject.toml", '[project]\ndependencies = ["nvidia-cublas-cu12"]\n'),
    ("PC009", "g.py", "import pynvml\n"),
    ("PC009", "g.py", "from pynvml import nvmlInit\n"),
    ("PC009", "g.py", "subprocess.run(['nvidia-smi'])\n"),
    ("PC009", "gpu.sh", "nvidia-smi --query-gpu=name --format=csv\n"),
    ("PC010", "s.py", "ok = torch.cuda.is_available()\n"),
    ("PC010", "s.py", "model = model.cuda()\n"),
    ("PC010", "s.py", "device = 'cuda'\n"),
]

NEGATIVE = [
    ("PC001", "ext/op.cpp", "void f() {}\n"),
    ("PC001", "setup.py", "from setuptools import Extension\n"),
    ("PC001", "notes.txt", "find_package(CUDA)\n"),  # regex is limited to build files
    ("PC002", "a.py", "import tensorrt_helpers\nimport mytensorrt\n"),
    ("PC002", "requirements.txt", "torch\n"),
    ("PC003", "m.py", "import flash_attention_utils\n"),
    ("PC003", "requirements.txt", "flash-attn-utils\n"),
    ("PC004", "d.py", "import nvidia\nimport cudf_like\n"),
    ("PC004", "requirements.txt", "pandas\n"),
    ("PC005", "t.py", "import torch.nn as nn\nx = 'nccls'\n"),
    ("PC005", "notes.md", "DistributedDataParallel\n"),
    ("PC006", "q.py", "dtype = 'float32'\nname = 'fp8x'\n"),
    ("PC006", "q.txt", "fp8\n"),
    ("PC007", "Dockerfile", "FROM rocm/pytorch:latest\nRUN echo nvidia/cuda\n"),
    ("PC007", "docker.txt", "FROM nvidia/cuda:12\n"),
    ("PC008", "requirements.txt", "torch\npynvml\nnvidia-dali-cuda120\n"),  # dali belongs to PC004
    ("PC008", "requirements.txt", "--index-url https://download.pytorch.org/whl/cpu\n"),
    ("PC008", "notes.py", "URL = 'https://download.pytorch.org/whl/cu121'\n"),
    ("PC009", "g.py", "import pynvml_utils\n"),
    ("PC009", "g.py", "try:\n    import pynvml\nexcept ImportError:\n    pynvml = None\n"),
    ("PC009", "requirements.txt", "pynvml\n"),  # by design, no dependency rule for pynvml
    ("PC009", "notes.md", "run nvidia-smi\n"),
    ("PC010", "s.py", "device = 'cpu'\nimport torch\n"),
    ("PC010", "s.txt", "torch.cuda.is_available()\n"),
]


def _ids(path: str, text: str) -> list[str]:
    return [f.rule_id for f in scan_file(path, text)]


def test_catalog_has_the_ten_rules():
    rules = {r.id: r for r in load_rules()}
    assert sorted(rules) == [f"PC{n:03d}" for n in range(1, 11)]
    assert {i for i, r in rules.items() if r.severity == "red"} == {
        "PC001",
        "PC002",
        "PC003",
        "PC004",
    }
    assert {i for i, r in rules.items() if r.severity == "green"} == {"PC010"}
    assert {i: r.fix for i, r in rules.items() if r.fix == "auto"} == {
        "PC007": "auto",
        "PC008": "auto",
    }
    assert rules["PC010"].action == (
        "No change needed. ROCm builds of PyTorch use the same torch.cuda API."
    )
    assert "HIPIFY" in rules["PC001"].action and "SCALE" in rules["PC001"].action
    for rule in rules.values():
        if rule.severity == "red":
            assert rule.fix == "manual"


@pytest.mark.parametrize("rule_id", [f"PC{n:03d}" for n in range(1, 11)])
def test_every_rule_has_positive_and_negative_samples(rule_id):
    assert any(r == rule_id for r, _p, _t in POSITIVE)
    assert any(r == rule_id for r, _p, _t in NEGATIVE)


@pytest.mark.parametrize(("rule_id", "path", "text"), POSITIVE)
def test_rule_fires(rule_id, path, text):
    assert rule_id in _ids(path, text)


@pytest.mark.parametrize(("rule_id", "path", "text"), NEGATIVE)
def test_rule_stays_silent(rule_id, path, text):
    assert rule_id not in _ids(path, text)


def test_file_glob_rule_needs_no_readable_content():
    findings = scan_file("ext/op.cu", None)
    assert [(f.rule_id, f.line) for f in findings] == [("PC001", None)]


def test_pc009_fix_depends_on_how_it_matched():
    fix = {
        f.line: f.fix
        for f in scan_file(
            "g.py",
            "import pynvml\n"  # 1: plain top-level import -> auto
            "from pynvml import nvmlInit\n"  # 2: from-import -> manual
            "def f():\n"
            "    import pynvml\n"  # 4: not top-level -> manual
            "    run('nvidia-smi')\n",  # 5: regex -> manual
        )
        if f.rule_id == "PC009"
    }
    assert fix == {1: "auto", 2: "manual", 4: "manual", 5: "manual"}


def test_pc008_is_auto_only_in_requirements_files_and_dockerfiles():
    req = scan_file("requirements.txt", "nvidia-cudnn-cu12\n")
    proj = scan_file("pyproject.toml", '[project]\ndependencies = ["nvidia-cudnn-cu12"]\n')
    assert [f.fix for f in req] == ["auto"] and [f.fix for f in proj] == ["manual"]


def test_pc010_reports_only_the_first_match_per_file():
    text = "import torch\nx = torch.cuda.is_available()\ny = 'cuda'\nm.cuda()\n"
    findings = [f for f in scan_file("s.py", text) if f.rule_id == "PC010"]
    assert [(f.line, f.fix, f.severity) for f in findings] == [(2, "none", "green")]


def test_findings_are_deduplicated_per_rule_file_line():
    # Both PC005 matchers hit line 1; only one finding may come out.
    findings = scan_file("t.py", "import torch.distributed  # for DistributedDataParallel\n")
    assert [(f.rule_id, f.line) for f in findings] == [("PC005", 1)]


def test_glob_matching():
    assert glob_match("a/b/op.cu", "**/*.cu") and glob_match("op.cu", "**/*.cu")
    assert glob_match("deep/setup.py", "setup.py") and not glob_match("deep/setup.pyc", "setup.py")
    assert glob_match("x/requirements-dev.txt", "@requirements")
    assert glob_match("x/Dockerfile.gpu", "@dockerfile")
    assert not glob_match("a/b/op.cuh", "**/*.cu")
