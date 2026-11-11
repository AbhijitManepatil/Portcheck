from __future__ import annotations

from portcheck.audit import parsers


def test_requirements():
    text = (
        "# comment\n"
        "torch>=2.0\n"
        "Nvidia_CuDNN.cu12==9.1.0.70  # pinned\n"
        "--extra-index-url https://download.pytorch.org/whl/cu121\n"
        "-r other.txt\n"
        "\n"
        "flash-attn ; python_version > '3.9'\n"
        "tensorrt[extra]==10.0 \\\n"
        "    --hash=sha256:abc\n"
        "https://example.test/pkg.whl\n"
    )
    deps = parsers.parse_requirements(text)
    assert [(d.name, d.line) for d in deps] == [
        ("torch", 2),
        ("nvidia-cudnn-cu12", 3),
        ("flash-attn", 7),
        ("tensorrt", 8),
    ]


def test_pyproject_dependencies_and_optional():
    text = (
        "[project]\n"
        'name = "x"\n'
        "dependencies = [\n"
        '  "torch",\n'
        '  "nvidia-cublas-cu12>=12",\n'
        "]\n"
        "[project.optional-dependencies]\n"
        'gpu = ["Flash_Attn==2.5", "cudf-cu12"]\n'
    )
    deps = parsers.parse_pyproject(text)
    assert [(d.name, d.line) for d in deps] == [
        ("torch", 4),
        ("nvidia-cublas-cu12", 5),
        ("flash-attn", 8),
        ("cudf-cu12", 8),
    ]
    assert parsers.parse_pyproject("not [valid toml") == []
    assert parsers.parse_pyproject("[tool.x]\na = 1\n") == []


def test_environment_yml_with_nested_pip():
    text = (
        "name: env\n"
        "channels: [rapidsai]\n"
        "dependencies:\n"
        "  - python=3.11\n"
        "  - rapidsai::cudf=24.02\n"
        "  - pip:\n"
        "      - nvidia-dali-cuda120\n"
        "      - tensorrt==10.0\n"
    )
    deps = parsers.parse_environment_yml(text)
    assert [(d.name, d.line) for d in deps] == [
        ("python", 4),
        ("cudf", 5),
        ("nvidia-dali-cuda120", 7),
        ("tensorrt", 8),
    ]
    assert parsers.parse_environment_yml(": : bad") == []
    assert parsers.parse_environment_yml("dependencies: nope\n") == []


def test_dependency_file_dispatch():
    assert parsers.parse_dependencies("requirements-dev.txt", "cuml\n")[0].name == "cuml"
    assert parsers.parse_dependencies("environment.yml", "dependencies: [cuml]\n")[0].name == "cuml"
    assert parsers.parse_dependencies("notes.txt", "cuml\n") == []


def test_dockerfile_names():
    for name in ("Dockerfile", "Dockerfile.gpu", "serve.Dockerfile"):
        assert parsers.is_dockerfile(name)
    for name in ("dockerfile.txt", "Dockerfiles", "README.md"):
        assert not parsers.is_dockerfile(name)


def test_dockerfile_from_lines():
    text = (
        "# syntax=docker/dockerfile:1\n"
        "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base\n"
        "RUN echo FROM nowhere\n"
        "from --platform=linux/amd64 nvcr.io/nvidia/pytorch:24.05-py3\n"
        "FROM base\n"
    )
    froms = parsers.parse_dockerfile(text)
    assert [(f.line, f.image) for f in froms] == [
        (2, "nvidia/cuda:12.4.1-runtime-ubuntu22.04"),
        (4, "nvcr.io/nvidia/pytorch:24.05-py3"),
        (5, "base"),
    ]


def test_replace_from_image_keeps_flags_and_alias():
    line = "FROM --platform=linux/amd64 nvidia/cuda:12 AS build\n"
    assert parsers.replace_from_image(line, "rocm/x:1") == (
        "FROM --platform=linux/amd64 rocm/x:1 AS build\n"
    )


def test_python_imports_with_ast():
    text = (
        "import os, tensorrt\n"
        "import torch.distributed as dist\n"
        "from nvidia import dali\n"
        "from . import local\n"
        "import pynvml\n"
        "def f():\n"
        "    import cudf\n"
        "try:\n"
        "    import cuml\n"
        "except ImportError:\n"
        "    cuml = None\n"
        "try:\n"
        "    import cugraph\n"
        "except ValueError:\n"
        "    pass\n"
    )
    imports = {i.module: i for i in parsers.parse_python_imports(text)}
    assert set(imports) == {
        "os",
        "tensorrt",
        "torch.distributed",
        "nvidia",
        "pynvml",
        "cudf",
        "cuml",
        "cugraph",
    }
    assert imports["tensorrt"].line == 1 and not imports["tensorrt"].plain  # two names
    assert not imports["torch.distributed"].plain  # aliased
    assert imports["nvidia"].kind == "from" and "nvidia.dali" in imports["nvidia"].candidates
    assert imports["pynvml"].plain and imports["pynvml"].top_level and imports["pynvml"].line == 5
    assert not imports["cudf"].top_level and imports["cudf"].line == 7
    assert imports["cuml"].guarded and not imports["cuml"].plain
    assert not imports["cugraph"].guarded  # the handler does not catch ImportError


def test_python_imports_fall_back_on_syntax_error():
    text = (
        "import tensorrt\n"
        "def broken(:\n"
        "    from flash_attn import flash_attn_func\n"
        "x = 'import not_an_import'\n"
        "from .relative import thing\n"
    )
    imports = parsers.parse_python_imports(text)
    assert [(i.module, i.line, i.kind, i.top_level) for i in imports] == [
        ("tensorrt", 1, "import", True),
        ("flash_attn", 3, "from", False),
    ]
    assert not any(i.plain for i in imports)  # fallback results are never auto-fixable
    assert "flash_attn.flash_attn_func" in imports[1].candidates


def test_normalise_name():
    assert parsers.normalise_name("Nvidia_CuDNN.cu12") == "nvidia-cudnn-cu12"
