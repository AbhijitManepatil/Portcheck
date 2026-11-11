"""Shared fixtures. Every test runs on CPU with no network access."""

from __future__ import annotations

import json
import random
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "examples" / "e2e-service"
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "e2e-service.findings.json"

TEST_ROCM_IMAGE = "registry.example.test/rocm-pytorch:test"
TEST_INDEX_URL = "https://wheels.example.test/whl/rocm-test"
TEST_CONFIG = f"target:\n  rocm_image: {TEST_ROCM_IMAGE}\n  torch_index_url: {TEST_INDEX_URL}\n"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt and braces: tell the Hugging Face libraries never to go online."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty root directory, selected through PORTCHECK_ROOT."""
    path = tmp_path / "root"
    path.mkdir()
    monkeypatch.setenv("PORTCHECK_ROOT", str(path))
    return path.resolve()


@pytest.fixture
def fixture_repo(root: Path) -> Path:
    """A copy of examples/e2e-service as the root. No portcheck.yaml yet."""
    shutil.copytree(FIXTURE_DIR, root, dirs_exist_ok=True)
    return root


@pytest.fixture
def test_config(root: Path) -> Path:
    path = root / "portcheck.yaml"
    path.write_text(TEST_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def slim(findings: list[dict]) -> list[dict]:
    keys = ("rule_id", "severity", "file", "line", "fix")
    return [{k: f[k] for k in keys} for f in findings]


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable sockets: any attempt to open one fails the test."""

    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted while sockets are disabled")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny randomly initialised GPT-2, built and saved locally (spec 11.1)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    tmp_dir = tmp_path_factory.mktemp("tiny-gpt2")
    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    GPT2LMHeadModel(cfg).save_pretrained(tmp_dir)
    return tmp_dir


@pytest.fixture
def tiny_workload(root: Path, tiny_model_dir: Path) -> str:
    """A workload in the root that points at the tiny model and uses input_ids prompts."""
    rng = random.Random(7)
    lines = [
        json.dumps({"input_ids": [rng.randint(2, 127) for _ in range(rng.randint(3, 9))]})
        for _ in range(6)
    ]
    (root / "tiny-prompts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "tiny-workload.yaml").write_text(
        f"model: {tiny_model_dir}\n"
        "revision: null\n"
        "task: text-generation\n"
        "dtype: float32\n"
        "prompts: tiny-prompts.jsonl\n"
        "max_new_tokens: 8\n"
        "batch_sizes: [1, 4]\n"
        "seed: 0\n",
        encoding="utf-8",
    )
    return "tiny-workload.yaml"


@pytest.fixture
def make_capture(root: Path):
    """Write a synthetic capture (no torch needed) and return its run id."""
    from portcheck.paths import create_run_dir
    from portcheck.schemas import BenchRow, Capture, Fingerprint
    from portcheck.validate.capture import write_capture

    def _make(
        label: str,
        logits: np.ndarray,
        ids: np.ndarray | None = None,
        sha: str = "a" * 64,
        dtype: str = "float32",
        fault: bool = False,
        tokens_per_second: float = 128.0,
        vendor: str = "cpu",
    ) -> str:
        logits = np.asarray(logits, dtype=np.float32)
        if ids is None:
            ids = np.tile(np.arange(4, dtype=np.int64), (logits.shape[0], 1))
        capture = Capture(
            label=label,
            workload={
                "model": "tiny",
                "revision": None,
                "task": "text-generation",
                "dtype": dtype,
                "prompts": "prompts.jsonl",
                "max_new_tokens": int(ids.shape[1]),
                "batch_sizes": [1, 4],
                "seed": 0,
            },
            workload_sha256=sha,
            fingerprint=Fingerprint(
                vendor=vendor,
                device_name=None if vendor == "cpu" else f"Test {vendor} GPU",
                device_count=0 if vendor == "cpu" else 1,
                torch_version="2.5.1",
                python_version="3.11.9",
                platform="Linux-test",
                portcheck_version="0.1.0",
            ),
            num_prompts=int(logits.shape[0]),
            fault_injected=fault,
            bench=[
                BenchRow(
                    batch_size=1,
                    tokens_generated=1600,
                    seconds=12.5,
                    tokens_per_second=tokens_per_second,
                    latency_p50_s=0.24,
                    latency_p95_s=0.31,
                ),
                BenchRow(
                    batch_size=4,
                    tokens_generated=6400,
                    seconds=16.0,
                    tokens_per_second=tokens_per_second * 3.125,
                    latency_p50_s=0.75,
                    latency_p95_s=0.82,
                ),
            ],
            peak_memory_bytes=None,
        )
        run_id, directory = create_run_dir("capture")
        write_capture(directory, capture, logits, ids)
        return run_id

    return _make


@pytest.fixture
def base_logits() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
