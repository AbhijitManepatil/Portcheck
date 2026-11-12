from __future__ import annotations

import subprocess
import sys
import types

import portcheck
from portcheck import fingerprint as fp_mod
from portcheck import tools
from portcheck.fingerprint import env_fingerprint


def _fake_torch(available: bool, hip: str | None, cuda: str | None) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "9.9.9"
    torch.version = types.SimpleNamespace(hip=hip, cuda=cuda)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: 2,
        get_device_name=lambda _i: "Fake GPU",
    )
    return torch


def test_cpu_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    fp = env_fingerprint()
    assert fp.vendor == "cpu"
    assert fp.torch_version is None and fp.cuda_version is None and fp.rocm_version is None
    assert fp.device_name is None and fp.driver_version is None
    assert fp.portcheck_version == portcheck.__version__
    assert fp.python_version and fp.platform


def test_this_machine_is_cpu_and_envelope_ok():
    env = tools.env_fingerprint()
    assert env["ok"] is True and env["run_id"] is None
    assert env["data"]["vendor"] == "cpu"


def test_amd_detected_from_hip(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "6.2.0", None))
    monkeypatch.setattr(fp_mod, "_run", lambda cmd: "Driver version: 6.8.5\n")
    fp = env_fingerprint()
    assert (fp.vendor, fp.rocm_version, fp.device_count, fp.device_name) == (
        "amd",
        "6.2.0",
        2,
        "Fake GPU",
    )
    assert fp.driver_version == "6.8.5"


def test_nvidia_detected_from_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, "12.4"))
    monkeypatch.setattr(fp_mod, "_run", lambda cmd: "550.54.15\n550.54.15\n")
    fp = env_fingerprint()
    assert fp.vendor == "nvidia" and fp.cuda_version == "12.4"
    assert fp.driver_version == "550.54.15"


def test_gpu_build_without_available_device_is_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, None, "12.4"))
    assert env_fingerprint().vendor == "cpu"


def test_driver_lookup_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, "12.4"))

    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(subprocess, "run", boom)
    fp = env_fingerprint()
    assert fp.vendor == "nvidia" and fp.driver_version is None


def test_broken_torch_never_raises(monkeypatch):
    torch = _fake_torch(True, None, "12.4")
    torch.cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError("driver exploded"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert env_fingerprint().vendor == "cpu"


def test_amd_driver_falls_back_to_the_kernel_module_version(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "7.15.26333", None))
    monkeypatch.setattr(fp_mod, "_run", lambda cmd: None)  # rocm-smi is not installed
    version_file = tmp_path / "version"
    version_file.write_text("6.12.12\n")
    monkeypatch.setattr(fp_mod, "AMDGPU_VERSION_FILE", str(version_file))
    assert env_fingerprint().driver_version == "6.12.12"
    monkeypatch.setattr(fp_mod, "AMDGPU_VERSION_FILE", str(tmp_path / "missing"))
    fp = env_fingerprint()
    assert fp.vendor == "amd" and fp.driver_version is None
