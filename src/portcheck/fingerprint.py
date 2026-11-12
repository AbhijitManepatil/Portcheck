"""Environment fingerprint (spec section 7.1). Never raises."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path

from portcheck import __version__
from portcheck.schemas import Fingerprint

_SMI_TIMEOUT_S = 5
AMDGPU_VERSION_FILE = "/sys/module/amdgpu/version"


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=False
        )
    except Exception:  # noqa: BLE001 - best effort by design
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _nvidia_driver() -> str | None:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[0] if lines else None


def _amd_driver() -> str | None:
    out = _run(["rocm-smi", "--showdriverversion"])
    if out:
        match = re.search(r"driver version\s*:\s*(\S+)", out, re.IGNORECASE)
        if match:
            return match.group(1)
    # PyTorch's ROCm wheels bundle the ROCm libraries, so rocm-smi is often not
    # installed. The kernel module reports its own version (seen on Azure NGads V620).
    try:
        version = Path(AMDGPU_VERSION_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return version or None


def env_fingerprint() -> Fingerprint:
    """Describe this machine: GPU vendor, library versions and driver."""
    vendor = "cpu"
    device_name: str | None = None
    device_count = 0
    torch_version: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    driver_version: str | None = None

    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optional; any import failure means "cpu"
        torch = None  # type: ignore[assignment]

    if torch is not None:
        try:
            torch_version = str(torch.__version__)
            rocm_version = getattr(torch.version, "hip", None)
            cuda_version = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                if rocm_version is not None:
                    vendor = "amd"
                elif cuda_version is not None:
                    vendor = "nvidia"
            if vendor != "cpu":
                device_count = int(torch.cuda.device_count())
                device_name = str(torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001 - never fail a fingerprint
            vendor, device_name, device_count = "cpu", None, 0

    if vendor == "nvidia":
        driver_version = _nvidia_driver()
    elif vendor == "amd":
        driver_version = _amd_driver()

    return Fingerprint(
        vendor=vendor,
        device_name=device_name,
        device_count=device_count,
        torch_version=torch_version,
        cuda_version=cuda_version,
        rocm_version=rocm_version,
        driver_version=driver_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        portcheck_version=__version__,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.stdout.write(env_fingerprint().model_dump_json(indent=2) + "\n")
