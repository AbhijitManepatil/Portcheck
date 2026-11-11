"""Config loading: packaged defaults overridden by ``<root>/portcheck.yaml`` (section 6.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portcheck.envelope import InvalidInputError
from portcheck.paths import get_root

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")
CONFIG_FILENAME = "portcheck.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetConfig(_Strict):
    rocm_image: str = ""
    torch_index_url: str = ""


class ParityConfig(_Strict):
    noise_multiplier: float = Field(gt=0)
    min_cosine: float
    min_top1_agreement: float
    max_abs_diff_floor: dict[str, float]


class BenchConfig(_Strict):
    warmup_batches: int = Field(ge=0)
    measured_batches: int = Field(ge=1)


class ScanConfig(_Strict):
    exclude: list[str]
    max_file_bytes: int = Field(gt=0)


class Config(_Strict):
    target: TargetConfig
    parity: ParityConfig
    bench: BenchConfig
    scan: ScanConfig


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(root: Path | None = None) -> Config:
    data = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    user_path = (root or get_root()) / CONFIG_FILENAME
    if user_path.is_file():
        try:
            user = yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise InvalidInputError(f"{CONFIG_FILENAME} is not valid YAML: {exc}") from exc
        if not isinstance(user, dict):
            raise InvalidInputError(f"{CONFIG_FILENAME} must contain a mapping at the top level.")
        data = _merge(data, user)
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise InvalidInputError(f"Invalid config in {CONFIG_FILENAME}: {problems}") from exc
