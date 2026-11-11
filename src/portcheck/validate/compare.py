"""``compare_captures`` (spec section 8.6). Needs only numpy and safetensors."""

from __future__ import annotations

import numpy as np

from portcheck.config import Config, load_config
from portcheck.envelope import CaptureMismatchError, ConfigMissingError, InvalidInputError
from portcheck.paths import get_root
from portcheck.schemas import (
    Check,
    CompareMetrics,
    CompareResult,
    NoiseFloor,
    Thresholds,
)
from portcheck.validate.capture import load_capture, load_tensors

_HUGE = 1e30  # stands in for "not finite" so results stay valid JSON


def _sig(value: float, digits: int = 8) -> float:
    """Round to significant digits so stored values and the report agree exactly."""
    value = float(value)
    if not np.isfinite(value):
        return _HUGE
    return float(f"{value:.{digits}g}")


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(max_abs_diff, mean_rel_error)`` between two logit arrays."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    if not (np.isfinite(a64).all() and np.isfinite(b64).all()):
        return _HUGE, _HUGE
    diff = np.abs(a64 - b64)
    denom = float(np.mean(np.abs(a64)))
    mean_diff = float(np.mean(diff))
    rel = mean_diff / denom if denom > 0 else (0.0 if mean_diff == 0 else _HUGE)
    return float(np.max(diff)), rel


def min_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    if not (np.isfinite(a64).all() and np.isfinite(b64).all()):
        return -1.0
    dot = np.sum(a64 * b64, axis=1)
    norms = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    both_zero = (np.linalg.norm(a64, axis=1) == 0) & (np.linalg.norm(b64, axis=1) == 0)
    cos = np.where(norms > 0, dot / np.where(norms > 0, norms, 1.0), np.where(both_zero, 1.0, 0.0))
    return float(np.clip(np.min(cos), -1.0, 1.0))


def greedy_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(exact_match_rate, mean_matching_prefix_tokens)`` for two id matrices."""
    equal = a == b
    exact = float(np.mean(np.all(equal, axis=1)))
    # Length of the common prefix: index of the first mismatch, or the row width.
    first_mismatch = np.where(equal.all(axis=1), equal.shape[1], np.argmin(equal, axis=1))
    return exact, float(np.mean(first_mismatch))


def _check_same_workload(ref_name: str, ref, other_name: str, other, tensors_a, tensors_b) -> None:
    if ref.workload_sha256 != other.workload_sha256:
        raise CaptureMismatchError(
            f"Captures '{ref_name}' and '{other_name}' were made from different workloads "
            "(workload_sha256 differs). Capture both with the same workload and prompts files."
        )
    for key in ("last_logits", "generated_ids"):
        if tensors_a[key].shape != tensors_b[key].shape:
            raise CaptureMismatchError(
                f"Captures '{ref_name}' and '{other_name}' have different '{key}' shapes "
                f"({tensors_a[key].shape} and {tensors_b[key].shape}); they do not come from "
                "the same model and workload."
            )


def compare_captures(
    reference: str,
    candidate: str,
    baseline_repeat: str | None = None,
    config: Config | None = None,
) -> tuple[CompareResult, list[str]]:
    """Compare two captures. Returns the result and the labels of fault-injected captures."""
    config = config or load_config(get_root())
    ref_name, ref, ref_dir = load_capture(reference)
    cand_name, cand, cand_dir = load_capture(candidate)
    ref_t, cand_t = load_tensors(ref_dir), load_tensors(cand_dir)
    _check_same_workload(ref_name, ref, cand_name, cand, ref_t, cand_t)

    a, b = ref_t["last_logits"], cand_t["last_logits"]
    max_abs, rel = diff_stats(a, b)
    top1 = float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))
    exact, prefix = greedy_stats(ref_t["generated_ids"], cand_t["generated_ids"])
    metrics = CompareMetrics(
        max_abs_diff=_sig(max_abs),
        mean_rel_error=_sig(rel),
        min_cosine=_sig(min_cosine(a, b)),
        top1_agreement=_sig(top1),
        greedy_exact_match_rate=_sig(exact),
        mean_matching_prefix_tokens=_sig(prefix),
    )

    faulty = [name for name, cap in ((ref_name, ref), (cand_name, cand)) if cap.fault_injected]
    noise = NoiseFloor()
    repeat_name: str | None = None
    if baseline_repeat is not None:
        repeat_name, repeat, repeat_dir = load_capture(baseline_repeat)
        repeat_t = load_tensors(repeat_dir)
        _check_same_workload(ref_name, ref, repeat_name, repeat, ref_t, repeat_t)
        if not (np.isfinite(a).all() and np.isfinite(repeat_t["last_logits"]).all()):
            # A non-finite noise floor would lift the threshold and let anything pass.
            raise InvalidInputError(
                f"The noise floor cannot be measured: '{ref_name}' or '{repeat_name}' contains "
                "NaN or infinite logits. Capture the reference and the repeat again."
            )
        noise_max, noise_rel = diff_stats(a, repeat_t["last_logits"])
        noise = NoiseFloor(max_abs_diff=_sig(noise_max), mean_rel_error=_sig(noise_rel))
        if repeat.fault_injected:
            faulty.append(repeat_name)

    dtype = str(ref.workload.get("dtype", "float32"))
    floors = config.parity.max_abs_diff_floor
    if dtype not in floors:
        raise ConfigMissingError(
            f"parity.max_abs_diff_floor has no entry for dtype '{dtype}'. Add one in portcheck.yaml."
        )
    thresholds = Thresholds(
        max_abs_diff=_sig(max(config.parity.noise_multiplier * noise.max_abs_diff, floors[dtype])),
        min_cosine=config.parity.min_cosine,
        min_top1_agreement=config.parity.min_top1_agreement,
    )
    checks = [
        Check(
            metric="max_abs_diff",
            value=metrics.max_abs_diff,
            threshold=thresholds.max_abs_diff,
            passed=metrics.max_abs_diff <= thresholds.max_abs_diff,
        ),
        Check(
            metric="min_cosine",
            value=metrics.min_cosine,
            threshold=thresholds.min_cosine,
            passed=metrics.min_cosine >= thresholds.min_cosine,
        ),
        Check(
            metric="top1_agreement",
            value=metrics.top1_agreement,
            threshold=thresholds.min_top1_agreement,
            passed=metrics.top1_agreement >= thresholds.min_top1_agreement,
        ),
    ]
    result = CompareResult(
        reference=ref_name,
        candidate=cand_name,
        baseline_repeat=repeat_name,
        dtype=dtype,
        metrics=metrics,
        noise_floor=noise,
        thresholds=thresholds,
        checks=checks,
        verdict="pass" if all(c.passed for c in checks) else "fail",
    )
    return result, faulty
