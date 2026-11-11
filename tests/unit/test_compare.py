from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from portcheck import tools
from portcheck.validate.compare import compare_captures, diff_stats, greedy_stats, min_cosine


def test_identical_arrays_pass(root, make_capture, base_logits):
    ref, cand = make_capture("ref", base_logits), make_capture("cand", base_logits.copy())
    result, faulty = compare_captures(ref, cand)
    assert result.verdict == "pass" and faulty == []
    assert result.metrics.max_abs_diff == 0.0 and result.metrics.mean_rel_error == 0.0
    assert result.metrics.min_cosine == 1.0 and result.metrics.top1_agreement == 1.0
    assert result.metrics.greedy_exact_match_rate == 1.0
    assert result.metrics.mean_matching_prefix_tokens == 4.0
    assert result.noise_floor.max_abs_diff == 0.0 and result.baseline_repeat is None
    assert [c.metric for c in result.checks] == ["max_abs_diff", "min_cosine", "top1_agreement"]
    assert all(c.passed for c in result.checks)
    assert (result.reference, result.candidate, result.dtype) == (ref, cand, "float32")


def test_large_perturbation_fails(root, make_capture, base_logits):
    noisy = base_logits + np.random.default_rng(1).normal(scale=2.0, size=base_logits.shape)
    result, _ = compare_captures(make_capture("ref", base_logits), make_capture("bad", noisy))
    assert result.verdict == "fail"
    assert {c.metric for c in result.checks if not c.passed} >= {"max_abs_diff", "min_cosine"}


def test_small_perturbation_below_floor_passes(root, make_capture, base_logits):
    result, _ = compare_captures(
        make_capture("ref", base_logits), make_capture("cand", base_logits + 0.0004)
    )
    assert result.verdict == "pass"
    assert result.metrics.max_abs_diff == pytest.approx(0.0004, rel=1e-3)
    assert result.thresholds.max_abs_diff == 0.001


def test_mismatched_workload_hashes_return_capture_mismatch(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits, sha="a" * 64)
    cand = make_capture("cand", base_logits, sha="b" * 64)
    env = tools.compare_captures(ref, cand)
    assert env["ok"] is False and env["error"]["code"] == "capture_mismatch"
    same = make_capture("same", base_logits, sha="a" * 64)
    assert tools.compare_captures(ref, same, baseline_repeat=cand)["error"]["code"] == (
        "capture_mismatch"
    )


def test_mismatched_shapes_return_capture_mismatch(root, make_capture, base_logits):
    env = tools.compare_captures(
        make_capture("ref", base_logits), make_capture("cand", base_logits[:5])
    )
    assert env["error"]["code"] == "capture_mismatch"


def test_threshold_is_max_of_multiplier_times_noise_and_floor(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits)
    # Noise 0.01 -> threshold 3 * 0.01 = 0.03, above the float32 floor of 0.001.
    noisy_repeat = make_capture("repeat", base_logits + 0.01)
    cand_in = make_capture("cand", base_logits + 0.02)
    cand_out = make_capture("cand2", base_logits + 0.05)
    result, _ = compare_captures(ref, cand_in, baseline_repeat=noisy_repeat)
    assert result.noise_floor.max_abs_diff == pytest.approx(0.01, rel=1e-3)
    assert result.noise_floor.mean_rel_error > 0
    assert result.thresholds.max_abs_diff == pytest.approx(0.03, rel=1e-3)
    assert result.thresholds.max_abs_diff == pytest.approx(3.0 * result.noise_floor.max_abs_diff)
    assert result.checks[0].passed and result.baseline_repeat == noisy_repeat
    assert not compare_captures(ref, cand_out, baseline_repeat=noisy_repeat)[0].checks[0].passed
    # Noise 0.0001 -> 3 * 0.0001 is below the floor, so the floor wins.
    quiet_repeat = make_capture("repeat2", base_logits + 0.0001)
    result, _ = compare_captures(ref, cand_in, baseline_repeat=quiet_repeat)
    assert result.thresholds.max_abs_diff == 0.001
    assert not result.checks[0].passed  # 0.02 is now outside the threshold
    # Without a repeat the same candidate is also judged against the floor.
    assert compare_captures(ref, cand_in)[0].thresholds.max_abs_diff == 0.001


def test_floor_depends_on_dtype_and_config(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits, dtype="float16")
    cand = make_capture("cand", base_logits + 0.01, dtype="float16")
    result, _ = compare_captures(ref, cand)
    assert result.dtype == "float16" and result.thresholds.max_abs_diff == 0.05
    assert result.checks[0].passed
    (root / "portcheck.yaml").write_text(
        "parity:\n  noise_multiplier: 10.0\n  max_abs_diff_floor:\n    float16: 0.001\n"
    )
    result, _ = compare_captures(ref, cand)
    assert result.thresholds.max_abs_diff == 0.001 and result.verdict == "fail"


def test_greedy_metrics_never_gate_the_verdict(root, make_capture, base_logits):
    ids_a = np.tile(np.arange(8, dtype=np.int64), (10, 1))
    ids_b = ids_a.copy()
    ids_b[:, 2:] = 99  # every row diverges at the third token
    result, _ = compare_captures(
        make_capture("ref", base_logits, ids=ids_a), make_capture("cand", base_logits, ids=ids_b)
    )
    assert result.metrics.greedy_exact_match_rate == 0.0
    assert result.metrics.mean_matching_prefix_tokens == 2.0
    assert result.verdict == "pass"
    assert "greedy_exact_match_rate" not in [c.metric for c in result.checks]


def test_top1_flip_fails(root, make_capture, base_logits):
    flipped = base_logits.copy()
    row = flipped[0]
    row[int(np.argsort(row)[-2])] = row.max() + 0.0005  # tiny change that flips the argmax
    result, _ = compare_captures(make_capture("ref", base_logits), make_capture("cand", flipped))
    assert result.metrics.top1_agreement == 0.9
    assert [c.passed for c in result.checks if c.metric == "top1_agreement"] == [False]
    assert result.verdict == "fail"


def test_fault_injection_is_mentioned_in_the_summary(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits)
    bad = make_capture("bad", base_logits + 5.0, fault=True)
    env = tools.compare_captures(ref, bad)
    assert env["ok"] and env["data"]["verdict"] == "fail"
    assert "Fault injected" in env["summary"] and bad in env["summary"]
    assert "Fault injected" not in tools.compare_captures(ref, ref)["summary"]


def test_non_finite_logits_fail_and_stay_valid_json(root, make_capture, base_logits):
    broken = base_logits.copy()
    broken[0, 0] = np.nan
    env = tools.compare_captures(make_capture("ref", base_logits), make_capture("nan", broken))
    assert env["ok"] and env["data"]["verdict"] == "fail"
    import json

    json.loads(json.dumps(env, allow_nan=False))


def test_capture_can_be_named_by_directory_path(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits)
    cand = make_capture("cand", base_logits)
    shutil.copytree(root / ".portcheck" / "runs" / cand, root / "copied" / "from-amd")
    env = tools.compare_captures(ref, "copied/from-amd")
    assert env["ok"] and env["data"]["candidate"] == "copied/from-amd"
    assert env["run_id"].startswith("compare-")
    assert (root / env["result_path"] / "result.json").is_file()
    assert tools.compare_captures(ref, "../elsewhere")["error"]["code"] == "path_outside_root"
    assert tools.compare_captures(ref, "no/such/dir")["error"]["code"] == "not_found"
    assert tools.compare_captures(ref, "capture-20200101T000000Z-abcdef")["error"]["code"] == (
        "not_found"
    )


def test_metric_helpers():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert diff_stats(a, b) == (2.0, pytest.approx((4.0 / 4) / (3.0 / 4)))
    assert min_cosine(a, b) == 0.0 and min_cosine(a, a) == pytest.approx(1.0)
    assert min_cosine(np.zeros((1, 2)), np.zeros((1, 2))) == 1.0
    ids = np.array([[1, 2, 3], [1, 2, 3]])
    other = np.array([[1, 2, 3], [9, 2, 3]])
    assert greedy_stats(ids, other) == (0.5, 1.5)


def test_compare_module_does_not_import_torch():
    code = (
        "import sys\n"
        "import portcheck.validate.compare\n"
        "import portcheck.cost, portcheck.report, portcheck.tools, portcheck.cli\n"
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
