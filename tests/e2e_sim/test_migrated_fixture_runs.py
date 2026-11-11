"""The migrated fixture service really starts and answers a request (on CPU).

This is the CPU stand-in for the release check "Migrated fixture serves a request".
Before migration, `gpu_stats.py` imports pynvml unconditionally, so the service
cannot even start on a machine without NVML. After the PC009 auto-fix it can.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from portcheck import tools


def _serve(repo: Path, model_dir: Path, prompt: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, E2E_MODEL=str(model_dir), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("USE_TRT", None)
    return subprocess.run(
        [sys.executable, "serve.py", prompt],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_migrated_fixture_serves_a_request(fixture_repo, test_config, tiny_text_workload):
    model_dir = fixture_repo / "tiny-text-model"

    if importlib.util.find_spec("pynvml") is None:
        before = _serve(fixture_repo, model_dir, "the cat sat")
        assert before.returncode != 0
        assert "No module named 'pynvml'" in before.stderr

    scan = tools.scan_repo(".")
    plan = tools.plan_migration(scan["run_id"])
    for step in plan["data"]["steps"]:
        if step["kind"] != "auto":
            continue
        dry = tools.apply_step(step["step_id"], plan["run_id"])
        done = tools.apply_step(
            step["step_id"], plan["run_id"], dry_run=False, diff_sha256=dry["data"]["diff_sha256"]
        )
        assert done["ok"] and done["data"]["applied"] is True

    after = _serve(fixture_repo, model_dir, "the cat sat")
    assert after.returncode == 0, after.stderr
    assert after.stdout.strip()  # it printed generated text

    # The usage error of the service still works.
    no_args = subprocess.run(
        [sys.executable, "serve.py"], cwd=fixture_repo, capture_output=True, text=True, timeout=60
    )
    assert no_args.returncode == 2 and "usage" in no_args.stderr
