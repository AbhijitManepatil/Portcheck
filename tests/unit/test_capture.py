from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from safetensors.numpy import load_file

from portcheck import tools
from portcheck.envelope import InvalidInputError
from portcheck.validate.workload import load_workload


def test_capture_on_cpu_writes_both_files_with_specified_shapes(root, tiny_workload):
    env = tools.capture_run(tiny_workload, "ref")
    assert env["ok"], env
    assert env["run_id"].startswith("capture-")
    directory = root / env["result_path"]
    assert sorted(p.name for p in directory.iterdir()) == ["capture.json", "capture.safetensors"]

    meta = json.loads((directory / "capture.json").read_text())
    assert meta == env["data"]
    assert set(meta) == {
        "label",
        "workload",
        "workload_sha256",
        "fingerprint",
        "num_prompts",
        "fault_injected",
        "bench",
        "peak_memory_bytes",
    }
    assert meta["label"] == "ref" and meta["num_prompts"] == 6 and meta["fault_injected"] is False
    assert meta["fingerprint"]["vendor"] == "cpu" and meta["peak_memory_bytes"] is None
    assert meta["workload"]["task"] == "text-generation" and meta["workload"]["max_new_tokens"] == 8
    assert len(meta["workload_sha256"]) == 64

    tensors = load_file(str(directory / "capture.safetensors"))
    assert set(tensors) == {"last_logits", "generated_ids"}
    assert tensors["last_logits"].dtype == np.float32 and tensors["last_logits"].shape == (6, 128)
    assert tensors["generated_ids"].dtype == np.int64 and tensors["generated_ids"].shape == (6, 8)
    assert np.isfinite(tensors["last_logits"]).all()
    ids = tensors["generated_ids"]
    assert ((ids >= -1) & (ids < 128)).all()
    for row in ids:  # -1 appears only as padding after an end-of-sequence token (id 1)
        pads = np.flatnonzero(row == -1)
        if len(pads):
            assert (row[pads[0] :] == -1).all() and row[pads[0] - 1] == 1

    assert [b["batch_size"] for b in meta["bench"]] == [1, 4]
    for bench in meta["bench"]:
        assert set(bench) == {
            "batch_size",
            "tokens_generated",
            "seconds",
            "tokens_per_second",
            "latency_p50_s",
            "latency_p95_s",
        }
        # 5 measured batches of up to 8 new tokens per row.
        assert 0 < bench["tokens_generated"] <= 5 * bench["batch_size"] * 8
        assert bench["seconds"] > 0 and bench["tokens_per_second"] > 0
        assert 0 < bench["latency_p50_s"] <= bench["latency_p95_s"]


def test_last_logits_match_a_direct_forward_pass(root, tiny_workload, tiny_model_dir):
    import torch
    from transformers import GPT2LMHeadModel

    env = tools.capture_run(tiny_workload, "ref")
    tensors = load_file(str(root / env["result_path"] / "capture.safetensors"))
    prompts = [json.loads(line)["input_ids"] for line in (root / "tiny-prompts.jsonl").open()]
    model = GPT2LMHeadModel.from_pretrained(tiny_model_dir).eval()
    with torch.inference_mode():
        for index, ids in enumerate(prompts):
            tokens = torch.tensor([ids])
            expected = model(tokens).logits[0, -1].numpy()
            np.testing.assert_allclose(tensors["last_logits"][index], expected, atol=1e-5)
            greedy = model.generate(tokens, do_sample=False, max_new_tokens=8, pad_token_id=1)[
                0, len(ids) :
            ].numpy()
            np.testing.assert_array_equal(tensors["generated_ids"][index][: len(greedy)], greedy)


def test_repeat_captures_are_identical_and_fault_injection_changes_outputs(root, tiny_workload):
    a = tools.capture_run(tiny_workload, "a")
    b = tools.capture_run(tiny_workload, "b")
    bad = tools.capture_run(tiny_workload, "bad", inject_fault=0.5)
    assert a["ok"] and b["ok"] and bad["ok"]
    assert bad["data"]["fault_injected"] is True and "Fault injected" in bad["summary"]
    assert a["data"]["workload_sha256"] == b["data"]["workload_sha256"]

    def logits(env):
        return load_file(str(root / env["result_path"] / "capture.safetensors"))["last_logits"]

    np.testing.assert_array_equal(logits(a), logits(b))
    assert np.abs(logits(a) - logits(bad)).max() > 0.1


def test_non_float32_dtype_falls_back_on_cpu_with_a_warning(root, tiny_workload):
    path = root / tiny_workload
    path.write_text(path.read_text().replace("dtype: float32", "dtype: float16"))
    env = tools.capture_run(tiny_workload, "half")
    assert env["ok"] and "Warning" in env["summary"] and "float32" in env["summary"]
    assert env["data"]["workload"]["dtype"] == "float16"


def test_missing_dependency_names_the_run_extra(root, tiny_workload, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    env = tools.capture_run(tiny_workload, "x")
    assert env["ok"] is False and env["error"]["code"] == "missing_dependency"
    assert "portcheck[run]" in env["error"]["message"]


def test_capture_input_errors(root, tiny_workload):
    assert tools.capture_run("missing.yaml", "x")["error"]["code"] == "not_found"
    assert tools.capture_run(tiny_workload, "bad label")["error"]["code"] == "invalid_input"
    assert (
        tools.capture_run(tiny_workload, "x", inject_fault=-1)["error"]["code"] == "invalid_input"
    )
    assert tools.capture_run("../w.yaml", "x")["error"]["code"] == "path_outside_root"
    assert not (root / ".portcheck").exists()  # failed captures leave nothing behind


def _write(root, body: str, prompts: str = '{"input_ids": [2, 3]}\n') -> str:
    (root / "p.jsonl").write_text(prompts)
    (root / "w.yaml").write_text(body)
    return "w.yaml"


def test_workload_validation(root, tiny_model_dir):
    good = f"model: {tiny_model_dir}\nprompts: p.jsonl\n"
    loaded = load_workload(_write(root, good))
    assert loaded.model_is_local and loaded.prompt_form == "input_ids" and loaded.num_prompts == 1
    assert loaded.workload.task == "text-generation" and loaded.workload.dtype == "float32"

    bad_bodies = [
        good + "task: summarization\n",
        good + "dtype: float64\n",
        good + "batch_sizes: []\n",
        good + "batch_sizes: [0]\n",
        good + "max_new_tokens: 0\n",
        good + "unknown_key: 1\n",
        "prompts: p.jsonl\n",
        "- not\n- a mapping\n",
        "model: some-org/some-model\nprompts: p.jsonl\n",  # hub id without a revision
    ]
    for body in bad_bodies:
        with pytest.raises(InvalidInputError):
            load_workload(_write(root, body))

    bad_prompts = [
        "",
        "not json\n",
        '{"text": "a"}\n{"input_ids": [2]}\n',  # mixed forms
        '{"input_ids": []}\n',
        '{"input_ids": [2, -1]}\n',
        '{"text": ""}\n',
        '{"other": 1}\n',
    ]
    for prompts in bad_prompts:
        with pytest.raises(InvalidInputError):
            load_workload(_write(root, good, prompts))


def test_hub_id_with_revision_is_accepted_without_touching_the_network(root, no_network):
    loaded = load_workload(
        _write(root, "model: some-org/some-model\nrevision: abc123\nprompts: p.jsonl\n")
    )
    assert not loaded.model_is_local and loaded.model_ref == "some-org/some-model"


def test_workload_hash_covers_workload_and_prompts(root, tiny_model_dir):
    body = f"model: {tiny_model_dir}\nprompts: p.jsonl\n"
    first = load_workload(_write(root, body)).sha256
    assert load_workload(_write(root, body)).sha256 == first
    assert load_workload(_write(root, body + "seed: 1\n")).sha256 != first
    assert load_workload(_write(root, body, '{"input_ids": [2, 4]}\n')).sha256 != first


def test_fixture_workload_needs_a_person_to_choose_the_model(fixture_repo):
    env = tools.capture_run("workload.yaml", "x")
    assert env["ok"] is False and env["error"]["code"] == "invalid_input"
    assert "revision" in env["error"]["message"]
    prompts = [json.loads(line) for line in (fixture_repo / "prompts.jsonl").open()]
    assert len(prompts) == 20 and all(set(p) == {"text"} for p in prompts)
