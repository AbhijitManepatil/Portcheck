"""Regression tests for defects found by an independent review of v0.1."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from conftest import TEST_INDEX_URL, TEST_ROCM_IMAGE
from portcheck import tools
from portcheck.audit import parsers
from portcheck.audit.scanner import scan_file
from portcheck.config import load_config
from portcheck.migrate import fixes
from portcheck.report import _code


def _pc(findings, rule):
    return [(f.line, f.fix, f.evidence) for f in findings if f.rule_id == rule]


# ---- 1. line numbers with form feeds and other non-newline separators ----


@pytest.mark.parametrize("sep", ["\x0c", "\x0b", "\x1c", "\x85", "\u2028"])
def test_only_real_newlines_count_as_line_breaks(sep):
    assert parsers.split_lines(f"a{sep}b\nc\r\nd\re") == [f"a{sep}b", "c", "d", "e"]
    assert parsers.split_lines("a\nb\n", keepends=True) == ["a\n", "b\n"]
    assert parsers.split_lines("") == [] and parsers.split_lines("x") == ["x"]
    text = f"# section{sep}\nFROM python:3\nFROM nvidia/cuda:12\n"
    assert _pc(scan_file("Dockerfile", text), "PC007") == [(3, "auto", "FROM nvidia/cuda:12")]


def test_pc009_fix_edits_the_right_line_after_a_form_feed(root):
    text = "# section\x0c\nimport os\nimport pynvml\nprint(os.getcwd())\n"
    assert _pc(scan_file("g.py", text), "PC009") == [(3, "auto", "import pynvml")]
    fixed = fixes.fix_pc009(text, "g.py", load_config())
    assert fixed.startswith("# section\x0c\nimport os\ntry:\n    import pynvml\n")
    assert fixed.endswith("    pynvml = None\nprint(os.getcwd())\n")
    compile(fixed, "g.py", "exec")

    literal = "s = 'a\u2028b'\nimport pynvml\n"
    fixed = fixes.fix_pc009(literal, "g.py", load_config())
    assert fixed.startswith("s = 'a\u2028b'\ntry:\n")
    compile(fixed, "g.py", "exec")


# ---- 2. an import that shares its line with the end of another statement ----


def test_import_after_a_multiline_statement_is_manual_and_untouched(root):
    text = "x = (1,\n     2); import pynvml\n"
    assert [fix for _line, fix, _e in _pc(scan_file("g.py", text), "PC009")] == ["manual"]
    assert fixes.fix_pc009(text, "g.py", load_config()) == text


# ---- 3. PC008 rewrites only a bare index URL ----


def test_pc008_does_not_rewrite_urls_that_point_at_a_file(root, test_config):
    cfg = load_config()
    text = (
        "--find-links https://download.pytorch.org/whl/cu121/torch_stable.html\n"
        "torch @ https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp311-linux_x86_64.whl\n"
    )
    assert fixes.fix_pc008(text, "requirements.txt", cfg) == text
    assert [fix for _l, fix, _e in _pc(scan_file("requirements.txt", text), "PC008")] == [
        "manual",
        "manual",
    ]


def test_pc008_rewrites_bare_index_urls_and_keeps_the_slash(root, test_config):
    cfg = load_config()
    text = (
        "--index-url https://download.pytorch.org/whl/cu121\n"
        "--extra-index-url https://download.pytorch.org/whl/cu118/\n"
        "torch\n"
    )
    assert fixes.fix_pc008(text, "requirements.txt", cfg) == (
        f"--index-url {TEST_INDEX_URL}\n--extra-index-url {TEST_INDEX_URL}/\ntorch\n"
    )
    assert [fix for _l, fix, _e in _pc(scan_file("requirements.txt", text), "PC008")] == [
        "auto",
        "auto",
    ]
    docker = 'RUN pip install torch --index-url "https://download.pytorch.org/whl/cu121"\n'
    assert fixes.fix_pc008(docker, "Dockerfile", cfg) == (
        f'RUN pip install torch --index-url "{TEST_INDEX_URL}"\n'
    )
    assert _pc(scan_file("Dockerfile", docker), "PC008")[0][1] == "auto"


def test_scanner_and_pc008_fix_agree(root, test_config):
    cfg = load_config()
    for line in (
        "--index-url https://download.pytorch.org/whl/cu121",
        "--index-url https://download.pytorch.org/whl/cu121/",
        "-f https://download.pytorch.org/whl/cu121/torch_stable.html",
        "torch @ https://download.pytorch.org/whl/cu118/torch-2.0.0.whl",
        "--index-url 'https://download.pytorch.org/whl/cu121'",
    ):
        text = line + "\n"
        auto = _pc(scan_file("requirements.txt", text), "PC008")[0][1] == "auto"
        assert (fixes.fix_pc008(text, "requirements.txt", cfg) != text) is auto, line


# ---- 4. a symlinked .portcheck must not take results outside the root ----


def test_symlinked_results_folder_is_refused(fixture_repo, tmp_path, make_capture, base_logits):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (fixture_repo / ".portcheck").symlink_to(outside, target_is_directory=True)
    for env in (
        tools.scan_repo("."),
        tools.plan_migration("scan-20200101T000000Z-abcdef"),
        tools.compare_captures(
            "capture-20200101T000000Z-abcdef", "capture-20200101T000000Z-abcdef"
        ),
        tools.build_report(scan_id="scan-20200101T000000Z-abcdef"),
        tools.list_runs(),
    ):
        assert env["ok"] is False and env["error"]["code"] == "path_outside_root", env
    assert list(outside.iterdir()) == []


def test_symlinked_run_folder_is_refused(root, tmp_path, make_capture, base_logits):
    cap = make_capture("ref", base_logits)
    runs = root / ".portcheck" / "runs"
    moved = tmp_path / "moved-capture"
    shutil.move(str(runs / cap), moved)
    (runs / cap).symlink_to(moved, target_is_directory=True)
    assert tools.estimate_cost(cap, 1.0)["error"]["code"] == "path_outside_root"
    assert tools.compare_captures(cap, cap)["error"]["code"] == "path_outside_root"
    assert tools.list_runs()["data"]["runs"] == []


# ---- 5. a non-finite repeat capture must not lift the threshold ----


def test_non_finite_repeat_capture_is_an_error(root, make_capture, base_logits):
    ref = make_capture("ref", base_logits)
    cand = make_capture("cand", base_logits * 1.02)
    broken = base_logits.copy()
    broken[0, 0] = np.nan
    repeat = make_capture("repeat", broken)
    assert tools.compare_captures(ref, cand)["data"]["verdict"] == "fail"
    env = tools.compare_captures(ref, cand, baseline_repeat=repeat)
    assert env["ok"] is False and env["error"]["code"] == "invalid_input"
    assert "noise floor" in env["error"]["message"]


# ---- 6. one pathological file must not abort the scan ----


def test_pathological_files_do_not_abort_the_scan(root):
    (root / "environment.yml").write_text("dependencies: " + "[" * 600 + "\n")
    (root / "pyproject.toml").write_text("x = " + "[" * 3000 + "\n")
    chain = "if a:\n    pass\n" + "".join(f"elif a == {i}:\n    pass\n" for i in range(1500))
    (root / "elif.py").write_text(chain + "import tensorrt\n")
    (root / "minus.py").write_text("x = " + "-" * 100_000 + "1\nimport tensorrt\n")
    (root / "ok.py").write_text("import tensorrt\n")
    env = tools.scan_repo(".")
    assert env["ok"], env
    files = {f["file"] for f in env["data"]["findings"] if f["rule_id"] == "PC002"}
    assert files == {"elif.py", "minus.py", "ok.py"}


# ---- 7. byte-order mark in dependency files ----


def test_bom_does_not_hide_the_first_requirement(root):
    (root / "requirements.txt").write_bytes(b"\xef\xbb\xbfnvidia-cudnn-cu12==9\n")
    findings = tools.scan_repo(".")["data"]["findings"]
    assert [(f["rule_id"], f["line"]) for f in findings] == [("PC008", 1)]


# ---- 8 and 9. PC009 details ----


def test_submodule_import_is_manual(root):
    text = "import pynvml.smi\n"
    assert [fix for _l, fix, _e in _pc(scan_file("g.py", text), "PC009")] == ["manual"]
    assert fixes.fix_pc009(text, "g.py", load_config()) == text


def test_trailing_comment_on_the_import_is_kept(root):
    fixed = fixes.fix_pc009("import pynvml  # noqa: F401\nx = 1\n", "g.py", load_config())
    assert fixed == (
        "try:\n"
        "    import pynvml  # noqa: F401\n"
        "except ImportError:  # not available on non-NVIDIA machines\n"
        "    pynvml = None\n"
        "x = 1\n"
    )


# ---- 10. Dockerfile parsing ----


def test_from_inside_heredocs_and_continuations_is_not_an_instruction(root, test_config):
    text = (
        "FROM python:3.12\n"
        "RUN cat <<EOF > /note.txt\n"
        "FROM nvidia/cuda:1\n"
        "EOF\n"
        "RUN echo start \\\n"
        "  # a comment inside the continued command\n"
        " FROM nvidia/cuda:2\n"
        "COPY <<-'DONE' /x\n"
        "\tFROM nvcr.io/nvidia/pytorch:3\n"
        "\tDONE\n"
        "FROM nvidia/cuda:12 AS real\n"
    )
    assert [(f.line, f.image) for f in parsers.parse_dockerfile(text)] == [
        (1, "python:3.12"),
        (11, "nvidia/cuda:12"),
    ]
    assert _pc(scan_file("Dockerfile", text), "PC007") == [
        (11, "auto", "FROM nvidia/cuda:12 AS real")
    ]
    fixed = fixes.fix_pc007(text, "Dockerfile", load_config())
    assert fixed == text.replace("FROM nvidia/cuda:12 AS real", f"FROM {TEST_ROCM_IMAGE} AS real")


def test_arg_based_and_registry_prefixed_images(root, test_config):
    text = (
        "ARG BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
        'ARG OTHER="python:3.12"\n'
        "FROM ${BASE} AS a\n"
        "FROM $OTHER AS b\n"
        "FROM docker.io/nvidia/cuda:12 AS c\n"
        "FROM ${UNSET:-nvidia/cuda:1}\n"
    )
    assert _pc(scan_file("Dockerfile", text), "PC007") == [
        (3, "manual", "FROM ${BASE} AS a"),  # found through the ARG default, not auto-fixed
        (5, "auto", "FROM docker.io/nvidia/cuda:12 AS c"),
    ]
    fixed = fixes.fix_pc007(text, "Dockerfile", load_config())
    assert fixed == text.replace("docker.io/nvidia/cuda:12", TEST_ROCM_IMAGE)


# ---- 11. decoding is greedy whatever the model's generation config says ----


def test_capture_ignores_beam_search_and_penalties_in_the_model_config(root, tiny_workload):
    from transformers import GenerationConfig

    plain = tools.capture_run(tiny_workload, "plain")
    workload = root / tiny_workload
    model_dir = next(
        line.split(": ", 1)[1]
        for line in workload.read_text().splitlines()
        if line.startswith("model:")
    )
    tuned = root / "tuned-model"
    shutil.copytree(model_dir, tuned)
    GenerationConfig(
        bos_token_id=0, eos_token_id=1, num_beams=3, repetition_penalty=5.0, no_repeat_ngram_size=2
    ).save_pretrained(tuned)
    workload.write_text(workload.read_text().replace(f"model: {model_dir}", "model: tuned-model"))
    other = tools.capture_run(tiny_workload, "tuned")
    assert plain["ok"] and other["ok"], (plain, other)

    a = load_file(str(root / plain["result_path"] / "capture.safetensors"))
    b = load_file(str(root / other["result_path"] / "capture.safetensors"))
    np.testing.assert_array_equal(a["generated_ids"], b["generated_ids"])
    np.testing.assert_array_equal(a["last_logits"], b["last_logits"])
    assert [r["tokens_generated"] for r in plain["data"]["bench"]] == [
        r["tokens_generated"] for r in other["data"]["bench"]
    ]


# ---- 12. a comment line never continues onto the next requirement ----


def test_comment_ending_in_a_backslash_does_not_swallow_the_next_line(root, test_config):
    text = "# note \\\nnvidia-cudnn-cu12==9\ntorch\n"
    assert [(d.name, d.line) for d in parsers.parse_requirements(text)] == [
        ("nvidia-cudnn-cu12", 2),
        ("torch", 3),
    ]
    assert fixes.fix_pc008(text, "requirements.txt", load_config()) == "# note \\\ntorch\n"


# ---- 13 and 14. report robustness ----


def test_report_quotes_shell_arguments_and_keeps_backticks(root):
    odd = root / "my dir; `x`"
    odd.mkdir()
    (odd / "a|b`c.py").write_text("import tensorrt\n")
    scan = tools.scan_repo("my dir; `x`")
    assert scan["ok"], scan
    report = tools.build_report(scan_id=scan["run_id"])
    markdown = (root / report["data"]["report_file"]).read_text()
    assert "portcheck scan 'my dir; `x`'" in markdown
    assert "`` a\\|b`c.py ``" in markdown
    assert _code("plain") == "`plain`" and _code("a`b") == "`` a`b ``"


def test_report_tolerates_a_damaged_capture(root, make_capture, base_logits):
    ref, cand = make_capture("ref", base_logits), make_capture("cand", base_logits)
    compare = tools.compare_captures(ref, cand)
    path = root / ".portcheck" / "runs" / cand / "capture.json"
    data = json.loads(path.read_text())
    del data["fingerprint"]
    path.write_text(json.dumps(data))
    report = tools.build_report(compare_id=compare["run_id"])
    assert report["ok"], report
    markdown = (root / report["data"]["report_file"]).read_text()
    assert "## Parity" in markdown and "`ref`" in markdown


# ---- 15. malformed capture tensors ----


@pytest.mark.parametrize(
    ("logits", "ids"),
    [
        (np.zeros((0, 8), np.float32), np.zeros((0, 4), np.int64)),  # no prompts
        (np.zeros((8,), np.float32), np.zeros((1, 4), np.int64)),  # 1-D logits
        (np.zeros((2, 8), np.float32), np.zeros((3, 4), np.int64)),  # row counts differ
    ],
)
def test_malformed_capture_tensors_are_invalid_input(root, make_capture, base_logits, logits, ids):
    good = make_capture("good", base_logits)
    bad = make_capture("bad", base_logits)
    save_file(
        {"last_logits": logits, "generated_ids": ids},
        str(root / ".portcheck" / "runs" / bad / "capture.safetensors"),
    )
    for env in (tools.compare_captures(good, bad), tools.compare_captures(bad, bad)):
        assert env["ok"] is False and env["error"]["code"] == "invalid_input", env


def test_wrong_typed_arguments_are_invalid_input(root):
    for env in (tools.scan_repo(None), tools.scan_repo(3), tools.capture_run(None, "x")):
        assert env["ok"] is False and env["error"]["code"] == "invalid_input", env
