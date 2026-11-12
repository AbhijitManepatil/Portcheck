# Spec deviations

Every place where the implementation differs from `IMPLEMENTATION_SPEC.md`, or where the
spec was ambiguous and a reading had to be chosen (spec section 3, rule 7).

## Dependencies and environment

1. **MCP SDK import.** The spec says `from mcp.server.fastmcp import FastMCP`. The current
   SDK (mcp 2.x) renamed that class to `mcp.server.mcpserver.MCPServer` and the old import
   raises. `server.py` tries the spec's import first and falls back to the new name. The
   tool decorator, `list_tools` and `run` are used the same way in both. Tested against
   mcp 2.3.0 and mcp 1.30.0.
2. **`tomli` on Python 3.10.** `pyproject.toml` parsing needs a TOML reader. `tomllib` is
   in the standard library from 3.11; for 3.10 the dependency
   `tomli; python_version < '3.11'` is added. No other dependency was added.
3. **`transformers` argument name.** The spec says `torch_dtype=<dtype>`. transformers 5
   renamed it to `dtype` and warns on the old name, so `capture.py` passes `dtype` on
   transformers 5 or newer and `torch_dtype` on older versions.
4. **Config defaults.** Both were looked up on 2026-11-11 and are kept on ROCm 10.0;
   the sources are in comments in `defaults.yaml`. `target.rocm_image` is
   `rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0` (Docker Hub).
   `target.torch_index_url` is `https://download.pytorch.org/whl/rocm10.0`, the newest
   stable ROCm index that holds torch wheels (the `rocm10.1` index had them only under
   `nightly/`). Both were confirmed to exist; neither was pulled, installed or run on an
   AMD GPU, which remains a human step (section 13, step 2).
5. **Lint rule.** `ruff` ignores E501 (line length); `ruff format` handles wrapping.

6. **Distribution name and install hint.** The spec's skill text says
   `pip install portcheck`. That name on PyPI belongs to an unrelated project (checked
   2026-11-11), so following it would install someone else's package. The owner chose the
   distribution name `gpu-portcheck`; the import package and the commands are still
   `portcheck`. Until the package is published, the skill and the README give
   `pip install git+https://github.com/AbhijitManepatil/Portcheck.git`.

## Layout

7. **Extra files.** `envelope.py` (error types and the return envelope), `tools.py` (the
   eight operations as envelope-returning functions, shared by the CLI and the server) and
   `defaults.yaml` are not in the spec's layout. Extra tests: `test_fingerprint.py`,
   `test_capture.py`, `test_cli.py`.

## Envelope and run folders

8. **`env_fingerprint` writes no run folder.** The spec says every tool call writes to a
   run folder, but lists no `fingerprint` run kind. The fingerprint is returned with
   `run_id: null` and `result_path: null`, and nothing is written.
9. **`apply_step` has no run kind either.** Its envelope carries the plan's `run_id` and
   folder. A dry run writes nothing anywhere. A successful apply also saves
   `applied-<step_id>.json` in the plan's folder as a record.
10. **Failure envelopes** include `run_id: null`, `result_path: null` and a `summary` equal
   to the error message, plus the specified `error` object and no `data`.
11. **Result files.** Scan, plan, compare and cost runs store the `data` object as
    `result.json`. A capture run stores `capture.json` and `capture.safetensors` only. A
    report run stores the report file plus a small `result.json`.

## Audit

12. **PC009 and guarded imports.** As the spec recommends, the PC009 `import` matcher
    ignores an import inside a `try` block whose handlers catch `ImportError`
    (also `ModuleNotFoundError`, `Exception`, or a bare `except`). This applies to PC009
    only: a guarded `import tensorrt` is still reported as red.
13. **`fix` is decided per finding.** A PC009 finding is `auto` only for a top-level
    `import pynvml` with no alias and no other name on the line; `from pynvml import ...`,
    nested or aliased imports and `nvidia-smi` matches are `manual`. A PC008 dependency
    finding is `auto` only in a requirements file; in `pyproject.toml` or
    `environment.yml` it is `manual`, because the spec defines the fix for requirements
    files only. Imports found by the syntax-error fallback are never `auto`.
14. **Filename-only rules ignore the size and encoding limits.** A `.cu` file that is
    larger than `scan.max_file_bytes` or is not UTF-8 is still reported by PC001. The spec
    states the exception for decoding failures; it was extended to oversize files because
    missing a red finding is the less safe outcome.
15. **`from` imports** match on the module and on `module.name`, so
    `from nvidia import dali` matches the `nvidia.dali` prefix.
16. **File globs.** A matcher glob without `/` (for example `setup.py`, `*.cmake`) matches
    the file name at any depth. "Requirements files and Dockerfiles" is expressed as the
    file classes `@requirements` and `@dockerfile` in `rules.yaml`.

## Migrate

17. **One step per (rule, file, kind).** The spec says one step per (rule, file). When a
    file has both auto and manual findings of the same rule (for example `import pynvml`
    and an `nvidia-smi` call), it gets one auto step and one manual step, so that an auto
    step never hides manual work.
18. **Fixes recompute from the current file.** Each fix re-detects what to change from the
    file's current text rather than from the line numbers saved in the scan. This is what
    makes the hash check meaningful: if the file changed after the dry run, the diff and
    its hash change and the apply is refused with `diff_mismatch`.
19. **PC008 details.** The URL replacement also runs in Dockerfiles (the rule matches the
    URL there). Deleting a requirement also deletes its backslash-continuation lines.
    `nvidia-dali*` lines are kept, because they belong to PC004 (red, manual).
20. **Nothing to change.** If a step's fix is already in place, the dry run returns an
    empty diff, and an apply with that diff's hash returns `applied: false`.
21. **The CLI takes both ids as positional arguments:** `portcheck apply <step_id> <plan_id>`.

## Validate

22. **Local model or hub id.** The spec does not say how to tell them apart. `model` is a
    local model if it is an existing directory (absolute, or relative to the workload
    file's directory, or to the root); otherwise it is a hub id and `revision` is
    required. A local model directory may be outside the root: it is only read.
23. **`workload_sha256`** is SHA-256 over the workload file bytes, one zero byte, then the
    prompts file bytes. A workload file that names the model by a machine-specific path
    therefore hashes differently on each machine; use a hub id or the same relative path.
24. **Fixture workload.** The spec leaves the fixture's model to a person (section 13,
    step 2). At the owner's request it is now pinned: `HuggingFaceTB/SmolLM2-135M`
    (Apache-2.0) at commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. No test downloads
    it; the `text` prompt path is tested offline with a tokenizer built locally.
25. **Bench token count.** New tokens are counted per row up to and including the first
    end-of-sequence token, so padding after an early stop is not counted as output.
26. **`capture_mismatch` also covers tensor shapes.** Captures whose tensors have different
    shapes are rejected with `capture_mismatch`, in addition to differing hashes.
27. **Stored precision.** Compare metrics are rounded to 8 significant digits before the
    checks are evaluated, and bench timings to 6 decimal places (tokens per second to 3),
    so the stored values, the checks and the report always agree exactly.
28. **Non-finite logits.** If a capture contains NaN or infinity, `max_abs_diff` and
    `mean_rel_error` are reported as `1e30` and `min_cosine` as `-1`, so the verdict is
    `fail` and the result stays valid JSON.

## Report

29. **One derived number.** The spec says the report must contain no number that is not in
    an input JSON file, and also asks for a cost ratio in the verdict line. The ratio
    (candidate cost divided by reference cost at batch size 1, rounded to 4 places) is the
    only number the report computes. `test_report.py` checks every other number against
    the inputs.
30. **Capture files are inputs.** The fingerprints for the Environment section are read
    from the `capture.json` of each capture named by the compare and cost results. If a
    capture folder is not on this machine, the report is built without it.
31. **Reproduce commands** show `<workload.yaml>` as a placeholder, because a capture does
    not record the path of its workload file; the workload hash is printed beside it.
32. **Cost roles.** With two cost results, the reference and candidate are taken from the
    compare result when it names the same captures, otherwise from the order of `cost_ids`.

## Hardening after an independent review

These go beyond the spec. Each has a regression test in `tests/unit/test_hardening.py`
or `tests/unit/test_review_findings.py`.

33. **Line breaks.** Only `\n`, `\r\n` and `\r` count as line ends, as in Python's own
    parser. `str.splitlines` also splits on form feeds and similar characters, which
    shifted line numbers and once made a fix edit the wrong line.
34. **Byte-order mark.** Files are decoded as `utf-8-sig`, so a BOM no longer hides a
    finding on line 1. `apply_step` keeps the BOM in the file it writes.
35. **Scan boundaries.** A directory that contains `pyvenv.cfg` (a virtualenv, whatever its
    name) is skipped, in addition to `scan.exclude`. A symlink that resolves outside the
    root is never read. A file too deeply nested to parse falls back to the line regex
    instead of aborting the scan.
36. **Results folder.** If `.portcheck`, `.portcheck/runs` or a run folder is a symlink that
    leaves the root, every tool returns `path_outside_root` and writes nothing.
37. **PC007 (Dockerfiles).** `FROM` text inside a heredoc body or a continued instruction
    is ignored. An optional `docker.io/` prefix is accepted (the spec's pattern is
    anchored at `nvidia/cuda` or `nvcr.io/`). An image that comes from an `ARG` default,
    such as `FROM ${BASE}`, is reported but is `manual`, because the argument can be
    overridden at build time.
38. **PC008 (URLs).** Only a bare index URL (`.../whl/cu121`, with or without a trailing
    slash) is rewritten. A URL that goes on to a file, such as `torch_stable.html` or a
    wheel, is reported as `manual` and left alone; rewriting it produced dead links. A
    comment line ending in a backslash no longer swallows the next requirement.
39. **PC009 (import guard).** The import must be `import pynvml` exactly, alone on its
    line: `import pynvml.smi`, or an import sharing a line with another statement, is
    `manual`. A trailing comment such as `# noqa` is kept on the guarded import. The
    scanner and the fix are tested to agree on what is auto-fixable.
40. **Atomic apply.** `apply_step` writes through a temporary file and a rename, and keeps
    the file's permission bits.
41. **Greedy decoding is forced.** `capture_run` passes `num_beams=1`,
    `repetition_penalty=1.0` and `no_repeat_ngram_size=0` with `do_sample=False`, so a
    model whose own generation config asks for beam search is still decoded greedily, as
    the spec requires.
42. **Context length.** A workload whose longest prompt plus `max_new_tokens` exceeds the
    model's context length is rejected with `invalid_input` before anything runs.
43. **Noise floor must be finite.** If the reference or the repeat capture contains NaN or
    infinity, `compare_captures` returns `invalid_input` instead of computing a huge noise
    floor that would let any candidate pass. Capture tensors must be non-empty 2-D arrays
    with matching row counts.
44. **Report.** Arguments in the reproduce commands are shell-quoted, values containing
    backticks keep them, and a damaged `capture.json` is skipped like a missing one.
45. **`portcheck runs`.** A ninth CLI command lists the stored results. It is CLI only: the
    MCP server still exposes exactly the 8 specified tools. Long finding tables in the
    CLI are capped at 50 rows; `--json` always has everything.

46. **Repeatability warning.** When a repeat capture is given and its throughput differs
    from the reference by more than 5% at any batch size (the release test's criterion),
    the `compare` summary says so and the report's cost section carries a warning. The
    report's warning is worded without figures, in keeping with item 29. The parity
    verdict is not affected.

## Found on a real AMD GPU

From the run recorded in `release-evidence/v0.1-azure-amd-v620/`.

47. **AMD driver version fallback.** The spec reads the AMD driver version from
    `rocm-smi`. PyTorch's ROCm wheels bundle ROCm, so `rocm-smi` is often not installed;
    the fingerprint then reads the kernel module's version from
    `/sys/module/amdgpu/version`.
48. **PC009 guidance.** The step detail and the rule's action still name the
    `if pynvml is not None` check the spec asks for, and now add that `nvmlInit()` raises
    `pynvml.NVMLError` when the package is installed on a machine without NVML, so that
    must be caught too. The migrated fixture hit exactly this in a container.
49. **`rocm_version` is the HIP version.** As the spec defines it (`torch.version.hip`),
    the field holds PyTorch's HIP version, for example `7.15.26333` for the ROCm 10.0
    build, not the ROCm release number.

## Not done (by design)

The full release test in spec section 13 has not been run. At the owner's request, the AMD
half was run on a rented Azure VM (`release-evidence/v0.1-azure-amd-v620/`); no NVIDIA GPU
was available, nothing was published to PyPI, and the repository's visibility was not
changed. See `docs/RELEASE_TEST.md`.
