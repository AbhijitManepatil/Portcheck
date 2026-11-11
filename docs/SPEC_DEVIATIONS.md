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
4. **Config defaults left empty.** `target.rocm_image` and `target.torch_index_url` ship
   empty. They could not be verified from the build environment (the PyTorch wheel host
   was blocked by its network policy), and the spec lists setting verified values as a
   human step (section 13, step 2). The fixes return `config_missing` until they are set.
5. **Lint rule.** `ruff` ignores E501 (line length); `ruff format` handles wrapping.

## Layout

6. **Extra files.** `envelope.py` (error types and the return envelope), `tools.py` (the
   eight operations as envelope-returning functions, shared by the CLI and the server) and
   `defaults.yaml` are not in the spec's layout. Extra tests: `test_fingerprint.py`,
   `test_capture.py`, `test_cli.py`.

## Envelope and run folders

7. **`env_fingerprint` writes no run folder.** The spec says every tool call writes to a
   run folder, but lists no `fingerprint` run kind. The fingerprint is returned with
   `run_id: null` and `result_path: null`, and nothing is written.
8. **`apply_step` has no run kind either.** Its envelope carries the plan's `run_id` and
   folder. A dry run writes nothing anywhere. A successful apply also saves
   `applied-<step_id>.json` in the plan's folder as a record.
9. **Failure envelopes** include `run_id: null`, `result_path: null` and a `summary` equal
   to the error message, plus the specified `error` object and no `data`.
10. **Result files.** Scan, plan, compare and cost runs store the `data` object as
    `result.json`. A capture run stores `capture.json` and `capture.safetensors` only. A
    report run stores the report file plus a small `result.json`.

## Audit

11. **PC009 and guarded imports.** As the spec recommends, the PC009 `import` matcher
    ignores an import inside a `try` block whose handlers catch `ImportError`
    (also `ModuleNotFoundError`, `Exception`, or a bare `except`). This applies to PC009
    only: a guarded `import tensorrt` is still reported as red.
12. **`fix` is decided per finding.** A PC009 finding is `auto` only for a top-level
    `import pynvml` with no alias and no other name on the line; `from pynvml import ...`,
    nested or aliased imports and `nvidia-smi` matches are `manual`. A PC008 dependency
    finding is `auto` only in a requirements file; in `pyproject.toml` or
    `environment.yml` it is `manual`, because the spec defines the fix for requirements
    files only. Imports found by the syntax-error fallback are never `auto`.
13. **Filename-only rules ignore the size and encoding limits.** A `.cu` file that is
    larger than `scan.max_file_bytes` or is not UTF-8 is still reported by PC001. The spec
    states the exception for decoding failures; it was extended to oversize files because
    missing a red finding is the less safe outcome.
14. **`from` imports** match on the module and on `module.name`, so
    `from nvidia import dali` matches the `nvidia.dali` prefix.
15. **File globs.** A matcher glob without `/` (for example `setup.py`, `*.cmake`) matches
    the file name at any depth. "Requirements files and Dockerfiles" is expressed as the
    file classes `@requirements` and `@dockerfile` in `rules.yaml`.

## Migrate

16. **One step per (rule, file, kind).** The spec says one step per (rule, file). When a
    file has both auto and manual findings of the same rule (for example `import pynvml`
    and an `nvidia-smi` call), it gets one auto step and one manual step, so that an auto
    step never hides manual work.
17. **Fixes recompute from the current file.** Each fix re-detects what to change from the
    file's current text rather than from the line numbers saved in the scan. This is what
    makes the hash check meaningful: if the file changed after the dry run, the diff and
    its hash change and the apply is refused with `diff_mismatch`.
18. **PC008 details.** The URL replacement also runs in Dockerfiles (the rule matches the
    URL there). Deleting a requirement also deletes its backslash-continuation lines.
    `nvidia-dali*` lines are kept, because they belong to PC004 (red, manual).
19. **Nothing to change.** If a step's fix is already in place, the dry run returns an
    empty diff, and an apply with that diff's hash returns `applied: false`.
20. **The CLI takes both ids as positional arguments:** `portcheck apply <step_id> <plan_id>`.

## Validate

21. **Local model or hub id.** The spec does not say how to tell them apart. `model` is a
    local model if it is an existing directory (absolute, or relative to the workload
    file's directory, or to the root); otherwise it is a hub id and `revision` is
    required. A local model directory may be outside the root: it is only read.
22. **`workload_sha256`** is SHA-256 over the workload file bytes, one zero byte, then the
    prompts file bytes. A workload file that names the model by a machine-specific path
    therefore hashes differently on each machine; use a hub id or the same relative path.
23. **Fixture workload.** `examples/e2e-service/workload.yaml` has a placeholder model and
    `revision: null`, because choosing and pinning the model is a human step. A capture on
    it returns `invalid_input` until a person fills it in.
24. **Bench token count.** New tokens are counted per row up to and including the first
    end-of-sequence token, so padding after an early stop is not counted as output.
25. **`capture_mismatch` also covers tensor shapes.** Captures whose tensors have different
    shapes are rejected with `capture_mismatch`, in addition to differing hashes.
26. **Stored precision.** Compare metrics are rounded to 8 significant digits before the
    checks are evaluated, and bench timings to 6 decimal places (tokens per second to 3),
    so the stored values, the checks and the report always agree exactly.
27. **Non-finite logits.** If a capture contains NaN or infinity, `max_abs_diff` and
    `mean_rel_error` are reported as `1e30` and `min_cosine` as `-1`, so the verdict is
    `fail` and the result stays valid JSON.

## Report

28. **One derived number.** The spec says the report must contain no number that is not in
    an input JSON file, and also asks for a cost ratio in the verdict line. The ratio
    (candidate cost divided by reference cost at batch size 1, rounded to 4 places) is the
    only number the report computes. `test_report.py` checks every other number against
    the inputs.
29. **Capture files are inputs.** The fingerprints for the Environment section are read
    from the `capture.json` of each capture named by the compare and cost results. If a
    capture folder is not on this machine, the report is built without it.
30. **Reproduce commands** show `<workload.yaml>` as a placeholder, because a capture does
    not record the path of its workload file; the workload hash is printed beside it.
31. **Cost roles.** With two cost results, the reference and candidate are taken from the
    compare result when it names the same captures, otherwise from the order of `cost_ids`.

## Not done (by design)

Nothing in spec section 13 was attempted. See `docs/RELEASE_TEST.md`.
