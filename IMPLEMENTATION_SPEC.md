# GPU Port Check: Implementation Spec for a Coding Agent

Version 0.1 spec, 2026-11-12. Working name: `portcheck`.

This file is the single source of truth for building v0.1. It is written for a coding agent. Follow the tasks in section 12 in order. Do not start a task until the previous task's acceptance tests pass.

---

## 1. What you are building

A Python package that checks whether a CUDA-based AI inference workload can move from NVIDIA to AMD ROCm, and proves it with measurements.

It has three surfaces over one core library:

1. A CLI, `portcheck`.
2. An MCP server, `portcheck-mcp`, exposing the same functions as 8 tools over stdio.
3. Three Claude skills (`SKILL.md` files) that tell an agent which tools to call.

Four capabilities:

| Capability | What it does | Needs a GPU |
| --- | --- | --- |
| Audit | Static scan of a repo for NVIDIA-only dependencies, graded red, yellow or green | No |
| Migrate | Applies known-safe edits as reviewable diffs | No |
| Validate | Captures model outputs on each machine, compares them offline | Yes, for capture only |
| Cost and report | Converts measured throughput to cost per million tokens, builds one report | No |

## 2. Non-goals for v0.1

Do not build any of these. If a task seems to need one, stop and report.

- Translating CUDA kernels (`.cu` files) to HIP or anything else.
- Training workloads, multi-GPU runs, distributed inference.
- Any hosted service, web UI, database, telemetry or analytics.
- Targets other than AMD ROCm (no Intel, no TPU).
- Tasks other than `text-generation` in the capture module. Keep the task field in the schema so others can be added later.

## 3. Rules for the implementing agent

1. Work task by task (section 12). Each task ends with `ruff check .` and `pytest -q` passing.
2. All tests in this spec must run on CPU with no network access. Never add a test that downloads a model.
3. Never hardcode a ROCm version, image tag or wheel index URL in code. They come from config (section 6.4). Before setting the default values in the config file, look up the current ones and note the source in a comment. If you cannot verify, leave the default empty and make the tool return a clear error asking the user to set it.
4. The server and CLI must make no network calls. The only permitted network use is the model download done by `transformers` inside `capture_run`.
5. Every file write by a tool must stay inside the root directory (section 6.1). Reject any path that resolves outside it.
6. Do not publish to PyPI, create a public GitHub repo, rent hardware or run the release test. Those are human steps (section 13).
7. If this spec is ambiguous or wrong, do not guess silently. Pick the safer reading, implement it, and add an entry to `docs/SPEC_DEVIATIONS.md` saying what you chose and why.
8. Report results truthfully. If a test is skipped or a feature is stubbed, say so in the task summary.

## 4. Tech stack

- Python 3.10 or newer. Type hints everywhere. `from __future__ import annotations`.
- Packaging: `pyproject.toml` with `hatchling`. `src/` layout.
- Core dependencies: `pydantic>=2`, `pyyaml`, `typer`, `numpy`, `safetensors`, `mcp` (the official MCP Python SDK; use `from mcp.server.fastmcp import FastMCP`).
- Optional extra `run`: `torch`, `transformers`. Import these lazily, only inside the capture module, so audit and migrate work without them.
- Optional extra `dev`: `pytest`, `ruff`, `torch`, `transformers`.
- Licence: Apache-2.0.
- No other dependencies without recording the reason in `docs/SPEC_DEVIATIONS.md`.

## 5. Repository layout

```
portcheck/
  pyproject.toml
  README.md
  LICENSE
  src/portcheck/
    __init__.py
    config.py            # section 6.4
    paths.py             # root resolution, path safety, run ids
    schemas.py           # all pydantic models (section 7)
    fingerprint.py       # env_fingerprint
    audit/
      __init__.py
      rules.yaml         # rule catalog (section 8.2)
      scanner.py         # scan_repo
      parsers.py         # requirements, pyproject, Dockerfile, Python AST
    migrate/
      __init__.py
      planner.py         # plan_migration
      fixes.py           # one function per auto-fix
      apply.py           # apply_step
    validate/
      __init__.py
      workload.py        # workload file loading
      capture.py         # capture_run
      compare.py         # compare_captures
    cost.py              # estimate_cost
    report.py            # build_report
    cli.py               # typer app
    server.py            # FastMCP app
  skills/
    gpu-port-audit/SKILL.md
    gpu-port-migrate/SKILL.md
    gpu-port-validate/SKILL.md
  examples/e2e-service/  # test fixture (section 10)
  tests/
    unit/
    e2e_sim/
    golden/e2e-service.findings.json
  docs/
    SPEC_DEVIATIONS.md
    RELEASE_TEST.md      # human runbook (section 13)
```

Entry points in `pyproject.toml`:

```toml
[project.scripts]
portcheck = "portcheck.cli:app"
portcheck-mcp = "portcheck.server:main"
```

## 6. Shared conventions

### 6.1 Root directory

The root is the environment variable `PORTCHECK_ROOT` if set, otherwise the current working directory. All `path` arguments are resolved relative to the root. A path that resolves outside the root raises `PathOutsideRootError`.

### 6.2 Results folder and run ids

Every tool call writes its full result to `<root>/.portcheck/runs/<run_id>/`.

- `run_id` format: `<kind>-<YYYYMMDDTHHMMSSZ>-<6 hex chars>`, where kind is one of `scan`, `plan`, `capture`, `compare`, `cost`, `report`.
- Any tool argument that names an earlier result accepts a `run_id`. Capture arguments also accept a path to a capture directory, so captures can be copied between machines.

### 6.3 Return envelope

Every CLI command (with `--json`) and every MCP tool returns this shape:

```json
{
  "ok": true,
  "run_id": "scan-20261112T101500Z-a1b2c3",
  "result_path": ".portcheck/runs/scan-20261112T101500Z-a1b2c3",
  "summary": "one sentence a person can read",
  "data": { }
}
```

On failure: `"ok": false`, plus `"error": {"code": "...", "message": "..."}` and no `data`. Tools never raise to the MCP client. Error codes: `path_outside_root`, `not_found`, `invalid_input`, `config_missing`, `missing_dependency`, `diff_mismatch`, `capture_mismatch`, `internal`.

### 6.4 Config

`config.py` loads defaults from `src/portcheck/defaults.yaml`, overridden by `<root>/portcheck.yaml` if present.

```yaml
target:
  rocm_image: ""          # e.g. a rocm/pytorch image tag; verify before setting a default
  torch_index_url: ""     # the PyTorch ROCm wheel index; verify before setting a default
parity:
  noise_multiplier: 3.0
  min_cosine: 0.999
  min_top1_agreement: 0.99
  max_abs_diff_floor:
    float32: 0.001
    float16: 0.05
    bfloat16: 0.05
bench:
  warmup_batches: 2
  measured_batches: 5
scan:
  exclude: [".git", ".venv", "node_modules", ".portcheck", "__pycache__"]
  max_file_bytes: 1000000
```

The parity numbers are initial defaults. They are calibrated by a human during the release test. Do not tune them to make a test pass.

## 7. Schemas

Define these as pydantic models in `schemas.py`. Field names are fixed; tests depend on them.

### 7.1 Fingerprint

```json
{
  "vendor": "nvidia | amd | cpu",
  "device_name": "string or null",
  "device_count": 1,
  "torch_version": "string or null",
  "cuda_version": "string or null",
  "rocm_version": "string or null",
  "driver_version": "string or null",
  "python_version": "3.11.9",
  "platform": "Linux-...",
  "portcheck_version": "0.1.0"
}
```

Detection, in `fingerprint.py`:

- If `torch` is not importable: vendor `cpu`, torch fields null. Do not fail.
- `rocm_version = getattr(torch.version, "hip", None)`; `cuda_version = torch.version.cuda`.
- If `torch.cuda.is_available()` and `rocm_version` is not null: vendor `amd`. Else if available and `cuda_version` is not null: vendor `nvidia`. Else `cpu`.
- `driver_version` is best effort: run `nvidia-smi --query-gpu=driver_version --format=csv,noheader` for nvidia. For amd, try `rocm-smi --showdriverversion` and parse; on any failure or timeout (5 s) return null. Never raise.

### 7.2 Finding and ScanResult

```json
{
  "rule_id": "PC007",
  "severity": "red | yellow | green",
  "title": "NVIDIA CUDA base image",
  "file": "Dockerfile",
  "line": 1,
  "evidence": "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04",
  "action": "Swap to a ROCm PyTorch image",
  "fix": "auto | manual | none"
}
```

```json
{
  "root": ".",
  "files_scanned": 42,
  "grade": "red | yellow | green",
  "counts": {"red": 2, "yellow": 3, "green": 1},
  "findings": [ ]
}
```

Grade is the worst severity present. Findings are sorted by severity (red first), then file, then line. `file` uses forward slashes and is relative to the scanned path. `line` is 1-based, or null for file-level findings.

### 7.3 Plan

```json
{
  "scan_id": "scan-...",
  "steps": [
    {
      "step_id": "S1",
      "rule_id": "PC007",
      "kind": "auto | manual",
      "title": "Swap base image in Dockerfile",
      "files": ["Dockerfile"],
      "detail": "what will change, or what the person must do"
    }
  ]
}
```

One step per (rule, file) pair. Auto steps first, then manual, each group ordered by rule id.

### 7.4 Workload file

```yaml
model: path/or/hub-id
revision: null            # commit hash; required when model is a hub id
task: text-generation
dtype: float32            # float32 | float16 | bfloat16
prompts: prompts.jsonl    # one JSON object per line
max_new_tokens: 32
batch_sizes: [1, 4]
seed: 0
```

Each line of the prompts file is either `{"text": "..."}` or `{"input_ids": [1, 2, 3]}`. A file must use one form throughout. The `input_ids` form needs no tokenizer and is what tests use.

### 7.5 Capture

A capture is a directory containing `capture.json` and `capture.safetensors`.

`capture.json`:

```json
{
  "label": "nvidia-ref",
  "workload": { },
  "workload_sha256": "hash of the workload file plus the prompts file",
  "fingerprint": { },
  "num_prompts": 50,
  "fault_injected": false,
  "bench": [
    {
      "batch_size": 1,
      "tokens_generated": 1600,
      "seconds": 12.5,
      "tokens_per_second": 128.0,
      "latency_p50_s": 0.24,
      "latency_p95_s": 0.31
    }
  ],
  "peak_memory_bytes": 123456789
}
```

`capture.safetensors` tensors:

- `last_logits`: float32, shape `[num_prompts, vocab_size]`. Logits at the last prompt position, before any generation.
- `generated_ids`: int64, shape `[num_prompts, max_new_tokens]`. Greedy decoding. Pad with `-1` after end of sequence.

### 7.6 CompareResult

```json
{
  "reference": "capture-...",
  "candidate": "capture-...",
  "baseline_repeat": "capture-... or null",
  "dtype": "float32",
  "metrics": {
    "max_abs_diff": 0.0004,
    "mean_rel_error": 0.00002,
    "min_cosine": 0.99999,
    "top1_agreement": 1.0,
    "greedy_exact_match_rate": 0.98,
    "mean_matching_prefix_tokens": 31.6
  },
  "noise_floor": {"max_abs_diff": 0.0, "mean_rel_error": 0.0},
  "thresholds": {"max_abs_diff": 0.001, "min_cosine": 0.999, "min_top1_agreement": 0.99},
  "checks": [
    {"metric": "max_abs_diff", "value": 0.0004, "threshold": 0.001, "passed": true}
  ],
  "verdict": "pass | fail"
}
```

### 7.7 CostResult

```json
{
  "capture": "capture-...",
  "price_per_hour": 2.5,
  "currency": "USD",
  "rows": [
    {
      "batch_size": 1,
      "tokens_per_second": 128.0,
      "latency_p50_s": 0.24,
      "latency_p95_s": 0.31,
      "cost_per_million_tokens": 5.43
    }
  ]
}
```

Formula: `cost_per_million_tokens = price_per_hour / (tokens_per_second * 3600) * 1_000_000`. Round to 4 decimal places.

## 8. Module specifications

### 8.1 `scan_repo(path: str = ".") -> ScanResult`

1. Walk files under `path`, skipping `scan.exclude` directories, files larger than `scan.max_file_bytes`, and files that fail UTF-8 decoding (except for filename-only rules).
2. Apply every rule in `rules.yaml`. A rule can produce many findings. Deduplicate on (rule_id, file, line).
3. Parsers in `parsers.py`:
   - Python imports: use `ast` to collect imported module names with line numbers. On a syntax error, fall back to a line regex for `^\s*(import|from)\s+([\w.]+)`.
   - Dependencies: parse `requirements*.txt`, `pyproject.toml` (`project.dependencies` and optional dependencies), and `environment.yml` (`dependencies` including the nested `pip` list). Normalise names to lowercase with `-`.
   - Dockerfiles: any file named `Dockerfile` or matching `Dockerfile.*` or `*.Dockerfile`. Extract each `FROM` line with its image reference.

### 8.2 Rule catalog (`rules.yaml`)

Rules are data. Each rule has `id`, `title`, `severity`, `action`, `fix`, and one or more matchers. Matcher kinds: `file_glob`, `import` (prefix match on dotted module names), `dependency` (glob on normalised name), `docker_from` (regex on image reference), `regex` (with a `files` glob list).

| id | Title | Severity | Fix | Matchers |
| --- | --- | --- | --- | --- |
| PC001 | Custom CUDA source | red | manual | file_glob `**/*.cu`, `**/*.cuh`; regex `CUDAExtension\|enable_language\(\s*CUDA\|find_package\(\s*CUDA` in `setup.py`, `CMakeLists.txt`, `*.cmake` |
| PC002 | TensorRT | red | manual | import `tensorrt`, `torch_tensorrt`, `tensorrt_llm`; dependency `tensorrt*`, `torch-tensorrt` |
| PC003 | FlashAttention package | red | manual | import `flash_attn`; dependency `flash-attn` |
| PC004 | NVIDIA-only library | red | manual | import `nvidia.dali`, `cudf`, `cuml`, `cugraph`; dependency `nvidia-dali*`, `cudf*`, `cuml*`, `cugraph*` |
| PC005 | Multi-GPU or NCCL use | yellow | manual | import `torch.distributed`; regex `DistributedDataParallel\|\bnccl\b` (case-insensitive) in `**/*.py`, `**/*.sh` |
| PC006 | Low-precision quantized path | yellow | manual | regex `\b(fp8\|float8\|fp4\|nvfp4)\b` (case-insensitive) in `**/*.py`, `**/*.yaml`, `**/*.yml`, `**/*.json` |
| PC007 | NVIDIA base image | yellow | auto | docker_from `^(nvidia/cuda\|nvcr\.io/)` |
| PC008 | NVIDIA pip pins | yellow | auto | dependency `nvidia-*` (excluding names matched by PC004); regex `download\.pytorch\.org/whl/cu\d+` in requirements files and Dockerfiles |
| PC009 | NVML or nvidia-smi call | yellow | auto for imports, manual otherwise | import `pynvml`; regex `nvidia-smi` in `**/*.py`, `**/*.sh` |
| PC010 | Plain torch.cuda usage | green | none | regex `torch\.cuda\.\|\.cuda\(\)\|["']cuda["']` in `**/*.py` |

PC010 produces at most one finding per file (the first match). Its action text is: "No change needed. ROCm builds of PyTorch use the same torch.cuda API."

Action text for red rules must say what the manual work is and must not imply the tool can do it. PC001's action names HIPIFY and SCALE as tools to evaluate.

### 8.3 `plan_migration(scan_id: str) -> Plan`

Build steps from the scan's findings as described in 7.3. Green findings produce no step. A PC009 finding from the `import` matcher is `auto`; one from the `nvidia-smi` regex is `manual`.

### 8.4 `apply_step(step_id: str, plan_id: str, dry_run: bool = True, diff_sha256: str | None = None)`

- With `dry_run=True`: compute the edit, return `{"diff": "<unified diff>", "diff_sha256": "<hash>", "applied": false}`. Write nothing to the repo.
- With `dry_run=False`: `diff_sha256` is required. Recompute the diff. If its hash differs from the one supplied, return error `diff_mismatch` and write nothing. Otherwise write the files and return `"applied": true`.
- Manual steps return error `invalid_input` with a message that the step is manual.

Auto-fixes in `fixes.py`. Each is a pure function from file text to new file text:

- **PC007**: replace the image reference on the matched `FROM` line with `target.rocm_image`, preserving any `AS <name>` suffix. If the config value is empty, return error `config_missing`.
- **PC008**: in requirements files, delete lines whose package name matches `nvidia-*`. Replace a `download.pytorch.org/whl/cu<digits>` URL with `target.torch_index_url`. If that config value is empty and a URL needs replacing, return `config_missing`.
- **PC009**: replace a top-level `import pynvml` statement with:

  ```python
  try:
      import pynvml
  except ImportError:  # not available on non-NVIDIA machines
      pynvml = None
  ```

  This fix covers the import only. The step's `detail` must list every line that calls `pynvml.` and say those call sites need a manual `if pynvml is not None` check. Imports that are not top-level, or use `from pynvml import ...`, are left as manual.

### 8.5 `capture_run(workload: str, label: str, inject_fault: float | None = None) -> Capture`

1. Load and validate the workload file. Compute `workload_sha256`.
2. Import `torch` and `transformers` lazily. If missing, return error `missing_dependency` naming the `run` extra.
3. Device: `cuda` if `torch.cuda.is_available()`, else `cpu`. Set `torch.manual_seed(seed)`.
4. Load the model with `AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=<dtype>)`, move to device, `eval()`. Load the tokenizer only if prompts use the `text` form. On `cpu`, if dtype is not `float32`, use `float32` and record a warning in the summary.
5. If `inject_fault` is set: add Gaussian noise with that standard deviation to the input embedding weight, using a generator seeded with `seed`. Set `fault_injected: true`. This exists only for negative tests.
6. Parity pass, batch size 1, under `torch.inference_mode()`: for each prompt, run one forward pass and store the last-position logits as float32 on CPU. Then run `generate` with `do_sample=False`, `max_new_tokens`, and store the new token ids.
7. Bench pass, for each batch size: build batches by cycling through the prompts (left-pad if needed; if the tokenizer has no pad token, use the EOS token id; with `input_ids` prompts of unequal length, pad with 0 and pass an attention mask). Run `bench.warmup_batches` unmeasured, then `bench.measured_batches` measured. Call `torch.cuda.synchronize()` before starting and stopping each timer when on `cuda`. Record per-batch seconds; compute tokens per second from total new tokens over total seconds, and p50 and p95 of per-batch latency.
8. Peak memory: `torch.cuda.max_memory_allocated()` on `cuda`, else null.
9. Write the capture directory and return.

### 8.6 `compare_captures(reference, candidate, baseline_repeat=None) -> CompareResult`

Needs only `numpy` and `safetensors`. Must not import torch.

1. If the `workload_sha256` values differ, return error `capture_mismatch`.
2. Metrics, with `a` = reference logits and `b` = candidate logits:
   - `max_abs_diff = max(|a - b|)`
   - `mean_rel_error = mean(|a - b|) / mean(|a|)`
   - `min_cosine` = the minimum, over prompts, of cosine similarity between rows
   - `top1_agreement` = fraction of prompts where `argmax(a) == argmax(b)`
   - `greedy_exact_match_rate` = fraction of prompts whose `generated_ids` rows are identical
   - `mean_matching_prefix_tokens` = mean length of the common prefix of the two rows
3. Noise floor: if `baseline_repeat` is given, compute `max_abs_diff` and `mean_rel_error` between reference and baseline_repeat. Otherwise zeros.
4. Thresholds:
   - `max_abs_diff` threshold = `max(noise_multiplier * noise_floor.max_abs_diff, max_abs_diff_floor[dtype])`
   - `min_cosine` and `min_top1_agreement` from config
5. Verdict is `pass` only if `max_abs_diff <= threshold`, `min_cosine >= min_cosine`, and `top1_agreement >= min_top1_agreement`. The two greedy metrics are reported but never gate the verdict, because one early token flip changes everything after it.
6. If either capture has `fault_injected: true`, include that fact in `summary`.

### 8.7 `estimate_cost(capture, price_per_hour: float, currency: str = "USD") -> CostResult`

Apply the formula in 7.7 to each bench row. Reject `price_per_hour <= 0` with `invalid_input`.

### 8.8 `build_report(scan_id=None, compare_id=None, cost_ids: list[str] | None = None, format="markdown")`

Produces `report.md` or `report.json` in a new `report-...` run folder. At least one input is required. Markdown sections, each included only when its input is present, in this order:

1. **Verdict line**: audit grade, parity verdict, and cost ratio (candidate cost divided by reference cost at batch size 1) when two cost results are given.
2. **Environment**: a table of both fingerprints.
3. **Audit findings**: a table of red and yellow findings; a count of green.
4. **Parity**: the checks table, then the noise floor, then the two greedy metrics labelled "informational".
5. **Performance and cost**: one table, a row per (machine, batch size).
6. **How to reproduce**: the exact CLI commands, with the workload hash.

The report must contain no number that is not present in an input JSON file.

### 8.9 CLI (`cli.py`)

One command per function, same names with dashes: `portcheck fingerprint`, `scan`, `plan`, `apply`, `capture`, `compare`, `cost`, `report`. Every command supports `--json` to print the envelope. Without `--json`, print the `summary` line and a short human-readable table. Exit code 0 when `ok` is true, 1 otherwise. `portcheck compare` also exits 2 when the verdict is `fail`.

`portcheck apply` defaults to dry run. Applying needs `--apply --diff-sha256 <hash>`.

### 8.10 MCP server (`server.py`)

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("portcheck")
```

Register 8 tools with these exact names and arguments: `env_fingerprint()`, `scan_repo(path=".")`, `plan_migration(scan_id)`, `apply_step(step_id, plan_id, dry_run=True, diff_sha256=None)`, `capture_run(workload, label, inject_fault=None)`, `compare_captures(reference, candidate, baseline_repeat=None)`, `estimate_cost(capture, price_per_hour, currency="USD")`, `build_report(scan_id=None, compare_id=None, cost_ids=None, format="markdown")`.

- Each tool is a thin wrapper: call the core function, catch all exceptions, return the envelope as a dict.
- Each tool has a docstring of 1 to 3 sentences saying what it does, what it needs first, and whether it writes files. Agents choose tools from these docstrings.
- `main()` runs the server over stdio. Nothing may be printed to stdout except protocol messages; send logs to stderr.

## 9. Skills

Create these three files. Keep each body under 60 lines. All logic stays in the server.

### 9.1 `skills/gpu-port-audit/SKILL.md`

````markdown
---
name: gpu-port-audit
description: Checks whether a repo that runs on NVIDIA CUDA can run on AMD ROCm. Use when the user asks if code can run on AMD, wants a CUDA dependency audit, or asks what blocks a move off NVIDIA.
---

# GPU port audit

1. If the `scan_repo` tool is not available, tell the user to install the server with
   `pip install portcheck` and add `portcheck-mcp` to their MCP config. Then stop.
2. Call `env_fingerprint`, then `scan_repo` on the repo root.
3. Answer in this order:
   - The grade in one sentence.
   - Up to three findings that matter most, red first, each with file and line.
   - What is safe: mention the green count in one line.
4. For red findings, name the manual work. Never say the tool can fix a red finding.
5. Offer the next step: "make this repo AMD-ready" (migrate) or "prove the results match" (validate).

Only quote findings returned by the tool. This skill is read-only.
````

### 9.2 `skills/gpu-port-migrate/SKILL.md`

Frontmatter description: "Applies the safe, mechanical changes that make a CUDA repo ready for AMD ROCm, as diffs the user approves. Use when the user asks to make a repo AMD-ready or to migrate it off NVIDIA."

Body steps:

1. Call `scan_repo`, then `plan_migration`.
2. List the steps, auto first, then manual.
3. For each auto step, call `apply_step` with `dry_run=true` and show the diff.
4. Ask the user to approve. Only after a yes, call `apply_step` with `dry_run=false` and the `diff_sha256` from the dry run.
5. On `config_missing`, tell the user which key to set in `portcheck.yaml` and stop.
6. Finish with the manual steps still open. For PC009, list the call sites that need a guard.

Rule: never apply a step the user has not seen as a diff.

### 9.3 `skills/gpu-port-validate/SKILL.md`

Frontmatter description: "Proves a model gives matching results on NVIDIA and AMD and compares cost. Use when the user asks whether outputs match across GPUs, wants a parity check, or asks what a workload costs on AMD versus NVIDIA."

Body steps:

1. Call `env_fingerprint` to learn which machine this is.
2. Check which captures exist under `.portcheck/runs/`. A full check needs three: a reference and a repeat from the NVIDIA machine, and a candidate from the AMD machine.
3. If this machine's capture is missing, call `capture_run` with the workload file.
4. If a capture from the other machine is missing, tell the user the exact command to run there and how to copy the capture folder back. Then stop.
5. With all three present, call `compare_captures`, then `estimate_cost` for each machine (ask the user for each hourly price), then `build_report`.
6. Lead with the verdict. State GPU, driver and ROCm or CUDA version beside every performance number.

Rules: quote numbers only from tool output. If the verdict is fail, show the failed checks and do not suggest loosening thresholds.

## 10. Test fixture: `examples/e2e-service`

A small inference service with known problems planted. Create exactly these files.

| File | Content | Expected finding |
| --- | --- | --- |
| `Dockerfile` | Line 1: `FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04`. Installs requirements, runs `serve.py`. | PC007 yellow, line 1 |
| `requirements.txt` | `torch`, `transformers`, `nvidia-cudnn-cu12==9.1.0.70`, `pynvml` | PC008 yellow on the `nvidia-cudnn-cu12` line |
| `serve.py` | CLI that loads the model named by env `E2E_MODEL`, reads a prompt from argv, prints generated text. Uses `torch.cuda.is_available()` and `"cuda"`. Imports `gpu_stats`. If env `USE_TRT=1`, imports `trt_backend`. | PC010 green |
| `gpu_stats.py` | Top-level `import pynvml`. Function `memory_used()` that returns `None` when `pynvml is None`, else queries NVML. | PC009 yellow on the import line, fix auto |
| `trt_backend.py` | `import tensorrt` and a stub function. Never imported unless `USE_TRT=1`. | PC002 red |
| `ext/fast_op.cu` | A 10-line CUDA kernel (element-wise add). Not built by default. | PC001 red |
| `workload.yaml`, `prompts.jsonl` | A text-generation workload with 20 short `text` prompts. | none |
| `README.md` | What the fixture is and that its problems are deliberate. | none |

`pynvml` in `requirements.txt` is not matched by any dependency rule, by design.

`tests/golden/e2e-service.findings.json` holds the expected findings as a list of `{rule_id, severity, file, line, fix}`. The scan must produce exactly this set: 2 red (PC001, PC002), 3 yellow (PC007, PC008, PC009), 1 green (PC010 in `serve.py`). Grade: red.

After the three auto steps are applied with a test config, a second scan must show no PC007 and no PC008 finding, and PC009 must no longer be reported as a bare import. Decide how PC009 treats the guarded import (recommended: the `import` matcher ignores imports inside a `try` block that catches `ImportError`) and record the choice in `docs/SPEC_DEVIATIONS.md`.

## 11. Test plan

### 11.1 Tiny local model for tests

Tests must not download anything. Add a pytest fixture `tiny_model_dir` that builds and saves a model locally:

```python
from transformers import GPT2Config, GPT2LMHeadModel
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=128, n_positions=64, n_embd=32, n_layer=2, n_head=2,
                 bos_token_id=0, eos_token_id=1)
GPT2LMHeadModel(cfg).save_pretrained(tmp_dir)
```

Test workloads point `model` at that directory and use `input_ids` prompts with ids in `[2, 127]`.

### 11.2 Unit tests (`tests/unit/`)

| Test | Asserts |
| --- | --- |
| `test_paths.py` | A path outside the root raises; run ids match the format |
| `test_parsers.py` | Requirements, pyproject, environment.yml, Dockerfile and AST parsing, including a file with a syntax error |
| `test_rules.py` | Each rule PC001 to PC010 fires on a minimal positive sample and stays silent on a negative sample |
| `test_scan_golden.py` | Scan of `examples/e2e-service` equals the golden file exactly |
| `test_plan.py` | Plan for the fixture has 3 auto steps and 2 manual steps, in the specified order |
| `test_apply.py` | Dry run writes nothing; apply with a wrong hash returns `diff_mismatch` and writes nothing; apply with the right hash changes only the expected lines; empty config returns `config_missing` |
| `test_compare.py` | Identical arrays pass; a large perturbation fails; mismatched workload hashes return `capture_mismatch`; threshold uses `max(multiplier * noise, floor)`; module does not import torch |
| `test_cost.py` | Formula checked against a hand-computed value; zero price rejected |
| `test_report.py` | Sections appear only when inputs are present; every number in the markdown appears in an input JSON |
| `test_server.py` | All 8 tool names are registered with the specified argument names; a tool given a bad path returns `ok: false` and does not raise |
| `test_no_network.py` | With sockets disabled by a fixture, `scan`, `plan`, `apply`, `compare`, `cost` and `report` all still work |

### 11.3 Simulated end-to-end (`tests/e2e_sim/test_flow.py`)

One test that runs the whole flow on CPU through the CLI functions, in a temp copy of the fixture:

1. `scan` gives grade red and matches golden.
2. `plan`, then dry-run and apply all 3 auto steps with a test config. Re-scan shows PC007 and PC008 gone.
3. `capture` three times with the tiny model: `ref`, `repeat`, `cand`.
4. `compare ref cand --baseline-repeat repeat` gives verdict `pass`.
5. `capture` again with `inject_fault=0.5` as `bad`. `compare ref bad` gives verdict `fail`.
6. `cost` on `ref` and `cand` with different prices; `report` with all inputs produces a markdown file containing all six sections.

A second test, `test_flow_mcp.py`, drives steps 1, 3 and 4 by calling the registered MCP tool functions, to prove the server path gives the same results as the CLI path.

## 12. Tasks, in order

Each task is done when its listed tests pass along with all earlier tests.

| Task | Build | Done when |
| --- | --- | --- |
| T1 | Repo scaffold, `pyproject.toml`, licence, `config.py`, `paths.py`, `schemas.py`, envelope helper | `test_paths.py` passes; `pip install -e .[dev]` works |
| T2 | `fingerprint.py` | Returns vendor `cpu` on a machine without a GPU and without torch; never raises |
| T3 | Fixture `examples/e2e-service` and the golden file | Files exist exactly as in section 10 |
| T4 | Parsers and rule engine, `rules.yaml`, `scan_repo` | `test_parsers.py`, `test_rules.py`, `test_scan_golden.py` pass |
| T5 | `plan_migration`, fixes, `apply_step` | `test_plan.py`, `test_apply.py` pass |
| T6 | Workload loading, `capture_run`, tiny model fixture | A capture on CPU produces both files with the specified shapes |
| T7 | `compare_captures` | `test_compare.py` passes |
| T8 | `estimate_cost`, `build_report` | `test_cost.py`, `test_report.py` pass |
| T9 | CLI | Every command works with and without `--json`; exit codes as specified |
| T10 | MCP server | `test_server.py` passes; server starts over stdio and lists 8 tools |
| T11 | Simulated end-to-end tests, no-network test | `tests/e2e_sim/` and `test_no_network.py` pass |
| T12 | Three skills, `README.md`, `docs/RELEASE_TEST.md` | README install steps work in a fresh virtual environment; skills reference only tools that exist |

After T12, write a final summary: what was built, any spec deviations, any test that is skipped or weaker than specified, and the config defaults you could not verify.

## 13. Human-only steps (do not attempt)

Write these into `docs/RELEASE_TEST.md` as a runbook, but do not perform them.

1. Confirm employment IP and outside-work terms in writing before any code is published.
2. Choose the open model for the fixture and pin its revision. Set verified values for `target.rocm_image` and `target.torch_index_url`.
3. Rent one NVIDIA and one AMD data-centre GPU.
4. Run the release scenario, driven by an agent through the MCP server and skills, with no CLI typed by hand:
   1. Audit the fixture.
   2. Plan, review diffs, apply the auto steps.
   3. Build and run the migrated fixture on the AMD machine.
   4. On NVIDIA: capture `ref` and `repeat`.
   5. On AMD: capture `cand`.
   6. Compare and price both.
   7. On AMD: capture with a fault injected; compare must fail.
   8. Build the report. Repeat steps 5 and 6 once.
   9. Repeat step 1 from a second MCP client.
5. Release only if every check passes:

| Check | Pass criterion |
| --- | --- |
| Audit accuracy | All 5 planted problems found; no red finding outside the golden file |
| Migration safety | Nothing written before approval; diffs touch only the 3 auto-fixable items |
| Build | Migrated fixture serves a request on the AMD machine |
| Parity, good case | Verdict `pass` |
| Parity, bad case | Verdict `fail` for the fault-injected capture |
| Repeatability | Throughput on the re-run within 5% of the first run |
| Report | Both fingerprints, all parity checks, cost per million tokens for both machines |
| Data handling | No network calls from the server other than the model download |
| Portability | Same findings from the second MCP client |

6. Save the transcript, fingerprints and report under `release-evidence/v0.1/`. If the good case fails, inspect the metrics and recalibrate; do not loosen thresholds just to pass.
7. Publish to GitHub and PyPI.

## 14. Definition of done for the agent

- All tasks T1 to T12 complete, `ruff check .` clean, `pytest -q` green with no network.
- `portcheck scan examples/e2e-service` prints grade red with 2 red, 3 yellow, 1 green.
- `portcheck-mcp` starts and lists exactly 8 tools.
- `docs/SPEC_DEVIATIONS.md` lists every place the implementation differs from this file, or says "none".
- Nothing in section 13 has been attempted.
