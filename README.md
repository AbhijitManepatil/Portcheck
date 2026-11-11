# Portcheck

GPU Port Check: an open-source, self-hosted toolkit that checks whether a CUDA-based AI
inference workload can move from NVIDIA to AMD ROCm, and proves it with measurements.

One core library with three surfaces:

- a CLI, `portcheck`
- an MCP server, `portcheck-mcp`, with the same 8 operations as tools over stdio
- three Claude skills in [`skills/`](skills/) that tell an agent which tools to call

| Capability | What it does | Needs a GPU |
| --- | --- | --- |
| Audit | Static scan of a repo for NVIDIA-only dependencies, graded red, yellow or green | No |
| Migrate | Applies known-safe edits as diffs you review and approve | No |
| Validate | Captures model outputs on each machine and compares them offline | For capture only |
| Cost and report | Converts measured throughput to cost per million tokens; builds one report | No |

Everything runs on your machine. The only network use is the model download that
`transformers` does during `capture` when the model is a hub id.

## Status

**v0.1, pre-release.** The code and its test suite are complete, but everything has been
run **on CPU only**, with a tiny random model. It has not yet been run on real NVIDIA or
AMD GPUs. That release test is a human step: see [`docs/RELEASE_TEST.md`](docs/RELEASE_TEST.md).
Until it is done:

- the parity thresholds are starting guesses, not calibrated values
- the default `target.rocm_image` and `target.torch_index_url` (both ROCm 10.0) were
  confirmed to exist, but have not been pulled, installed or run on an AMD GPU
- the package is not on PyPI yet; install it from this repository. Its distribution name
  is `gpu-portcheck`. **Do not run `pip install portcheck`:** that name on PyPI belongs to
  a different, unrelated project

The validation flow has also been run on CPU with a real model, the fixture's pinned
SmolLM2-135M: three captures compared as `pass`, and a fault-injected capture as `fail`.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/AbhijitManepatil/Portcheck.git
cd Portcheck
python -m venv .venv && source .venv/bin/activate
pip install .
```

That is enough for audit, migrate, compare, cost and report. To run captures, also install
PyTorch and transformers. Install the PyTorch build for your GPU first (CUDA or ROCm), then:

```bash
pip install '.[run]'
```

For development: `pip install -e '.[dev]'`, then `ruff check .` and `pytest -q`.

## Quick start

All paths are relative to the root: the `PORTCHECK_ROOT` environment variable if set,
otherwise the current directory. Tools refuse any path outside the root. Every result is
saved under `.portcheck/runs/<run_id>/`. Add `--json` to any command for the full result.

### Audit

```bash
portcheck scan examples/e2e-service
```

```
Grade red: 2 red, 3 yellow, 1 green findings in 9 files.
severity  rule   file              line  fix     title
red       PC001  ext/fast_op.cu          manual  Custom CUDA source
red       PC002  trt_backend.py    3     manual  TensorRT
yellow    PC007  Dockerfile        1     auto    NVIDIA base image
yellow    PC009  gpu_stats.py      3     auto    NVML or nvidia-smi call
yellow    PC008  requirements.txt  3     auto    NVIDIA pip pins
green     PC010  serve.py          29    none    Plain torch.cuda usage
```

Red means manual porting work the tool cannot do. Yellow needs a change or a check. Green
is safe as it is.

### Migrate

```bash
portcheck plan <scan_run_id>
portcheck apply S1 <plan_run_id>                               # dry run: prints a diff and its hash
portcheck apply S1 <plan_run_id> --apply --diff-sha256 <hash>  # writes the change
```

`apply` never writes without the hash of the diff you reviewed. If the file changed since
the dry run, it refuses. Only three fixes are automatic: the Docker base image (PC007),
NVIDIA pip pins (PC008) and guarding `import pynvml` (PC009). PC007 uses `target.rocm_image`;
PC008 uses `target.torch_index_url` when a CUDA wheel URL is present (see Configuration). If
either value is set to empty, the fix that needs it returns `config_missing`.

### Validate

Write a workload file (see `examples/e2e-service/workload.yaml`):

```yaml
model: org/model-name     # a hub id, or a local model directory
revision: <commit hash>   # required for a hub id
task: text-generation
dtype: float32            # float32 | float16 | bfloat16
prompts: prompts.jsonl    # {"text": "..."} or {"input_ids": [1, 2, 3]} per line
max_new_tokens: 32
batch_sizes: [1, 4]
seed: 0
```

```bash
# on the NVIDIA machine
portcheck capture workload.yaml --label nvidia-ref
portcheck capture workload.yaml --label nvidia-repeat
# on the AMD machine
portcheck capture workload.yaml --label amd-cand
```

Copy the `.portcheck/runs/capture-...` folders onto one machine (any machine; comparing
needs no GPU), then:

```bash
portcheck compare <ref> <cand> --baseline-repeat <repeat>
```

The verdict is `pass` only if all three checks hold: the largest logit difference is within
`max(3 x the reference machine's own run-to-run noise, a fixed floor for the dtype)`, the
minimum cosine similarity is at least 0.999, and the top-1 token agrees on at least 99% of
prompts. Greedy-decoding match is reported but never gates the verdict. `compare` exits
with code 2 when the verdict is `fail`.

### Cost and report

```bash
portcheck cost <ref> --price-per-hour 2.50
portcheck cost <cand> --price-per-hour 1.90
portcheck report --scan <scan_id> --compare <compare_id> --cost <cost_ref_id> --cost <cost_cand_id>
```

The report (`report.md`, or `--format json`) has the verdict line, both machines'
fingerprints, the audit findings, the parity checks, performance and cost per machine and
batch size, and the commands to reproduce it.

## Commands

| CLI command | MCP tool | What it does |
| --- | --- | --- |
| `portcheck fingerprint` | `env_fingerprint` | GPU vendor, device, driver, PyTorch and CUDA or ROCm versions |
| `portcheck scan [PATH]` | `scan_repo` | Audit a repo |
| `portcheck plan SCAN_ID` | `plan_migration` | Ordered steps: auto first, then manual |
| `portcheck apply STEP_ID PLAN_ID` | `apply_step` | Dry-run diff, or apply with `--apply --diff-sha256` |
| `portcheck capture WORKLOAD --label L` | `capture_run` | Save logits, greedy tokens and timings |
| `portcheck compare REF CAND` | `compare_captures` | Parity verdict |
| `portcheck cost CAPTURE --price-per-hour P` | `estimate_cost` | Cost per million tokens |
| `portcheck report` | `build_report` | One report from earlier results |

Exit codes: 0 on success, 1 on any error, 2 from `compare` when the verdict is `fail`.

## MCP server

`portcheck-mcp` runs over stdio. Add it to your MCP client's config, with the root set to
the repo you want to work on:

```json
{
  "mcpServers": {
    "portcheck": {
      "command": "portcheck-mcp",
      "env": { "PORTCHECK_ROOT": "/path/to/your/repo" }
    }
  }
}
```

Every tool returns the same envelope and never raises:

```json
{
  "ok": true,
  "run_id": "scan-20261112T101500Z-a1b2c3",
  "result_path": ".portcheck/runs/scan-20261112T101500Z-a1b2c3",
  "summary": "one sentence a person can read",
  "data": {}
}
```

On failure `ok` is false and `error` holds `code` and `message`. Codes:
`path_outside_root`, `not_found`, `invalid_input`, `config_missing`, `missing_dependency`,
`diff_mismatch`, `capture_mismatch`, `internal`.

## Skills

Copy the folders in [`skills/`](skills/) into your agent's skills directory:

- `gpu-port-audit`: "can this repo run on AMD?" Read-only.
- `gpu-port-migrate`: applies the auto steps, one approved diff at a time.
- `gpu-port-validate`: captures, compares, prices and reports.

## Configuration

Defaults are in `src/portcheck/defaults.yaml`. Override any key in `portcheck.yaml` in the
root directory:

```yaml
target:
  # Both defaults are on ROCm 10.0; sources and check dates are in defaults.yaml.
  rocm_image: "rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0"  # used by the PC007 fix
  torch_index_url: "https://download.pytorch.org/whl/rocm10.0"  # replaces CUDA wheel URLs (PC008)
parity:
  noise_multiplier: 3.0
  min_cosine: 0.999
  min_top1_agreement: 0.99
  max_abs_diff_floor: { float32: 0.001, float16: 0.05, bfloat16: 0.05 }
bench:
  warmup_batches: 2
  measured_batches: 5
scan:
  exclude: [".git", ".venv", "node_modules", ".portcheck", "__pycache__"]
  max_file_bytes: 1000000
```

## What the audit looks for

| Rule | Finding | Severity | Fix |
| --- | --- | --- | --- |
| PC001 | Custom CUDA source (`.cu`, `.cuh`, CUDA in build files) | red | manual |
| PC002 | TensorRT | red | manual |
| PC003 | FlashAttention package | red | manual |
| PC004 | NVIDIA-only library (DALI, cuDF, cuML, cuGraph) | red | manual |
| PC005 | Multi-GPU or NCCL use | yellow | manual |
| PC006 | Low-precision quantized path (FP8, FP4) | yellow | manual |
| PC007 | NVIDIA base image | yellow | auto |
| PC008 | NVIDIA pip pins | yellow | auto |
| PC009 | NVML or `nvidia-smi` call | yellow | auto for a plain import, else manual |
| PC010 | Plain `torch.cuda` usage | green | none needed |

Rules are data: see `src/portcheck/audit/rules.yaml`.

## Limits of v0.1

- It does not translate CUDA kernels. It flags them and points to HIPIFY and SCALE.
- Inference only, single GPU, `text-generation` only. No training or multi-GPU runs.
- The only target is AMD ROCm.
- A static scan can miss dependencies that are loaded dynamically. A `pass` from `compare`
  covers the prompts in your workload file, not every input.

## Project documents

- [`IMPLEMENTATION_SPEC.md`](IMPLEMENTATION_SPEC.md): the build spec.
- [`docs/SPEC_DEVIATIONS.md`](docs/SPEC_DEVIATIONS.md): where the implementation differs from the spec, and why.
- [`docs/RELEASE_TEST.md`](docs/RELEASE_TEST.md): the human runbook for the two-GPU release test.
- [`HANDOFF.md`](HANDOFF.md): project context and open points for the owner.

## Licence

Apache-2.0. See [`LICENSE`](LICENSE).
