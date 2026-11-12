# GPU Port Check report

**Verdict:** Parity: **fail**.

## Environment

| Field | `amd-ref` (reference) | `amd-bad` (candidate) | `amd-repeat` (baseline repeat) |
| --- | --- | --- | --- |
| capture | `capture-20261112T182250Z-8d4507` | `capture-20261112T183650Z-c21ed0` | `capture-20261112T182904Z-7bdd9c` |
| `vendor` | amd | amd | amd |
| `device_name` | AMD Radeon Pro V620 MxGPU | AMD Radeon Pro V620 MxGPU | AMD Radeon Pro V620 MxGPU |
| `device_count` | 1 | 1 | 1 |
| `torch_version` | 2.15.0+rocm10.0 | 2.15.0+rocm10.0 | 2.15.0+rocm10.0 |
| `cuda_version` | n/a | n/a | n/a |
| `rocm_version` | 7.15.26333 | 7.15.26333 | 7.15.26333 |
| `driver_version` | n/a | n/a | n/a |
| `python_version` | 3.12.16 | 3.12.16 | 3.12.16 |
| `platform` | Linux-6.8.0-1070-azure-x86_64-with-glibc2.35 | Linux-6.8.0-1070-azure-x86_64-with-glibc2.35 | Linux-6.8.0-1070-azure-x86_64-with-glibc2.35 |
| `portcheck_version` | 0.1.0 | 0.1.0 | 0.1.0 |

**Fault injected:** `capture-20261112T183650Z-c21ed0` ran with deliberately corrupted weights (negative test).

## Parity

Compare `compare-20261112T183652Z-006c05`: reference `capture-20261112T182250Z-8d4507`, candidate `capture-20261112T183650Z-c21ed0`, dtype `float32`. Verdict: **fail**.

| Check | Value | Threshold | Result |
| --- | --- | --- | --- |
| `max_abs_diff` | 149.40211 | at most 0.001 | FAIL |
| `min_cosine` | -0.51040887 | at least 0.999 | FAIL |
| `top1_agreement` | 0.0 | at least 0.99 | FAIL |

Noise floor (measured against the repeat capture `capture-20261112T182904Z-7bdd9c`): `max_abs_diff` 0.0, `mean_rel_error` 0.0.

Mean relative error between reference and candidate: 3.630691.

Greedy decoding (informational, never gates the verdict):

- `greedy_exact_match_rate` (informational): 0.0
- `mean_matching_prefix_tokens` (informational): 0.0

## How to reproduce

Workload hash (`workload_sha256`, covering the workload file and its prompts file): `929659904c410ee8689f3c9b76b117994f93d3277d620076cd864df48f4f7b30`

```bash
portcheck capture <workload.yaml> --label amd-ref   # on AMD Radeon Pro V620 MxGPU; produced capture-20261112T182250Z-8d4507
portcheck capture <workload.yaml> --label amd-bad --inject-fault <std>   # on AMD Radeon Pro V620 MxGPU; produced capture-20261112T183650Z-c21ed0
portcheck capture <workload.yaml> --label amd-repeat   # on AMD Radeon Pro V620 MxGPU; produced capture-20261112T182904Z-7bdd9c
portcheck compare capture-20261112T182250Z-8d4507 capture-20261112T183650Z-c21ed0 --baseline-repeat capture-20261112T182904Z-7bdd9c
portcheck report --compare compare-20261112T183652Z-006c05
```

`<workload.yaml>` is the workload file whose hash is shown above. Run ids are created fresh on each run, so substitute the ids your own commands print.
