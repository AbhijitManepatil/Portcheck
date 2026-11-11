<!--
Sample output of `portcheck report`, kept as an example of what a report looks like.

How it was made (2026-11-11): the full flow on the fixture in examples/e2e-service with
the pinned model HuggingFaceTB/SmolLM2-135M and the shipped defaults: scan, plan, apply
the 3 auto steps, four captures (ref, repeat, cand, and one with --inject-fault 0.5),
compare, cost with made-up prices (2.50 and 1.90 USD per hour), report.

Read it with care:
- All three captures ran on the SAME CPU-only machine. "reference" and "candidate" are
  just labels here. This report says nothing about NVIDIA versus AMD.
- Parity is exact (all differences 0.0) because one machine repeated the same
  deterministic computation. Across two different GPUs the differences will not be zero.
- The repeatability warning in the cost section is real: on this shared virtual machine
  two back-to-back captures differed by 8% in throughput at batch size 1.

Everything below this comment is the tool's output, unedited.
-->

# GPU Port Check report

**Verdict:** Audit grade: **red**. Parity: **pass**. Cost ratio (candidate / reference, batch size 1): **0.8726**.

## Environment

| Field | `ref` (reference) | `cand` (candidate) | `repeat` (baseline repeat) |
| --- | --- | --- | --- |
| capture | `capture-20261111T205312Z-6c8296` | `capture-20261111T205432Z-e12d9f` | `capture-20261111T205351Z-1767f7` |
| `vendor` | cpu | cpu | cpu |
| `device_name` | n/a | n/a | n/a |
| `device_count` | 0 | 0 | 0 |
| `torch_version` | 2.15.0+cu132 | 2.15.0+cu132 | 2.15.0+cu132 |
| `cuda_version` | 13.2 | 13.2 | 13.2 |
| `rocm_version` | n/a | n/a | n/a |
| `driver_version` | n/a | n/a | n/a |
| `python_version` | 3.13.18 | 3.13.18 | 3.13.18 |
| `platform` | Linux-6.18.44-fc-v378-x86_64-with-glibc2.39 | Linux-6.18.44-fc-v378-x86_64-with-glibc2.39 | Linux-6.18.44-fc-v378-x86_64-with-glibc2.39 |
| `portcheck_version` | 0.1.0 | 0.1.0 | 0.1.0 |

## Audit findings

Scan `scan-20261111T205232Z-4cb413` of `examples/e2e-service`: grade **red**, 9 files scanned.

| Severity | Rule | Title | File | Line | Fix | Action |
| --- | --- | --- | --- | --- | --- | --- |
| red | PC001 | Custom CUDA source | `ext/fast_op.cu` | n/a | manual | Port this CUDA source by hand. Evaluate HIPIFY (AMD's CUDA-to-HIP translator) and SCALE (a toolchain that compiles CUDA for AMD GPUs) as starting points, then rebuild and test on the AMD machine. portcheck does not translate kernels. |
| red | PC002 | TensorRT | `trt_backend.py` | 3 | manual | TensorRT runs only on NVIDIA GPUs. A person must replace this code path with one that works on ROCm, for example plain PyTorch, ONNX Runtime with the MIGraphX or ROCm provider, or MIGraphX directly, and re-validate the outputs. |
| yellow | PC007 | NVIDIA base image | `Dockerfile` | 1 | auto | Swap the base image to a ROCm PyTorch image (target.rocm_image in portcheck.yaml). |
| yellow | PC009 | NVML or nvidia-smi call | `gpu_stats.py` | 3 | auto | NVML and nvidia-smi exist only on NVIDIA machines. Guard the import so the code still loads elsewhere, and make each call site handle a missing NVML (on AMD, rocm-smi or amd-smi report the same kind of data). |
| yellow | PC008 | NVIDIA pip pins | `requirements.txt` | 3 | auto | Remove the NVIDIA-only pip packages and point any CUDA wheel index at the PyTorch ROCm index (target.torch_index_url in portcheck.yaml). |

Green findings (no change needed): 1

## Parity

Compare `compare-20261111T205513Z-2569f0`: reference `capture-20261111T205312Z-6c8296`, candidate `capture-20261111T205432Z-e12d9f`, dtype `float32`. Verdict: **pass**.

| Check | Value | Threshold | Result |
| --- | --- | --- | --- |
| `max_abs_diff` | 0.0 | at most 0.001 | pass |
| `min_cosine` | 1.0 | at least 0.999 | pass |
| `top1_agreement` | 1.0 | at least 0.99 | pass |

Noise floor (measured against the repeat capture `capture-20261111T205351Z-1767f7`): `max_abs_diff` 0.0, `mean_rel_error` 0.0.

Mean relative error between reference and candidate: 0.0.

Greedy decoding (informational, never gates the verdict):

- `greedy_exact_match_rate` (informational): 1.0
- `mean_matching_prefix_tokens` (informational): 32.0

## Performance and cost

| Machine | Batch size | Tokens per second | `latency_p50_s` | `latency_p95_s` | Price per hour | Cost per million tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `ref` (cpu) reference | 1 | 38.768 | 0.835662 | 0.866433 | 2.5 USD | 17.9128 USD |
| `ref` (cpu) reference | 4 | 95.214 | 1.372111 | 1.379104 | 2.5 USD | 7.2935 USD |
| `cand` (cpu) candidate | 1 | 33.766 | 0.95546 | 0.987808 | 1.9 USD | 15.6305 USD |
| `cand` (cpu) candidate | 4 | 91.214 | 1.403444 | 1.434501 | 1.9 USD | 5.7861 USD |

**Warning: timings are not repeatable.** The reference machine's two captures disagree on throughput by more than the release test's repeatability limit. Capture again on an idle machine before trusting the performance and cost figures above. The parity verdict is not affected.

## How to reproduce

Workload hash (`workload_sha256`, covering the workload file and its prompts file): `929659904c410ee8689f3c9b76b117994f93d3277d620076cd864df48f4f7b30`

```bash
portcheck scan examples/e2e-service
portcheck capture <workload.yaml> --label ref   # on cpu; produced capture-20261111T205312Z-6c8296
portcheck capture <workload.yaml> --label cand   # on cpu; produced capture-20261111T205432Z-e12d9f
portcheck capture <workload.yaml> --label repeat   # on cpu; produced capture-20261111T205351Z-1767f7
portcheck compare capture-20261111T205312Z-6c8296 capture-20261111T205432Z-e12d9f --baseline-repeat capture-20261111T205351Z-1767f7
portcheck cost capture-20261111T205312Z-6c8296 --price-per-hour 2.5 --currency USD
portcheck cost capture-20261111T205432Z-e12d9f --price-per-hour 1.9 --currency USD
portcheck report --scan scan-20261111T205232Z-4cb413 --compare compare-20261111T205513Z-2569f0 --cost cost-20261111T205513Z-30ac9c --cost cost-20261111T205514Z-992235
```

`<workload.yaml>` is the workload file whose hash is shown above. Run ids are created fresh on each run, so substitute the ids your own commands print.
