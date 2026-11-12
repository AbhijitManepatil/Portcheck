# GPU Port Check report

**Verdict:** Audit grade: **red**. Parity: **pass**. Cost ratio (candidate / reference, batch size 1): **8.6044**.

## Environment

| Field | `cpu-ref` (reference) | `amd-ref` (candidate) | `cpu-repeat` (baseline repeat) |
| --- | --- | --- | --- |
| capture | `capture-20261112T182951Z-08bc13` | `capture-20261112T182250Z-8d4507` | `capture-20261112T183037Z-5aaebf` |
| `vendor` | cpu | amd | cpu |
| `device_name` | n/a | AMD Radeon Pro V620 MxGPU | n/a |
| `device_count` | 0 | 1 | 0 |
| `torch_version` | 2.15.0+rocm10.0 | 2.15.0+rocm10.0 | 2.15.0+rocm10.0 |
| `cuda_version` | n/a | n/a | n/a |
| `rocm_version` | 7.15.26333 | 7.15.26333 | 7.15.26333 |
| `driver_version` | n/a | n/a | n/a |
| `python_version` | 3.12.16 | 3.12.16 | 3.12.16 |
| `platform` | Linux-6.8.0-1070-azure-x86_64-with-glibc2.35 | Linux-6.8.0-1070-azure-x86_64-with-glibc2.35 | Linux-6.8.0-1070-azure-x86_64-with-glibc2.35 |
| `portcheck_version` | 0.1.0 | 0.1.0 | 0.1.0 |

## Audit findings

Scan `scan-20261112T181539Z-7bbb8a` of `examples/e2e-service`: grade **red**, 9 files scanned.

| Severity | Rule | Title | File | Line | Fix | Action |
| --- | --- | --- | --- | --- | --- | --- |
| red | PC001 | Custom CUDA source | `ext/fast_op.cu` | n/a | manual | Port this CUDA source by hand. Evaluate HIPIFY (AMD's CUDA-to-HIP translator) and SCALE (a toolchain that compiles CUDA for AMD GPUs) as starting points, then rebuild and test on the AMD machine. portcheck does not translate kernels. |
| red | PC002 | TensorRT | `trt_backend.py` | 3 | manual | TensorRT runs only on NVIDIA GPUs. A person must replace this code path with one that works on ROCm, for example plain PyTorch, ONNX Runtime with the MIGraphX or ROCm provider, or MIGraphX directly, and re-validate the outputs. |
| yellow | PC007 | NVIDIA base image | `Dockerfile` | 1 | auto | Swap the base image to a ROCm PyTorch image (target.rocm_image in portcheck.yaml). |
| yellow | PC009 | NVML or nvidia-smi call | `gpu_stats.py` | 3 | auto | NVML and nvidia-smi exist only on NVIDIA machines. Guard the import so the code still loads elsewhere, and make each call site handle a missing NVML (on AMD, rocm-smi or amd-smi report the same kind of data). |
| yellow | PC008 | NVIDIA pip pins | `requirements.txt` | 3 | auto | Remove the NVIDIA-only pip packages and point any CUDA wheel index at the PyTorch ROCm index (target.torch_index_url in portcheck.yaml). |

Green findings (no change needed): 1

## Parity

Compare `compare-20261112T183651Z-caa862`: reference `capture-20261112T182951Z-08bc13`, candidate `capture-20261112T182250Z-8d4507`, dtype `float32`. Verdict: **pass**.

| Check | Value | Threshold | Result |
| --- | --- | --- | --- |
| `max_abs_diff` | 0.00011539459 | at most 0.001 | pass |
| `min_cosine` | 1.0 | at least 0.999 | pass |
| `top1_agreement` | 1.0 | at least 0.99 | pass |

Noise floor (measured against the repeat capture `capture-20261112T183037Z-5aaebf`): `max_abs_diff` 0.0, `mean_rel_error` 0.0.

Mean relative error between reference and candidate: 2.8519695e-06.

Greedy decoding (informational, never gates the verdict):

- `greedy_exact_match_rate` (informational): 1.0
- `mean_matching_prefix_tokens` (informational): 32.0

## Performance and cost

| Machine | Batch size | Tokens per second | `latency_p50_s` | `latency_p95_s` | Price per hour | Cost per million tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `cpu-ref` (cpu) reference | 1 | 26.381 | 1.214322 | 1.216059 | 0.635 USD | 6.6862 USD |
| `cpu-ref` (cpu) reference | 4 | 113.48 | 1.126818 | 1.132429 | 0.635 USD | 1.5544 USD |
| `amd-ref` (AMD Radeon Pro V620 MxGPU) candidate | 1 | 3.066 | 10.462384 | 10.505199 | 0.635 USD | 57.5306 USD |
| `amd-ref` (AMD Radeon Pro V620 MxGPU) candidate | 4 | 11.591 | 11.041873 | 11.087739 | 0.635 USD | 15.2177 USD |

Peak GPU memory: `amd-ref`: 590854656 bytes.

## How to reproduce

Workload hash (`workload_sha256`, covering the workload file and its prompts file): `929659904c410ee8689f3c9b76b117994f93d3277d620076cd864df48f4f7b30`

```bash
portcheck scan examples/e2e-service
portcheck capture <workload.yaml> --label cpu-ref   # on cpu; produced capture-20261112T182951Z-08bc13
portcheck capture <workload.yaml> --label amd-ref   # on AMD Radeon Pro V620 MxGPU; produced capture-20261112T182250Z-8d4507
portcheck capture <workload.yaml> --label cpu-repeat   # on cpu; produced capture-20261112T183037Z-5aaebf
portcheck compare capture-20261112T182951Z-08bc13 capture-20261112T182250Z-8d4507 --baseline-repeat capture-20261112T183037Z-5aaebf
portcheck cost capture-20261112T182951Z-08bc13 --price-per-hour 0.635 --currency USD
portcheck cost capture-20261112T182250Z-8d4507 --price-per-hour 0.635 --currency USD
portcheck report --scan scan-20261112T181539Z-7bbb8a --compare compare-20261112T183651Z-caa862 --cost cost-20261112T183835Z-a427ee --cost cost-20261112T183835Z-8b0b95
```

`<workload.yaml>` is the workload file whose hash is shown above. Run ids are created fresh on each run, so substitute the ids your own commands print.
