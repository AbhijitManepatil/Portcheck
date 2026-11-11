# e2e-service: portcheck test fixture

A small text-generation service with NVIDIA-only problems **planted on purpose**.
`portcheck` uses it as the audit, migration and release-test fixture. Do not
"fix" this folder: the tests compare a scan of it to
`tests/golden/e2e-service.findings.json`.

| File | Planted problem | Expected finding |
| --- | --- | --- |
| `Dockerfile` | NVIDIA CUDA base image | PC007, yellow, auto-fixable |
| `requirements.txt` | NVIDIA pip pin (`nvidia-cudnn-cu12`) | PC008, yellow, auto-fixable |
| `gpu_stats.py` | Top-level `import pynvml` | PC009, yellow, auto-fixable |
| `trt_backend.py` | `import tensorrt` | PC002, red, manual |
| `ext/fast_op.cu` | Custom CUDA kernel | PC001, red, manual |
| `serve.py` | Plain `torch.cuda` usage | PC010, green, no change needed |

`workload.yaml` and `prompts.jsonl` describe the validation workload (20 short
prompts). The model and its revision are left for a person to choose before the
release test.

Try it from the repo root:

    portcheck scan examples/e2e-service
