# Real-GPU test on Azure, AMD Radeon PRO V620 (2026-11-12)

This is evidence from the first run of portcheck on a real AMD GPU. It covers the AMD
half of the release test in `docs/RELEASE_TEST.md`. **It is not the full release test:**
no NVIDIA GPU was available, so there is no NVIDIA-versus-AMD comparison here.

## What ran

| Item | Value |
| --- | --- |
| Cloud | Azure, East US 2, VM size `Standard_NG8ads_V620_v1` (8 vCPUs, a quarter of one GPU) |
| GPU | AMD Radeon PRO V620, virtual function (`1002:73ae`), 7568 MB of GPU memory |
| OS | Ubuntu 22.04.5, kernel `6.8.0-1070-azure` |
| GPU driver | AMD `amdgpu-dkms` 6.4.4 (kernel module 6.12.12), with the firmware noted below |
| PyTorch | `2.15.0+rocm10.0` from `https://download.pytorch.org/whl/rocm10.0` (the repo's default index) |
| Model | `HuggingFaceTB/SmolLM2-135M` at the pinned commit, float32, 20 prompts |
| Code | branch `claude/magical-ptolemy-xy13dp` at commit `7b35313` |
| Retail price | 0.635 USD per hour (Azure retail price API, Linux, pay as you go) |

The VM was driven through Azure's run-command, without SSH. It existed for about 70
minutes and was deleted afterwards with its resource group.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Fingerprint identifies the machine | vendor `amd`, device `AMD Radeon Pro V620 MxGPU` | `results/cap-amd-ref.json` |
| Audit of the fixture | grade red: 2 red, 3 yellow, 1 green, as in the golden file | `results/scan.json` |
| Migrate | 3 auto steps applied from reviewed hashes; re-scan shows 0 yellow | `results/apply-S*.json`, `results/rescan.json` |
| Migrated service answers a request on the AMD GPU (virtualenv) | yes, exit 0 | `logs/serve.txt`, `logs/flow.log` |
| Migrated fixture builds on the default ROCm image | yes, `docker build` succeeded | `logs/docker.txt` |
| Migrated container answers a request on the AMD GPU | **partly:** it printed the generated text, then exited 1 in the NVML call | `logs/docker.txt` |
| Parity, CPU against AMD GPU (same VM) | **pass**: `max_abs_diff` 0.00011539459, threshold 0.001; cosine 1.0; top-1 1.0; all greedy outputs identical | `results/cmp-cpu-vs-amd.json` |
| Parity, AMD GPU against its own repeat | pass: all differences 0.0 | `results/cmp-amd-vs-amd.json` |
| Parity, fault-injected capture | **fail**, as it must (exit code 2) | `results/cmp-amd-vs-bad.json`, `report-fault-injected.md` |
| Repeatability of throughput | GPU 3.066 and 3.076 tokens/s at batch size 1 (0.3% apart); CPU 26.381 and 26.736 (1.3% apart) | `results/cap-*.json` |
| Report | built from the real results | `report.md` |

`report.md` compares the VM's CPU (reference) with its AMD GPU (candidate), both priced at
the VM's hourly rate.

## What this run taught us

1. **The shipped defaults work on real AMD hardware.** The default wheel index gave a
   PyTorch that sees and uses the V620, and the default ROCm image built the fixture.
2. **CPU and AMD GPU agree closely.** The largest logit difference was about 0.000115
   in float32, roughly a ninth of the 0.001 floor. The thresholds were not changed. This
   is one model on one GPU; it is a data point, not a calibration.
3. **A guarded import is not enough for NVML.** In the container `pynvml` is installed,
   so the import guard added by the PC009 fix passes, and `pynvml.nvmlInit()` then raises
   `NVMLError_LibraryNotFound`. The plan already marked the call sites as manual work, but
   its advice (`if pynvml is not None`) was incomplete. The step detail and the rule text
   now also say to catch `pynvml.NVMLError`.
4. **`driver_version` was empty.** PyTorch's ROCm wheels bundle ROCm, so `rocm-smi` was
   not installed. The fingerprint now falls back to the kernel module's version
   (`results/fingerprint-after-fix.json` shows `6.12.12`, taken on this VM after the fix).
   The captures above were made before the fix, so they show no driver version.
5. **`rocm_version` is the HIP version.** The spec defines it as `torch.version.hip`,
   which reports `7.15.26333` for the ROCm 10.0 build. It is not the ROCm release number.
6. **This GPU size is slow for this workload.** A quarter of a V620 generated about 3
   tokens per second at batch size 1, against about 26 on the VM's 8 CPU cores. A 4096 by
   4096 matrix product took about the same time on both. The numbers say this VM size is
   a place to check correctness, not performance; they say nothing about full AMD GPUs.

## Not covered

- No NVIDIA GPU: the subscription had no NVIDIA quota. NVIDIA-versus-AMD parity and cost
  are still untested.
- One model, one dtype (float32), one GPU model, a partial GPU.
- The run was driven by an agent through the CLI on the VM, not through the MCP server.

## Driver setup that worked

See `docs/AZURE_V620_SETUP.md`. The short version: AMD's public `amdgpu-dkms` 6.4.4, plus
two firmware files (`sienna_cichlid_cap.bin`, `sienna_cichlid_vcn.bin`) taken from the
Azure-specific driver build that Microsoft's V710 guide points to. With the public
firmware the driver fails to initialise on this VM.
