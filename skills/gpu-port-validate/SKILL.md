---
name: gpu-port-validate
description: Proves a model gives matching results on NVIDIA and AMD and compares cost. Use when the user asks whether outputs match across GPUs, wants a parity check, or asks what a workload costs on AMD versus NVIDIA.
---

# GPU port validate

1. Call `env_fingerprint` to learn which machine this is.
2. Check which captures exist under `.portcheck/runs/`. A full check needs three: a
   reference and a repeat from the NVIDIA machine, and a candidate from the AMD machine.
   With a shell, `portcheck runs --kind capture` lists them with label and machine.
3. If this machine's capture is missing, call `capture_run` with the workload file.
4. If a capture from the other machine is missing, tell the user the exact command to run
   there (`portcheck capture <workload.yaml> --label <name>`) and how to copy the capture
   folder back into `.portcheck/runs/`. Then stop.
5. With all three present, call `compare_captures` (reference, candidate, and the repeat
   as `baseline_repeat`), then `estimate_cost` for each machine (ask the user for each
   hourly price), then `build_report`.
6. Lead with the verdict. State GPU, driver and ROCm or CUDA version beside every
   performance number.

Rules: quote numbers only from tool output. If the verdict is fail, show the failed checks
and do not suggest loosening thresholds.
