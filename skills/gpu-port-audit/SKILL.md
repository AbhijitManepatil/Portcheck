---
name: gpu-port-audit
description: Checks whether a repo that runs on NVIDIA CUDA can run on AMD ROCm. Use when the user asks if code can run on AMD, wants a CUDA dependency audit, or asks what blocks a move off NVIDIA.
---

# GPU port audit

1. If the `scan_repo` tool is not available, tell the user to install the server with
   `pip install git+https://github.com/AbhijitManepatil/Portcheck.git` and add
   `portcheck-mcp` to their MCP config. Then stop. Do not suggest `pip install portcheck`:
   that name on PyPI belongs to a different project.
2. Call `env_fingerprint`, then `scan_repo` on the repo root.
3. Answer in this order:
   - The grade in one sentence.
   - Up to three findings that matter most, red first, each with file and line.
   - What is safe: mention the green count in one line.
4. For red findings, name the manual work. Never say the tool can fix a red finding.
5. Offer the next step: "make this repo AMD-ready" (migrate) or "prove the results match" (validate).

Only quote findings returned by the tool. This skill is read-only.
