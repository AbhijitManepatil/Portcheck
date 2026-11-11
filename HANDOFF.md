# Handoff to Claude Code: GPU Port Check (`portcheck`)

Written 2026-11-12. Read this first, then `IMPLEMENTATION_SPEC.md` (same project). The spec is the source of truth; this file is the context around it.

## What the owner asked for

Abhijit (GitHub: `AbhijitManepatil`) asked to:

1. Create a new repo under https://github.com/AbhijitManepatil
2. Build the complete application end to end from `IMPLEMENTATION_SPEC.md`
3. Push everything to that repo

## Where things stand

- **The repo exists:** https://github.com/AbhijitManepatil/Portcheck. It holds only this file, `IMPLEMENTATION_SPEC.md` and a short `README.md`.
- **No application code is written yet.** Start at task T1.
- The spec is finished: `IMPLEMENTATION_SPEC.md`, 14 sections, tasks T1 to T12 in section 12.
- A human-readable spec and MVP plan also exists as a Claude doc: https://claude.ai/artifact/Tf1SygBUQ7gQkj8CjEMmwe

## What the product is

An open-source, self-hosted Python toolkit that checks whether a CUDA-based AI inference workload can move from NVIDIA to AMD ROCm, and proves it with measurements. One core library with three surfaces: a CLI (`portcheck`), an MCP server (`portcheck-mcp`, 8 tools over stdio), and three Claude skills. Four capabilities: audit, migrate (safe edits only), validate (output parity), cost and report.

## Decisions already made

- **Packaging:** free GitHub repo plus MCP server and skills. Not a hosted service.
- **The value is validation, not code translation.** Kernel translation is left to HIPIFY and SCALE; the tool only flags it.
- **Everything runs on the user's machine.** No network calls except the model download during capture.
- **Stack:** Python 3.10+, `hatchling`, `pydantic>=2`, `typer`, `pyyaml`, `numpy`, `safetensors`, official `mcp` SDK (`FastMCP`). `torch` and `transformers` are an optional extra, imported lazily.
- **Licence:** Apache-2.0.
- **Parity verdict:** threshold is `max(3 x noise floor, fixed minimum per dtype)`, plus cosine and top-1 minimums. Greedy-decoding match is informational only.
- **Tests:** all on CPU, no network, using a tiny randomly initialised GPT-2 built locally.

## Open points for the owner (ask, do not guess)

1. **Package name.** The repo is `Portcheck`; the Python package is `portcheck`. PyPI availability is unchecked.
2. **Visibility.** Do not change the repo's visibility. The spec's own week-0 step is for the owner to confirm employment IP and outside-work terms in writing before publishing.
3. **ROCm image tag and PyTorch ROCm wheel index URL.** Deliberately left empty in the spec. Look up current values and cite the source, or leave empty.
4. **Parity thresholds** are starting guesses, to be calibrated in the real two-GPU release test.

## Do not do

These are human-only steps (spec section 13): renting GPUs, running the real NVIDIA and AMD release test, publishing to PyPI, and making the repo public without the owner saying so.

## Environment notes from the previous session

- Python 3.13 was present. `torch`, `transformers`, `typer` were not installed; `pip install` works.
- The spec's layout shows a top-level `portcheck/` folder; that is the repo root itself, so `pyproject.toml` and `src/` go directly at the root of this repo.
- Commit as the owner's configured git identity where one exists.

## Suggested kickoff prompt for Claude Code

```
Read HANDOFF.md and IMPLEMENTATION_SPEC.md at the root of this repo
(AbhijitManepatil/Portcheck). Implement tasks T1 to T12 from section 12 of the
spec in order, building the package at the repo root. After each task, run
`ruff check .` and `pytest -q`, commit, and push. Follow the rules in section 3.
Do not attempt anything in section 13. When done, give me the final summary the
spec asks for, including any deviations and anything skipped.
```

## Background: why this shape

The idea was pressure-tested before the spec was written. The findings that shaped it:

- The problem statement is the brief for AMD's developer hackathons on lablab.ai; dozens of near-identical "migration copilot" projects exist, and AMD ships HIPIFY and ROCm.ai free.
- Plain PyTorch inference already ports easily with ROCm. The unsolved part is trustworthy, reproducible proof on the customer's own model: same numbers, what speed, what cost.
- Enterprises will not send weights or prompts to an outsider, so the tool must be self-hosted.
- Recommended path: ship open source, publish a few reports, and only treat it as a business if teams ask to have it run on their own models.
