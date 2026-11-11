---
name: gpu-port-migrate
description: Applies the safe, mechanical changes that make a CUDA repo ready for AMD ROCm, as diffs the user approves. Use when the user asks to make a repo AMD-ready or to migrate it off NVIDIA.
---

# GPU port migrate

1. Call `scan_repo`, then `plan_migration` with the scan's `run_id`.
2. List the steps, auto first, then manual.
3. For each auto step, call `apply_step` with `dry_run=true` and show the diff.
4. Ask the user to approve. Only after a yes, call `apply_step` with `dry_run=false` and
   the `diff_sha256` from the dry run.
5. On `config_missing`, tell the user which key to set in `portcheck.yaml`
   (`target.rocm_image` or `target.torch_index_url`) and stop.
6. Finish with the manual steps still open. For PC009, list the call sites that need a
   guard (they are in the step's `detail`).

Rule: never apply a step the user has not seen as a diff.

If `apply_step` returns `diff_mismatch`, the files changed since the dry run: run the
dry run again and show the new diff before asking again.
