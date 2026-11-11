# Release test runbook (human steps)

These steps are **for a person**. The coding agent that built v0.1 did not perform any of
them: no GPU was rented, nothing was run on NVIDIA or AMD hardware, nothing was published
to PyPI and the repository's visibility was not changed.

Everything in this repo has so far been exercised only on CPU with a tiny random model.
The release test is what turns "the code works" into "the tool can be trusted".

## 1. Before publishing anything

Confirm your employment IP and outside-work terms **in writing** before any code is
published or the repository is made public.

## 2. Choose the model and set verified config values

1. Choose the open model for the fixture and pin its revision (a commit hash). Edit
   `examples/e2e-service/workload.yaml`: set `model` to the hub id and `revision` to the
   commit hash. Until you do, `capture` on that file returns `invalid_input`.
2. Look up and verify the current values, then put them in `portcheck.yaml` in the
   directory you run from (and, if you want them as shipped defaults, in
   `src/portcheck/defaults.yaml` with a comment naming the source):

   ```yaml
   target:
     rocm_image: "<a rocm/pytorch image tag you have pulled and run>"
     torch_index_url: "<the PyTorch ROCm wheel index URL for that ROCm version>"
   ```

   `rocm_image` ships with a default tag that was confirmed to exist on Docker Hub on
   2026-11-11 but was never pulled or run; pull it, run it on the AMD machine and replace
   it if it does not suit the GPU and driver. `torch_index_url` ships empty: with it
   empty, the PC008 fix returns `config_missing` when a CUDA wheel URL is present.

## 3. Rent hardware

Rent one NVIDIA and one AMD data-centre GPU. On each machine install the PyTorch build
for that vendor first, then install this repo with `pip install '.[run]'` from a checkout.

## 4. Run the release scenario

Drive every step with an agent through the MCP server (`portcheck-mcp`) and the three
skills in `skills/`. Do not type CLI commands by hand: the point is to test the agent path.

1. Audit the fixture (`examples/e2e-service`).
2. Plan, review the diffs, apply the auto steps.
3. Build and run the migrated fixture on the AMD machine.
4. On NVIDIA: capture `ref` and `repeat`.
5. On AMD: capture `cand`. (Copy capture folders between machines by copying
   `.portcheck/runs/capture-...` directories; a capture can also be named by path.)
6. Compare and price both.
7. On AMD: capture with a fault injected; the compare must fail.
8. Build the report. Repeat steps 5 and 6 once.
9. Repeat step 1 from a second MCP client.

Use the same workload file and prompts file on both machines, byte for byte, with `model`
written as a hub id (not a machine-specific absolute path). Otherwise the
`workload_sha256` values differ and `compare_captures` returns `capture_mismatch`.

## 5. Release criteria

Release only if every check passes.

| Check | Pass criterion |
| --- | --- |
| Audit accuracy | All 5 planted problems found; no red finding outside the golden file |
| Migration safety | Nothing written before approval; diffs touch only the 3 auto-fixable items |
| Build | Migrated fixture serves a request on the AMD machine |
| Parity, good case | Verdict `pass` |
| Parity, bad case | Verdict `fail` for the fault-injected capture |
| Repeatability | Throughput on the re-run within 5% of the first run |
| Report | Both fingerprints, all parity checks, cost per million tokens for both machines |
| Data handling | No network calls from the server other than the model download |
| Portability | Same findings from the second MCP client |

## 6. Save the evidence

Save the transcript, fingerprints and report under `release-evidence/v0.1/`.

If the good case fails, inspect the metrics and recalibrate. The parity thresholds in
`defaults.yaml` are starting guesses. **Do not loosen thresholds just to pass.**

## 7. Publish

Publish to GitHub and PyPI. **The name `portcheck` is already taken on PyPI** by an
unrelated project (checked 2026-11-11: versions 0.2.0 to 0.7.0 exist). Choose another
distribution name, set it as `name` in `pyproject.toml`, and update the install command in
`README.md` and in `skills/gpu-port-audit/SKILL.md`. The import name and the `portcheck`
and `portcheck-mcp` commands can stay as they are.
