# Release test runbook (human steps)

These steps are **for a person**. The coding agent that built v0.1 did not perform any of
them: no GPU was rented, nothing was run on NVIDIA or AMD hardware, nothing was published
to PyPI and the repository's visibility was not changed.

Everything in this repo has so far been exercised only on CPU with a tiny random model.
The release test is what turns "the code works" into "the tool can be trusted".

## 1. Before publishing anything

Confirm your employment IP and outside-work terms **in writing** before any code is
published or the repository is made public.

## 2. Confirm the model and the config values

1. The fixture workload (`examples/e2e-service/workload.yaml`) is pinned to the open model
   `HuggingFaceTB/SmolLM2-135M` (Apache-2.0) at commit
   `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It has been captured and compared on CPU.
   Keep it, or replace `model` and `revision` with the model you want to release against.
2. The shipped defaults in `src/portcheck/defaults.yaml` are both on ROCm 10.0:

   ```yaml
   target:
     rocm_image: "rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0"
     torch_index_url: "https://download.pytorch.org/whl/rocm10.0"
   ```

   Both were confirmed to exist on 2026-11-11 (sources are in comments in that file).
   Neither was pulled, installed or run on an AMD GPU. Pull the image and install from
   the index on the AMD machine; if they do not suit its GPU and driver, or a newer
   release is out, override them in `portcheck.yaml` in the directory you run from.

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

Publish to GitHub and PyPI. The name `portcheck` on PyPI belongs to an unrelated project,
so the distribution is named `gpu-portcheck` in `pyproject.toml` (free on PyPI when checked
on 2026-11-11; check again before uploading). The import name and the `portcheck` and
`portcheck-mcp` commands are unchanged. After the first upload, change the install commands
in `README.md` and `skills/gpu-port-audit/SKILL.md` from the GitHub URL to
`pip install gpu-portcheck`.
