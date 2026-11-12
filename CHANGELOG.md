# Changelog

## 0.1.0 (unreleased)

Not yet published. Tested on CPU and on one AMD GPU (see below). Nothing has run on an
NVIDIA GPU, so the two-GPU release test in `docs/RELEASE_TEST.md` is still open.

### Built from the spec

- Audit: rule catalog PC001 to PC010, `portcheck scan`.
- Migrate: `portcheck plan` and `portcheck apply`, with three auto-fixes applied only
  against the hash of a reviewed diff.
- Validate: `portcheck capture` and `portcheck compare`, with a pass or fail parity verdict.
- Cost and report: `portcheck cost` and `portcheck report`.
- MCP server `portcheck-mcp` with 8 tools, and three Claude skills.

### After the first build

- Defaults for the ROCm image and the PyTorch ROCm wheel index (both ROCm 10.0), with
  sources.
- The fixture workload is pinned to SmolLM2-135M at a commit.
- The distribution is named `gpu-portcheck`; the import name and commands are `portcheck`.

### First run on a real AMD GPU

Azure `Standard_NG8ads_V620_v1`; evidence in `release-evidence/v0.1-azure-amd-v620/`.

- Verified: fingerprint, audit, migration, capture, compare, cost and report on an AMD
  Radeon PRO V620 with the default ROCm wheel index and image. CPU against AMD GPU passed
  parity; the fault-injected capture failed, as it must.
- Fixed: `driver_version` was empty on AMD when `rocm-smi` is not installed; it now falls
  back to the kernel module's version.
- Fixed: the PC009 guidance now says to catch `pynvml.NVMLError`, after the migrated
  fixture failed in a container where `pynvml` was installed without NVML.
- Not done: nothing has run on an NVIDIA GPU.

### Hardening

Found by an independent review and by hostile-input tests; details in
`docs/SPEC_DEVIATIONS.md`, items 33 to 45.

- Fixed: a fix could edit the wrong line in files containing form feeds or similar
  characters.
- Fixed: a UTF-8 byte-order mark hid findings on line 1.
- Fixed: the PC008 fix rewrote URLs that point at a file into dead links, and dropped the
  slash after an index URL.
- Fixed: a symlinked `.portcheck` folder let results be written outside the root.
- Fixed: a repeat capture containing NaN lifted the parity threshold so that anything
  passed.
- Fixed: one pathological file could abort a whole scan.
- Fixed: `FROM` text inside Dockerfile heredocs and continued commands was treated as an
  instruction; images given through `ARG` or with a `docker.io/` prefix were missed.
- Fixed: the PC009 fix dropped trailing comments and could break a line shared with
  another statement; the scanner now marks those cases manual.
- Fixed: decoding was not forced to greedy when a model's own generation config asked for
  beam search or penalties.
- Added: a warning in `compare` and the report when the reference machine's own two
  captures disagree on throughput by more than 5%.
- Added: `portcheck runs`, atomic writes in `apply`, a context-length check before
  capture, virtualenv and out-of-root symlink skipping in scans, shell-quoted reproduce
  commands, and a CI workflow.
