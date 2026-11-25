# Profiling environment contract

GEAK main already discovers rocprofv3 and rocprof-compute in the workload image.
GEAK's pip/bootstrap setup does not install the ROCm SDK, compute profiler or ATT
decoder. Reuse the workload images selected by `ci/docker_setup/`; prepare matching
ROCm packages or a versioned derived image before running a workflow. This does
not require changing CI scheduling or adding dependencies to the Gluon pack.

## Inspect in the execution environment

```bash
# run inside the workload container / environment that will profile
python3 kernel_workflow/scripts/kernel_tools/profiler_deps.py
```

The JSON report distinguishes absent tools from broken Python/library installs,
records CLI versions, and reports trace/PMC/compute support independently.
`llvm_dual_load.status = duplicate_inode` means the image will abort every rocprofv3 mode
once torch/Triton is loaded (see "Known abort"); ATT is then reported `llvm_dual_load`. It runs
no kernel and acquires no GPU lease. Exit zero means the report was produced;
read its statuses. `cli_available` does not establish GPU permissions, supported
counter names or a valid measured window. A missing/broken external container is
an error, never a fallback to inspecting the host.

`capture.sh` checks ATT dependencies in its actual locus and writes
`att_dependencies.json`. `rocprofv3_safe.sh` runs in the workload environment and
writes the same report before an ATT attempt. Ordinary profile/trace and CPU-only
work do not install or require the optional decoder.

## ATT and SDK pairing

- For a CLI exposing `--att-library-path`, check the configured library directories
  (`ROCPROF_ATT_LIBRARY_PATH`, `LD_LIBRARY_PATH`, ROCm lib paths). A candidate must
  load in that environment and expose `rocprof_trace_decoder_parse_data` (V1).
  An empty file, missing shared-library dependency or V2-only library is rejected
  as `decoder_incompatible`. A loadable V1 library is `external_candidate`, and its
  directory is passed explicitly to rocprofv3. This check does not certify the
  complete ABI or target compatibility.
- A CLI exposing ATT without a separate library-path option is recorded as
  `sdk_managed_unverified`. New SDKs can link/manage the decoder; absence of an
  external `.so` is not grounds for rejecting them. Let the SDK attempt the
  requested collection and validate the decoded artifacts.
- `decoder_absent`, `decoder_incompatible`, `unsupported_cli` and probe errors
  degrade the ATT layer with the reason retained. They do not become zero counters.

ROCm >= 7.13 / TheRock 10.0 images bundle `librocprof-trace-decoder` (0.2.0 with
rocprofiler-sdk 1.3.5 in `geak-gluon-ab:tri38-att`) in `/opt/rocm/lib`, where rocprofv3 finds it
without any environment variable; `ROCPROF_ATT_LIBRARY_PATH` is only needed for a decoder that
lives elsewhere. The standalone `ROCm/rocprof-trace-decoder` repository is deprecated and its
binaries are older; decoder and SDK sources are in `ROCm/rocm-systems/projects/`.

Pin ROCm, rocprofiler-sdk and decoder to a tested release combination in the image
inventory. Decoder packaging and ABI vary by release; inspect the installed
SDK's CLI and that release's documentation rather than inferring compatibility
from a version threshold or the presence of a `.so`. Do not install a floating
development decoder into an older SDK image. Preserve the selected release's
license; legacy prebuilt binaries and newer source releases have different packaging.

These are checks and instructions, **not an ATT installer**. Prepare the
SDK, compute analyzer and any external decoder while building the workload image,
then run the preflight inside that image. The optional AMD matrix calculator has
no ATT dependency.

Upstream sources:
[rocprofiler-sdk](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk),
[decoder](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprof-trace-decoder),
[legacy decoder migration](https://github.com/ROCm/rocprof-trace-decoder).

## Compute analyzer

Install rocprofiler-compute from the ROCm repository matching the workload or a
tested source revision under
[rocm-systems](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute).
Keep its Python requirements in the corresponding analyzer environment. For 3.4,
`pandas<3` avoids the known string/object dtype conversion failure. This constraint
is not a GEAK-wide or TraceLens requirement. CLI success still requires a small
real collection and analyzer run before a batch is considered validated.

## Known abort: two LLVM copies in one process

Symptom, in every rocprofv3 mode (`--kernel-trace`, `--pmc`, `--att`) once torch/Triton is imported:
`Option 'spirv-expand-step' registered more than once`, then `LLVM ERROR: inconsistency in registered
CommandLine options` and SIGABRT; the process then ignores SIGTERM until killed (`timeout -s KILL`).
`rocprofv3_safe.sh` watches for this line, stops the run and records `llvm_dual_load` (exit 7);
`capture.sh` exits 7 after its static layer.

Cause (TheRock images): `libamd_comgr.so.3`, which carries the SPIR-V translator, is loaded twice --
once through the rocprofv3 tool's RPATH (`/opt/rocm` = `_rocm_sdk_devel/lib`) and once by torch's
`rocm_sdk` preload (`_rocm_sdk_core/lib`). The files are byte-identical but have different inodes.

Fix the image, not the profiler call: make each identical file under `_rocm_sdk_devel/lib` resolve to
the `_rocm_sdk_core/lib` inode (idempotent; run as root, `S` = the venv's site-packages):

```bash
C="$S/_rocm_sdk_core/lib"; D="$S/_rocm_sdk_devel/lib"; cd "$C"
find . -type f -name '*.so*' -print0 | while IFS= read -r -d '' f; do
  d="$D/${f#./}"; [ -f "$d" ] && [ ! -L "$d" ] && cmp -s "$f" "$d" || continue
  [ "$(stat -L -c %i "$f")" = "$(stat -L -c %i "$d")" ] || ln -sf "$C/${f#./}" "$d"
done
```
- In a Dockerfile use **symlinks** (as above): hard links do not survive image layering (the
  core copy stays in a lower layer and the inodes differ again after `docker build`).
- In a running container (as root) hard links (`ln -f`) also work.
- `LD_LIBRARY_PATH=<core>/lib...` does not help (the tool's RPATH wins), and `LD_PRELOAD` of the core
  comgr fails differently (rocprofiler error 16, configuration outside the valid period).

Check with `profiler_deps.py` (`llvm_dual_load.status`): `single_inode` after the fix.

## Acceptance beyond the dependency probe

Before declaring an image ready for GPU campaigns, run a small workload through
trace, any required PMC groups, ATT decoding, and the selected analyzer. Check
actual dispatch/decoded output, target and device identity, errors and coverage.
For ATT, loadability or CLI support alone is explicitly unverified. Keep these GPU
results alongside the image digest and tool versions; CPU smoke cannot supply them.

## Shared collection

`profile_kernel.sh` remains the public baseline/reprofile entry and is unchanged.
Extra counter/ATT passes go through `capture.sh` or `rocprofv3_safe.sh` directly, which
handle allocation checks, timeout and one ATT SIMD retry. External-container capture goes
through `capture.sh` and the shared primitive's locus routing. A successful process must
also produce matching valid artifacts. Read `rocprofv3_safe.json`: collected, partial,
unavailable and failed are different outcomes. A dispatch trace does not establish
hardware-counter support.
