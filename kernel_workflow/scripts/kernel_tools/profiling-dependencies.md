# Profiling environment contract

GEAK main already discovers rocprofv3 and rocprof-compute in the workload image.
GEAK's pip/bootstrap setup does not install the ROCm SDK, compute profiler or ATT
decoder. Reuse the workload images selected by `ci/docker_setup/`; prepare matching
ROCm packages or a versioned derived image before running a workflow. This does
not require changing CI scheduling or adding dependencies to the Gluon pack.

## Inspect in the execution environment

```bash
bash kernel_workflow/scripts/kernel_tools/profiler_preflight.sh
# With an external container (mount the GEAK tool tree and configure locus paths):
TILE_KERNEL_CONTAINER=my-workload \
  bash kernel_workflow/scripts/kernel_tools/profiler_preflight.sh
```

The JSON report distinguishes absent tools from broken Python/library installs,
records CLI versions, and reports trace/PMC/compute support independently. It runs
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

Pin ROCm, rocprofiler-sdk and decoder to a tested release combination in the image
inventory. The old decoder repository says ROCm >=7.13 includes it by default;
current rocm-systems development also has a V2 API. Do not install a floating
development decoder into an older SDK image. Preserve the selected release's
license; old prebuilt binaries and the newer MIT source have different packaging.

Upstream sources:
[rocprofiler-sdk](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk),
[decoder](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprof-trace-decoder),
[legacy decoder migration](https://github.com/ROCm/rocprof-trace-decoder).

## Compute analyzer and native roofline

Install rocprofiler-compute from the ROCm repository matching the workload or a
tested source revision under
[rocm-systems](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute).
Keep its Python requirements in the corresponding analyzer environment. For 3.4,
`pandas<3` avoids the known string/object dtype conversion failure. This constraint
is not a GEAK-wide or TraceLens requirement. CLI success still requires a small
real collection and analyzer run before a batch is considered validated.

Where the native `kernel-roofline` skill is installed, its `references/installation.md`
owns TraceLens setup, frozen versions and calibration requirements. TL-only does
not need compute or ATT. Shared `roofline.sh` preserves that separation; this
dependency check does not add a counter pass, change frozen peaks or supply a
replacement roofline calculator. Use the existing GEAK GPU lease for actual
profiling and preserve native measured-window/source contracts.

## Acceptance beyond the dependency probe

Before declaring an image ready for GPU campaigns, run a small workload through
trace, any required PMC groups, ATT decoding, and the selected analyzer. Check
actual dispatch/decoded output, target and device identity, errors and coverage.
For ATT, loadability or CLI support alone is explicitly unverified. Keep these GPU
results alongside the image digest and tool versions; CPU smoke cannot supply them.
