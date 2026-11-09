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
inventory. Decoder packaging and ABI vary by release; inspect the installed
SDK's CLI and that release's documentation rather than inferring compatibility
from a version threshold or the presence of a `.so`. Do not install a floating
development decoder into an older SDK image. Preserve the selected release's
license; legacy prebuilt binaries and newer source releases have different packaging.

This PR provides checks and instructions, **not an ATT installer**. Prepare the
SDK, compute analyzer and any external decoder while building the workload image,
then run the preflight inside that image. The optional AMD matrix calculator has
its own [pinned preparation flow](dependencies.md); it has no ATT dependency.
`hw_sources.sh decode` uses `isa_spec_cli` for static instruction bytes, which is
separate from decoding a captured ATT trace.

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

The native `kernel-roofline` runner is not bundled by this PR or by the reviewed
GEAK main. `roofline.sh` looks under `GEAK_KERNEL_ROOFLINE_DIR` (default:
`e2e_workflow/knowledge/analysis_skills/kernel-roofline`) for its scripts and
returns 3 when missing. Install a compatible external runner explicitly; its
versioned `references/installation.md` owns TraceLens setup and calibration.
The existing e2e legacy roofline and TraceLens-prior import remain separate paths. TL-only does
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

## Shared collection and task payloads

`profile_kernel.sh` remains the public baseline/reprofile entry. Its optional
PMC/ATT/SPI layers and `capture.sh` use `rocprofv3_safe.sh` for rocprofv3 execution,
including allocation checks, timeout and one ATT SIMD retry. Run the main
`profile_kernel.sh` inside the workload environment, as main already requires;
it refuses `TILE_KERNEL_CONTAINER` to prevent mixing a host baseline with
container optional layers. External-container capture goes through `capture.sh`
and the shared primitive's locus routing. A successful process must also produce matching valid artifacts. Read
`rocprofv3_safe.json` and `profile_layers.json`: collected, partial, unavailable
and failed are different outcomes; a failed optional layer does not fail the
main profile. A dispatch trace does not establish hardware-counter support.

`capture.sh --payload FILE` uses the public
[`profile_payload.py`](profile_payload.py) and
[`profile_payload.schema.json`](profile_payload.schema.json). Its existing
`tile.profile-payload/1` schema identifier is retained for compatibility. Language
packs supply payloads and may keep forwarding shims; no public collector imports
a pack. The validator checks source SHA, execution locus, allowed host roots and
argv; payload environment entries cannot override device/allocation/locus control.
The PCI BDF and image inventory fields are supplied provenance, not independently
verified hardware attestation. For a live identity check use hardware_context.py.

`kernel_breakdown.py` displays static instructions, profiler metrics and ATT
observations in their own units. It does not invent idle-cycle estimates or rank
optimizations. Ideal matrix-instruction cadence is unknown unless the caller
provides a reference for that instruction and architecture.
