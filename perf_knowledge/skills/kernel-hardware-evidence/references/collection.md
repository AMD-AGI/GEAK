# Choose and collect evidence

Start with the measured full call, then choose the smallest useful diagnostic.
Use the task's supplied commands, resource limits and recorded tool-support checks.
Installed tools or passing CPU checks do not establish GPU collection support.

| Uncertainty | Useful evidence |
|---|---|
| Which stage or implementation costs time? | Matched dispatch trace, launch counts and source/code-object attribution |
| Reuse, traffic or materialized intermediates? | Source/IR/ISA dataflow, then documented counters at the relevant interface |
| Arithmetic, communication or latency hiding? | Actual instructions, launch geometry, resources, spills; targeted stall/occupancy metrics |
| Roofline fraction? | Matched work/traffic/time and an applicable capacity; see [roofline](roofline.md) |

Keep correctness, warmup, timing and profiling phases identifiable. Runtime guards,
input reuse or graph capture can select different paths. Trace/configuration
coverage must establish the path being discussed; names and heuristic-choice logs
alone do not. Record unattributed launches instead of silently discarding them.

Inspect actual fields and definitions in raw artifacts. For text use `rg -n` and
read the surrounding section; for JSON/CSV select named rows/fields with Python.
Read coverage/error fields as well as the metric. Missing, unsupported and measured
zero differ. Preserve counter units, device/subunit reductions, invocation counts
and replay intervals before deriving rates. Match each counter to the relevant
call where the capture permits; narrow claims when it does not.

Before attributing event duration, cross-rank waiting or stragglers to a bottleneck,
check timestamp/clock semantics and the meaning of overlapping recorded intervals,
including graph replay. Inspect the cross-rank join rule: justify same-operation
identity and expected participants using per-rank counts/order and available
group/message metadata. Matching names or ordinals, or uniformly missing fields,
are insufficient on their own. Index offsets and nearest-time matches can diagnose
mispairing but do not validate a replacement join. Keep unresolved events for
exploration and narrow timing or bottleneck claims that depend on them.

## AMD tool recipes

Check the installed CLI first; documentation for `latest` may describe another
version. These rocprofiler-compute (Compute) 3.7 command examples require a tested
local setup. Replace `COMPUTE`, `ROCPROFV3` and `RUNNER` with its pinned commands;
the examples alone do not establish GPU support.

```text
COMPUTE --version
COMPUTE profile --help
ROCPROFV3 --version
ROCPROFV3 --list-avail
COMPUTE profile --block 4 --no-roof --format-rocprof-output csv --output-directory FRESH_PROFILE -- RUNNER
COMPUTE analyze --path COLLECTED_PROFILE --block 4
```

Block 4 provides roofline events; block 2 is the theoretical speed-of-light panel.
Use the task's available profile for the question. `--no-roof` avoids implicit calibration;
attach matching calibration before calling a comparison empirical. Fresh collection
needs a fresh output directory: existing `pmc_perf.csv` or `roofline.csv` can be
reused silently. Reanalysis must keep the original source/device/phase identity.

Known version-sensitive checks, relevant only when that path is used:

- Compute 3.4 has affected gfx950 memory-clock/read formulas and MFMA calibration.
  The [3.5 clock](https://github.com/ROCm/rocm-systems/commit/793732a04e3753e7245a4e11721179e7a13b2a94),
  [3.5 read](https://github.com/ROCm/rocm-systems/commit/aa5dfb98f96ab080de97259df64ba0d36f6796b4)
  and [3.6 MFMA](https://github.com/ROCm/rocm-systems/commit/c89eb9929b672b98b5dd2933f4fa4c4e8f89addb)
  corrections affect different stages. Reanalysis cannot supply missing events
  or repair the calibration binary that ran. An older BF16 shortfall belongs to
  that benchmark binary/device; do not apply a universal twofold correction.
- Reviewed Compute 3.7 (`therock-7.14`) BF16 selectors used F16 in two places. Check the actual
  plotted operands against `MFMABF16Flops`; a label or version number is insufficient.
- The reviewed 3.7 (`therock-7.14`) full calibration LDS benchmark had an out-of-bounds first read.
  Use the selected calibration benchmarks tested on that host; do not automatically
  fall back to `--bench-only` or `--roof-only` to obtain an unrelated capacity.
- A missing capacity remains unknown. Retain other supported comparisons rather
  than substituting another dtype or discarding all measured points.
- Counter names and scaling are version-specific. Verify the installed definition
  of `SQ_INSTS_VALU_MFMA_MOPS_F8` before converting it to FLOPs; its name and an
  instruction shape alone do not establish the multiplier. Do not assume a
  `MfmaFlopsF8` field exists. Retain the event definition and conversion with the result.

See the pinned [profile-mode documentation](https://github.com/ROCm/rocm-systems/blob/therock-7.14/projects/rocprofiler-compute/docs/how-to/profile/mode.rst).
Verify support with the installed executable, device and execution setup.

## Collection failures

Preserve the error and partial artifacts; distinguish preparation, execution,
analysis and publication failures from numerical failure. Reanalyze or republish
retained data when possible instead of rerunning successful GPU work. Check the
actual failure before changing dependencies or limits. Use owned writable working,
output and cache directories; some ROCm 7.2 paths also require `ROCPROF_TMPDIR`.
Keep profiler dependencies separate from benchmark imports.

Wait for the profiling job to finish and use the returned artifact paths; a host-only path
or successful CLI exit does not prove complete accessible output. Verify the
selected device and expected files/fields. Pending or failed cleanup belongs to
the existing operation owner, not a new allocation or reset clock.
