# Reuse deterministic TraceLens evidence

Use the existing TraceLens Python reports for numerical evidence and the engineer
loop for interpretation and experiments. This is an optional route when the task
supplies applicable inference traces/reports and an installed tool; it does not
install TraceLens, expand source authority or reserve another GPU.

## Reuse the bound inputs and outputs

Follow references in `profiling_summary`, `codebase_context`, the COMMANDMENT or
task handoff. Check the actual tables and their definitions against the question:
source/backend/configuration, loaded model state, shapes/dtypes/strides, workload
recipe, rank/topology, phase/window and physical device. A checkpoint name alone
does not establish the transformed weights or mutable state used by the callable;
use the supplied capture/state records. One rank's trace does not imply TP=1.
Identify missing bindings and keep conclusions conditional where they matter.

Reuse matching raw CSVs and their report configuration before recalculating.
When analysis settings change but the trace still applies, rerun the report on
that trace rather than recollecting GPU activity. Refresh only the evidence that
a changed source, state, workload, device or question makes inapplicable. Retain
the original report and identify a new result directory for a changed analysis.

## Use the existing report entry point

Prefer the task's supplied command or resolved config. The standard inference
entry below runs Python analysis without a model call. Resolve the placeholders
from actual artifacts and check the installed version's options before first use:

```bash
mkdir -p "$TL_REPORT_DIR"
set -o pipefail
TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path "$TL_REPLAY_TRACE" \
  --capture_folder "$TL_CAPTURE_DIR" \
  --gpu_arch_json_path "$TL_ARCH_JSON" \
  --output_csvs_dir "$TL_REPORT_DIR" \
  --include_unlinked_kernels --enable_kernel_summary \
  2>&1 | tee "$TL_REPORT_DIR/report.log"
```

For graph replay, `capture_folder` supplies matching startup/capture metadata;
the CLI uses its existing classifier and capture/replay merger. Preserve the
capture files and `execution_details.json`, which can be reused as an input.
Reuse that classification only with its matching capture-file and rank lineage;
the CLI skips classification when `execution_details.json` already exists.
The classifier writes `execution_details.json` when absent; for frozen/read-only
capture inputs, use an authorized writable copy and preserve the originals.
Omit this option for an eager trace or when capture metadata is unavailable; a
replay-only report can retain useful timings while its operation/model coverage
remains incomplete. Do not substitute warmup timings for measured replay timings.

The same module exposes `generate_perf_report_pytorch(...)`; a caller performing
graph analysis passes the existing merger's result as `augmented_tree`. A supplied
`TraceLens_generate_perf_report_from_config` command is another existing route.
Reuse these paths rather than rebuilding the graph join or computing a separate
roofline in prose. Bind the actual arch JSON or installed extension to the device
and label specification versus empirical capacities. If no applicable hardware
model is available, retain the timings/work estimates and leave the roofline
unknown; do not substitute another product's capacities.

## Preserve meaning and acquire only what is needed

Keep references to raw tables/logs, command/resolved config, TraceLens and extension
versions, arch input and trace/capture/state identities in ordinary evidence notes
or existing `evidence_context.evidence_refs`. Record what remains unlinked or
unmodeled and the timing denominator/rank/phase. Missing rows or model values do
not mean zero cost; an emitted report is not proof of complete coverage.

TraceLens useful-work/semantic-byte models are not measurements of executed
instructions or HBM traffic. Per-operation roofline percentages, timeline activity,
summed GPU durations and request-serving throughput have different denominators.
Keep modeled savings, including fusion assumptions, separate from measured gains.
Inspect the pinned field implementation when a conversion or inference matters.

If the current decision needs missing shapes, state, rank coverage, graph joins or
hardware measurements, identify the exact observation and why it separates the
competing explanations. Use existing authorized capture/profile/build tools when
available; otherwise send that specific request to the task's collection owner
and record it as unresolved. Keep working on independent ideas. No fixed profiling
sequence or repeat collection per edit is required. The native correctness,
unprofiled timing and required serving/quality evaluations still judge the result.
