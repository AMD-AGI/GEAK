#!/usr/bin/env bash
# Called by the standard worker for a qualified frozen task using native CLI tools.
set -euo pipefail
: "${PROMPT_OUT:?}" "${TASK_DIR:?}" "${KERNEL_SRC_ALL:?}" "${GPU_IDS:?}"
GEAK_ROOT="${GEAK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
SCRIPTS="$GEAK_ROOT/workload/scripts"
EDITABLE="$(for file in $KERNEL_SRC_ALL; do printf '  %s/kernel_src/%s\n' "$TASK_DIR" "$file"; done)"
UT_WALL="$(python3 - "$TASK_DIR/run/ut_result.json" <<'PY2'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("wall_s", "unknown"))
except (OSError, ValueError):
    print("unknown")
PY2
)"
cat > "$PROMPT_OUT" <<EOF
Optimize the GPU task in $TASK_DIR using your native coding tools. This is attempt $ATTEMPT; the best
measured whole-task speedup so far is ${BEST_SPEEDUP}x against the original implementation.
Your attempt has ${ATTEMPT_CAP} seconds; keep useful source changes on disk before it ends.

The profile stage has already qualified this task's frozen harness and recorded baseline
case timings. Read that evidence; focus this attempt on kernel changes instead of building
a new harness. Work directly with the existing build path and editable device sources.
This method requires no Workflow tool or kernel_workflow.js runtime. Keep the work in this
one runner, and serialize GPU experiments. The container exposes the assigned logical GPU
set [$GPU_IDS]; preserve that visibility. For a multi-rank task, launch the real collective
on this entire set. A pool lock that picks one GPU cannot validate a multi-GPU collective.
The outer scheduler waits for workers to stop before cold verification. Do not change
allocations or start other servers.

Read meta.json, cases.py, unittest.py and the declared source/build adapters first. Preserve
the input/output contract, graph lifecycle and all served shape regimes. For a collective,
also preserve rank count, synchronization and peer addressing. Every rank must succeed,
and the slowest rank determines collective latency. Inspect the existing negative controls before changing their anchors.

Only these declared files may change:
$EDITABLE
All other source and harness files, including meta.json, cases.py, unittest.py, recorded
inputs, baseline references, binding adapters and build scaffolding, are frozen. Changes
outside the declaration are rejected and restored by the worker. Preserve nested paths.
Use $EVAL_DIR for scratch experiments. Small diagnostic probes can guide implementation,
but only the frozen whole-task measurement can establish correctness and speedup.

The last complete task test took ${UT_WALL} seconds. The worker will run that entire frozen
measurement after this attempt exits. Use bounded diagnostic probes during your coding
budget. Start a full test yourself only if enough time remains for its recorded duration
plus a handoff; leave a promising candidate promptly so the worker can score it.
The authoritative check, when there is enough time, is:
  python3 "$SCRIPTS/run_task_entry.py" "$TASK_DIR"
Do not run from the task directory as "python3 unittest.py": that can shadow stdlib unittest.
The entry may report CORRECTNESS: PASS and GEAK_GEOMEAN_SPEEDUP, or a JSON verdict with
serving-weighted timings in run/ut_result.json. Read the full gate result. A compilation
failure or a missing rank is a failure, never a successful numerical negative control.
Optimize the sum of baseline_ms * observed_calls divided by the sum of candidate_ms *
observed_calls across all cases. Preserve the weighting. The worker scores the full frozen task after your attempt, and cold verification checks
the saved result independently. A gain must clear $KEEP_THRESHOLD and beat the current best; do not treat noise or a single favorable shape as improvement.

Keep implementation changes in kernel_src so the worker can measure, save or revert them.
When you have a correct promising candidate, leave it in the declared sources promptly.
If no candidate improves the task, retain the prior best. Finish with a short HANDOFF.md
at $RUN_DIR/HANDOFF.md: experiments, exact checks, results, and the next useful direction.
Do not claim an end-to-end gain; that requires a separate serving replay.

Read prior evidence before repeating an experiment:
  $RUN_DIR/HANDOFF.md
  $RUN_DIR/rejected/
  $RUN_DIR/best_measure.log
Missing files mean there is no prior evidence yet.

${EXTRA_STEER:-}
EOF
