#!/usr/bin/env bash
# Build the prompt for one attempt and write it to $PROMPT_OUT. Called by scripts/worker.sh.
#
# Every input arrives through the environment (see the boundary diagram in SKILL.md):
#   K TASK_DIR KERNEL_SRC ATTEMPT BEST_SPEEDUP KEEP_THRESHOLD RUN_DIR EVAL_DIR
#   ATTEMPT_CAP PROMPT_OUT
# GEAK_ROOT is also needed (exported by the scheduler; it falls back to four levels up
# from this file).
set -euo pipefail

: "${PROMPT_OUT:?PROMPT_OUT}"
: "${TASK_DIR:?TASK_DIR}"
: "${KERNEL_SRC:?KERNEL_SRC}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="${GEAK_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
WF_DIR="$GEAK_ROOT/kernel_workflow"
WF_JS="$WF_DIR/kernel_workflow.js"
[ -f "$WF_JS" ] || { echo "[prompt] cannot find $WF_JS" >&2; exit 2; }

BUDGET="${BUDGET:-4}"
MIN_IMPROVE="${MIN_IMPROVE:-0.05}"
KEEP_THRESHOLD="${KEEP_THRESHOLD:-1.045}"
BEST_SPEEDUP="${BEST_SPEEDUP:-1.0}"
ATTEMPT="${ATTEMPT:-1}"
GPU_IDS="${GPU_IDS:-0}"
EXTRA_STEER="${EXTRA_STEER:-}"

# Hints from the task directory save the agent a detour, but their absence must not be
# fatal.
HINT=""
if [ -f "$TASK_DIR/meta.json" ]; then
  HINT="$(python3 - "$TASK_DIR/meta.json" <<'PY' 2>/dev/null || true
import json, sys
m = json.load(open(sys.argv[1]))
bits = []
for k in ("kernel", "module", "pct_gpu_time", "bound_type", "attainable_speedup",
          "roofline_pct", "num_cases", "calls"):
    if m.get(k) is not None:
        bits.append(f"- {k}: {m[k]}")
print("\n".join(bits))
PY
)"
fi

cat > "$PROMPT_OUT" <<EOF
Optimize a Triton kernel using GEAK's kernel workflow. This is attempt $ATTEMPT; the best
speedup achieved on this kernel so far is ${BEST_SPEEDUP}x (relative to the original
implementation).

Run this workflow:

  $WF_JS

Parameters:
  kernel_path        = $TASK_DIR/kernel_src/$KERNEL_SRC
  workflow_dir       = $WF_DIR
  eval_dir           = $EVAL_DIR
  exp_root           = $RUN_DIR
  budget             = $BUDGET
  min_improve        = $MIN_IMPROVE
  gpu_ids            = $GPU_IDS
  apply_to_original  = "true"
  task               = see "Task" below

apply_to_original must be "true": changes have to land back in kernel_src/ so that the
next attempt starts from your result and builds on it.

## Task

Make $TASK_DIR/kernel_src/$KERNEL_SRC faster with correctness fully preserved.
Check yourself with \`cd $TASK_DIR && python3 unittest.py\`; it prints CORRECTNESS and
GEAK_GEOMEAN_SPEEDUP.

$( [ -n "$HINT" ] && printf 'What is known about this kernel:\n%s\n' "$HINT" )

## Four hard constraints

1. **The harness is frozen.** Only kernel_src/$KERNEL_SRC may be edited. unittest.py,
   cases.py and meta.json must not change by a single byte. A speedup measured against a
   modified harness is worthless and will be rejected unchanged by the cold
   re-verification gate.

2. **The noise band is $KEEP_THRESHOLD.** Any "improvement" below that number is
   measurement noise. In the previous campaign four kernels whose code md5 was identical
   to the original re-measured at 1.0246-1.0453. Do not chase noise.

3. **It must generalize across the whole length sweep.** cases.py holds the entire shape
   sweep, and the aggregate is sum(baseline_ms x calls) / sum(candidate_ms x calls), not
   the best case. Tabulating for one shape, or branching on shape to skip the work, only
   makes that aggregate worse.

4. **Every output must be written.** The UT leg allocates from a virgin allocator, so
   empty pages read back as zero — which means "skip the part of the output nobody reads"
   passes every correctness gate and then breaks in the real service. One such kernel was
   caught in the previous campaign. Do not skip any part of the output to save bandwidth.

## Finishing

The attempt is capped at ${ATTEMPT_CAP}s, timed externally, and you will be terminated
when it expires. So: **the moment you have a version that is confirmed faster and
correct, land it in kernel_src/** — do not batch everything into one final commit.
Uncommitted progress is lost at termination; what you committed is what the next attempt
continues from.
${EXTRA_STEER}
EOF

echo "[prompt] attempt $ATTEMPT -> $PROMPT_OUT ($(wc -c < "$PROMPT_OUT") bytes)"
