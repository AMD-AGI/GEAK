#!/usr/bin/env bash
# Prompt generator for the raw path. Same inputs as geak-kernel-workflow/prompt.sh (see
# stages/optimize/README.md).
set -euo pipefail

: "${PROMPT_OUT:?PROMPT_OUT}"
: "${TASK_DIR:?TASK_DIR}"
: "${KERNEL_SRC:?KERNEL_SRC}"

KEEP_THRESHOLD="${KEEP_THRESHOLD:-1.045}"
BEST_SPEEDUP="${BEST_SPEEDUP:-1.0}"
ATTEMPT="${ATTEMPT:-1}"
ATTEMPT_CAP="${ATTEMPT_CAP:-1500}"
EXTRA_STEER="${EXTRA_STEER:-}"

HINT=""
if [ -f "$TASK_DIR/meta.json" ]; then
  HINT="$(python3 - "$TASK_DIR/meta.json" <<'PY' 2>/dev/null || true
import json, sys
m = json.load(open(sys.argv[1]))
bits = [f"- {k}: {m[k]}" for k in
        ("kernel", "module", "pct_gpu_time", "bound_type", "attainable_speedup",
         "roofline_pct", "num_cases", "calls") if m.get(k) is not None]
print("\n".join(bits))
PY
)"
fi

cat > "$PROMPT_OUT" <<EOF
Optimize a Triton kernel. This is attempt $ATTEMPT; the best result on this kernel so far
is ${BEST_SPEEDUP}x (relative to the original implementation).

**The only file you may modify**: $TASK_DIR/kernel_src/$KERNEL_SRC

**Self-check**: cd $TASK_DIR && python3 unittest.py
It prints:
  CORRECTNESS: PASS|FAIL
  GEAK_GEOMEAN_SPEEDUP <x>     <- cumulative speedup against the **original**
The aggregate is sum(baseline_ms x calls) / sum(candidate_ms x calls) over the entire
shape sweep in cases.py, not the best case.

$( [ -n "$HINT" ] && printf 'What is known:\n%s\n' "$HINT" )

## How to work

1. Read kernel_src/$KERNEL_SRC and cases.py first; understand the real shape distribution
   and where the current implementation is losing time.
2. Run unittest.py once to get the current number — do not guess the state of things by
   reading code.
3. Change one thing, re-measure. One variable at a time, or you will not know which edit
   did the work.
4. The moment you have a version that is **correct and genuinely faster**, leave it in
   kernel_src/; revert immediately if it is slower or wrong.

## Four hard constraints

1. **The harness is frozen.** unittest.py / cases.py / meta.json must not change by a
   single byte. Numbers measured against a modified harness are rejected by the cold
   re-verification gate.
2. **The noise band is $KEEP_THRESHOLD.** Any "improvement" below that is noise. In the
   previous campaign four kernels whose code md5 was identical to the original
   re-measured at 1.0246-1.0453.
3. **It must generalize across the whole length sweep.** Tabulating for a single shape,
   or branching on shape to skip the work, only looks worse under the aggregate.
4. **Every output must be written.** The UT leg allocates from a virgin allocator and
   empty pages read back as zero, so "skip the output nobody reads" passes every
   correctness gate and then breaks in production — one such kernel was caught in the
   previous campaign.

## Finishing

The cap is ${ATTEMPT_CAP}s, timed externally, and you are terminated when it expires. So
**commit as you go** rather than batching to the end. Progress not left in kernel_src/ is
lost; what you leave behind is what the next attempt continues from.
${EXTRA_STEER}
EOF

echo "[prompt] attempt $ATTEMPT -> $PROMPT_OUT ($(wc -c < "$PROMPT_OUT") bytes)"
