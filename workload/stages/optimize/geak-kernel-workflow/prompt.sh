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
KERNEL_SRC_ALL="${KERNEL_SRC_ALL:-$KERNEL_SRC}"
EDITABLE_LIST="$(for f in $KERNEL_SRC_ALL; do printf 'kernel_src/%s ' "$f"; done)"
# Everything already tried on this kernel and rejected, with what it measured. The
# attempt only sees its own run otherwise, and a rediscovered edit costs a full attempt:
# the harness keys its per-bucket timing checkpoint on the source snapshot, so a
# byte-identical candidate is not re-timed at all, it returns the cached numbers.
# The previous attempt's own handoff, if it wrote one. An attempt that self-reverts
# leaves no diff, so this is the only carrier for what it measured and refuted.
HANDOFF_TXT="$(python3 - "$RUN_DIR" <<'PY'
import os
import sys

run = sys.argv[1]


def _read(path, limit):
    try:
        t = open(path, errors="replace").read()
    except OSError:
        return ""
    if not t.strip():
        return ""
    return t if len(t) <= limit else t[:limit] + "\n[...truncated, read the file]\n"


p = os.path.join(run, "HANDOFF.md")
head = _read(p, 6000)
if head:
    print("### Handoff from the previous attempt (%s)\n" % p)
    print(head)

# One hop is not enough for findings about the harness: those stay true for every later
# attempt, while HANDOFF.md is replaced each time. Carry the one before as well, trimmed,
# and name the rest so they can be read on demand.
hd = os.path.join(run, "handoffs")
olds = []
if os.path.isdir(hd):
    def _key(name):
        bits = name.split("_")
        try:
            return int(bits[1].split(".")[0])
        except (IndexError, ValueError):
            return -1
    olds = sorted((f for f in os.listdir(hd) if f.startswith("attempt_")), key=_key)
if len(olds) > 1:
    prev = _read(os.path.join(hd, olds[-2]), 2500)
    if prev:
        print("\n### Handoff from the attempt before that (%s), trimmed\n"
              % os.path.join(hd, olds[-2]))
        print(prev)
if olds:
    print("\nEvery handoff on this kernel, oldest first -- read one if a direction you are"
          "\nconsidering looks like it may already have been tried or trapped:")
    for f in olds:
        print("  - " + os.path.join(hd, f))
arts = []
for d in sorted(os.listdir(run)) if os.path.isdir(run) else []:
    c = os.path.join(run, d, "candidates")
    if d.startswith("attempt_") and os.path.isdir(c):
        for f in sorted(os.listdir(c)):
            arts.append(os.path.join(c, f))
if arts:
    print("\nWorking candidate sources kept by earlier attempts -- do not rebuild them:")
    for a in arts[-12:]:
        print("  - " + a)
PY
)"

# What landed and stayed. An attempt can be in rejected/ and still be the kept one: a
# measurement the reader misread is reverted first and recovered after. The status file
# is the authority on which attempt the current kernel_src came from.
KEPT_TXT="$(python3 - "$RUN_DIR" "$K" <<'PY'
import json, os, sys

run, k = sys.argv[1], sys.argv[2]
p = os.path.join(os.path.dirname(os.path.dirname(run)), "state", k + ".status.json")
try:
    d = json.load(open(p))
except (OSError, ValueError):
    sys.exit(0)
a = d.get("best_attempt")
sp = d.get("best_speedup")
if not a or not isinstance(sp, (int, float)) or sp <= 1.0:
    sys.exit(0)
iv = d.get("best_speedup_interval")
fig = "%.4fx" % sp
if isinstance(iv, (list, tuple)) and len(iv) == 2:
    fig = "%.4fx (measured interval [%.4f, %.4f]; the low end is the figure of record)" % (
        sp, iv[0], iv[1])
print("### What is already in kernel_src -- this is your starting point, not a draft\n")
print("attempt %s landed and was kept at %s. The sources you are editing already contain"
      % (a, fig))
print("it. Build on it. Do not revert it, and do not treat it as someone else's leftover"
      "\nwork to clean up -- reverting it is a measured regression.\n")
note = d.get("best_note")
if note:
    print("  provenance: %s\n" % note)
PY
)"
LEDGER="$(python3 - "$RUN_DIR/rejected" "$RUN_DIR" "$K" <<'PY'
import glob, json, os, sys

d = sys.argv[1]
kept = None
try:
    _p = os.path.join(os.path.dirname(os.path.dirname(sys.argv[2])), "state",
                      sys.argv[3] + ".status.json")
    kept = json.load(open(_p)).get("best_attempt")
except (OSError, ValueError, IndexError):
    pass
rows = []
for p in sorted(glob.glob(os.path.join(d, "attempt_*.json"))):
    try:
        rows.append(json.load(open(p)))
    except (OSError, ValueError):
        pass
rows.sort(key=lambda r: r.get("attempt", 0))
# The kept attempt is not a rejected one, whatever the ledger file says.
rows = [r for r in rows if kept is None or r.get("attempt") != kept]
for r in rows[-8:]:
    m = r.get("measured")
    c = r.get("clean")
    bits = ["attempt %s: measured %s" % (r.get("attempt"), "none" if m is None else "%.4f" % m)]
    if c is not None and r.get("single_pair_buckets"):
        bits.append("%.4f over the repeated buckets (%d measured once)"
                    % (c, r["single_pair_buckets"]))
    bits.append("%s" % r.get("verdict", "rejected"))
    if r.get("changed_lines", 0):
        bits.append("%d changed line(s), diff: %s"
                    % (r["changed_lines"], r.get("diff")))
    else:
        # No diff means the attempt restored the sources itself. What it learned is in
        # its own artifacts, not in an empty patch file.
        cut = "rc=124" in (r.get("verdict") or "")
        bits.append(
            ("no code change landed -- it was cut off at the attempt cap with pristine "
             "sources, so its direction is unfinished rather than refuted; its work is "
             "under attempt_%s/" % r.get("attempt")) if cut else
            ("no code change landed -- it reverted its own work; what it measured "
             "is in the handoff above and under attempt_%s/" % r.get("attempt")))
    print("  - " + "; ".join(bits))
PY
)"

SPLIT_NOTE=""
if [ "$(echo "$KERNEL_SRC_ALL" | wc -w)" -gt 1 ]; then
  SPLIT_NOTE="This seam is split across more than one file. **All of these are editable**, and
they are snapshotted and reverted together, so a change may span them:
  $EDITABLE_LIST
kernel_path below names the primary one because the workflow takes a single path; make the
change wherever it belongs and the measurement still sees it."
fi
# The slot script the prompt tells the agent to call, resolved here so no path is
# guessed inside the prompt text.
WORKLOAD_SCRIPTS="${WORKLOAD_SCRIPTS:-$GEAK_ROOT/workload/scripts}"
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
$( [ -n "$KEPT_TXT" ] && printf '%s\n' "
$KEPT_TXT" )
$( [ -n "$HANDOFF_TXT" ] && printf '%s\n' "
$HANDOFF_TXT" )
$( [ -n "$LEDGER" ] && printf '%s\n%s\n' "
### Already tried on this kernel and rejected -- read the diffs before you decide

$LEDGER
Each of these is reverted: kernel_src is back to the state before it. Do not re-derive
one of them. If you believe a rejected direction was right and was rejected for a
measurement reason rather than a kernel reason, say so and measure it differently --
re-submitting the same bytes returns the same cached numbers without re-timing." "" )
$( [ -n "$SPLIT_NOTE" ] && printf '%s\n' "$SPLIT_NOTE" )
Check yourself with:

  python3 $WORKLOAD_SCRIPTS/run_task_entry.py $TASK_DIR

Run it that way, not as \`cd $TASK_DIR && python3 unittest.py\`: the task directory holds
its own unittest.py, which shadows the standard library package and makes the run die
inside torch with "No module named 'unittest.mock'". It prints one of three shapes, and
all three are the same verdict -- the one the scheduler reads to decide KEEP or revert:

  1. \`CORRECTNESS: PASS\` plus \`GEAK_GEOMEAN_SPEEDUP <x>\`.
  2. A JSON line {"exit": 0, "weighted": <x>}: exit 0 means every gate passed
     (correctness, negative controls, tail contracts) and \`weighted\` is the
     serving-weighted speedup across the whole sweep.
  3. A bare gate map -- {"legs_differ": true, ..., "timing": true} -- followed by
     \`UT PASS\`. All true plus UT PASS is a pass. This shape prints no speedup at all:
     the numbers are in $TASK_DIR/ut_result.json, and when the prefix history behind the
     dominant buckets was never captured there is no single aggregate, only an interval
     (serving_weighted_speedup_px_causal and _px_alloc). The low end is the figure of
     record. Ignore captured_inputs_only_speedup -- it covers a small slice of the call
     mass and is not the serving number.

If you get shape 3, you have not failed and you have not measured nothing. Read
ut_result.json.

$( [ -n "$HINT" ] && printf 'What is known about this kernel:\n%s\n' "$HINT" )

## Four hard constraints

1. **The harness is frozen.** Only ${EDITABLE_LIST}may be edited. unittest.py,
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
