#!/usr/bin/env bash
# One kernel, one GPU, optimizing in a loop until TARGET seconds have been credited.
#
# This file is **shared machinery**, not an optimization method: the attempt loop,
# credit/heartbeat, KEEP/revert, and resumable accounting. The method itself comes from
# the skill selected for the optimize stage, through two hooks:
#
#   $SKILL_ROOT/prompt.sh    required. Builds this attempt's prompt into $PROMPT_OUT.
#   $SKILL_ROOT/measure.sh   optional. Measures the current kernel_src and prints
#                            STATUS=/SPEEDUP=. Without it the default below applies
#                            (running the task's own frozen unittest.py).
#
# Runs inside a **one-container-per-GPU** sandbox (the container mounts exactly one
# /dev/dri/renderD*, so no other GPU is reachable from within). All durable state lives
# on shared storage, so losing a node costs at most the current attempt — the scheduler
# restarts elsewhere and resumes from the credited seconds and the committed kernel_src.
set -uo pipefail

K="${1:?kernel name}"
SLOT="${2:-unknown}"

: "${WORK_ROOT:?WORK_ROOT}"      # this stage's work root (= $OUT/work)
: "${TASK_DIR:?TASK_DIR}"        # this kernel's frozen-harness task directory
: "${SKILL_ROOT:?SKILL_ROOT}"    # the skill selected for the optimize stage
: "${RUNNER:?RUNNER}"            # runner script, see runners/CONTRACT.md

TARGET="${TARGET:-172800}"               # optimization budget per kernel; 48 h default
# Nodes get reclaimed. On a churning node a long attempt rarely reaches its
# measure/commit step, and all of its work is discarded. Keep attempts short enough to
# finish inside one node lifetime: several small committed steps beat one long
# uncommitted one, especially since apply_to_original makes each attempt build on the
# last anyway. Measured: 2 h cap -> 69 attempts / 11 keeps (16%); 1500 s cap -> 85
# attempts / 22 keeps (26%), in under half the wall-clock.
ATTEMPT_CAP="${ATTEMPT_CAP:-1500}"
ATTEMPT_MIN="${ATTEMPT_MIN:-600}"
# Keep threshold, calibrated by measurement rather than quoted from documentation: four
# kernels in the previous campaign whose source was md5-identical to the original
# re-measured at 1.0246-1.0453, so anything below this is noise, not speedup.
KEEP_THRESHOLD="${KEEP_THRESHOLD:-1.045}"
KEEP_MARGIN="${KEEP_MARGIN:-0.005}"      # and it must beat the current best by this much
MEASURE_TIMEOUT="${MEASURE_TIMEOUT:-1800}"

RUN="$WORK_ROOT/runs/$K"
LOG="$WORK_ROOT/logs/$K"
STATE="$WORK_ROOT/state"
CREDIT="$STATE/$K.credit"
HB="$STATE/$K.hb"
STATUS="$STATE/$K.status.json"

mkdir -p "$RUN/snapshots" "$LOG" "$STATE"
note() { echo "[$(date '+%F %T')] $*" >> "$LOG/worker.log"; }

[ -d "$TASK_DIR/kernel_src" ] || { note "FATAL no kernel_src/ under $TASK_DIR"; exit 2; }
# Regular files only, .py first: a __pycache__/ created by a previous attempt sorts ahead
# of the kernel and would otherwise be treated as the editable source.
KSRC="$(cd "$TASK_DIR/kernel_src" && ls -p | grep -v '/$' | grep -v '^__' | grep '\.py$' \
        | head -1)"
[ -n "$KSRC" ] || KSRC="$(cd "$TASK_DIR/kernel_src" && ls -p | grep -v '/$' | grep -v '^__' | head -1)"
[ -n "$KSRC" ] || { note "FATAL kernel_src/ holds no editable file"; exit 2; }

[ -f "$CREDIT" ] || echo 0 > "$CREDIT"
BASE="$(cat "$CREDIT" 2>/dev/null || echo 0)"
case "$BASE" in ''|*[!0-9]*) BASE=0 ;; esac
START="$(date +%s)"

# Heartbeat + credit ticker. The scheduler treats a stale .hb as a dead worker, and
# .credit is the only reason an interruption resumes instead of starting over.
#
# The ticker re-checks its parent every cycle. The trap below does **not** run under
# SIGKILL — observed in practice: kill -9 the worker and the ticker is orphaned, keeps
# crediting time to a worker that died long ago and keeps refreshing its heartbeat, so
# the scheduler never notices the death and an entire slice of budget evaporates.
# Zombies count as dead too: after -9 and before the parent reaps, `kill -0` still
# succeeds against a zombie.
PARENT=$$
(
  while :; do
    st="$(awk '{print $3}' "/proc/$PARENT/stat" 2>/dev/null)"
    case "${st:-gone}" in Z|X|gone) exit 0 ;; esac
    now="$(date +%s)"
    echo "$now" > "$HB"
    echo $(( BASE + now - START )) > "$CREDIT"
    sleep 20
  done
) &
TICKER=$!
trap 'kill "$TICKER" 2>/dev/null' EXIT INT TERM

# Default measurement: run the task's own frozen entry and read what it contracts to print.
# The baseline leg is always the **originally installed module**, not your previous change,
# so this speedup is always the cumulative figure against the real baseline.
#
# The entry is launched through scripts/run_task_entry.py rather than with `cd $TASK_DIR &&
# python3 unittest.py`: a task directory contains its own `unittest.py`, which lands on
# sys.path[0] and shadows the standard library package, and the failure surfaces far away
# ("No module named 'unittest.mock'") as an empty measurement rather than as an error.
#
# Two output contracts are accepted, because both kinds of harness exist in the wild:
#   * `CORRECTNESS: PASS` plus `GEAK_GEOMEAN_SPEEDUP <x>`
#   * a JSON line {"exit": 0, "weighted": <x>} -- the form this pipeline's extraction skill
#     emits, where exit 0 means every gate passed (correctness, negative controls, tail
#     contracts) and `weighted` is the serving-weighted speedup over the whole shape sweep.
default_measure() {
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ( timeout "$MEASURE_TIMEOUT" python3 "$here/run_task_entry.py" "$TASK_DIR" unittest.py ) \
      > "$RUN/.measure.log" 2>&1
  local sp ok
  sp="$(grep -oE 'GEAK_GEOMEAN_SPEEDUP [0-9.]+' "$RUN/.measure.log" | tail -1 | awk '{print $2}')"
  ok=0; grep -q '^CORRECTNESS: PASS' "$RUN/.measure.log" && ok=1
  if [ -z "$sp" ]; then
    read -r ok sp <<<"$(python3 - "$RUN/.measure.log" <<'PY'
import json, sys
ok, sp = 0, ""
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if "exit" not in d:
        continue
    # A gate failure is not a measurement: report STATUS=0 and let the worker revert.
    ok = 1 if d.get("exit") == 0 else 0
    w = d.get("weighted")
    sp = "" if w in (None, "") else repr(float(w))
print(ok, sp)
PY
)"
  fi
  echo "STATUS=${ok:-0} SPEEDUP=${sp:-}"
}

measure() {
  if [ -x "$SKILL_ROOT/measure.sh" ] || [ -f "$SKILL_ROOT/measure.sh" ]; then
    TASK_DIR="$TASK_DIR" RUN_DIR="$RUN" MEASURE_TIMEOUT="$MEASURE_TIMEOUT" \
      bash "$SKILL_ROOT/measure.sh"
  else
    default_measure
  fi
}

BEST="$(python3 -c "
import json,os
p='$STATUS'
print(json.load(open(p)).get('best_speedup') or 1.0 if os.path.exists(p) else 1.0)
" 2>/dev/null || echo 1.0)"
case "$BEST" in ''|None) BEST=1.0 ;; esac

note "start kernel=$K slot=$SLOT skill=$(basename "$SKILL_ROOT") base_credit=${BASE}s target=${TARGET}s src=$KSRC best=${BEST}x"

write_status() {  # status credited best attempts [measured rc]
  python3 - "$STATUS" "$@" <<'PY'
import json, os, sys
p, st, cred, best, n = sys.argv[1:6]
extra = sys.argv[6:]
d = json.load(open(p)) if os.path.exists(p) else {}
d.update(status=st, credited_seconds=int(cred), best_speedup=float(best),
         attempts=int(n), slot=os.environ.get("SLOT", d.get("slot")))
if extra:
    d.setdefault("history", []).append(
        {"attempt": int(n), "slot": os.environ.get("SLOT"),
         "measured": extra[0], "rc": int(extra[1])})
json.dump(d, open(p, "w"), indent=1)
PY
}
export SLOT

while :; do
  now="$(date +%s)"; CRED=$(( BASE + now - START ))
  if [ "$CRED" -ge "$TARGET" ]; then
    note "budget exhausted (${CRED}s >= ${TARGET}s) -- stopping"
    N="$(ls "$RUN" 2>/dev/null | grep -c '^attempt_')"
    write_status done "$CRED" "$BEST" "$N"
    touch "$STATE/$K.done"
    exit 0
  fi

  N=$(( $(ls "$RUN" 2>/dev/null | grep -c '^attempt_') + 1 ))
  REMAIN=$(( TARGET - CRED ))
  CAP=$(( REMAIN < ATTEMPT_CAP ? REMAIN : ATTEMPT_CAP ))
  [ "$CAP" -lt "$ATTEMPT_MIN" ] && CAP="$ATTEMPT_MIN"
  EVAL_DIR="$RUN/attempt_$N"
  mkdir -p "$EVAL_DIR"
  cp -a "$TASK_DIR/kernel_src/$KSRC" "$RUN/snapshots/before_attempt_$N"
  note "attempt $N: cap=${CAP}s credited=${CRED}s best=${BEST}x"

  # --- skill hook: build this attempt's prompt.
  PROMPT_OUT="$RUN/.prompt_$N.txt"
  if ! env K="$K" TASK_DIR="$TASK_DIR" KERNEL_SRC="$KSRC" ATTEMPT="$N" \
        BEST_SPEEDUP="$BEST" KEEP_THRESHOLD="$KEEP_THRESHOLD" \
        RUN_DIR="$RUN" EVAL_DIR="$EVAL_DIR" ATTEMPT_CAP="$CAP" \
        PROMPT_OUT="$PROMPT_OUT" \
        bash "$SKILL_ROOT/prompt.sh" >> "$LOG/worker.log" 2>&1; then
    note "FATAL $SKILL_ROOT/prompt.sh failed"
    exit 2
  fi
  [ -s "$PROMPT_OUT" ] || { note "FATAL prompt.sh did not write $PROMPT_OUT"; exit 2; }

  # --- run one unattended session through the runner. The time budget is enforced from
  #     the **outside** with timeout; the runner sets none of its own (contract §4).
  timeout --signal=TERM --kill-after=120 "$CAP" \
      bash "$RUNNER" "$PROMPT_OUT" "$TASK_DIR" "$LOG/attempt_$N.log"
  RC=$?
  note "attempt $N runner rc=$RC"
  if [ "$RC" = 78 ]; then
    # Unmet precondition. Ten thousand retries give the same result; do not spend budget.
    note "FATAL runner returned 78 (unmet precondition), see $LOG/attempt_$N.log"
    write_status blocked "$CRED" "$BEST" "$N"
    exit 78
  fi

  # --- measure, then decide KEEP or revert.
  M="$(measure)"
  SP="$(echo "$M" | sed -n 's/.*SPEEDUP=\([0-9.]*\).*/\1/p')"
  OK="$(echo "$M" | sed -n 's/.*STATUS=\([01]\).*/\1/p')"
  OK="${OK:-0}"

  if [ -z "$SP" ]; then
    note "attempt $N: no measurement -- reverting"
    cp -a "$RUN/snapshots/before_attempt_$N" "$TASK_DIR/kernel_src/$KSRC"
  else
    KEEP="$(python3 -c "
sp, best = $SP, $BEST
print(1 if $OK and sp >= $KEEP_THRESHOLD and sp > best + $KEEP_MARGIN else 0)")"
    if [ "$KEEP" = 1 ]; then
      note "attempt $N: KEEP ${SP}x (was ${BEST}x)"
      BEST="$SP"
      cp -a "$TASK_DIR/kernel_src/$KSRC" "$RUN/best_kernel_src"
      cp -a "$RUN/.measure.log" "$RUN/best_measure.log" 2>/dev/null
    else
      note "attempt $N: revert (correct=$OK measured=${SP} best=${BEST}x threshold=${KEEP_THRESHOLD})"
      cp -a "$RUN/snapshots/before_attempt_$N" "$TASK_DIR/kernel_src/$KSRC"
    fi
  fi

  now="$(date +%s)"
  write_status running "$(( BASE + now - START ))" "$BEST" "$N" "${SP:-null}" "$RC"
done
