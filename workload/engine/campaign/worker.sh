#!/usr/bin/env bash
# One kernel, one GPU, optimizing in a loop until TARGET seconds have been credited.
#
# This file is **shared machinery**, not an optimization method: the attempt loop,
# credit/heartbeat, KEEP/revert, and resumable accounting. The method itself comes from
# the skill selected for the optimize stage, through two hooks:
#
#   $SKILL_ROOT/prompt.sh    required. Builds this attempt's prompt into $PROMPT_OUT.
#   $SKILL_ROOT/measure.sh   required. Runs the method's measurement and prints
#                            STATUS= SPEEDUP= CLEAN= SINGLE=.
#
# Runs inside a **one-container-per-GPU** sandbox (the container mounts exactly one
# /dev/dri/renderD*, so no other GPU is reachable from within). All durable state lives
# on shared storage, so losing a node costs at most the current attempt — the scheduler
# restarts elsewhere and resumes from the credited seconds and the committed kernel_src.
set -uo pipefail

K="${1:?kernel name}"
export K
SLOT="${2:-unknown}"

: "${WORK_ROOT:?WORK_ROOT}"      # this stage's work root (= $OUT/work)
: "${TASK_DIR:?TASK_DIR}"        # this kernel's frozen-harness task directory
: "${SKILL_ROOT:?SKILL_ROOT}"    # the skill selected for the optimize stage
: "${RUNNER:?RUNNER}"            # runner script, see runners/CONTRACT.md

TARGET="${TARGET:-172800}"               # optimization budget per kernel; 48 h default
REQUIRE_FULL_BUDGET="${REQUIRE_FULL_BUDGET:-0}"
# Nodes get reclaimed. On a churning node a long attempt rarely reaches its
# measure/commit step, and all of its work is discarded. Keep attempts short enough to
# finish inside one node lifetime: several small committed steps beat one long
# uncommitted one, especially since apply_to_original makes each attempt build on the
# last anyway. Measured: 2 h cap -> 69 attempts / 11 keeps (16%); 1500 s cap -> 85
# attempts / 22 keeps (26%), in under half the wall-clock.
ATTEMPT_CAP="${ATTEMPT_CAP:-1500}"
ATTEMPT_MIN="${ATTEMPT_MIN:-600}"

# A long campaign outlives the numbers it started with: the scheduler fixes these at
# dispatch out of its own environment, and correcting them would otherwise mean killing
# the stage. An optional file in WORK_ROOT wins, so a cap can be retuned from measured
# evidence mid-run. Non-numeric content is ignored rather than trusted.
read_override() {  # <file> <current>
  local v
  [ -f "$1" ] || { echo "$2"; return; }
  v="$(tr -dc '0-9' < "$1" | head -c 9)"
  [ -n "$v" ] && [ "$v" -gt 0 ] 2>/dev/null && echo "$v" || echo "$2"
}
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
SOURCE_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/source_tree.py"
RETAIN_MEASUREMENT="$(dirname "$SOURCE_TREE")/../evidence/retain_measurement.py"
STATE_IO="$(dirname "$SOURCE_TREE")/durable_write.py"

write_status() {  # status [credited best attempts [measured rc]]
  python3 - "$STATE_IO" "$STATUS" "$@" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
from durable_write import write_text
p, st = sys.argv[2:4]
extra = sys.argv[7:]
d = json.load(open(p)) if os.path.exists(p) else {"credited_seconds": 0}
d["status"] = st
if len(sys.argv) > 4:
    cred, best, n = sys.argv[4:7]
    d.update(credited_seconds=int(cred), best_speedup=float(best),
             attempts=int(n), slot=os.environ.get("SLOT", d.get("slot")))
if extra:
    rec = {"attempt": int(n), "slot": os.environ.get("SLOT"),
           "measured": extra[0], "rc": int(extra[1])}
    if len(extra) > 3:
        # The same serving weighting over the buckets the harness measured more than
        # once, and how many it measured only once. A gap between the two is the harness
        # freezing a first pair that fell outside its undecided band, not the edit.
        rec["measured_repeated_buckets_only"] = extra[2]
        rec["single_pair_buckets"] = int(extra[3])
    d.setdefault("history", []).append(rec)
write_text(p, json.dumps(d, indent=1) + "\n")
PY
}

if [ ! -f "$SKILL_ROOT/measure.sh" ]; then
  echo "[worker] required measurement hook is missing: $SKILL_ROOT/measure.sh" >&2
  write_status blocked || exit 78
  exit 78
fi

PENDING_EVIDENCE="$RUN/.evidence_pending"
exec 201>"$RUN/.measurement.lock" || exit 78

# Preparation can outlive scheduler startup grace. It proves liveness without
# crediting recovery, archival or frozen-harness work as optimization time.
PARENT=$$
PARENT_START="$(awk '{print $22}' "/proc/$PARENT/stat")" || exit 78
ticker() {  # Optional BASE START enables credit only after preparation succeeds.
  local state started now pause="" stopping=0
  exec 201>&-
  # A graceful join waits for any current metadata write before switching phases.
  # Interruptible sleep avoids adding the 20-second cadence to every startup/exit.
  trap 'stopping=1' INT TERM
  trap '[ -z "$pause" ] || { kill "$pause" 2>/dev/null; wait "$pause" 2>/dev/null; }' EXIT
  while [ "$stopping" -eq 0 ]; do
    read -r state started < <(awk '{print $3, $22}' "/proc/$PARENT/stat" 2>/dev/null)
    case "${state:-gone}" in Z|X|gone) exit 0 ;; esac
    [ "${started:-}" = "$PARENT_START" ] || exit 0
    now="$(date +%s)"
    if [ "$#" -eq 2 ] && ! python3 "$STATE_IO" "$CREDIT" "$(( $1 + now - $2 ))"; then
      echo "[worker] accounting storage failed; previous complete counters were retained" >&2
      exit 1
    fi
    if ! python3 "$STATE_IO" "$HB" "$now"; then
      echo "[worker] heartbeat storage failed; previous complete counters were retained" >&2
      exit 1
    fi
    [ "$stopping" -eq 0 ] || exit 0
    sleep 20 &
    pause=$!
    [ "$stopping" -eq 0 ] || exit 0
    if wait "$pause"; then pause=""; fi
  done
  exit 0
}
ticker_running() {
  # A failed ticker may have been reaped during lengthy preparation; its reused
  # PID must never identify an unrelated process as our job or receive our signal.
  jobs -pr | grep -qx -- "$TICKER"
}
stop_ticker() {
  ticker_running && kill "$TICKER" 2>/dev/null || return 78
  wait "$TICKER" || return 78
}
cleanup_ticker() {
  if ticker_running; then kill "$TICKER" 2>/dev/null || :; fi
  wait "$TICKER" 2>/dev/null || :
}
ticker &
TICKER=$!
trap cleanup_ticker EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

lock_measurement() {
  # A bounded operational wait lets a surviving manual wrapper finish copying.
  # Expiry blocks the task; it never permits overwriting unretained evidence.
  if ! flock -x -w "${MEASUREMENT_LOCK_TIMEOUT:-300}" 201; then
    note "FATAL another measurement/retention still owns this task"
    exit 78
  fi
}
recover_pending() {
# A failed/interrupted copy must not be followed by another agent or sweep that
# overwrites the only remaining working evidence. Keep its partial archive visible.
if [ -e "$PENDING_EVIDENCE" ] || [ -L "$PENDING_EVIDENCE" ]; then
  note "recovering pending evidence before any working files change: $(cat "$PENDING_EVIDENCE")"
  if ! python3 "$RETAIN_MEASUREMENT" recover "$TASK_DIR" "$PENDING_EVIDENCE" \
       --raw-log "$RUN/.measure.log" --lock-timeout "$MEASURE_TIMEOUT" >> "$LOG/worker.log" 2>&1; then
    note "FATAL evidence retention is still incomplete; stopping without overwriting working evidence"
    exit 78
  fi
fi
}
lock_measurement
recover_pending
if [ ! -f "$RUN/pre_search_evidence.path" ]; then
  INITIAL_EVIDENCE="$(python3 "$RETAIN_MEASUREMENT" begin "$TASK_DIR" "$RUN/evidence" \
                       --label pre-search --pending "$PENDING_EVIDENCE" \
                       2>> "$LOG/worker.log")" || exit 2
  python3 "$RETAIN_MEASUREMENT" retain "$TASK_DIR" "$INITIAL_EVIDENCE" \
      --raw-log "$RUN/.measure.log" --complete-pointer "$RUN/pre_search_evidence.path" \
      --lock-timeout "$MEASURE_TIMEOUT" \
      >> "$LOG/worker.log" 2>&1 || exit 2
  rm "$PENDING_EVIDENCE" || exit 2
fi
FROZEN="$RUN/frozen_harness"
python3 "$SOURCE_TREE" freeze "$TASK_DIR" "$FROZEN" >> "$LOG/worker.log" 2>&1 || {
  note "FATAL cannot snapshot the frozen harness"
  exit 2
}
restore_frozen() {
  python3 "$SOURCE_TREE" restore-frozen "$TASK_DIR" "$FROZEN" >> "$LOG/worker.log" 2>&1 || {
    note "FATAL cannot restore frozen harness"
    exit 2
  }
}
if ! python3 "$SOURCE_TREE" check-frozen "$TASK_DIR" "$FROZEN" >> "$LOG/worker.log" 2>&1; then
  note "restoring frozen harness left modified by an interrupted attempt"
  restore_frozen
fi
if ! KSRC_ALL="$(python3 "$SOURCE_TREE" editable "$TASK_DIR" 2>> "$LOG/worker.log")"; then
  note "FATAL invalid editable source metadata"
  exit 2
fi
KSRC="${KSRC_ALL%% *}"
note "editable leaves from meta.json: $KSRC_ALL"
copy_sources() {
  python3 "$SOURCE_TREE" copy "$1" "$2" >> "$LOG/worker.log" 2>&1 || {
    note "FATAL source tree copy failed: $1 -> $2"
    exit 2
  }
}

# Join the heartbeat-only phase before reading credit or starting its clock. A
# failed preparation ticker must not be replaced by a healthy accounting ticker.
stop_ticker || {
  echo "[worker] preparation heartbeat stopped; preserve state and repair storage before resuming" >&2
  exit 78
}

BASE="$(python3 - "$STATE_IO" "$CREDIT" "$STATUS" <<'PY_CREDIT'
import os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
from durable_write import read_credit_pair
print(read_credit_pair(sys.argv[2], sys.argv[3]))
PY_CREDIT
)" || exit 78
[ -f "$CREDIT" ] || python3 "$STATE_IO" "$CREDIT" "$BASE" || exit 78
START="$(date +%s)"

ticker "$BASE" "$START" &
TICKER=$!

measure() {
  TASK_DIR="$TASK_DIR" RUN_DIR="$RUN" MEASURE_TIMEOUT="$MEASURE_TIMEOUT" \
    MEASURE_NATIVE_RC_FILE="$MEASURE_NATIVE_RC_FILE" \
    bash "$SKILL_ROOT/measure.sh" 201>&-
}

measure_and_retain() {  # <sweep number>; leaves the unchanged measure output in M
  local archive measure_rc
  recover_pending
  archive="$(python3 "$RETAIN_MEASUREMENT" begin "$TASK_DIR" "$EVAL_DIR/worker_measurements" \
                --label "sweep-$1" --pending "$PENDING_EVIDENCE" \
                2>> "$LOG/worker.log")" || return 2
  # Keep the previous compatibility log without mislabelling it as this sweep's
  # native log when a custom measure hook only writes to stdout/stderr.
  if [ -e "$RUN/.measure.log" ] || [ -L "$RUN/.measure.log" ]; then
    mv "$RUN/.measure.log" "$archive/previous_measure.log" || return 2
  fi
  MEASURE_NATIVE_RC_FILE="$archive/native_exit_code"
  measure > "$archive/measure.stdout.log" 2> "$archive/measure.stderr.log"
  measure_rc=$?
  M="$(cat "$archive/measure.stdout.log")" || return 2
  local native_args=()
  [ -f "$MEASURE_NATIVE_RC_FILE" ] && native_args=(--native-exit "$(cat "$MEASURE_NATIVE_RC_FILE")")
  python3 "$RETAIN_MEASUREMENT" retain "$TASK_DIR" "$archive" \
      --raw-log "$RUN/.measure.log" --measure-exit "$measure_rc" "${native_args[@]}" \
      --lock-timeout "$MEASURE_TIMEOUT" \
      >> "$LOG/worker.log" 2>&1 || return 2
  rm "$PENDING_EVIDENCE" || return 2
  note "attempt $N: sweep $1 evidence retained at $archive"
  return "$measure_rc"
}

BEST="$(python3 -c "
import json,os
p='$STATUS'
print(json.load(open(p)).get('best_speedup') or 1.0 if os.path.exists(p) else 1.0)
" 2>/dev/null || echo 1.0)"
case "$BEST" in ''|None) BEST=1.0 ;; esac

note "start kernel=$K slot=$SLOT skill=$(basename "$SKILL_ROOT") base_credit=${BASE}s target=${TARGET}s src=$KSRC best=${BEST}x"

record_rejected() {  # <attempt> <measured> <clean> <single> <reason>
  # What this attempt changed, and what it measured, kept where the next prompt can read
  # it. Without this the only trace of a rejected edit is that kernel_src is back to the
  # snapshot, and the next attempt is free to rediscover it: the harness keys its bucket
  # checkpoint on the source snapshot, so an identical edit is not even re-timed -- it
  # returns the cached numbers and the attempt is spent for nothing.
  local n="$1" meas="$2" clean="$3" single="$4" reason="$5" d="$RUN/rejected"
  mkdir -p "$d"
  : > "$d/attempt_$n.diff"
  for f in $KSRC_ALL; do
    diff -u "$SNAP/$f" "$TASK_DIR/kernel_src/$f" >> "$d/attempt_$n.diff" 2>/dev/null
  done
  python3 - "$d/attempt_$n.json" "$n" "$meas" "$clean" "$single" "$reason" \
           "$d/attempt_$n.diff" <<'PY'
import json, os, sys
p, n, meas, clean, single, reason, diff = sys.argv[1:8]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


changed = 0
if os.path.exists(diff):
    changed = sum(1 for ln in open(diff, errors="replace")
                  if ln[:1] in "+-" and ln[:3] not in ("+++", "---"))
json.dump({"attempt": int(n), "measured": num(meas), "clean": num(clean),
           "single_pair_buckets": int(single or 0), "verdict": reason,
           "diff": diff, "changed_lines": changed}, open(p, "w"), indent=1)
PY
}

export SLOT

while :; do
  if ! ticker_running; then
    echo "[worker] accounting ticker stopped; preserve state and repair storage before resuming" >&2
    exit 78
  fi
  now="$(date +%s)"; CRED=$(( BASE + now - START ))
  if [ "$CRED" -ge "$TARGET" ]; then
    note "budget exhausted (${CRED}s >= ${TARGET}s) -- stopping"
    N="$(ls "$RUN" 2>/dev/null | grep -c '^attempt_')"
    write_status done "$CRED" "$BEST" "$N" || exit 78
    touch "$STATE/$K.done"
    exit 0
  fi

  N=$(( $(ls "$RUN" 2>/dev/null | grep -c '^attempt_') + 1 ))
  REMAIN=$(( TARGET - CRED ))
  ATTEMPT_CAP="$(read_override "$WORK_ROOT/ATTEMPT_CAP" "$ATTEMPT_CAP")"
  MEASURE_TIMEOUT="$(read_override "$WORK_ROOT/MEASURE_TIMEOUT" "$MEASURE_TIMEOUT")"
  # An attempt is agent time plus the sweep that scores it. Budget only the first part,
  # or the last attempt of a task hands over a candidate that never gets a verdict.
  RESERVE="$(read_override "$WORK_ROOT/MEASURE_RESERVE" "$(( MEASURE_TIMEOUT + 180 ))")"
  # A reserve bigger than the budget it protects is not a reserve. And a task that has
  # not made an attempt yet has nothing to protect: let it start.
  [ "$RESERVE" -gt "$(( TARGET / 3 ))" ] && RESERVE="$(( TARGET / 3 ))"
  DONE_ATTEMPTS="$(ls "$RUN" 2>/dev/null | grep -c '^attempt_')"
  if [ "$REQUIRE_FULL_BUDGET" != 1 ] && [ "$DONE_ATTEMPTS" -gt 0 ] && [ "$REMAIN" -lt "$(( ATTEMPT_MIN + RESERVE ))" ]; then
    note "stopping: ${REMAIN}s left, not enough for a ${ATTEMPT_MIN}s attempt plus its \
${RESERVE}s measurement -- an unscored candidate is worth nothing"
    write_status done "$CRED" "$BEST" "$DONE_ATTEMPTS" || exit 78
    touch "$STATE/$K.done"
    exit 0
  fi
  AGENT_REMAIN=$(( REMAIN - RESERVE ))
  # A full-budget campaign finishes one last bounded attempt and its measurement,
  # even when that extends past TARGET. It never pads credit by sleeping until TARGET.
  CAP=$(( AGENT_REMAIN < ATTEMPT_CAP ? AGENT_REMAIN : ATTEMPT_CAP ))
  [ "$CAP" -lt "$ATTEMPT_MIN" ] && CAP="$ATTEMPT_MIN"
  EVAL_DIR="$RUN/attempt_$N"
  mkdir -p "$EVAL_DIR"
  SNAP="$RUN/snapshots/before_attempt_$N"
  mkdir -p "$SNAP"
  copy_sources "$TASK_DIR/kernel_src" "$SNAP"
  note "attempt $N: cap=${CAP}s credited=${CRED}s best=${BEST}x"

  # --- skill hook: build this attempt's prompt.
  PROMPT_OUT="$RUN/.prompt_$N.txt"
  if ! env K="$K" TASK_DIR="$TASK_DIR" KERNEL_SRC="$KSRC" KERNEL_SRC_ALL="$KSRC_ALL" \
        ATTEMPT="$N" \
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
  # Stamp the handoff before the attempt, so afterwards we can tell whether the agent
  # wrote one itself. Its own handoff outranks our capture of its closing message.
  HO="$RUN/HANDOFF.md"
  HO_BEFORE=""
  [ -f "$HO" ] && HO_BEFORE="$(md5sum "$HO" 2>/dev/null | cut -d' ' -f1)"
  flock -u 201 || exit 78
  timeout --signal=TERM --kill-after=120 "$CAP" \
      bash "$RUNNER" "$PROMPT_OUT" "$TASK_DIR" "$LOG/attempt_$N.log" 201>&-
  RC=$?
  note "attempt $N runner rc=$RC"
  lock_measurement
  recover_pending

  # The session's closing output is the handoff. prompt.sh reads HANDOFF.md and the
  # ledger promises it, so an attempt that refutes a direction without landing a diff
  # has somewhere to leave what it learned.
  if [ -s "$LOG/attempt_$N.log" ]; then
    mkdir -p "$RUN/handoffs"
    python3 - "$LOG/attempt_$N.log" "$RUN/handoffs/attempt_$N.md" "$N" "$RC" <<'PY' \
      || note "attempt $N: could not capture a handoff (non-fatal)"
import sys

src, dst, n, rc = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    text = open(src, errors="replace").read()
except OSError:
    sys.exit(1)
# Drop the runner's own framing; what is left is what the session said.
body = "\n".join(l for l in text.splitlines()
                 if not l.startswith("[runner:")).strip()
if not body:
    sys.exit(0)
LIMIT = 8000
if len(body) > LIMIT:
    body = body[-LIMIT:]
    body = "[...truncated, full session output in %s]\n\n" % src + body
with open(dst, "w") as f:
    f.write("# Handoff from attempt %s (runner rc=%s)\n\n" % (n, rc))
    f.write("Written by the worker from the attempt's own closing output. Treat the\n"
            "measurements in it as measured and the rest as that attempt's reasoning.\n\n")
    f.write(body + "\n")
PY
    if [ -s "$RUN/handoffs/attempt_$N.md" ]; then
      HO_AFTER=""
      [ -f "$HO" ] && HO_AFTER="$(md5sum "$HO" 2>/dev/null | cut -d' ' -f1)"
      if [ -n "$HO_AFTER" ] && [ "$HO_AFTER" != "$HO_BEFORE" ]; then
        # The attempt wrote its own handoff. Keep it, and append the capture rather
        # than replacing -- the two say different things.
        cp "$HO" "$RUN/handoffs/attempt_${N}_agent.md"
        {
          printf '\n\n---\n\n## Closing output of attempt %s (captured by the worker)\n\n' "$N"
          sed -e '1,4d' "$RUN/handoffs/attempt_$N.md"
        } >> "$HO"
      else
        cp "$RUN/handoffs/attempt_$N.md" "$HO"
      fi
    fi
  fi
  if [ "$RC" = 78 ]; then
    # Unmet precondition. Ten thousand retries give the same result; do not spend budget.
    note "FATAL runner returned 78 (unmet precondition), see $LOG/attempt_$N.log"
    copy_sources "$SNAP" "$TASK_DIR/kernel_src"
    restore_frozen
    write_status blocked "$CRED" "$BEST" "$N"
    exit 78
  fi

  if ! FROZEN_CHANGES="$(python3 "$SOURCE_TREE" check-frozen "$TASK_DIR" "$FROZEN" 2>&1)"; then
    note "attempt $N: $FROZEN_CHANGES -- rejecting and restoring"
    record_rejected "$N" "" "" "0" "$FROZEN_CHANGES"
    copy_sources "$SNAP" "$TASK_DIR/kernel_src"
    restore_frozen
    now="$(date +%s)"
    write_status running "$(( BASE + now - START ))" "$BEST" "$N" "null" "$RC" "null" "0"
    continue
  fi

  # --- measure, then decide KEEP or revert.
  parse_measurement() {  # <measure output>
    SP="$(echo "$1" | sed -n 's/.*SPEEDUP=\([0-9.eE+-]*\).*/\1/p')"
    OK="$(echo "$1" | sed -n 's/.*STATUS=\([01]\).*/\1/p')"
    CLEAN="$(echo "$1" | sed -n 's/.*CLEAN=\([0-9.eE+-]*\).*/\1/p')"
    SINGLE="$(echo "$1" | sed -n 's/.*SINGLE=\([0-9]*\).*/\1/p')"
    OK="${OK:-0}"; SINGLE="${SINGLE:-0}"
  }

  # An attempt that was killed at the cap -- or that decided against every edit it tried
  # -- leaves kernel_src byte-identical to the snapshot. Measuring that spends a full
  # sweep to rediscover 1.0x. There is no candidate here; record it and move on.
  SOURCE_CHANGES="$(python3 "$SOURCE_TREE" check "$SNAP" "$TASK_DIR/kernel_src" \
      --allowed "$KSRC_ALL" 2>&1)"
  CHANGED=$?
  if [ "$CHANGED" -gt 1 ]; then
    note "attempt $N: $SOURCE_CHANGES -- restoring complete source tree"
    record_rejected "$N" "" "" "0" "$SOURCE_CHANGES"
    copy_sources "$SNAP" "$TASK_DIR/kernel_src"
    now="$(date +%s)"
    write_status running "$(( BASE + now - START ))" "$BEST" "$N" "null" "$RC" "null" "0"
    continue
  fi
  if [ "$CHANGED" = 0 ]; then
    note "attempt $N: kernel_src is byte-identical to the snapshot (runner rc=$RC) -- \
no candidate to measure, skipping the sweep"
    record_rejected "$N" "" "" "0" "no edit (runner rc=$RC)"
    now="$(date +%s)"
    write_status running "$(( BASE + now - START ))" "$BEST" "$N" "null" "$RC" "null" "0"
    continue
  fi

  # A sweep past this point is bounded by MEASURE_TIMEOUT and is the only thing this
  # attempt was for. Mark it so the scheduler's budget check does not kill it halfway.
  : > "$STATE/$K.measuring"
  if measure_and_retain 1; then
    :
  else
    MEASURE_STEP_RC=$?
    rm -f "$STATE/$K.measuring"
    note "FATAL measurement hook or evidence retention failed; stopping before overwrite or revert"
    write_status blocked || exit 78
    exit "$MEASURE_STEP_RC"
  fi
  rm -f "$STATE/$K.measuring"
  parse_measurement "$M"
  [ -n "$CLEAN" ] && note "attempt $N: measured ${SP:-none} (harness serving-weighted); \
${CLEAN} over the ${SINGLE:+non-}single-pair buckets, ${SINGLE} bucket(s) measured once"

  # A bucket the harness measured once is one unrepeated sample, and it keeps exactly the
  # first pairs that fell outside its undecided band. When the two figures straddle the
  # keep threshold, the verdict rests on those samples alone -- that is the one case where
  # a second full measurement can change the answer, so pay for it there and nowhere else.
  if [ "$OK" = 1 ] && [ -n "$SP" ] && [ -n "$CLEAN" ] && [ "${SINGLE:-0}" -gt 0 ]; then
    STRADDLE="$(python3 -c "
print(1 if $SP < $KEEP_THRESHOLD <= $CLEAN else 0)")"
    if [ "$STRADDLE" = 1 ]; then
      note "attempt $N: ${SP} and ${CLEAN} straddle the ${KEEP_THRESHOLD} threshold on \
${SINGLE} single-pair bucket(s) -- re-measuring once, and deciding on that"
      : > "$STATE/$K.measuring"
      if measure_and_retain 2; then
        :
      else
        MEASURE_STEP_RC=$?
        rm -f "$STATE/$K.measuring"
        note "FATAL second-sweep hook or evidence retention failed; stopping before overwrite or revert"
        write_status blocked || exit 78
        exit "$MEASURE_STEP_RC"
      fi
      M2="$M"
      rm -f "$STATE/$K.measuring"
      parse_measurement "$M2"
      note "attempt $N: re-measured ${SP:-none} (clean ${CLEAN:-none}, ${SINGLE} single-pair)"
    fi
  fi

  if ! python3 "$SOURCE_TREE" check-frozen "$TASK_DIR" "$FROZEN" >> "$LOG/worker.log" 2>&1; then
    note "attempt $N: measurement changed the frozen harness -- rejecting"
    restore_frozen
    OK=0; SP=""
  fi
  python3 "$SOURCE_TREE" check "$SNAP" "$TASK_DIR/kernel_src" \
      --allowed "$KSRC_ALL" >> "$LOG/worker.log" 2>&1
  if [ "$?" -gt 1 ]; then
    note "attempt $N: measurement left invalid source edits -- rejecting"
    OK=0; SP=""
  fi

  if [ -z "$SP" ]; then
    note "attempt $N: no measurement -- reverting"
    record_rejected "$N" "" "" "0" "no measurement"
    copy_sources "$SNAP" "$TASK_DIR/kernel_src"
  else
    KEEP="$(python3 -c "
sp, best = $SP, $BEST
print(1 if $OK and sp >= $KEEP_THRESHOLD and sp > best + $KEEP_MARGIN else 0)")"
    if [ "$KEEP" = 1 ]; then
      note "attempt $N: KEEP ${SP}x (was ${BEST}x)"
      BEST="$SP"
      mkdir -p "$RUN/best_kernel_src"
      copy_sources "$TASK_DIR/kernel_src" "$RUN/best_kernel_src"
      cp -a "$RUN/.measure.log" "$RUN/best_measure.log" 2>/dev/null
    else
      # Separate "the harness would not run" from "a gate said no". The negative
      # control is built by text-patching kernel_src at a literal source anchor, so an
      # edit that rewrites the anchored line aborts the run before anything is measured.
      # Reporting that as a gate failure sends the next attempt to fix a kernel that was
      # never evaluated.
      REFUSED=0
      if grep -qE 'anchor is not unique|refusing to patch blind' "$RUN/.measure.log" \
         2>/dev/null; then
        REFUSED=1
      fi
      if [ "$REFUSED" = 1 ]; then
        ANCHOR_MUT="$(sed -n 's/.*RuntimeError: mutation \([A-Za-z0-9_]*\).*/\1/p' \
          "$RUN/.measure.log" 2>/dev/null | tail -1)"
        note "attempt $N: revert -- the harness refused to run (negative control \
${ANCHOR_MUT:-mutation} lost its source anchor); nothing was measured"
        record_rejected "$N" "" "$CLEAN" "${SINGLE:-0}" \
          "the harness refused to run: the ${ANCHOR_MUT:-negative-control} mutation \
text-patches kernel_src at a literal source anchor and this edit removed it, so the run \
aborted before any gate or timing. The kernel was NOT judged -- do not read this as a \
correctness or speed failure. Keep the anchored expression and neutralise it at trace \
time instead (a folded constant leaves the control armed), then this same candidate can \
be measured"
      else
        note "attempt $N: revert (correct=$OK measured=${SP} best=${BEST}x threshold=${KEEP_THRESHOLD})"
        record_rejected "$N" "$SP" "$CLEAN" "${SINGLE:-0}" \
            "$( [ "$OK" = 1 ] && echo "below the keep threshold" || echo "a gate failed" )"
      fi
      copy_sources "$SNAP" "$TASK_DIR/kernel_src"
    fi
  fi

  now="$(date +%s)"
  write_status running "$(( BASE + now - START ))" "$BEST" "$N" "${SP:-null}" "$RC" \
      "${CLEAN:-null}" "${SINGLE:-0}" || exit 78
done
