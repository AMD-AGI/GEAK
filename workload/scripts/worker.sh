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
#   * a JSON line {"exit": 0, "weighted": <x>}, where exit 0 means every gate passed
#     (correctness, negative controls, tail contracts) and `weighted` is the
#     serving-weighted speedup over the whole shape sweep;
#   * a JSON *document* {"ut_result": "run/ut_result.json", "ok": true, "gates": {...}} --
#     what the AgentX extraction harness emits. It is pretty-printed, so it has to be read
#     as a whole document, and the speedup is not in it at all: it lives in the referenced
#     ut_result.json under gates.timing.aggregate.primary (weighted, falling back to the
#     geomean). `ok` is the conjunction of the six frozen-harness gates.
#
# Accepting only the first two cost a real measurement on 2026-10-19: all six gates passed,
# the sweep returned a serving-weighted 0.95806 over 13/13 buckets, and the attempt was
# recorded as unmeasured. A regression is a result; "no measurement" is a broken harness.
# Reporting the second when the first happened teaches the optimizer nothing and spends
# the budget anyway.
default_measure() {
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ( timeout "$MEASURE_TIMEOUT" python3 "$here/run_task_entry.py" "$TASK_DIR" unittest.py ) \
      > "$RUN/.measure.log" 2>&1
  local measure_rc=$?
  if [ "$measure_rc" -ne 0 ]; then
    echo "STATUS=0 SPEEDUP= CLEAN= SINGLE=0"
    return
  fi
  local sp ok
  sp="$(grep -oE 'GEAK_GEOMEAN_SPEEDUP [0-9.]+' "$RUN/.measure.log" | tail -1 | awk '{print $2}')"
  ok=0; grep -q '^CORRECTNESS: PASS' "$RUN/.measure.log" && ok=1
  if [ -z "$sp" ]; then
    read -r ok sp clean single <<<"$(python3 - "$RUN/.measure.log" "$TASK_DIR" <<'PY'
import json, os, sys

log, task_dir = sys.argv[1], sys.argv[2]
text = open(log, errors="replace").read()


def docs(s):
    """Every JSON object in the log, pretty-printed ones included.

    The harness writes its summary with indent=2, so a line-wise reader sees only "{".
    Scanning with raw_decode from each "{" at the start of a line costs nothing and
    handles both shapes.
    """
    dec = json.JSONDecoder()
    i = 0
    while True:
        i = s.find("{", i)
        if i < 0:
            return
        if i and s[i - 1] not in "\n\r":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(s, i)
        except ValueError:
            i += 1
            continue
        yield obj
        i = end


def aggregate_speedup(path):
    """(primary, clean, single_pair_count) from an agentx_ut_result document.

    `primary` is the harness's own serving-weighted figure. `clean` re-weights the same
    way over the buckets it measured more than once: the harness stops repeating a bucket
    whose first pair falls outside its undecided band, so a single-pair bucket is one
    unrepeated sample and per-pair noise on a collective seam reaches 12%.
    """
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None, None, 0
    timing = ((d.get("gates") or {}).get("timing") or {})
    agg = timing.get("aggregate") or {}
    primary = None
    for key in ("primary", "weighted", "geomean"):
        v = agg.get(key)
        if isinstance(v, (int, float)) and v > 0:
            primary = float(v)
            break
    if primary is None:
        # Interval form: the harness reports two serving aggregates because the prefix
        # history behind the dominant buckets was never recorded, and refuses to pick
        # one. Decide on the low end -- the high end would assert the model the harness
        # just said it cannot support.
        ends = [float(v) for k, v in timing.items()
                if k.startswith("serving_weighted_speedup_")
                and isinstance(v, (int, float)) and v > 0]
        iv = timing.get("serving_weighted_speedup_interval")
        if isinstance(iv, (list, tuple)):
            ends += [float(v) for v in iv if isinstance(v, (int, float)) and v > 0]
        if ends:
            primary = min(ends)
    cases = (timing.get("per_case") or [])
    rows = (timing.get("rows") or [])
    reps = {c.get("sig"): c.get("reps") for c in (cases or rows)}
    num = den = 0.0
    single = 0
    if not cases and rows:
        # The interval variant carries no weights, only measured rows. Count how many
        # rest on a single pair so the worker can say so; the aggregate stays the
        # harness's own.
        single = sum(1 for r in rows if (r.get("reps") or 0) < 2)
        return primary, None, single
    for c in (agg.get("per_case") or []):
        if not c.get("included"):
            continue
        w, sp = c.get("weight"), c.get("speedup")
        if not isinstance(w, (int, float)) or not isinstance(sp, (int, float)):
            continue
        if (reps.get(c.get("sig")) or 0) >= 2:
            num += w * sp
            den += w
        else:
            single += 1
    clean = (num / den) if den > 0 else None
    return primary, clean, single


ok, sp, clean, single = 0, "", "", 0
for d in docs(text):
    if "exit" in d:
        # A gate failure is not a measurement: report STATUS=0 and let the worker revert.
        ok = 1 if d.get("exit") == 0 else 0
        w = d.get("weighted")
        sp = "" if w in (None, "") else repr(float(w))
        break
    if "gates" not in d and d and all(isinstance(v, bool) for v in d.values()):
        # Bare gate map: the pass/fail verdict is the document itself, and the numbers
        # are in the task's own ut_result.json. "UT PASS" in the log is the harness's
        # verdict; require the gates to agree with it.
        ok = 1 if (d.values() and all(d.values()) and "UT PASS" in text) else 0
        for cand in ("ut_result.json", os.path.join("run", "ut_result.json")):
            p = os.path.join(task_dir, cand)
            if os.path.exists(p):
                v, c, single = aggregate_speedup(p)
                if v is not None:
                    sp = repr(v)
                if c is not None:
                    clean = repr(c)
                break
        break
    if "ut_result" in d and "gates" in d:
        gates = d.get("gates") or {}
        # `ok` is the harness's own verdict; require the gate map to agree, so a future
        # gate the summary forgets to fold in cannot pass silently.
        ok = 1 if (d.get("ok") is True and all(v is True for v in gates.values())) else 0
        p = d["ut_result"]
        if not os.path.isabs(p):
            p = os.path.join(task_dir, p)
        v, c, single = aggregate_speedup(p)
        if v is not None:
            sp = repr(v)
        if c is not None:
            clean = repr(c)
        break
print(ok, sp, clean, single)
PY
)"
  fi
  echo "STATUS=${ok:-0} SPEEDUP=${sp:-} CLEAN=${clean:-} SINGLE=${single:-0}"
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

write_status() {  # status credited best attempts [measured rc]
  python3 - "$STATUS" "$@" <<'PY'
import json, os, sys
p, st, cred, best, n = sys.argv[1:6]
extra = sys.argv[6:]
d = json.load(open(p)) if os.path.exists(p) else {}
d.update(status=st, credited_seconds=int(cred), best_speedup=float(best),
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
  ATTEMPT_CAP="$(read_override "$WORK_ROOT/ATTEMPT_CAP" "$ATTEMPT_CAP")"
  MEASURE_TIMEOUT="$(read_override "$WORK_ROOT/MEASURE_TIMEOUT" "$MEASURE_TIMEOUT")"
  # An attempt is agent time plus the sweep that scores it. Budget only the first part,
  # or the last attempt of a task hands over a candidate that never gets a verdict.
  RESERVE="$(read_override "$WORK_ROOT/MEASURE_RESERVE" "$(( MEASURE_TIMEOUT + 180 ))")"
  # A reserve bigger than the budget it protects is not a reserve. And a task that has
  # not made an attempt yet has nothing to protect: let it start.
  [ "$RESERVE" -gt "$(( TARGET / 3 ))" ] && RESERVE="$(( TARGET / 3 ))"
  DONE_ATTEMPTS="$(ls "$RUN" 2>/dev/null | grep -c '^attempt_')"
  if [ "$DONE_ATTEMPTS" -gt 0 ] && [ "$REMAIN" -lt "$(( ATTEMPT_MIN + RESERVE ))" ]; then
    note "stopping: ${REMAIN}s left, not enough for a ${ATTEMPT_MIN}s attempt plus its \
${RESERVE}s measurement -- an unscored candidate is worth nothing"
    write_status done "$CRED" "$BEST" "$DONE_ATTEMPTS"
    touch "$STATE/$K.done"
    exit 0
  fi
  AGENT_REMAIN=$(( REMAIN - RESERVE ))
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
  timeout --signal=TERM --kill-after=120 "$CAP" \
      bash "$RUNNER" "$PROMPT_OUT" "$TASK_DIR" "$LOG/attempt_$N.log"
  RC=$?
  note "attempt $N runner rc=$RC"

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
    SP="$(echo "$1" | sed -n 's/.*SPEEDUP=\([0-9.]*\).*/\1/p')"
    OK="$(echo "$1" | sed -n 's/.*STATUS=\([01]\).*/\1/p')"
    CLEAN="$(echo "$1" | sed -n 's/.*CLEAN=\([0-9.]*\).*/\1/p')"
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
  M="$(measure)"
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
      cp -a "$RUN/.measure.log" "$RUN/.measure_1.log" 2>/dev/null
      : > "$STATE/$K.measuring"
      M2="$(measure)"
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
      "${CLEAN:-null}" "${SINGLE:-0}"
done
