#!/usr/bin/env bash
# wait_for.sh - launch a long command so its completion is OBSERVABLE, and wait on that
# observation instead of on a timer.
#
# THE GAP THIS FILLS. An agent that backgrounds a command is told it will be notified when the
# command exits. An agent launched by a workflow (a GEAK role such as the deep_engineer) does not
# reliably receive that notification: across a measured campaign, hundreds of background launches
# produced zero completion events and zero `BashOutput` reads inside those agents' transcripts. With no event to wait on, the only remaining move is to guess a duration and sleep --
# and the rational guess is a large one, because sleeping too little costs another turn of
# generation while sleeping too much costs nothing the agent can see.
#
# What that produces is not a small inefficiency. Timers get set in minutes while the jobs they
# cover finish in tens of seconds, so most of the waiting is spent on a job that has already exited,
# and nothing in the transcript reveals it: the sleep returns, the log looks finished, and the agent
# proceeds. The failure is invisible precisely because it always "works".
#
# THE FIX IS AN ARTIFACT, NOT A LONGER SLEEP. `--launch` runs the command under `setsid` and, when
# it exits, writes its exit code to `<log>.rc` via a temp file and `mv`, so the marker appears
# atomically and only after the command is done. `--await` then polls for that file on a short
# interval. Polling a marker every few seconds is cheap in a way that sleeping is not: it costs one
# `test -e` per interval and it stops within one interval of the actual exit.
#
# WHY A FILE AND NOT A PID. `kill -0 $PID` answers "is it alive", which is necessary and not
# sufficient -- a job blocked in a GPU queue is alive right up to the moment it is dropped, and a
# job that died in one second is indistinguishable from one that never started. The `.rc` file
# carries the exit STATUS, so "finished" and "finished badly" are different answers, and it survives
# the process, so a waiter that starts late still learns the outcome.
#
# NEVER MATCH THE WAITER AGAINST ITSELF. A `pgrep -f <pattern>` whose pattern also appears in the
# polling command matches the poller, so the predicate is true forever and the loop runs to its
# ceiling. That is why this tool waits on a marker rather than on a pattern.
#
# USAGE
#   wait_for.sh --run --log <path> [--fg-s N] -- <cmd...>
#       THE DEFAULT. Runs <cmd> in the foreground and returns its exit code, like a plain call --
#       but launched under a marker, so if it overruns --fg-s it is ALREADY detached: this prints
#       how to await it and returns 124. You do not have to know the duration beforehand.
#
#   wait_for.sh --launch --log <path> [--tag <name>] -- <cmd...>
#       Starts <cmd> detached. Prints the log path, the pid and the rc path. Returns immediately.
#
#   wait_for.sh --await --log <path> [--max-s N] [--interval-s N] [--quiet]
#       Blocks until <log>.rc exists or the ceiling elapses. Prints `rc=<n> elapsed=<s>` plus the
#       tail of the log, and exits with the command's own exit code (or 124 on timeout, which means
#       STILL RUNNING and not failed). Unset, the ceiling is AWAIT_MAX_S (570 s) rather than the
#       shorter `--run` foreground window: this job is already backgrounded, so the bound is the
#       heartbeat, not the caller's stall detector. On timeout it prints the exact --status and
#       doubled --await commands, so a job of unknown length is bracketed rather than guessed at.
#
#   wait_for.sh --status --log <path>
#       One line, no blocking: `running` / `done rc=<n>` / `missing`.
#
# Run BOTH halves through this tool. A command launched with a bare `&` leaves no marker, so
# `--await` on it can only time out.
#
# PREFER `--run`. Splitting launch from await forces a duration guess BEFORE the job has run, which
# is the one moment nobody can make it: guess short and an overrunning job is abandoned, guess long
# and a two-second probe pays for a marker plus a second call. Under that trade the cheap move is a
# plain foreground call with a blind sleep behind it -- exactly what this tool replaces. `--run`
# removes the guess by making the decision afterwards, from what the job actually did.
set -uo pipefail

# THE WAIT MUST STAY UNDER THE CALLER'S STALL DETECTOR, AND MUST BE AUDIBLE WHILE IT WAITS.
#
# A blocking wait that prints nothing is indistinguishable from a hung agent. The harness above this
# tool kills an agent that produces no output for 180 s, so a silent 590 s await is not a long wait --
# it is a guaranteed kill a third of the way in, and the agent loses everything it had reasoned
# through. Measured on one fleet run: of 45 tool calls that ran past 180 s, 28 were `--await`, and the
# agents holding them were reaped.
#
# THE BUDGET IS THE WAIT PLUS THE THINKING THAT FOLLOWS IT, NOT THE WAIT ALONE. What the detector
# measures is the gap between two pieces of output, and a returning tool call does not end that gap --
# the agent still has to reason about the result before it emits anything. Measured on the same run,
# that reasoning is p90 = 38 s and reaches 232 s. So a window sized right up against 180 s is still a
# kill: 150 + 38 already overruns it. 120 s leaves the deliberation room to land inside the budget,
# and the cost of being wrong is only a second call.
#
# The split follows from that. Short work stays in the FOREGROUND, because a round trip through a
# marker costs a turn and most commands here finish in seconds. Anything still running at the
# foreground deadline is ALREADY detached under its marker, so it converts to background at no cost
# and is collected later. A heartbeat every 45 s keeps the caller audible either way -- silence is
# what gets reaped, not duration.
MODE=""; LOG=""; TAG=""; MAX_S=120; INTERVAL_S=5; QUIET=0; FG_S=""; TAIL_N=20; HEARTBEAT_S=45
MAX_S_SET=0
# `--run`'s foreground window and `--await`'s blocking ceiling are DIFFERENT quantities and shared one
# default, which is why a sweep could not be awaited without a guess. The window above must stay under
# the caller's stall detector, because overrunning it converts a long job into a killed agent. The
# ceiling below is "how long am I willing to block on a job I have ALREADY backgrounded", and the
# answer is bounded by the heartbeat rather than by the detector -- silence is what gets reaped, not
# duration, and `--await` emits a line every HEARTBEAT_S throughout.
#
# 570 is the value measured callers had already worked out for themselves: a Wave-2 session passed
# `--max-s 560` and then `570`, and both still timed out with the sweep running, because a config
# sweep on a slow reader takes tens of minutes. A short default made every long await a guess, and a
# guess that returns 124 looks identical to one that was simply too small.
AWAIT_MAX_S=570
while [ $# -gt 0 ]; do
  case "$1" in
    --run)        MODE=run;    shift ;;
    --launch)     MODE=launch; shift ;;
    --await)      MODE=await;  shift ;;
    --status)     MODE=status; shift ;;
    --log)        LOG="${2:-}"; shift 2 ;;
    --tag)        TAG="${2:-}"; shift 2 ;;
    --max-s)      MAX_S="${2:-}"; MAX_S_SET=1; shift 2 ;;
    --interval-s) INTERVAL_S="${2:-}"; shift 2 ;;
    --fg-s)       FG_S="${2:-}"; shift 2 ;;
    --tail)       TAIL_N="${2:-}"; shift 2 ;;
    --quiet)      QUIET=1; shift ;;
    --selftest)   MODE=selftest; shift ;;
    --help|-h)    sed -n '2,56p' "$0"; exit 0 ;;
    --)           shift; break ;;
    *)            echo "wait_for.sh: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
done

# HOW MUCH OF THE LOG COMES BACK. Every byte echoed here lands in the caller's context and stays
# there for the rest of its life, so this is a context budget, not a display choice. The default is
# small on purpose: the answer to "did it work" is the rc, and the tail is for the case where it did
# not. A caller that needs more asks for more (`--tail 200`), and one that needs none says `--tail 0`
# -- which is the right setting for a job whose output it is about to parse from the file anyway.
_tail_log() {
  [ "${TAIL_N:-20}" -gt 0 ] 2>/dev/null || return 0
  tail -n "${TAIL_N}" "$1"
}

_rc_path()  { printf '%s.rc' "$1"; }
_pid_path() { printf '%s.pid' "$1"; }

do_launch() {
  [ -n "$LOG" ] || { echo "wait_for.sh --launch needs --log" >&2; exit 2; }
  [ $# -gt 0 ]  || { echo "wait_for.sh --launch needs a command after --" >&2; exit 2; }
  mkdir -p "$(dirname "$LOG")" || exit 2
  local rc pid_f
  rc="$(_rc_path "$LOG")"; pid_f="$(_pid_path "$LOG")"
  # A stale marker from a previous run would make --await return instantly with the OLD status.
  # Moved aside rather than rm'd (GEAK convention for scripts roles may run).
  local _stale; _stale="$(date +%s)_$$"
  [ -e "$rc" ] && mv -f -- "$rc" "$rc.stale_$_stale"
  [ -e "$pid_f" ] && mv -f -- "$pid_f" "$pid_f.stale_$_stale"
  # setsid: survives the agent's shell going away between turns. The rc is written by the SAME
  # subshell that ran the command, via temp+mv, so a reader never sees a half-written marker and
  # never sees one before the command is actually finished.
  setsid bash -c '
    "$@" >>"$0" 2>&1
    printf "%s\n" "$?" > "$0.rc.tmp" && mv -f "$0.rc.tmp" "$0.rc"
  ' "$LOG" "$@" </dev/null >/dev/null 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_f"
  echo "launched${TAG:+ [$TAG]} pid=$pid log=$LOG rc=$rc"
  echo "  await with: bash $0 --await --log $LOG"
}

# --run: THE MODE THAT NEEDS NO GUESS.
#
# `--launch` and `--await` are two acts, so using them correctly requires answering "how long will
# this take?" BEFORE running it -- and that is the one thing nobody knows yet. Guess "short" and a
# job that overruns the turn is lost; guess "long" and a two-second probe costs a marker, a second
# call and a polling interval. Faced with that, the cheap move is a plain foreground call, and the
# fallback when it overruns is a blind sleep -- which is the behaviour this tool exists to replace.
#
# So: always launch under the marker, then wait in the FOREGROUND. If the job finishes inside the
# foreground window -- which is the common case -- this returns exactly like a plain call, with the
# command's own exit code, and the marker was free. If it overruns, the job is ALREADY detached and
# already writing its rc, so nothing is lost: this prints how to await it and returns 124, and the
# caller can go do other work and collect the result later.
#
# The classification still happens, but the TOOL makes it, after the fact, from what the job
# actually did -- instead of the caller making it beforehand from a guess.
do_run() {
  [ -n "$LOG" ] || { echo "wait_for.sh --run needs --log" >&2; exit 2; }
  [ $# -gt 0 ]  || { echo "wait_for.sh --run needs a command after --" >&2; exit 2; }
  # Default foreground window: just under the caller's own per-call ceiling, so we hand back control
  # while the caller can still act on it rather than being cut off mid-wait.
  # Same reasoning as MAX_S: a foreground window longer than the caller's stall detector converts a
  # long job into a killed agent. Overrunning is cheap here -- the job is already detached and its rc
  # is still coming -- so the default errs short.
  local fg="${FG_S:-$MAX_S}"
  do_launch "$@" >/dev/null || return $?
  local rc_f t0 rc
  rc_f="$(_rc_path "$LOG")"; t0=$SECONDS
  while :; do
    if [ -e "$rc_f" ]; then
      rc="$(head -1 "$rc_f" 2>/dev/null | tr -d '"'"'[:space:]'"'"')"
      [ "$QUIET" -eq 1 ] || {
        echo "rc=${rc:-?} elapsed=$((SECONDS - t0))s log=$LOG"
        [ -s "$LOG" ] && [ "${TAIL_N:-20}" -gt 0 ] 2>/dev/null && { echo "--- tail ---"; _tail_log "$LOG"; }
      }
      # The job's OWN status. Returning a flat 0 here would make --run report every finished job as
      # a success, which is the exact confusion the marker exists to prevent: "it finished" and "it
      # finished well" are different answers. A malformed marker is not a success either.
      case "$rc" in ''|*[!0-9]*) return 1 ;; *) return "$rc" ;; esac
    fi
    if [ "$QUIET" -ne 1 ] && [ $(( (SECONDS - t0) % HEARTBEAT_S )) -lt "$INTERVAL_S" ] \
       && [ $((SECONDS - t0)) -ge "$HEARTBEAT_S" ]; then
      echo "  ...running $((SECONDS - t0))s: $(basename "$LOG")"
    fi
    if [ $((SECONDS - t0)) -ge "$fg" ]; then
      # Not a failure and not a loss: it is detached and its rc is still coming.
      echo "STILL RUNNING after ${fg}s -- the job is detached and will write its own rc." >&2
      echo "  do other work, then: bash $0 --await --log $LOG" >&2
      echo "  or check without blocking: bash $0 --status --log $LOG" >&2
      return 124
    fi
    sleep "$INTERVAL_S"
  done
}

do_await() {
  [ -n "$LOG" ] || { echo "wait_for.sh --await needs --log" >&2; exit 2; }
  local rc_f t0 waited rc ceiling
  # The caller's explicit --max-s always wins. Unset, this is a BACKGROUNDED job's ceiling and not
  # the foreground window, so it uses the larger default -- see AWAIT_MAX_S.
  ceiling="$MAX_S"
  [ "$MAX_S_SET" -eq 1 ] || ceiling="$AWAIT_MAX_S"
  rc_f="$(_rc_path "$LOG")"; t0=$SECONDS
  while :; do
    if [ -e "$rc_f" ]; then
      rc="$(head -1 "$rc_f" 2>/dev/null | tr -d '[:space:]')"
      waited=$((SECONDS - t0))
      [ "$QUIET" -eq 1 ] || {
        echo "rc=${rc:-?} elapsed=${waited}s log=$LOG"
        [ -s "$LOG" ] && [ "${TAIL_N:-20}" -gt 0 ] 2>/dev/null && { echo "--- tail ---"; _tail_log "$LOG"; }
      }
      case "$rc" in ''|*[!0-9]*) return 1 ;; *) return "$rc" ;; esac
    fi
    # A heartbeat, because SILENCE is what gets an agent reaped, not elapsed time. One line every
    # HEARTBEAT_S keeps the caller visibly alive to whatever is watching it, and tells a human
    # reading the log that the wait is progressing rather than wedged.
    if [ "$QUIET" -ne 1 ] && [ $(( (SECONDS - t0) % HEARTBEAT_S )) -lt "$INTERVAL_S" ] \
       && [ $((SECONDS - t0)) -ge "$HEARTBEAT_S" ]; then
      echo "  ...waiting $((SECONDS - t0))s for $(basename "$LOG") (job still running)"
    fi
    if [ $((SECONDS - t0)) -ge "$ceiling" ]; then
      # A timeout is NOT a result. Say which one it is so the caller does not read it as failure:
      # the job may still be running, and its rc will appear later.
      #
      # AND HAND BACK THE EXACT NEXT COMMAND. "re-await with a larger --max-s" left the caller to
      # invent a number, which is the same guess `--run` exists to remove: one measured session went
      # 560 then 570 and timed out both times, because the job was tens of minutes long and nothing
      # in the message said so. The suggestion doubles, so a job of unknown length is bracketed in a
      # few calls instead of being probed one guess at a time.
      local next=$((ceiling * 2))
      echo "TIMEOUT after ${ceiling}s: no ${rc_f} yet. The job is STILL RUNNING and nothing was" >&2
      echo "  lost -- its rc will appear when it finishes. Next, in order of cost:" >&2
      echo "    bash $0 --status --log $LOG                       # ask, without blocking" >&2
      echo "    bash $0 --await --log $LOG --max-s ${next}        # keep waiting (doubled)" >&2
      [ -s "$LOG" ] && [ "${TAIL_N:-20}" -gt 0 ] 2>/dev/null && { echo "--- tail so far ---" >&2; _tail_log "$LOG" >&2; }
      return 124
    fi
    sleep "$INTERVAL_S"
  done
}

do_status() {
  [ -n "$LOG" ] || { echo "wait_for.sh --status needs --log" >&2; exit 2; }
  local rc_f pid_f rc pid
  rc_f="$(_rc_path "$LOG")"; pid_f="$(_pid_path "$LOG")"
  if [ -e "$rc_f" ]; then
    rc="$(head -1 "$rc_f" 2>/dev/null | tr -d '[:space:]')"
    echo "done rc=${rc:-?}"; return 0
  fi
  if [ -e "$pid_f" ]; then
    pid="$(head -1 "$pid_f" 2>/dev/null | tr -d '[:space:]')"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "running pid=$pid"; return 0; fi
    # No rc and no process: the job died without the wrapper completing (killed -9, box rebooted).
    # Reporting this as `running` would hang a waiter forever.
    echo "vanished pid=$pid (no rc file and no live process)"; return 0
  fi
  echo "missing (never launched through wait_for.sh, or a different --log)"; return 0
}

selftest() {
  local td rc out
  td="$(mktemp -d)"; trap 'rm -rf "$td"' RETURN
  local ok=0 bad=0
  _chk() { if [ "$2" = "$3" ]; then ok=$((ok+1)); echo "  ok: $1"; else bad=$((bad+1)); echo "  FAIL: $1 (got '$2' want '$3')"; fi; }

  # A fast job: --await must return its status, not a timer.
  bash "$0" --launch --log "$td/a.log" -- bash -c 'echo hello; exit 0' >/dev/null
  bash "$0" --await --log "$td/a.log" --interval-s 1 --quiet; rc=$?
  _chk "a successful job returns rc 0" "$rc" "0"
  _chk "and its output is in the log" "$(cat "$td/a.log")" "hello"

  # A FAILING job must be distinguishable from a successful one -- `kill -0` cannot do this.
  bash "$0" --launch --log "$td/b.log" -- bash -c 'exit 7' >/dev/null
  bash "$0" --await --log "$td/b.log" --interval-s 1 --quiet; rc=$?
  _chk "a failing job returns ITS exit code" "$rc" "7"

  # The waiter must stop within an interval of the exit, not sleep out a ceiling.
  bash "$0" --launch --log "$td/c.log" -- bash -c 'sleep 2' >/dev/null
  local t0=$SECONDS
  bash "$0" --await --log "$td/c.log" --interval-s 1 --max-s 60 --quiet
  local waited=$((SECONDS - t0))
  if [ "$waited" -le 6 ]; then ok=$((ok+1)); echo "  ok: waits for the exit, not the ceiling (${waited}s for a 2s job)"
  else bad=$((bad+1)); echo "  FAIL: took ${waited}s for a 2s job"; fi

  # A timeout is 124 and must NOT be mistaken for the job's own failure.
  bash "$0" --launch --log "$td/d.log" -- bash -c 'sleep 30' >/dev/null
  bash "$0" --await --log "$td/d.log" --interval-s 1 --max-s 2 --quiet 2>/dev/null; rc=$?
  _chk "a timeout is 124, not a job failure" "$rc" "124"
  _chk "and --status still says it is running" "$(bash "$0" --status --log "$td/d.log" | cut -d' ' -f1)" "running"

  # A stale rc from a previous run must not satisfy the next --await instantly.
  echo 0 > "$td/e.log.rc"
  bash "$0" --launch --log "$td/e.log" -- bash -c 'sleep 1; exit 3' >/dev/null
  bash "$0" --await --log "$td/e.log" --interval-s 1 --max-s 30 --quiet; rc=$?
  _chk "a stale rc is cleared at launch" "$rc" "3"

  # --run needs no duration guess: a short job returns like a plain foreground call...
  bash "$0" --run --log "$td/f.log" --interval-s 1 --quiet -- bash -c 'echo done; exit 0'; rc=$?
  _chk "--run returns a fast job's own rc, like a plain call" "$rc" "0"
  _chk "and its output is captured" "$(cat "$td/f.log")" "done"
  bash "$0" --run --log "$td/g.log" --interval-s 1 --quiet -- bash -c 'exit 5'; rc=$?
  _chk "--run returns a FAILING job's own rc, not a generic 1" "$rc" "5"

  # ...and an overrun is neither a failure nor a loss: the job was launched under a marker from the
  # start, so it keeps running detached and its rc is collectable afterwards. This is the property
  # that makes the mode safe to use without knowing the duration -- without it, a caller who
  # guessed wrong would have to re-run the job.
  bash "$0" --run --log "$td/h.log" --interval-s 1 --fg-s 2 --quiet \
       -- bash -c 'sleep 6; exit 3' 2>/dev/null; rc=$?
  _chk "--run reports an overrun as 124 (still running), not as a failure" "$rc" "124"
  _chk "and the job survives the handback" \
       "$(bash "$0" --status --log "$td/h.log" | cut -d' ' -f1)" "running"
  bash "$0" --await --log "$td/h.log" --interval-s 1 --max-s 30 --quiet; rc=$?
  _chk "and its real rc is collected later -- nothing was lost or re-run" "$rc" "3"

  # HOW MUCH LOG COMES BACK IS A CONTEXT BUDGET. Whatever is echoed here lands in the caller's
  # context permanently, so the caller must be able to size it -- and `--tail 0` must mean none,
  # not "fall back to the default", which is the bug an unset-vs-zero test catches.
  bash "$0" --run --log "$td/t.log" --interval-s 1 --tail 3 -- bash -c 'seq 1 40' > "$td/t.out" 2>&1
  _chk "--tail N returns exactly N lines of log" "$(sed -n '/--- tail ---/,$p' "$td/t.out" | tail -n +2 | wc -l | tr -d ' ')" "3"
  bash "$0" --run --log "$td/z.log" --interval-s 1 --tail 0 -- bash -c 'seq 1 40' > "$td/z.out" 2>&1
  _chk "--tail 0 returns NO log at all, only the rc" "$(grep -c 'tail' "$td/z.out")" "0"
  _chk "and --tail 0 still reports the exit code" "$(grep -c '^rc=0' "$td/z.out")" "1"

  # SILENCE IS WHAT GETS REAPED. The harness above this tool kills an agent that emits nothing for
  # 180 s, so a wait that blocks quietly is a timer counting down to a kill -- and it takes the
  # agent's whole context with it, not just the wait. Two properties keep that from happening: the
  # default window ends BEFORE the detector fires, and output continues while the job runs.
  # THE BUDGET IS THE WAIT PLUS THE THINKING AFTER IT. A returning tool call does not reset the
  # detector -- the agent still has to reason about the result before it emits anything, and that
  # reasoning was measured at p90 = 38 s. So the window must leave room for it: 150 s would pass a
  # naive "< 180" check and still be reaped at 150 + 38. Asserted as the SUM, with the measured
  # deliberation as the term, because that is the quantity the detector actually sees.
  _chk "the foreground window plus a p90 deliberation stays inside the 180 s detector" \
       "$( [ $((MAX_S + 38)) -lt 180 ] && echo yes || echo no )" "yes"
  _chk "and the window is still long enough that short work is not pushed to a second call" \
       "$( [ "$MAX_S" -ge 60 ] && echo yes || echo no )" "yes"
  bash "$0" --run --log "$td/hb.log" --interval-s 2 --fg-s 100 -- bash -c 'sleep 95' > "$td/hb.out" 2>&1
  _chk "a long wait emits a heartbeat instead of going silent" \
       "$( [ "$(grep -c 'running' "$td/hb.out")" -ge 1 ] && echo yes || echo no )" "yes"
  # The gap between heartbeats has to be comfortably inside the detector, not merely nonzero.
  _chk "and the heartbeat interval is under the detector" \
       "$( [ "$HEARTBEAT_S" -lt 120 ] && echo yes || echo no )" "yes"
  _chk "--quiet still suppresses it, for a caller parsing the output" \
       "$(bash "$0" --run --log "$td/hq.log" --interval-s 2 --fg-s 100 --quiet -- bash -c 'sleep 50' 2>&1 | grep -c running)" "0"

  # --await's CEILING IS NOT --run's FOREGROUND WINDOW. Sharing one default is why a config sweep
  # could not be awaited without a guess: one measured session passed 560, then 570, and timed out
  # both times with the job still running.
  _chk "an unset --await ceiling is the backgrounded default, not the fg window" \
       "$( [ "$AWAIT_MAX_S" -gt "$MAX_S" ] && echo yes || echo no )" "yes"
  _chk "and it is bounded by the heartbeat rather than the stall detector" \
       "$( [ "$HEARTBEAT_S" -lt 120 ] && echo yes || echo no )" "yes"
  bash "$0" --launch --log "$td/aw.log" -- bash -c 'sleep 30' >/dev/null
  bash "$0" --await --log "$td/aw.log" --interval-s 1 --max-s 2 > "$td/aw.out" 2>&1; rc=$?
  _chk "an explicit --max-s still wins over that default" "$rc" "124"
  # The timeout must hand back the NEXT COMMAND, not leave the caller to invent a number.
  _chk "a timeout prints the non-blocking --status command" \
       "$(grep -c -- '--status --log' "$td/aw.out")" "1"
  _chk "and a DOUBLED --await, so an unknown length is bracketed not guessed" \
       "$(grep -c -- '--max-s 4' "$td/aw.out")" "1"
  _chk "and it says the job is still running rather than failed" \
       "$(grep -c 'STILL RUNNING' "$td/aw.out")" "1"

  _chk "an unlaunched log reports missing" "$(bash "$0" --status --log "$td/nope.log" | cut -d' ' -f1)" "missing"

  echo ""
  if [ "$bad" -eq 0 ]; then
    echo "[wait_for] SELFTEST PASS -- $ok checks: an exit code is observable, a failure is not a"
    echo "           success, a timeout is not a failure, and the wait ends with the job."
    return 0
  fi
  echo "[wait_for] SELFTEST FAIL -- $bad of $((ok+bad)) checks failed."
  return 1
}

case "$MODE" in
  run)      do_run "$@" ;;
  launch)   do_launch "$@" ;;
  await)    do_await ;;
  status)   do_status ;;
  selftest) selftest ;;
  *)        echo "wait_for.sh: need --run, --launch, --await, --status or --selftest (try --help)" >&2; exit 2 ;;
esac
