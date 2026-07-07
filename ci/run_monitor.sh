#!/usr/bin/env bash
# Host-side liveness monitor (watchdog) for one L1 GEAK e2e run.
#
# WHY: a wedged GPU / NFS stall / OOM loop can hang the run for hours until the
# job's wall-clock timeout, wasting the runner. This process watches the run's
# log from OUTSIDE the container and, every INTERVAL, asks a Claude session
# (host login, no tools) to judge CONTINUE vs KILL. On a confirmed KILL it stops
# the container so the CI step returns non-zero (red) fast instead of limping to
# a false-green no_gain.
#
# SAFETY: runs ON THE HOST only. It never enters the GPU container and never
# touches the GPU. Claude gets NO tools — just the log tail + factual context in,
# a two-line verdict out — so it cannot run anything. If `claude` is missing the
# monitor no-ops (the run proceeds unwatched, protected by the wall-clock).
#
# Usage: run_monitor.sh <container_name> <run_log_path> <out_dir> [docker_pid]
set -uo pipefail

CONTAINER="${1:?container name}"
LOG="${2:?run.log path}"
OUT_DIR="${3:?out dir}"
DOCKER_PID="${4:-}"

INTERVAL="${GEAK_MONITOR_INTERVAL_S:-300}"        # normal poll cadence (5 min)
RECHECK_S="${GEAK_MONITOR_RECHECK_S:-60}"         # faster re-poll while confirming a KILL
CONFIRM="${GEAK_MONITOR_CONFIRM:-2}"              # consecutive KILL votes required to act
MODEL="${GEAK_MONITOR_MODEL:-claude-opus-4-8}"
TAIL_LINES="${GEAK_MONITOR_TAIL_LINES:-300}"
CALL_CAP="${GEAK_MONITOR_CALL_TIMEOUT_S:-180}"    # cap a single claude call
STARTUP_GRACE_S="${GEAK_MONITOR_STARTUP_GRACE_S:-300}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$HERE/monitor_prompt.md"
VERDICT="$OUT_DIR/monitor_verdict.json"
MON_LOG="$OUT_DIR/monitor.log"

mkdir -p "$OUT_DIR" 2>/dev/null || true
log() { printf '[monitor %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$MON_LOG" >&2; }

# Resolve the host claude binary (default machine login). No-op if unavailable.
CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
[ -z "$CLAUDE_BIN" ] && [ -x "$HOME/.local/bin/claude" ] && CLAUDE_BIN="$HOME/.local/bin/claude"
if [ -z "$CLAUDE_BIN" ]; then
  log "no 'claude' on host; monitor disabled (run continues, wall-clock is the backstop)"
  exit 0
fi
[ -f "$PROMPT_FILE" ] || { log "missing prompt $PROMPT_FILE; monitor disabled"; exit 0; }

json_str() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()))'; }
container_running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }

log "started: container=$CONTAINER model=$MODEL interval=${INTERVAL}s confirm=${CONFIRM} log=$LOG"

# Wait for the container to come up before treating "not running" as "finished",
# so we don't exit during the (healthcheck + image pull) startup window.
seen_running=0
start_epoch=$(date +%s)
kill_streak=0
last_size=0

while true; do
  if container_running; then
    seen_running=1
  else
    if [ "$seen_running" = "1" ]; then
      log "container $CONTAINER no longer running; run finished — monitor exiting"
      exit 0
    fi
    if [ $(( $(date +%s) - start_epoch )) -ge "$STARTUP_GRACE_S" ]; then
      log "container never came up within ${STARTUP_GRACE_S}s; monitor exiting"
      exit 0
    fi
    sleep 10; continue
  fi

  # Factual context: how much has the log grown since last check? (stall signal)
  now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ -f "$LOG" ]; then
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    age_s=$(( $(date +%s) - mtime ))
  else
    size=0; age_s=-1
  fi
  delta=$(( size - last_size ))
  last_size="$size"
  tail_txt="$(tail -n "$TAIL_LINES" "$LOG" 2>/dev/null || echo '(no log yet)')"

  prompt="$(cat "$PROMPT_FILE")

=== CONTEXT (facts computed on the host) ===
now_utc: $now_utc
check_interval_s: $INTERVAL
log_bytes_added_since_last_check: $delta
log_last_modified_age_s: $age_s
current_kill_streak: $kill_streak (need $CONFIRM consecutive KILL votes to act)
=== RUN LOG (last $TAIL_LINES lines of $LOG) ===
$tail_txt
=== END LOG ==="

  resp="$(timeout "$CALL_CAP" "$CLAUDE_BIN" -p "$prompt" --model "$MODEL" </dev/null 2>>"$MON_LOG")" || {
    log "claude call failed/timed out (cap ${CALL_CAP}s); retry next interval"
    sleep "$INTERVAL"; continue
  }

  verdict="$(printf '%s' "$resp" | grep -oiE 'VERDICT:[[:space:]]*(CONTINUE|KILL)' | tail -n1 | grep -oiE '(CONTINUE|KILL)' | tr '[:lower:]' '[:upper:]')"
  reason="$(printf '%s' "$resp" | grep -iE 'REASON:' | tail -n1 | sed -E 's/.*REASON:[[:space:]]*//')"
  log "verdict=${verdict:-<none>} reason=${reason:-<none>}"

  if [ "$verdict" = "KILL" ]; then
    kill_streak=$(( kill_streak + 1 ))
    if [ "$kill_streak" -lt "$CONFIRM" ]; then
      log "KILL vote ${kill_streak}/${CONFIRM}; re-checking in ${RECHECK_S}s before acting"
      sleep "$RECHECK_S"; continue
    fi
    log "KILL confirmed (${kill_streak}/${CONFIRM}) -> stopping container $CONTAINER"
    printf '{"action":"kill","container":%s,"reason":%s,"kill_streak":%d,"ts":%s}\n' \
      "$(printf '%s' "$CONTAINER" | json_str)" \
      "$(printf '%s' "${reason:-unspecified}" | json_str)" \
      "$kill_streak" \
      "$(printf '%s' "$now_utc" | json_str)" > "$VERDICT"
    docker kill "$CONTAINER" 2>>"$MON_LOG" || true
    [ -n "$DOCKER_PID" ] && kill "$DOCKER_PID" 2>/dev/null || true
    exit 0
  fi

  # CONTINUE (or unparseable — treat conservatively as CONTINUE): reset streak.
  [ "$kill_streak" -ne 0 ] && log "progress/continue -> resetting kill streak"
  kill_streak=0
  sleep "$INTERVAL"
done
