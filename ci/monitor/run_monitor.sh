#!/usr/bin/env bash
# Host-side liveness monitor (watchdog) for one L1 GEAK e2e run.
#
# WHY: a wedged GPU / NFS stall / OOM loop can hang the run for hours until the
# job's wall-clock timeout, wasting an exclusive GPU node. This process watches
# the run from OUTSIDE the container and, on a confirmed WEDGE, stops the
# container so the CI step goes red fast instead of limping to the wall clock
# (or to a false-green no_gain).
#
# TWO MODES (GEAK_MONITOR_MODE):
#   * stall  — DETERMINISTIC, no deps. Declares a wedge ONLY on positive evidence
#              of no work: the log is flat AND the GPUs are idle AND the container
#              CPU is idle, sustained for GEAK_STALL_KILL_S and confirmed CONFIRM
#              times. A long silent-but-working leg (bench/build/profile) keeps GPU
#              or CPU busy, so it is NEVER killed. If GPU utilisation cannot be
#              measured (no rocm-smi/amd-smi) it CANNOT prove "idle" and so
#              degrades to warn-only — it will never kill on a guess.
#   * claude — LLM arbiter. Every INTERVAL feeds the log tail + factual context to
#              a tool-less `claude -p` session that votes CONTINUE/KILL.
#
# SAFETY: runs ON THE HOST only; never enters the container, never touches the
# GPU (read-only rocm-smi/docker-stats sampling). Bias is strongly toward
# CONTINUE — a false kill throws away hours of good work and the wall clock is
# the ultimate backstop.
#
# Usage: run_monitor.sh <container_name> <run_log_path> <out_dir> [docker_pid]
set -uo pipefail

CONTAINER="${1:?container name}"
LOG="${2:?run.log path}"
OUT_DIR="${3:?out dir}"
DOCKER_PID="${4:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Tunables live in ci/config.sh; source it so the monitor is self-sufficient
# (it's normally launched by run_local.sh, which already exported these).
# shellcheck source=/dev/null
[ -f "$HERE/../config.sh" ] && source "$HERE/../config.sh"

MODE="${GEAK_MONITOR_MODE:-stall}"
INTERVAL="$GEAK_MONITOR_INTERVAL_S"        # normal poll cadence
RECHECK_S="$GEAK_MONITOR_RECHECK_S"        # faster re-poll while confirming a KILL
CONFIRM="$GEAK_MONITOR_CONFIRM"            # consecutive KILL votes required to act
MODEL="$GEAK_MONITOR_MODEL"
TAIL_LINES="$GEAK_MONITOR_TAIL_LINES"
CALL_CAP="$GEAK_MONITOR_CALL_TIMEOUT_S"    # cap a single claude call
STARTUP_GRACE_S="$GEAK_MONITOR_STARTUP_GRACE_S"
STALL_KILL_S="${GEAK_STALL_KILL_S:-2700}"
STALL_GPU_PCT="${GEAK_STALL_GPU_UTIL_PCT:-5}"
STALL_CPU_PCT="${GEAK_STALL_CPU_PCT:-5}"
PROMPT_FILE="$HERE/monitor_prompt.md"
VERDICT="$OUT_DIR/monitor_verdict.json"
MON_LOG="$OUT_DIR/monitor.log"

mkdir -p "$OUT_DIR" 2>/dev/null || true
log() { printf '[monitor %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$MON_LOG" >&2; }

json_str() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()))'; }
container_running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }

# ---- activity samplers (stall mode) ----------------------------------------
# Max GPU utilisation across devices, as an integer percent. Echoes "" (unknown)
# when no GPU tool is available — the caller treats unknown as "cannot prove idle".
gpu_util_max() {
  local out max=""
  if command -v rocm-smi >/dev/null 2>&1; then
    # "GPU use (%)" rows; grab all integers in the use column.
    out="$(rocm-smi --showuse 2>/dev/null | grep -oiE 'GPU use \(%\)[^0-9]*[0-9]+' | grep -oE '[0-9]+$')"
  elif command -v amd-smi >/dev/null 2>&1; then
    out="$(amd-smi monitor -u 2>/dev/null | grep -oE '[0-9]+' )"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    out="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | grep -oE '[0-9]+')"
  else
    echo ""; return 0
  fi
  [ -n "$out" ] || { echo ""; return 0; }
  max="$(printf '%s\n' "$out" | sort -n | tail -1)"
  echo "${max:-}"
}

# Container CPU percent (integer). Echoes "" (unknown) if docker stats fails.
container_cpu_pct() {
  command -v docker >/dev/null 2>&1 || { echo ""; return 0; }
  local p
  p="$(docker stats --no-stream --format '{{.CPUPerc}}' "$CONTAINER" 2>/dev/null | tr -d '% ' )"
  [ -n "$p" ] || { echo ""; return 0; }
  printf '%.0f\n' "$p" 2>/dev/null || echo ""
}

# ---- mode preflight --------------------------------------------------------
if [ "$MODE" = claude ]; then
  CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
  [ -z "$CLAUDE_BIN" ] && [ -x "$HOME/.local/bin/claude" ] && CLAUDE_BIN="$HOME/.local/bin/claude"
  if [ -z "$CLAUDE_BIN" ]; then
    log "MODE=claude but no 'claude' on host; monitor disabled (run continues, wall-clock is the backstop)"
    exit 0
  fi
  [ -f "$PROMPT_FILE" ] || { log "missing prompt $PROMPT_FILE; monitor disabled"; exit 0; }
elif [ "$MODE" != stall ]; then
  log "unknown GEAK_MONITOR_MODE='$MODE' (want stall|claude); monitor disabled"
  exit 0
fi

# ---- per-mode decision: sets globals `verdict` (CONTINUE|KILL) and `reason` --
decide_claude() {  # args: delta age_s
  local delta="$1" age_s="$2" now_utc tail_txt resp prompt
  now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
    verdict=""; reason="claude call failed/timed out"; return 0; }
  verdict="$(printf '%s' "$resp" | grep -oiE 'VERDICT:[[:space:]]*(CONTINUE|KILL)' | tail -n1 | grep -oiE '(CONTINUE|KILL)' | tr '[:lower:]' '[:upper:]')"
  reason="$(printf '%s' "$resp" | grep -iE 'REASON:' | tail -n1 | sed -E 's/.*REASON:[[:space:]]*//')"
}

decide_stall() {  # args: delta age_s ; uses stall_start_epoch (global)
  local delta="$1" age_s="$2" now gpu cpu stall_s
  now="$(date +%s)"
  # Progress? any log growth resets the stall clock.
  if [ "$delta" -gt 0 ]; then stall_start_epoch="$now"; fi
  stall_s=$(( now - stall_start_epoch ))
  verdict="CONTINUE"; reason="log active (Δ=${delta}B, idle ${stall_s}s < ${STALL_KILL_S}s)"
  [ "$stall_s" -ge "$STALL_KILL_S" ] || return 0

  # Log has been flat long enough — now REQUIRE positive idle evidence before killing.
  gpu="$(gpu_util_max)"; cpu="$(container_cpu_pct)"
  if [ -z "$gpu" ]; then
    verdict="CONTINUE"
    reason="log flat ${stall_s}s but GPU util unmeasurable (no rocm-smi/amd-smi) -> cannot prove wedge; warn-only"
    log "WARN: $reason"
    return 0
  fi
  local gpu_idle=0 cpu_idle=0
  [ "$gpu" -le "$STALL_GPU_PCT" ] && gpu_idle=1
  { [ -n "$cpu" ] && [ "$cpu" -le "$STALL_CPU_PCT" ]; } && cpu_idle=1
  if [ "$gpu_idle" = 1 ] && [ "$cpu_idle" = 1 ]; then
    verdict="KILL"
    reason="wedge: log flat ${stall_s}s, GPU ${gpu}%<=${STALL_GPU_PCT}% idle, CPU ${cpu:-?}%<=${STALL_CPU_PCT}% idle"
  else
    verdict="CONTINUE"
    reason="log flat ${stall_s}s but still working (GPU ${gpu}%, CPU ${cpu:-unknown}%)"
  fi
}

log "started: mode=$MODE container=$CONTAINER interval=${INTERVAL}s confirm=${CONFIRM} log=$LOG${MODEL:+ model=$MODEL}"
[ "$MODE" = stall ] && log "stall thresholds: kill after ${STALL_KILL_S}s flat AND GPU<=${STALL_GPU_PCT}% AND CPU<=${STALL_CPU_PCT}%"

# Wait for the container to come up before treating "not running" as "finished",
# so we don't exit during the (healthcheck + image pull) startup window.
seen_running=0
start_epoch=$(date +%s)
stall_start_epoch=$(date +%s)
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

  verdict=""; reason=""
  if [ "$MODE" = claude ]; then decide_claude "$delta" "$age_s"; else decide_stall "$delta" "$age_s"; fi
  if [ -z "$verdict" ]; then
    log "no verdict (${reason:-unknown}); retry next interval"
    sleep "$INTERVAL"; continue
  fi
  log "verdict=$verdict reason=${reason:-<none>}"

  if [ "$verdict" = "KILL" ]; then
    kill_streak=$(( kill_streak + 1 ))
    if [ "$kill_streak" -lt "$CONFIRM" ]; then
      log "KILL vote ${kill_streak}/${CONFIRM}; re-checking in ${RECHECK_S}s before acting"
      sleep "$RECHECK_S"; continue
    fi
    log "KILL confirmed (${kill_streak}/${CONFIRM}) -> stopping container $CONTAINER"
    printf '{"action":"kill","mode":%s,"container":%s,"reason":%s,"kill_streak":%d,"ts":%s}\n' \
      "$(printf '%s' "$MODE" | json_str)" \
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
