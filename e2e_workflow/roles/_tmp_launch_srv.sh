#!/usr/bin/env bash
# Manual keep-warm launcher that reuses the sglang adapter's launch semantics,
# so ref & cand legs launch identically to bench_e2e.sh. Args via env:
#   MODEL PORT GPU TP MEM_FRACTION EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH LOG
set -u
HERE="/wekafs/GEAK/e2e_workflow/scripts"
export BACKEND=sglang
HOST=127.0.0.1
BASE_URL="http://${HOST}:${PORT}"
PROFILE_DIR="/tmp/geak_prof_${PORT}"; mkdir -p "$PROFILE_DIR"
PROFILE=0; PROFILE_NUM_STEPS=1; SEED=0; ISL=1024; OSL=1024; CONC=64
export MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH
export PROFILE_DIR PROFILE PROFILE_NUM_STEPS SEED ISL OSL CONC BASE_URL LOG
# shellcheck disable=SC1090
. "$HERE/adapters/sglang.sh"
adapter_launch
echo "SERVER_PID=$SERVER_PID"
echo "$SERVER_PID" > "${LOG}.pid"
# wait health up to ~10min; fail fast on fatal markers
for i in $(seq 1 120); do
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then echo "HEALTHY after ~$((i*5))s"; exit 0; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "SERVER_DIED"; tail -n 40 "$LOG"; exit 2; fi
  if grep -Eq 'CUDA out of memory|HIP out of memory|watchdog timeout|Capturing cuda graph failed|PassManager::run failed|FATAL' "$LOG" 2>/dev/null; then
    echo "FATAL_MARKER"; tail -n 40 "$LOG"; exit 3; fi
  sleep 5
done
echo "HEALTH_TIMEOUT"; tail -n 40 "$LOG"; exit 4
