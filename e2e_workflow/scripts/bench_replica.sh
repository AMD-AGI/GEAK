#!/usr/bin/env bash
# Run one isolated-server measurement replica through bench_e2e.sh's existing
# single-server implementation.  The parent bench_e2e.sh scheduler may invoke
# this process twice for one replica (one retry), but every invocation is a
# complete fresh launch -> full warmup -> full measurement -> teardown lifecycle.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_E2E="${BENCH_E2E:-$HERE/bench_e2e.sh}"

if [ ! -f "$BENCH_E2E" ]; then
  echo "!!! bench_replica.sh cannot find bench_e2e.sh at $BENCH_E2E" >&2
  exit 3
fi
if [ "${REUSE_SERVER:-0}" = "1" ]; then
  echo "!!! An isolated replica cannot reuse a server." >&2
  exit 4
fi
if [ "${PROFILE:-0}" = "1" ]; then
  echo "!!! Profiling is not supported inside an isolated timed replica." >&2
  exit 4
fi

# Re-enter the legacy body rather than recursively scheduling more replicas.
export GEAK_REPEAT_MODE=legacy
export GEAK_ISOLATED_REPLICA=1
export REPEATS=1
export REUSE_SERVER=0
export PROFILE=0
export BENCH_COLD_FINAL=0
export BENCH_OUTER_WARMUP_FULL_ROUND=1
export BENCH_REQUIRE_SUCCESS=1

# InferenceX performs an internal client warmup inside each adapter invocation.
# The isolated protocol has exactly two invocations (outer warmup + measure), and
# both must use 2*CONC internal warmups with deterministic seed zero.
if [ "${BENCH_CLIENT:-native}" = "inferencex" ]; then
  _conc="${CONC:-64}"
  case "$_conc" in
    ''|*[!0-9]*|0)
      echo "!!! CONC must be a positive integer for an InferenceX replica (got '$_conc')." >&2
      exit 4 ;;
  esac
  export NUM_WARMUPS=$((2 * _conc))
  export SEED=0
fi

exec bash "$BENCH_E2E"
