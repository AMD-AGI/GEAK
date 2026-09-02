#!/usr/bin/env bash
# Run one warm-reuse measurement replica through bench_e2e.sh's single-server
# implementation.  Each invocation owns a fresh server, runs one discarded
# full workload warmup, measures once against that same server, then lets the
# parent script's teardown trap reclaim it.  It never attaches to a server
# owned by another replica or another E2E arm.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_E2E="${BENCH_E2E:-$HERE/bench_e2e.sh}"

if [ ! -f "$BENCH_E2E" ]; then
  echo "!!! bench_warm_replica.sh cannot find bench_e2e.sh at $BENCH_E2E" >&2
  exit 3
fi
if [ "${REUSE_SERVER:-0}" = "1" ]; then
  echo "!!! A warm-reuse replica must launch its own server; REUSE_SERVER=1 is unsafe." >&2
  exit 4
fi
if [ "${PROFILE:-0}" = "1" ]; then
  echo "!!! Profiling is not supported inside a timed warm-reuse replica." >&2
  exit 4
fi

# Re-enter the legacy single-server body rather than recursively scheduling
# more replicas.  REUSE_SERVER remains 0: the body launches once and leaves
# that GEAK-owned server running across its warmup and timed rounds.
export GEAK_REPEAT_MODE=legacy
unset GEAK_ISOLATED_REPLICA
export GEAK_WARM_REUSE_REPLICA=1
export REPEATS=1
export REUSE_SERVER=0
export PROFILE=0
export BENCH_COLD_FINAL=0
export BENCH_OUTER_WARMUP_FULL_ROUND=1
export BENCH_REQUIRE_SUCCESS=1

# Retain the established InferenceX compute-warm contract for both the
# discarded full round and the measured round.
_conc="${CONC:-64}"
case "$_conc" in
  ''|*[!0-9]*|0)
    echo "!!! CONC must be a positive integer for a warm-reuse replica (got '$_conc')." >&2
    exit 4 ;;
esac
export NUM_WARMUPS=$((2 * _conc))
export SEED=0

exec bash "$BENCH_E2E"
