#!/usr/bin/env bash
# Within-instance spread: N consecutive bench runs against ONE server instance.
#
# This is deliberately NOT the same quantity as ab_campaign.sh measures. That rig restarts
# the server for every single measurement, so every number it produces is a first-run
# against a cold instance. This one holds the instance fixed and varies only the run index,
# which is the only way to see the settle curve -- whether throughput climbs, falls or
# holds over the first few runs against a given server.
#
# The two spreads have to be quoted separately, and an A/B is only honest if both arms sit
# at the same point on the settle curve. ab_campaign.sh satisfies that by always taking
# run #1.
#
#   RUNS=5 analysis/spread_within.sh
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-5}"
PREFIX="${PREFIX:-spread_within}"

./scripts/launch_server.sh --stop >/dev/null 2>&1
sleep 5
echo "[spread] starting one server instance, then $RUNS consecutive runs against it"
if ! ./scripts/launch_server.sh > "/tmp/${PREFIX}_launch.log" 2>&1; then
    echo "[spread] launch FAILED -- see /tmp/${PREFIX}_launch.log" >&2
    tail -20 "/tmp/${PREFIX}_launch.log" >&2
    exit 1
fi

for r in $(seq 1 "$RUNS"); do
    echo "[spread] run $r/$RUNS  ($(date +%H:%M:%S))"
    TAG="${PREFIX}_r${r}" ./scripts/run_bench.sh > "/tmp/${PREFIX}_bench_${r}.log" 2>&1 \
        || echo "[spread] run $r FAILED"
done

./scripts/launch_server.sh --stop >/dev/null 2>&1
echo "[spread] done"
