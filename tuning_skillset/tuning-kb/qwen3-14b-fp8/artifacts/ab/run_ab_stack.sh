#!/usr/bin/env bash
# Interleaved across-restart A/B driver.
#
# tuning-core/measurement.md Rule 6b: on gfx950 a back-to-back sweep measures
# clock drift as if it were a code difference, so the arms must be interleaved.
# Rule 3b: a change that needs a server restart is bounded below by the
# restart-to-restart spread, so each arm has to be re-measured across several
# fresh instances rather than several runs of one instance.
#
# Layout: A B A B, each instance a fresh server, RUNS benchmark runs per
# instance. Comparisons are made position-by-position (run 1 vs run 1, ...),
# because within-instance throughput drifts monotonically downward over the
# first runs and an unmatched comparison folds that drift into the delta.
#
# Usage: run_ab.sh <fileA> <fileB> <target-path> <label> [rounds] [runs]
set -euo pipefail

FILE_A=$1; FILE_B=$2; TARGET=unused; LABEL=$3
ROUNDS=${4:-2}; RUNS=${5:-3}
ROOT=/home/ethany/tuning_workspace/experiment_standalone/qwen3_14b_fp8_tuning
cd "$ROOT"

for r in $(seq 1 "$ROUNDS"); do
  for arm in A B; do
    src=$FILE_A; [ "$arm" = B ] && src=$FILE_B
    echo "=== round $r arm $arm: running $src"
    "$src"
    ./scripts/launch_server.sh --stop >/dev/null 2>&1 || true
    sleep 10
    ./scripts/launch_server.sh >/dev/null 2>&1
    for i in $(seq 1 "$RUNS"); do
      TAG="${LABEL}_${arm}${r}_r${i}" ./scripts/run_bench.sh >/dev/null 2>&1
      echo "    round $r arm $arm run $i done"
    done
  done
done
echo "=== A/B complete"
