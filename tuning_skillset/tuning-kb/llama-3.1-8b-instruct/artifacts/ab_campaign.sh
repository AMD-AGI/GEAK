#!/usr/bin/env bash
# Interleaved restart-to-restart A/B for the tuned-GEMM rows.
#
# The local noise floor is 1.49% (spread of 4 baseline runs) and the predicted effect is
# ~0.56%, so a 1v1 comparison cannot resolve it. This runs N paired rounds and flips the
# arm order every round (base,cand / cand,base / ...) so the gfx950 clock ramp and any
# other monotonic drift cancel instead of landing entirely on whichever arm goes first.
#
# Each arm is a full restart: the GEMM table is consumed at HIP-graph capture time, so the
# config MUST be swapped with the server down and /tmp/aiter_configs removed (the merged
# table is regenerated only when absent -- the deploy script does that).
#
# Exactly ONE bench run per server start, always run #1. That is not incidental: within one
# server instance throughput falls monotonically (analysis/spread_within.sh measured 1.20%
# over 5 consecutive runs, still falling), while run #1 across restarts is stable to 0.09%.
# Taking run #1
# every time is what makes the two arms comparable.
#
#   DEPLOY=analysis/deploy_down_flydsl.py ROUNDS=6 PREFIX=abd analysis/ab_campaign.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROUNDS="${ROUNDS:-6}"
DEPLOY="${DEPLOY:-analysis/deploy_gemm_rows.py}"
PREFIX="${PREFIX:-ab}"

run_arm() {   # $1 = base|cand   $2 = round
    local arm="$1" r="$2"
    echo "=== [$(date +%H:%M:%S)] round $r arm=$arm ==="
    ./scripts/launch_server.sh --stop >/dev/null 2>&1
    python3 "$DEPLOY" "$([ "$arm" = cand ] && echo apply || echo revert)" >/dev/null
    if ! ./scripts/launch_server.sh > "/tmp/${PREFIX}_launch_${arm}_${r}.log" 2>&1; then
        echo "  LAUNCH FAILED -- see /tmp/${PREFIX}_launch_${arm}_${r}.log"; tail -5 "/tmp/${PREFIX}_launch_${arm}_${r}.log"; return 1
    fi
    TAG="${PREFIX}_${arm}_r${r}" ./scripts/run_bench.sh > "/tmp/${PREFIX}_bench_${arm}_${r}.log" 2>&1
    # round 2 addition: keep the scheduler log per arm. launch_server.sh writes to a fixed path
    # and truncates it on the next start, so a per-arm copy is the only way to check afterwards
    # that a patch was actually engaged during the *benchmark* (not merely under a probe load),
    # and to read the per-prefill-batch `input throughput` the scheduler prints.
    cp /tmp/sglang_server_llama_3_1_8b_instruct.log "/tmp/${PREFIX}_srv_${arm}_${r}.log" 2>/dev/null
    local d; d=$(ls -td results/${PREFIX}_${arm}_r${r}_* 2>/dev/null | head -1)
    local t; t=$(python3 -c "import json,sys; print(f\"{json.load(open('$d/inferencex_result.json'))['output_throughput']:.3f}\")" 2>/dev/null)
    echo "  $arm r$r -> ${t:-FAILED} tok/s   ($d)"
    ./scripts/launch_server.sh --stop >/dev/null 2>&1
}

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then run_arm base "$r"; run_arm cand "$r"
    else                          run_arm cand "$r"; run_arm base "$r"; fi
done
echo "=== [$(date +%H:%M:%S)] campaign done ==="
