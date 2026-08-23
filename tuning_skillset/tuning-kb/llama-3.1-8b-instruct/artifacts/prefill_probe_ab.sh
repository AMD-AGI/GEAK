#!/usr/bin/env bash
# Restart-paired A/B of the *prefill phase only*, off-contract.
#
# Why this exists: the end-to-end benchmark moved 0.0% for a prefill-attention swap that is
# 34.7% faster in isolation, so the question is which link in the chain is wrong. SGLang's own
# scheduler logs `input throughput (token/s)` for every prefill batch, which is a direct,
# n=93-per-run measurement of exactly the phase the patch touches -- far more sensitive than a
# whole-run throughput number in which prefill is only ~29% of the wall clock.
#
# Same discipline as analysis/ab_campaign.sh: one measurement per freshly started server, arm
# order flipped every round. Uses analysis/probe_load.py (workload *shape*, not the sealed
# benchmark) so no result JSON is produced and nothing here can be mistaken for a contract run.
#
#   ROUNDS=3 analysis/prefill_probe_ab.sh
set -u
cd "$(dirname "$0")/.."
ROUNDS="${ROUNDS:-3}"
OUT="${OUT:-/tmp/prefill_probe_ab}"
SRVLOG=/tmp/sglang_server_llama_3_1_8b_instruct.log
mkdir -p "$OUT"

run_arm() {
    local arm="$1" r="$2"
    echo "=== [$(date +%T)] round $r arm=$arm ==="
    ./scripts/launch_server.sh --stop >/dev/null 2>&1
    python3 analysis/deploy_ragged_prefill.py "$([ "$arm" = cand ] && echo apply || echo revert)"
    if ! ./scripts/launch_server.sh >"$OUT/launch_${arm}_${r}.log" 2>&1; then
        echo "  LAUNCH FAILED"; return 1
    fi
    python3 analysis/probe_load.py --n 192 --conc 64 --osl 1024 >"$OUT/load_${arm}_${r}.json"
    cp "$SRVLOG" "$OUT/server_${arm}_${r}.log"
    ./scripts/launch_server.sh --stop >/dev/null 2>&1
    python3 analysis/prefill_stats.py "$OUT/server_${arm}_${r}.log" "$arm" "$r"
}

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then a=base b=cand; else a=cand b=base; fi
    run_arm "$a" "$r"
    run_arm "$b" "$r"
done
