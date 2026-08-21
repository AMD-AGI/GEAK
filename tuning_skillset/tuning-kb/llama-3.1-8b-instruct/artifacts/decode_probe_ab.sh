#!/usr/bin/env bash
# Restart-paired A/B of the *decode phase only*, off-contract. Lead 1 from the brief: the Triton
# unified/gluon decode attention against the aiter MFMA paged kernel.
#
# The switch is a source-level boolean (aiter_backend.py:247-250) that happens to read
# SGLANG_USE_AITER_UNIFIED_ATTN, so an env var is enough to *measure* it; if it wins it ships as a
# source default, not as an env var. Enabling it changes only the decode branch for this model --
# forward_extend's unified_attention call is gated on is_target_verify() (EAGLE), and the
# unified metadata layout is built only inside the is_decode_or_idle() branch -- so it composes
# with the shipped prefill patch, which stays applied in both arms.
#
# Instrument is analysis/decode_stats.py on the scheduler's own per-batch `gen throughput` line,
# for the same reason prefill_probe_ab.sh reads `input throughput`: decode is 71% of the wall, so
# whole-run throughput would dilute a decode-only delta and mix in prefill noise.
#
#   ROUNDS=3 analysis/decode_probe_ab.sh
set -u
cd "$(dirname "$0")/.."
ROUNDS="${ROUNDS:-3}"
OUT="${OUT:-/tmp/decode_probe_ab}"
SRVLOG=/tmp/sglang_server_llama_3_1_8b_instruct.log
mkdir -p "$OUT"

run_arm() {
    local arm="$1" r="$2"
    echo "=== [$(date +%T)] round $r arm=$arm ==="
    ./scripts/launch_server.sh --stop >/dev/null 2>&1
    if [ "$arm" = triton ]; then
        export SGLANG_USE_AITER_UNIFIED_ATTN=1
    else
        unset SGLANG_USE_AITER_UNIFIED_ATTN
    fi
    if ! ./scripts/launch_server.sh >"$OUT/launch_${arm}_${r}.log" 2>&1; then
        echo "  LAUNCH FAILED (see $OUT/launch_${arm}_${r}.log)"; return 1
    fi
    python3 analysis/probe_load.py --n 192 --conc 64 --osl 1024 >"$OUT/load_${arm}_${r}.json"
    cp "$SRVLOG" "$OUT/server_${arm}_${r}.log"
    ./scripts/launch_server.sh --stop >/dev/null 2>&1
    python3 analysis/decode_stats.py "$OUT/server_${arm}_${r}.log" "$arm" "$r"
    python3 -c "
import json,sys
d=json.load(open('$OUT/load_${arm}_${r}.json'))
print('    probe wall %.2f s' % d['wall_s'])" 2>/dev/null || true
}

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then a=paged b=triton; else a=triton b=paged; fi
    run_arm "$a" "$r"
    run_arm "$b" "$r"
done
echo "=== [$(date +%T)] decode probe done ==="
