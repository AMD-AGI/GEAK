#!/usr/bin/env bash
# final_launch.sh — reproduce the OPTIMIZED Qwen-Qwen3.5-27B-FP8 sglang server + e2e throughput bench.
#
# Accepted optimizations (this run, MI300X/gfx942, sglang 0.5.11, TP=1 single GPU, ISL/OSL/conc=1024/1024/64):
#   1) --attention-backend triton           (config, parity ok)
#   2) --kv-cache-dtype fp8_e4m3             (config, LOSSY KV — parity-gated; coherent in probes)
#   3) FlyDSL fp8 a8w8 blockscale down-proj GEMM overlay  (authored; gate ACCEPTED; +14.17% matched A/B)
#
# Measured (same-session, single GPU):
#   true baseline (no config, no flydsl):                 ~988-993 tok/s
#   FlyDSL isolated (matched A/B, ref 1170 -> cand 1336): +14.17%   (the rigorous, locked number)
#   full stack vs baseline (988.03 -> 1352.97):           +36.94%   (2-rep; base noisier — see report caveats)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$HERE/.." && pwd)"

export BACKEND=sglang
export MODEL="${MODEL:-/wekafs/models/Qwen-Qwen3.5-27B-FP8/}"
export TP=1
export GPU="${GPU:-0}"
export MEM_FRACTION="${MEM_FRACTION:-0.78}"     # FlyDSL needs headroom for the preshuffled-weight cache
export ISL="${ISL:-1024}" OSL="${OSL:-1024}" CONC="${CONC:-64}"
export REPEATS="${REPEATS:-2}"

# THE ACCEPTED STACK
export EXTRA_SERVER_ARGS="--attention-backend triton --kv-cache-dtype fp8_e4m3 --watchdog-timeout 900"
export EXTRA_ENV=""
export OVERLAY_PYTHONPATH="$EVAL_DIR/overlay/cand_flydsl_blockscale_gemm"   # FlyDSL down-proj GEMM (reversible)
export OUT_DIR="${OUT_DIR:-$HERE/final_bench}"

echo "=== final_launch.sh: optimized Qwen-Qwen3.5-27B-FP8 (triton + fp8-kv + FlyDSL) ==="
echo "    flags: $EXTRA_SERVER_ARGS   overlay: $OVERLAY_PYTHONPATH"
echo "    TP=$TP GPU=$GPU mem=$MEM_FRACTION ISL/OSL/conc=$ISL/$OSL/$CONC repeats=$REPEATS"
bash "$EVAL_DIR/bench_e2e.sh"
