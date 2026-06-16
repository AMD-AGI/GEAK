#!/usr/bin/env bash
# ============================================================================
# final_launch.sh — reproduce the optimized Qwen3.5-27B-FP8 sglang server
#                    and benchmark its e2e serving throughput.
#
# Spec deliverable: complete patch + launch/benchmark script.
#
# Optimization bundle (this run):
#   * Config flag : --attention-backend triton   (accepted Tier-0 flag)
#   * Overlay     : authored FlyDSL fused fp8 a8w8 blockscale GEMM core, rebound
#                   over the down-proj seam (N=5120, K=17408) via a lazy
#                   meta-path sitecustomize seam (no site-packages edit).
#
# Measured (tight 2-launch A/B, GPU0, TP=1, mem-fraction 0.85,
#           ISL/OSL/conc 1024/1024/64):
#   true baseline        : 931.593 tok/s
#   accepted (ref leg)   : 953.139 tok/s  (--attention-backend triton)
#   optimized (cand leg) : 1525.857 tok/s (+ FlyDSL down-proj)  => +63.8% vs base
#
# SERVING INVARIANT: TP=1 on a SINGLE GPU. Do NOT set TP>1 or multi-GPU here.
# ============================================================================
set -uo pipefail

# --- locate the bundle / shared harness ----------------------------------
EVAL_DIR="${EVAL_DIR:-/wekafs/zihao/2026/geak_cc/PerfSkills/exp/e2e_Qwen-Qwen3.5-27B-FP8_20260613_195618_371691_18852}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-$EVAL_DIR/bench_e2e.sh}"        # backend-agnostic dispatcher + adapters/

# --- the optimized config (overlay + flags + env) ------------------------
export BACKEND=sglang                           # selects adapters/sglang.sh
OVERLAY="${OVERLAY:-$HERE/overlay}"             # FlyDSL core + sitecustomize seam
ACCEPTED_FLAGS="--attention-backend triton"     # accepted Tier-0 config flag
ACCEPTED_ENV=""                                 # no extra env this run

# --- model / serving knobs (serving invariant: TP=1, single GPU) ---------
MODEL_PATH="${MODEL:-/wekafs/models/Qwen-Qwen3.5-27B-FP8/}"
GPU="${GPU:-0}"
TP=1
MEM_FRACTION="${MEM_FRACTION:-0.85}"

# --- workload ------------------------------------------------------------
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"; CONC="${CONC:-64}"
REPEATS="${REPEATS:-3}"
OUT_DIR="${OUT_DIR:-$HERE/final_bench}"
# pin a port in the allowed range (sglang grpc_port = port+10000 must be <= 65535)
PORT="${PORT:-31123}"

echo ">>> Optimized launch+bench"
echo "    overlay : $OVERLAY"
echo "    flags   : $ACCEPTED_FLAGS"
echo "    env     : ${ACCEPTED_ENV:-<none>}"
echo "    model   : $MODEL_PATH   GPU=$GPU TP=$TP mem-fraction=$MEM_FRACTION"
echo "    workload: ISL/OSL/conc=$ISL/$OSL/$CONC repeats=$REPEATS port=$PORT"
echo

BACKEND=sglang \
OUT_DIR="$OUT_DIR" GPU="$GPU" TP="$TP" MODEL="$MODEL_PATH" \
MEM_FRACTION="$MEM_FRACTION" PORT="$PORT" \
ISL="$ISL" OSL="$OSL" CONC="$CONC" REPEATS="$REPEATS" PROFILE=0 \
OVERLAY_PYTHONPATH="$OVERLAY" \
EXTRA_SERVER_ARGS="$ACCEPTED_FLAGS" \
EXTRA_ENV="$ACCEPTED_ENV" \
  bash "$BENCH"

echo
echo ">>> Done. Summary: $OUT_DIR/bench_summary.json"
