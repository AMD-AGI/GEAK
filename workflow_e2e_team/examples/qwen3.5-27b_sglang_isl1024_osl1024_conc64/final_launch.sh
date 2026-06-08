#!/usr/bin/env bash
# final_launch.sh — reproduce the OPTIMIZED Qwen3.5-27B serving config and measure e2e throughput.
#
# Model:    /wekafs/models/Qwen-Qwen3.5-27B   (sglang 0.5.11, ROCm/MI300X gfx942, tp=1, bf16)
# Workload: ISL/OSL/conc = 1024 / 1024 / 64  (prefill-dominated)
#
# ACCEPTED optimization stack (vs baseline 1485.432 tok/s):
#   * config flag: --attention-backend triton   (no source edit, no overlay)
#   * overlay:     EMPTY   (no kernel patch accepted — all editable-Triton kernels were
#                            Amdahl-NULL on the e2e gate; see ./final_patch.diff)
#
# Drives the shared scripts/bench_e2e.sh (warm server + N timed repeats + median) with the sglang
# adapter. Self-contained: just `bash final_launch.sh`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$HERE/.." && pwd)"

# --- the deliverable config ---
export BACKEND="${BACKEND:-sglang}"
MODEL="${MODEL:-/wekafs/models/Qwen-Qwen3.5-27B}"
GPU="${GPU:-0}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"; CONC="${CONC:-64}"
REPEATS="${REPEATS:-7}"

# accepted flags / env (the whole optimization)
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---attention-backend triton}"
EXTRA_ENV="${EXTRA_ENV:-}"
# EMPTY overlay -> stock sglang install. (Left unset on purpose; no kernel patch accepted.)
OVERLAY_PYTHONPATH="${OVERLAY_PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-$HERE/bench_out}"
mkdir -p "$OUT_DIR"

echo "=== final optimized config ==="
echo "  flags:   $EXTRA_SERVER_ARGS"
echo "  env:     ${EXTRA_ENV:-<none>}"
echo "  overlay: ${OVERLAY_PYTHONPATH:-<none (empty)>}"
echo "  ISL/OSL/conc=$ISL/$OSL/$CONC  repeats=$REPEATS  gpu=$GPU"
echo

BACKEND="$BACKEND" OUT_DIR="$OUT_DIR" GPU="$GPU" MODEL="$MODEL" \
  ISL="$ISL" OSL="$OSL" CONC="$CONC" REPEATS="$REPEATS" PROFILE=0 \
  OVERLAY_PYTHONPATH="$OVERLAY_PYTHONPATH" \
  EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS" EXTRA_ENV="$EXTRA_ENV" \
  bash "$EVAL_DIR/bench_e2e.sh"

echo
echo "=== summary: $OUT_DIR/bench_summary.json ==="
cat "$OUT_DIR/bench_summary.json" 2>/dev/null || true
