#!/usr/bin/env bash

set -euo pipefail


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SERVER_ONLY=0
for arg in "$@"; do
    if [ "$arg" = "--server-only" ]; then
        SERVER_ONLY=1
    fi
done

export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_NEW_CA=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export USE_AITER_COMM=1
export SGLANG_ALT_STREAM=True

# --- Defaults: port/model/TP match launch_server below ---
MODEL="${MODEL:-/wekafs/ethany/GEMM_tuning_test10/qwen3_14b_fp8}"
TP="${TP:-2}"
CONC="${CONC:-16}"
ISL="${ISL:-2048}"
OSL="${OSL:-256}"
PORT="${PORT:-8000}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-3}"
NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULTIPLIER))
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/${TIMESTAMP}}"
SERVER_LOG="${RESULT_DIR}/server.log"
KEEP_SERVER="${KEEP_SERVER:-0}"

mkdir -p "$RESULT_DIR"

kill_sglang() {
    local wait_s="${SERVER_KILL_WAIT_S:-10}"
    ps aux | grep "[p]ython3 -m sglang" | awk '{print $2}' | xargs -r kill -TERM 2>/dev/null || true
    sleep "$wait_s"
    ps aux | grep "[p]ython3 -m sglang" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    ps aux | grep "[m]ultiprocessing.spawn" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 2
}

wait_for_health() {
    local port=$1 log_file=$2 pid=$3 timeout=${4:-3600}
    local start_ts elapsed
    start_ts=$(date +%s)
    echo "Waiting for SGLang on port ${port} (timeout=${timeout}s)..."
    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: Server PID $pid exited. Last log lines:"
            tail -40 "$log_file" || true
            return 1
        fi
        if curl -s --max-time 5 "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "Server ready (health OK)."
            return 0
        fi
        elapsed=$(( $(date +%s) - start_ts ))
        if [ "$elapsed" -gt "$timeout" ]; then
            echo "ERROR: Health check timed out. Last log lines:"
            tail -40 "$log_file" || true
            return 1
        fi
        sleep 5
    done
}

cleanup() {
    if [ "${KEEP_SERVER:-0}" != "1" ] && [ "$SERVER_ONLY" != 1 ]; then
        kill_sglang || true
    fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "GEMM_tuning_test4 SGLang (397B aiter launch + run_qwen14b bench flow)"
echo "MODEL=$MODEL  TP=$TP  PORT=$PORT"
echo "Workload: CONC=$CONC ISL=$ISL OSL=$OSL prompts=$NUM_PROMPTS (${NUM_PROMPTS_MULTIPLIER}x CONC)"
echo "RESULT_DIR=$RESULT_DIR"
echo "============================================================"

kill_sglang || true

echo ""
echo "[1/2] Starting SGLang server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --tp-size "$TP" \
    --mem-fraction-static 0.9 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --attention-backend aiter \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --max-running-requests 4096 \
    --chunked-prefill-size 32768 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_health "$PORT" "$SERVER_LOG" "$SERVER_PID"

if [ "$SERVER_ONLY" = 1 ] || [ "${SKIP_BENCH:-0}" = 1 ]; then
    KEEP_SERVER=1
    echo ""
    echo "Server-only mode: PID=$SERVER_PID  log=$SERVER_LOG"
    echo "Stop with: kill $SERVER_PID  (or re-run without --server-only after manual stop)"
    trap - EXIT
    exit 0
fi

BENCH_JSON="${RESULT_DIR}/bench_serving_${TP}tp_conc${CONC}_isl${ISL}_osl${OSL}.json"


echo ""
echo "[2/2] bench_serving (same host/port/concurrency shape as run_qwen14b.sh)..."
python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --request-rate inf \
    --warmup-requests "${WARMUP_REQUESTS:-$((CONC * 2))}" \
    --output-file "$BENCH_JSON" \
    ${BENCH_EXTRA_ARGS:-}

echo ""
echo "Done. Server log: $SERVER_LOG"
echo "Bench output: $BENCH_JSON"

