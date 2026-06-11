#!/usr/bin/env bash
# Throughput benchmark for Qwen3.5-27B on sglang (MI300X)
# ISL/OSL = 1024 / 1024, concurrency = 64
set -euo pipefail

MODEL=${MODEL:-/wekafs/models/Qwen-Qwen3.5-27B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
TP=${TP:-1}                 # tensor parallel size
GPU=${GPU:-0}              # which GPU(s) to use, e.g. "0" or "0,1"

ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONC=${CONC:-64}
NUM_PROMPTS=${NUM_PROMPTS:-$((CONC * 5))}   # 5 rounds per slot for a stable measurement

# ---- profiling ----
PROFILE=${PROFILE:-1}                 # 1 = capture a torch-profiler trace after the main bench
PROFILE_DIR=${PROFILE_DIR:-$(pwd)/profile_qwen35_27b}
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-5}   # decode steps to profile (keeps the trace small)
PROFILE_PROMPTS=${PROFILE_PROMPTS:-$CONC}   # one round at full concurrency is enough for a trace

BASE_URL="http://${HOST}:${PORT}"
LOG=server_qwen35_27b.log

echo "Model:        $MODEL"
echo "GPU(s):       $GPU  (TP=$TP)"
echo "ISL/OSL:      $ISL / $OSL"
echo "Concurrency:  $CONC   num-prompts: $NUM_PROMPTS"
echo

# ---- launch server ----
mkdir -p "$PROFILE_DIR"
echo ">>> Launching sglang server (log: $LOG) ..."
echo "    profiler dir: $PROFILE_DIR (PROFILE=$PROFILE)"
HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR" \
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --tp-size "$TP" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  > "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo ">>> Shutting down server (pid $SERVER_PID) ..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- wait for readiness ----
echo ">>> Waiting for server to become healthy ..."
for i in $(seq 1 120); do   # up to ~10 min
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
    echo ">>> Server is up after ${i}x5s."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!!! Server process died. Last log lines:"; tail -n 40 "$LOG"; exit 1
  fi
  sleep 5
done
curl -sf "${BASE_URL}/health" >/dev/null 2>&1 || { echo "!!! Server not healthy in time."; tail -n 40 "$LOG"; exit 1; }

# ---- run benchmark ----
echo ">>> Running bench_serving ..."
python -m sglang.bench_serving \
  --backend sglang \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --random-range-ratio 1.0 \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONC" \
  --output-file bench_qwen35_27b_result.jsonl

echo ">>> Throughput result appended to bench_qwen35_27b_result.jsonl"

# ---- profiling run (separate, so profiler overhead doesn't taint throughput numbers) ----
if [ "$PROFILE" = "1" ]; then
  echo ">>> Running profiling bench (torch profiler, ${PROFILE_NUM_STEPS} decode steps) ..."
  python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$PROFILE_PROMPTS" \
    --max-concurrency "$CONC" \
    --profile \
    --profile-num-steps "$PROFILE_NUM_STEPS" \
    --profile-output-dir "$PROFILE_DIR" \
    --profile-prefix qwen35_27b
  echo ">>> Profile traces written to: $PROFILE_DIR"
  echo "    Inspect with: open chrome://tracing  (or https://ui.perfetto.dev) and load the .trace.json.gz"
  ls -lh "$PROFILE_DIR" | tail -n +1
fi

echo ">>> Done."
