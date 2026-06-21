#!/usr/bin/env bash
# gsm8k accuracy A/B: serve the BASELINE (no overlay) and a CANDIDATE (OVERLAY_PYTHONPATH) one at a time,
# run the gsm8k subset eval against each, print both scores + the delta. NO `rm` (kill by PID only).
#   accuracy_ab.sh <MODEL> <OVERLAY_DIR> <GPUS> <PORT> <LIMIT> <OUTDIR>
set -uo pipefail
MODEL="${1:?model path}"; OVERLAY="${2:?overlay dir (cand)}"; GPUS="${3:-0,1,2,3}"; PORT="${4:-30900}"
LIMIT="${5:-200}"; OUT="${6:-/tmp/gsm8k_ab_$$}"; TP=$(echo "$GPUS" | awk -F, '{print NF}')
SELF="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES="$GPUS" VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_BREAKABLE_CUDAGRAPH=0

serve_flags=(--port "$PORT" --tensor-parallel-size "$TP" --block-size 128 --no-enable-prefix-caching
  --language-model-only --max-model-len 9216 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice
  --gpu-memory-utilization 0.9 --trust-remote-code)

run_one() {  # $1=tag  $2=overlay_pythonpath(empty for baseline)
  local tag="$1" ovl="$2" log="$OUT/server_$1.log" pid
  echo ">>> [$tag] launching vLLM (overlay='${ovl:-NONE}') on GPU $GPUS port $PORT"
  ( [ -n "$ovl" ] && export PYTHONPATH="$ovl:${PYTHONPATH:-}"; exec vllm serve "$MODEL" "${serve_flags[@]}" ) > "$log" 2>&1 &
  pid=$!
  for _ in $(seq 1 720); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo ">>> [$tag] server ready"; break; }
    kill -0 "$pid" 2>/dev/null || { echo "!!! [$tag] server exited early; tail:"; tail -20 "$log"; return 1; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "!!! [$tag] not healthy"; tail -20 "$log"; kill "$pid" 2>/dev/null; return 1; }
  echo ">>> [$tag] running gsm8k (limit=$LIMIT, 5-shot, greedy)"
  python3 "$SELF/gsm8k_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
      --limit "$LIMIT" --out "$OUT/gsm8k_$tag.json" 2>>"$OUT/gsm8k_$tag.stderr" | tee "$OUT/gsm8k_$tag.txt"
  echo ">>> [$tag] shutting down server (pid $pid)"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # belt: kill orphaned TP workers (renamed VLLM::Worker, PPID 1) — kill by name, no rm
  pkill -KILL -f "VLLM::Worker" 2>/dev/null; sleep 3
}

run_one baseline ""        || exit 1
run_one cand     "$OVERLAY" || exit 1
B=$(grep -oE "GSM8K_EXACT_MATCH=[0-9.]+" "$OUT/gsm8k_baseline.txt" | cut -d= -f2)
C=$(grep -oE "GSM8K_EXACT_MATCH=[0-9.]+" "$OUT/gsm8k_cand.txt" | cut -d= -f2)
echo "================ gsm8k A/B (n=$LIMIT) ================"
echo "  baseline exact_match = $B"
echo "  cand     exact_match = $C"
python3 -c "b=$B; c=$C; d=c-b; print(f'  delta = {d:+.4f}  ({\"ACCEPT\" if d>=-0.01 else \"REJECT\"} at tol -1%)')" 2>/dev/null
echo "  artifacts: $OUT"
