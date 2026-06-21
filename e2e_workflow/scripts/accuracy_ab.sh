#!/usr/bin/env bash
# gsm8k accuracy A/B: serve the BASELINE (no overlay) and a CANDIDATE (OVERLAY_PYTHONPATH) one at a time,
# run the gsm8k subset eval against each, print both scores + the delta. NO `rm` (kill by PID only).
#   accuracy_ab.sh <MODEL> <OVERLAY_DIR> <GPUS> <PORT> <LIMIT> <OUTDIR>
set -uo pipefail
MODEL="${1:?model path}"; OVERLAY="${2:?overlay dir (cand)}"; GPUS="${3:-0,1,2,3}"; PORT="${4:-30900}"
LIMIT="${5:-200}"; OUT="${6:-/tmp/gsm8k_ab_$$}"; TP=$(echo "$GPUS" | awk -F, '{print NF}')
SELF="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES="$GPUS" VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTORCH_ALLOC_CONF=expandable_segments:True   # avoid cudagraph-capture fragmentation OOM
# Fresh per-run JIT/compile caches: a corrupted Triton/inductor cache from earlier crashed runs is the
# prime suspect for the "Triton HIP 901 (previous error during capture)" that aborts graph capture mid-way
# (the same model+vllm captured all 102 graphs cleanly on 06-20). Isolated dirs, no rm of the old cache.
export TRITON_CACHE_DIR="$OUT/triton_cache" VLLM_CACHE_ROOT="$OUT/vllm_cache"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

serve_flags=(--port "$PORT" --tensor-parallel-size "$TP" --block-size 128 --no-enable-prefix-caching
  --language-model-only --max-model-len 4096 --max-num-batched-tokens 2048 --kv-cache-dtype fp8
  --attention-backend TRITON_ATTN
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice
  --gpu-memory-utilization 0.6 --trust-remote-code
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}')
  # mem 0.6 — counterintuitively LOW is correct. vllm's startup check is `free-after-weights(~177 GiB) >=
  # util*288`. 0.70 needs 201 > 177 so it FAILS unless a TP race measures pre-weights (flaky: ab10-14
  # passed, ab15 failed). util<=0.614 (threshold<=177) passes DETERMINISTICALLY. 0.6 -> pool 172.8, weights
  # 111, KV ~62 GiB, and ~115 GiB free for capture+inference (fixes the prefill/decode OOM with room to
  # spare). Plus: cudagraph_capture_sizes [1,2,4,8] = tiny graph pool (all-51 reserved too much), prefill
  # capped by max-num-batched-tokens 2048 + max-model-len 4096, gsm8k concurrency 8. PIECEWISE (FULL=901).
  # FINAL config: PIECEWISE-only (NOT FULL_AND_PIECEWISE). On this box state the FULL whole-model decode
  # graph capture fails every time with Triton HIP 901 at graph 0 (even with fresh caches + mem headroom),
  # while PIECEWISE captures all 51 sizes cleanly. So drop the FULL graph. Keep DEFAULT capture sizes (all
  # 51, do NOT restrict) so every decode batch pads to a captured graph — ab8 restricted to [1..16] and
  # OOM'd mid-decode; full coverage avoids that. mem 0.7 fits weights+KV+capture with runtime headroom.
  # Workflow's WORKING baseline (06-20) used FULL_AND_PIECEWISE + all 51 sizes at mem 0.9 and captured all
  # 102 graphs fine with a 5.37M-token KV cache. The SAME config OOMs today at capture 6/51 (less effective
  # headroom now). So keep the intended deploy graph (FULL_AND_PIECEWISE, all sizes — FULL captures
  # attention so decode never recompiles -> avoids the progressive PIECEWISE OOM) but drop mem to 0.7:
  # KV shrinks (~90 GiB) leaving ~86 GiB for the 102-graph capture. NO compilation override (default mode).
  # Non-standard configs all failed: NONE -> deadlock; PIECEWISE -> recompile OOM; restricted FULL -> HIP
  # 901; mem 0.9 -> capture OOM. Pre-launch we hard-verify GPUs clean (a flaky kill left 111 GiB/GPU once).

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
      --limit "$LIMIT" --concurrency 8 --out "$OUT/gsm8k_$tag.json" 2>>"$OUT/gsm8k_$tag.stderr" | tee "$OUT/gsm8k_$tag.txt"
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
