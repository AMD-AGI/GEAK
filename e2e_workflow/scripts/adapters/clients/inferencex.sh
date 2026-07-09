#!/usr/bin/env bash
# InferenceX bench CLIENT adapter (identical to Hyperloom / Magpie).
#
# This is NOT a serving backend — the server is still launched by the BACKEND
# adapter (sglang/vllm). This file ONLY redefines adapter_bench so the timed
# benchmark is driven by the EXACT same client Hyperloom uses: InferenceX's
# utils/bench_serving/benchmark_serving.py (OpenAI-compatible --backend vllm),
# with the same dataset / warmups / percentile metrics. That removes the
# "different bench client" residual when comparing GEAK numbers to a
# Hyperloom Magpie baseline.
#
# Enable with:  BENCH_CLIENT=inferencex  (bench_e2e.sh sources this and preserves
# the backend's native bench as adapter_bench_native for profiling).
#
# Requires: $INFERENCEX_PATH (or $INFERENCEX_BENCH_SERVING pointing straight at
# benchmark_serving.py). bench_e2e.sh exports MODEL/BASE_URL/ISL/OSL/CONC/SEED/
# NUM_WARMUPS/RANDOM_RANGE_RATIO/PROFILE_DIR.

_ix_resolve_bench_py() {
  if [ -n "${INFERENCEX_BENCH_SERVING:-}" ] && [ -f "${INFERENCEX_BENCH_SERVING}" ]; then
    echo "${INFERENCEX_BENCH_SERVING}"; return 0
  fi
  local root="${INFERENCEX_PATH:-}"
  for cand in \
    "${root}/utils/bench_serving/benchmark_serving.py" \
    "${root}/benchmarks/benchmark_serving.py" \
    "${root}/benchmark_serving.py"; do
    [ -n "$root" ] && [ -f "$cand" ] && { echo "$cand"; return 0; }
  done
  return 1
}

# adapter_bench NUMP MAXC PROF — run ONE InferenceX bench; append a canonical
# result line to $RESULT_JSONL (output_throughput / mean_ttft_ms / mean_tpot_ms).
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"

  # The portable InferenceX client has no server-trace hook; for the profile
  # round fall back to the backend's native bench (set up by bench_e2e.sh).
  if [ "$PROF" = "1" ] && declare -F adapter_bench_native >/dev/null; then
    adapter_bench_native "$NUMP" "$MAXC" 1
    return $?
  fi

  local py="${PYTHON_BIN:-python3}"
  local bench_py
  if ! bench_py="$(_ix_resolve_bench_py)"; then
    echo "!!! inferencex client: benchmark_serving.py not found. Set INFERENCEX_PATH or INFERENCEX_BENCH_SERVING." >&2
    return 5
  fi

  # Dedicated dir so a robust "newest json" fallback can't grab an unrelated file
  # (some benchmark_serving.py builds ignore --result-filename and auto-name).
  local res_dir="${OUT_DIR:-$PROFILE_DIR}/ix_client"
  mkdir -p "$res_dir"
  local res_name="ix_bench_$$_${RANDOM}"
  local num_warmups="${NUM_WARMUPS:-$(( MAXC < 8 ? MAXC : 8 ))}"

  # --backend vllm: OpenAI-compatible client regardless of the actual serving
  # stack (matches Hyperloom). --request-rate inf + --max-concurrency: the same
  # saturation driver. --ignore-eos + fixed seed + random-range-ratio: identical
  # dataset semantics.
  "$py" "$bench_py" \
    --model "$MODEL" \
    --backend vllm \
    --base-url "$BASE_URL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio "${RANDOM_RANGE_RATIO:-0}" \
    --num-prompts "$NUMP" \
    --max-concurrency "$MAXC" \
    --request-rate inf \
    --ignore-eos \
    --num-warmups "$num_warmups" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --seed "$SEED" \
    --save-result \
    --result-dir "$res_dir" \
    --result-filename "${res_name}.json" || return $?

  local res_json="$res_dir/${res_name}.json"
  if [ ! -f "$res_json" ]; then
    # build ignored --result-filename: take the newest json it just wrote
    res_json="$(ls -t "$res_dir"/*.json 2>/dev/null | head -n1)"
  fi
  if [ -n "$res_json" ] && [ -f "$res_json" ]; then
    "$py" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$res_json" \
      >> "$RESULT_JSONL" 2>/dev/null || cat "$res_json" >> "$RESULT_JSONL"
    mv "$res_json" "$res_json.consumed" 2>/dev/null || true   # no `rm` (approval prompt); move aside
  else
    echo "!!! inferencex client: no result file in $res_dir" >&2
    return 6
  fi
}
