# sglang serving adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads env exported by the dispatcher: MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS
#   EXTRA_ENV OVERLAY_PYTHONPATH PROFILE PROFILE_DIR PROFILE_NUM_STEPS BASE_URL RESULT_JSONL LOG
#   ISL OSL CONC SEED
# Sets SERVER_PID (global) in adapter_launch. Append canonical result lines to $RESULT_JSONL.

adapter_default_port() { echo 30000; }

adapter_launch() {
  # Raise the scheduler watchdog by default: an authored/JIT kernel (FlyDSL/triton-author) overlaid on
  # the path JIT-compiles on first prefill, which can exceed sglang's default watchdog and kill the
  # server before CUDA-graph capture. Harmless for stock runs. Only add it if the caller didn't already
  # set one in EXTRA_SERVER_ARGS (override via WATCHDOG_TIMEOUT=...; set empty to disable).
  local _wd=""
  case " $EXTRA_SERVER_ARGS " in
    *" --watchdog-timeout "*) _wd="" ;;
    *) [ -n "${WATCHDOG_TIMEOUT:-600}" ] && _wd="--watchdog-timeout ${WATCHDOG_TIMEOUT:-600}" ;;
  esac
  # shellcheck disable=SC2086
  env $EXTRA_ENV \
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    python -m sglang.launch_server \
      --model-path "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --tp-size "$TP" \
      --trust-remote-code \
      --mem-fraction-static "$MEM_FRACTION" \
      $_wd \
      $EXTRA_SERVER_ARGS \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
}

adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }

# adapter_bench NUM_PROMPTS MAX_CONC PROFILE_FLAG
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"
  local extra=()
  if [ "$PROF" = "1" ]; then
    extra=(--profile --profile-num-steps "$PROFILE_NUM_STEPS"
           --profile-output-dir "$PROFILE_DIR" --profile-prefix e2e)
  fi
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
    --num-prompts "$NUMP" --max-concurrency "$MAXC" \
    --seed "$SEED" \
    --output-file "$RESULT_JSONL" "${extra[@]}"
  # sglang.bench_serving appends a result json line (output_throughput, median_ttft_ms, median_tpot_ms)
  # to --output-file, which is exactly the dispatcher's canonical schema. Nothing else to do.
}
