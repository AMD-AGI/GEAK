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
  # Pin GPU_ARCHS so aiter's JIT (chip_info.get_gfx_list) takes the env branch instead of
  # _detect_native() — the latter shells to rocm_agent_enumerator -> rocminfo PER cold-build worker
  # (~77 per import), which hang under GPU/KFD contention and pile up into a box-degrading storm
  # (observed: 561 procs, e2e throughput halved). Detect once here; honor a caller-set value.
  local _ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"
  # shellcheck disable=SC2086
  env $EXTRA_ENV \
    ${_ga:+GPU_ARCHS=$_ga} \
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    python -m sglang.launch_server \
      --model-path "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --tp-size "$TP" \
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
  # Optional request-rate (req/s) to STAGGER arrivals so sequences sit at different prefill/decode
  # phases — used by the steady-state profiling path. Empty => inf (max_concurrency still caps).
  [ -n "${REQUEST_RATE:-}" ] && extra+=(--request-rate "$REQUEST_RATE")
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
    --num-prompts "$NUMP" --max-concurrency "$MAXC" \
    --seed "$SEED" \
    --output-file "$RESULT_JSONL" "${extra[@]}"
  # sglang.bench_serving appends a result json line (output_throughput, median_ttft_ms, median_tpot_ms)
  # to --output-file, which is exactly the dispatcher's canonical schema. Nothing else to do.
}

# adapter_profile_window — capture a profiler window on the ALREADY-RUNNING, warm, mid-load server via
# sglang's HTTP profiler, so the trace reflects the real continuous-batching steady-state mix (prefill
# chunks + decode interleaved) instead of a cold prefill ramp. record_shapes=true so the parser gets
# Input Dims for shape attribution. Called by bench_e2e.sh AFTER a sustained background load is warm.
adapter_profile_window() {
  local before after
  before=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  # num_steps set => the server records that many forward steps then auto-saves (async; returns at once).
  if ! curl -sf -X POST "${BASE_URL}/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\":\"${PROFILE_DIR}\",\"num_steps\":${PROFILE_NUM_STEPS},\"record_shapes\":true}" \
        >/dev/null 2>&1; then
    echo "!!! /start_profile request failed (sglang HTTP profiler unavailable?)" >&2
    return 1
  fi
  # wait for a NEW trace to land (server saves after num_steps forward passes)
  local deadline=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
    [ "$after" -gt "$before" ] && { sleep 2; return 0; }   # +2s for the write to flush
    sleep 3
  done
  # num_steps may not be honored on some builds — force a stop and re-check
  curl -sf -X POST "${BASE_URL}/stop_profile" >/dev/null 2>&1 || true
  sleep 3
  after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ]
}
