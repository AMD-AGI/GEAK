# vllm serving adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads the same env the dispatcher exports; sets SERVER_PID in adapter_launch; appends canonical
# result lines to $RESULT_JSONL.
#
# VERSION NOTE (read scripts/../knowledge/preflight.md): the vllm CLI surface drifts across releases.
#   - `vllm serve` and `vllm bench serve` exist on current vllm (>=0.6 / v1). On older builds the
#     equivalents are `python -m vllm.entrypoints.openai.api_server` and
#     `python benchmarks/benchmark_serving.py` (needs the repo checkout).
#   - `--gpu-memory-utilization` is the vllm analogue of sglang's `--mem-fraction-static`.
#   - profiling: vllm >=0.19 moved torch-profiler config from the VLLM_TORCH_PROFILER_DIR env var to the
#     `--profiler-config` CLI flag. adapter_launch probes vllm.config.ProfilerConfig to pick the path:
#     present -> --profiler-config; absent (<0.19) -> the env var (passing the flag would abort argparse).
#   - fusion capture: GEAK_FUSION_TRACE=1 (via EXTRA_ENV) switches the profiler to the KernelFusion
#     evidence mode: with_stack ON (module hierarchy for layer boundaries) and a short iteration-bounded
#     window (GEAK_FUSION_MAX_ITERS). Ordinary Profile rounds keep with_stack OFF.
#   - phase annotations are free: gpu_worker wraps every execute_model in annotate_profile()
#     unconditionally, emitting `execute_context_<nreq>(<ntok>)_generation_<nreq>(<ntok>)` into
#     gpu_user_annotation -- the dialect parse_profile._seg reads (measured on v0.27.1/gfx942).
# The Director's preflight step should smoke-test these two commands on the target image and record
# any needed EXTRA_SERVER_ARGS BEFORE the run relies on them. This adapter targets the current CLI.

adapter_default_port() { echo 8000; }

# ---------------------------------------------------------------------------
# KernelFusion capture mode.  Set by the orchestrator via EXTRA_ENV
# (GEAK_FUSION_TRACE=1).  Fusion has a DIFFERENT evidence goal from the native Top-N
# profiler: it needs Python/module spans and a short window that contains BOTH serving
# phases, not a long statistical sample.  See bench_e2e.sh's sizing block.
# A KEY=value the caller put in EXTRA_ENV: that reaches the server process, not this shell,
# so a setting the adapter itself must read (e.g. VLLM_PROFILE_WITH_STACK) is looked up here.
_extra_env_value() {
  local kv
  for kv in ${EXTRA_ENV:-}; do
    case "$kv" in "$1="*) echo "${kv#*=}"; return 0 ;; esac
  done
  return 1
}

_geak_fusion_capture() {
  case " ${EXTRA_ENV:-} " in *" GEAK_FUSION_TRACE=1 "*) return 0 ;; esac
  [ "${GEAK_FUSION_TRACE:-0}" = "1" ]
}

adapter_launch() {
  # Pin GPU_ARCHS so aiter's JIT skips rocm_agent_enumerator/_detect_native (see sglang.sh / gpu_lock.sh).
  local _ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"
  # Enable the server-side torch profiler version-portably. No PROFILE_DIR -> off. The ProfilerConfig
  # schema is strict (extra=forbid) and aborts the server on an unknown key, so probe its fields and emit
  # only what the installed build declares. JSON held in an array so it stays one argument.
  local -a _prof=()
  local -a _prof_env=()
  if [ -n "${PROFILE_DIR:-}" ]; then
    # with_stack: off for ordinary Profile rounds (stacks are the biggest per-event cost), on for the
    # KernelFusion capture (semantics needs the nn.Module hierarchy for layer boundaries).
    # Override with VLLM_PROFILE_WITH_STACK=true|false.
    local _stack_override="${VLLM_PROFILE_WITH_STACK:-$(_extra_env_value VLLM_PROFILE_WITH_STACK)}"
    local _stack="${_stack_override:-false}"
    local _max_iters="${PROFILE_MAX_ITERS:-64}"
    local _delay_iters="${PROFILE_DELAY_ITERS:-0}"
    if _geak_fusion_capture; then
      _stack="${_stack_override:-true}"
      # vllm has no profile_by_stage, so ONE window must hold both phases. A saturated server
      # interleaves them, but one iteration (the sglang setting) would capture a single phase.
      # If Phase 1 reports a single phase, raise this.
      _max_iters="${GEAK_FUSION_MAX_ITERS:-16}"
      # Skip this many engine iterations before the profiler actually starts.
      #
      # Landing the window in DECODE by wall-clock warmup is a race, and it is one you
      # lose quietly: at ISL 8192 / CONC 64 the prefill ramp is ~32 steps, so a short
      # warmup captures 24 iterations of pure prefill (Phase 1 then reports
      # `mixed_trace_no_decode_steps_in_window`), while a warmup long enough to clear the
      # ramp can outlive the background load entirely and fall back to a cold-start
      # capture -- prefill again, by a different route. Both were observed on MiniMax-M3.
      #
      # delay_iterations counts ENGINE STEPS, so it clears the ramp deterministically
      # regardless of how slow those steps are. Set it to at least
      # ceil(CONC*ISL/max_num_batched_tokens) to skip the prefill ramp.
      _delay_iters="${GEAK_FUSION_DELAY_ITERS:-$_delay_iters}"
    fi
    local _prof_fields
    _prof_fields="$(python3 - <<'PY' 2>/dev/null
names=set()
try:
    from vllm.config import ProfilerConfig
    import dataclasses
    try:
        names |= {f.name for f in dataclasses.fields(ProfilerConfig)}
    except Exception:
        pass
    names |= set(getattr(ProfilerConfig, "model_fields", {}) or {})
    names |= set(getattr(ProfilerConfig, "__annotations__", {}) or {})
    print(" ".join(sorted(names)))
except Exception:
    pass
PY
)"
    _has() { case " $_prof_fields " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
    if [ -n "$_prof_fields" ]; then
      local _json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_record_shapes\":true"
      _has torch_profiler_with_stack && _json="$_json,\"torch_profiler_with_stack\":$_stack"
      # 0.26+: max_iterations self-stops the profiler after N worker steps, bounding the buffer.
      if _has max_iterations; then
        _json="$_json,\"max_iterations\":$_max_iters"
      elif _geak_fusion_capture; then
        echo "!!! fusion capture: this build's ProfilerConfig has no max_iterations; the window" >&2
        echo "!!! falls back to GEAK_FUSION_WINDOW_SEC timing (trace size is NOT iteration-bounded)" >&2
      fi
      _has delay_iterations && _json="$_json,\"delay_iterations\":$_delay_iters"
      _has ignore_frontend  && _json="$_json,\"ignore_frontend\":true"
      # 0.26+: per-iteration prefill/decode annotation the parser uses for the phase split. Cheap.
      _has detailed_trace_annotation && _json="$_json,\"detailed_trace_annotation\":true"
      # 0.26+ decode-shape capture: opt-in (hardcodes with_stack+profile_memory, needs mem validation).
      if [ "${PROFILE_CAPTURE_TRACES:-0}" = "1" ]; then
        _has capture_torch_profiler && _json="$_json,\"capture_torch_profiler\":true"
      fi
      _json="$_json}"
      _prof=(--profiler-config "$_json")
    else
      # <0.19: no flag; the time window is the only bound.
      _prof_env=(VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
                 VLLM_TORCH_PROFILER_WITH_STACK="$([ "$_stack" = true ] && echo 1 || echo 0)")
    fi
    # A with_stack window stops and flushes INSIDE a worker RPC. On a large model that can
    # outlive vLLM's 300s execute_model RPC timeout and kill the engine: Kimi-K2.5 TP8 on
    # vLLM 0.21, max_iterations=192 with stacks -> "RPC call to sample_tokens timed out"
    # exactly 300s in, no trace written. Raise the bound for profiling servers only, unless
    # the caller set it.
    if [ "$_stack" = true ] && [ -z "${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-}" ] \
        && [ -z "$(_extra_env_value VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)" ]; then
      _prof_env+=(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${GEAK_PROFILE_STACK_RPC_TIMEOUT_S:-1800}")
    fi
  fi
  # Launch through $SERVER_LAUNCH_PREFIX (adapter contract): it puts the server in its
  # own session so teardown can prove the process group is ours. Empty when unset.
  # shellcheck disable=SC2086
  ${SERVER_LAUNCH_PREFIX:-} env $EXTRA_ENV \
    ${_ga:+GPU_ARCHS=$_ga} \
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    "${_prof_env[@]}" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    vllm serve "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$MEM_FRACTION" \
      "${_prof[@]}" \
      $EXTRA_SERVER_ARGS \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
}

adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }

# adapter_bench NUM_PROMPTS MAX_CONC PROFILE_FLAG
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"
  # Not in PROFILE_DIR: a background-load bench can finish after the trace manifest is built
  # and leave its .json among the traces (trace discovery matches that suffix), and PROFILE_DIR
  # is empty on unprofiled runs.
  local res_json
  res_json="$(mktemp "${TMPDIR:-/tmp}/vllm_bench_XXXXXX.json")"
  local extra=()
  [ "$PROF" = "1" ] && extra=(--profile)
  # Custom-tokenizer models (e.g. Kimi-K2.6) need the bench client to trust remote code to load
  # the tokenizer; mirror the server's trust setting (BENCH_TRUST_REMOTE_CODE from the dispatcher).
  [ "${BENCH_TRUST_REMOTE_CODE:-0}" = "1" ] && extra+=(--trust-remote-code)
  # GREEDY (--temperature 0) + --ignore-eos: deterministic, fixed-length OSL output. This is the
  # correct protocol for optimization work — it makes throughput reproducible, output parity byte-exact,
  # and speculative-decoding (MTP/EAGLE) acceptance meaningful (recent vllm dropped the temp==0 default).
  vllm bench serve \
    --backend vllm --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --num-prompts "$NUMP" --max-concurrency "$MAXC" \
    --seed "$SEED" --temperature 0 --ignore-eos \
    --save-result --result-filename "$res_json" "${extra[@]}"
  # vllm writes ONE result object (keys: output_throughput, median_ttft_ms, median_tpot_ms, ...).
  # Append it as a single jsonl line into the dispatcher's canonical results file.
  if [ -f "$res_json" ]; then
    python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$res_json" \
      >> "$RESULT_JSONL" 2>/dev/null || cat "$res_json" >> "$RESULT_JSONL"
    rm -f "$res_json"
  fi
}

# adapter_profile_window — capture a window on the warm, mid-load server via vllm's HTTP profiler (needs
# the profiler enabled at launch). Unlike sglang, /start_profile takes no num_steps: it runs until
# /stop_profile, so the window is time-controlled (start, sleep, stop) and the trace flushes on stop.
adapter_profile_window() {
  local before after
  before=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  if ! curl -sf -X POST "${BASE_URL}/start_profile" >/dev/null 2>&1; then
    echo "!!! /start_profile request failed (vllm torch profiler not enabled at launch?)" >&2
    return 1
  fi
  # 0.26+ self-stops at max_iterations, so this sleep is a safety cap; on <0.26 it is the only bound.
  if _geak_fusion_capture; then
    sleep "${GEAK_FUSION_WINDOW_SEC:-20}"
  else
    sleep "${PROFILE_WINDOW_SEC:-20}"
  fi
  # /stop_profile flushes the trace; the server waits for the flush, so give curl a generous timeout.
  curl -s --max-time "${PROFILE_WINDOW_TIMEOUT:-180}" -X POST "${BASE_URL}/stop_profile" \
    >/dev/null 2>&1 || echo "!!! /stop_profile request errored (checking for a trace anyway)" >&2
  # wait for a NEW trace to land (flush is async on some builds even after the stop returns)
  local deadline=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
    [ "$after" -gt "$before" ] && { sleep 2; return 0; }   # +2s for the write to flush
    sleep 3
  done
  after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ]
}
