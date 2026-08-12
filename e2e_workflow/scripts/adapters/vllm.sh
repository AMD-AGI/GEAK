# vllm serving adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads the same env the dispatcher exports; sets SERVER_PID in adapter_launch; appends canonical
# result lines to $RESULT_JSONL.
#
# VERSION NOTE (read scripts/../knowledge/preflight.md): the vllm CLI surface drifts across releases.
#   - `vllm serve` and `vllm bench serve` exist on current vllm (>=0.6 / v1). On older builds the
#     equivalents are `python -m vllm.entrypoints.openai.api_server` and
#     `python benchmarks/benchmark_serving.py` (needs the repo checkout).
#   - `--gpu-memory-utilization` is the vllm analogue of sglang's `--mem-fraction-static`.
#   - profiling: vllm >=0.19 MOVED torch-profiler config from the VLLM_TORCH_PROFILER_DIR env var to the
#     `--profiler-config` CLI flag (the env var is now an UNKNOWN var -> warned + ignored -> NO trace is
#     written, so TraceLens gets no input). We emit `--profiler-config '{"profiler":"torch",...}'` so the
#     bench's `--profile` dumps a *.pt.trace.json.gz into PROFILE_DIR.
#     CROSS-VERSION: we DON'T blindly pass `--profiler-config` — old (<0.19) builds' argparse rejects the
#     unknown flag and the server never starts. We detect support by importing vllm.config.ProfilerConfig
#     (only present on builds that have the flag): if it imports -> use the flag (new builds); otherwise
#     fall back to the VLLM_TORCH_PROFILER_DIR env (old builds). This probe is device-independent (unlike
#     `vllm serve --help[=all]`, which initializes config/device and CRASHES on a driver-less host -> empty
#     output -> false-negative -> profiling silently lost) and far cheaper (no full server spin-up).
# The Director's preflight step should smoke-test these two commands on the target image and record
# any needed EXTRA_SERVER_ARGS BEFORE the run relies on them. This adapter targets the current CLI.

adapter_default_port() { echo 8000; }

adapter_launch() {
  # Pin GPU_ARCHS so aiter's JIT skips rocm_agent_enumerator/_detect_native (see sglang.sh / gpu_lock.sh).
  local _ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"
  # Enable the server-side torch profiler in a version-portable way. Two mutually-exclusive paths:
  #   new vllm (>=0.19): pass --profiler-config (the env var is rejected/ignored there).
  #   old vllm (<0.19) : pass VLLM_TORCH_PROFILER_DIR env (the CLI flag does NOT exist -> argparse would
  #                      abort the launch, so we MUST NOT pass it on old builds).
  # We pick the path by importing ProfilerConfig (device-independent capability probe). The JSON is held
  # in an array so it stays ONE argument (no word-split / brace-expansion). When PROFILE_DIR is unset,
  # profiling is off: the array is empty and we don't export the env var.
  local -a _prof=()
  local -a _prof_env=()
  if [ -n "${PROFILE_DIR:-}" ]; then
    # Field-granular capability probe (mirrors Hyperloom PR #1157's _vllm_profiler_is_native): the
    # ProfilerConfig schema is strict (pydantic extra=forbid) and ABORTS the server on an unknown key,
    # so we may only emit fields the INSTALLED build actually declares. Print the union of dataclass
    # fields / pydantic model_fields / __annotations__ so this works across the 0.19->0.26 schema churn.
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
      # New build (>=0.19): pass --profiler-config, built one field at a time so an older schema that
      # lacks the 0.26 knobs still launches. record_shapes stays on for Input Dims.
      local _json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_record_shapes\":true"
      # #398 primary fix: stacks default True upstream and are the biggest per-event cost (sglang.sh
      # already turns them off). Dropping them is the single largest cut to the profiler event buffer.
      _has torch_profiler_with_stack && _json="$_json,\"torch_profiler_with_stack\":false"
      # 0.26+ step control: max_iterations auto-stops the profiler after N worker steps (wrapper.py
      # step()->_call_stop), turning the wall-clock window step-bounded and capping the buffer by
      # construction. delay_iterations skips cold start; ignore_frontend is their required companion
      # (silences the high-overhead warning when delay/max are set).
      _has max_iterations   && _json="$_json,\"max_iterations\":${PROFILE_MAX_ITERS:-64}"
      _has delay_iterations && _json="$_json,\"delay_iterations\":${PROFILE_DELAY_ITERS:-0}"
      _has ignore_frontend  && _json="$_json,\"ignore_frontend\":true"
      # 0.26+ shape param riding along in the same config: emits the execute_..._context_N(sq..sk..)_
      # generation_N(...) dialect that parse_profile.py's _seg/_classify_step already parse, activating
      # the measured prefill/decode split. Emit-side cheap (one annotation per ITERATION, not per op).
      _has detailed_trace_annotation && _json="$_json,\"detailed_trace_annotation\":true"
      # 0.26+ rank0 decode-shape capture: OPT-IN only (PROFILE_CAPTURE_TRACES=1). Upstream hardcodes
      # with_stack+profile_memory for this path, so it needs memory validation on a 512 GiB pod first.
      if [ "${PROFILE_CAPTURE_TRACES:-0}" = "1" ]; then
        _has capture_torch_profiler && _json="$_json,\"capture_torch_profiler\":true"
      fi
      _json="$_json}"
      _prof=(--profiler-config "$_json")
    else
      # Old build (<0.19): the CLI flag does not exist (argparse would abort), so use the env var. No
      # with_stack/step knob here -> the shorter time window (bench_e2e.sh) is the only buffer bound.
      _prof_env=(VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR")
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
  local res_json="$PROFILE_DIR/.vllm_bench_$$_${RANDOM}.json"
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

# adapter_profile_window — capture a profiler window on the ALREADY-RUNNING, warm, mid-load server via
# vllm's HTTP profiler, so the trace reflects the real continuous-batching steady-state mix (prefill
# chunks + decode interleaved) instead of the cold prefill ramp `vllm bench serve --profile` would catch.
# Requires the server to have been launched with the torch profiler enabled (adapter_launch does:
# --profiler-config / VLLM_TORCH_PROFILER_DIR, with record_shapes=true so the parser gets Input Dims).
#
# DIFFERS FROM sglang: vllm's /start_profile takes NO num_steps — it runs until /stop_profile. So the
# window is TIME-controlled: start, sleep PROFILE_WINDOW_SEC of steady-state load, then stop. The trace
# is flushed on /stop_profile (the server blocks until the flush completes), so we allow a long curl
# timeout and then confirm a new trace landed.
adapter_profile_window() {
  local before after
  before=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  if ! curl -sf -X POST "${BASE_URL}/start_profile" >/dev/null 2>&1; then
    echo "!!! /start_profile request failed (vllm torch profiler not enabled at launch?)" >&2
    return 1
  fi
  # profile a steady-state window of this duration. On 0.26+ the profiler self-stops at max_iterations
  # (adapter_launch), so this sleep is just a safety upper bound; on <0.26 it is the only buffer bound.
  sleep "${PROFILE_WINDOW_SEC:-20}"
  # /stop_profile flushes the trace; the server waits for the flush, so give curl a generous timeout.
  curl -s --max-time "${PROFILE_WINDOW_TIMEOUT:-180}" -X POST "${BASE_URL}/stop_profile" \
    >/dev/null 2>&1 || echo "!!! /stop_profile request errored (checking for a trace anyway)" >&2
  # wait for a NEW trace to land (flush is async on some builds even after the stop returns)
  local deadline=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
    [ "$after" -gt "$before" ] && { sleep 2; _prune_nonrank0_traces; return 0; }   # +2s for the write to flush
    sleep 3
  done
  after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ] && _prune_nonrank0_traces
  [ "$after" -gt "$before" ]
}

# Drop the traces parse_profile.py never consumes: rank>=1 WORKER traces (TP/EP) and the *.async_llm.*
# engine trace (python_function only, no kernels). Only rank0 is read downstream (roles/profiler.md).
# Pruning saves disk + makes the dir unambiguous for the single-file parser. This does NOT cut the
# concurrent buffer peak (all workers still buffer during the window) — max_iterations bounds that.
# DENYLIST, not allowlist: a rank0 trace, or a TP=1 trace with NO rank marker at all, is always kept, so
# we never delete the only trace. Only ordinary *.trace.json* files are touched (a capture_traces/ subdir
# and non-trace files are left). PROFILE_KEEP_ALL_RANKS=1 disables pruning (diagnostics).
_prune_nonrank0_traces() {
  [ "${PROFILE_KEEP_ALL_RANKS:-0}" = "1" ] && return 0
  local f
  for f in "$PROFILE_DIR"/*.trace.json*; do
    [ -f "$f" ] || continue
    case "$f" in *rank0*|*rank-0*) continue ;; esac      # never touch rank0
    case "$f" in
      *rank[1-9]*|*rank-[1-9]*|*.async_llm.*) rm -f "$f" ;;
    esac
  done
}
