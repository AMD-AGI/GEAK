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
    if python3 -c 'from vllm.config import ProfilerConfig' 2>/dev/null; then
      # PROFILE_ACTIVE_ITERS: widen the profiler capture window (default vllm=5 steps). Set large to
      # cover the whole short bench so trace and probe span the SAME steps -> count 绝对值可直接比.
      local _ai="${PROFILE_ACTIVE_ITERS:-}"
      local _aij=""
      [ -n "$_ai" ] && _aij=",\"active_iterations\":$_ai"
      _prof=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_record_shapes\":true$_aij}")
    else
      _prof_env=(VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR")
    fi
  fi
  # shellcheck disable=SC2086
  env $EXTRA_ENV \
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
