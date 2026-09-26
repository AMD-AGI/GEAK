# ATOM serving adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads the same env the dispatcher exports; sets SERVER_PID in adapter_launch; appends canonical
# result lines to $RESULT_JSONL.
#
# ATOM is AMD's MI3xx serving stack (`python3 -m atom.entrypoints.openai_server`). It speaks the
# OpenAI-compatible surface (/v1/completions, /v1/chat/completions, /v1/models) plus /health and
# /start_profile,/stop_profile, so OpenAI-compatible bench clients drive it with `--backend vllm`.
#
# FLAG SURFACE — probed on-box, NOT guessed. Source of truth for this file is
#   python3 -m atom.entrypoints.openai_server --help
# on rocm/atom:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom20260511. Notable facts that
# differ from the sglang/vllm adapters and that you MUST keep in mind when editing:
#
#   * TWO PORTS. `--server-port` is the HTTP/OpenAI port that $BASE_URL must point at. `--port` is the
#     ENGINE-INTERNAL port (default 8006) used for engine IPC. Passing the HTTP port as `--port` gives
#     a server that never answers on $BASE_URL. We derive the internal port from PORT so two concurrent
#     ATOM servers on one box do not collide; override with ATOM_ENGINE_PORT.
#   * MEM_FRACTION maps to `--gpu-memory-utilization` (0.0-1.0) — the dimension IS supported, so unlike
#     a stack without the knob we pass it through rather than dropping it.
#   * `--kv_cache_dtype {bf16,fp8}` uses UNDERSCORES (not --kv-cache-dtype). ATOM's own default is bf16;
#     this adapter defaults to fp8 to match the InferenceX MI355X recipes it was written for. Override
#     with KV_CACHE_DTYPE=bf16.
#   * Profiling is a CLI flag, `--torch-profiler-dir`, NOT the VLLM_TORCH_PROFILER_DIR env var. ATOM
#     validates the directory exists at config time and ABORTS if it does not, so we mkdir -p first.
#     Each rank writes its own subdir: $PROFILE_DIR/rank_<N>/ (or dp<D>_tp<N>/ under DP), so anything
#     hunting for traces must recurse. Traces are named *.pt.trace.json.gz.
#   * ATOM_PROFILER_MORE=1 turns on record_shapes/with_stack/profile_memory inside the rank profiler.
#     record_shapes is what makes a trace usable for shape-driven kernel work, so adapter_profile_window
#     asks for it by default (ATOM_PROFILER_MORE=0 to opt out of the extra trace weight).
#
# TUNING SURFACE the Config Tuner can drive through EXTRA_SERVER_ARGS / EXTRA_ENV (all probed above):
#   args: --cudagraph-capture-sizes --level(0-3 compile) --max-num-batched-tokens --max-num-seqs
#         --enable-tbo[prefill|all] --all2all-backend[high-throughput|low-latency] --enforce-eager
#         --enable_prefix_caching --scheduler-delay-factor --block-size --method{mtp,eagle3}
#         --num-speculative-tokens
#   env:  ATOM_USE_TRITON_MLA ATOM_USE_TRITON_MOE ATOM_USE_TRITON_GEMM (Triton kernel paths — these are
#         the hooks an OVERLAY_PYTHONPATH kernel actually lands on), ATOM_ENABLE_*_FUSION,
#         ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD, ATOM_MOE_GU_ITLV, ATOM_USE_UNIFIED_ATTN,
#         ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE, ATOM_V4_BACKEND.

adapter_default_port() { echo 8888; }

# ---- ATOM worker-reaping supervisor -------------------------------------------------------------
# WHY: ATOM starts its per-rank ModelRunners with multiprocessing.spawn. Those workers do NOT die on
# the SIGTERM that server_teardown.sh sends to the process group -- the leader exits promptly, the
# workers reparent to pid 1 and keep running, each still holding its share of weights + KV cache
# (~277 GiB/GPU for DSR1 fp8 on MI355X). server_teardown.sh's escalation is gated on the LEADER still
# being alive after the grace window ("kill -0 $SERVER_PID || break"), so a leader that exits cleanly
# means SIGKILL is never sent to the group and the whole box stays pinned. The next launch then dies
# with "HIP out of memory ... 64.00 MiB is free" -- observed on this box before this supervisor existed.
#
# FIX: interpose a tiny supervisor as the group leader. It forwards SIGTERM to the server, waits for
# it to actually exit, and then SIGKILLs the remaining process group -- i.e. the leaked workers. It
# stays alive until that is done, which ALSO keeps server_teardown.sh's own escalation path valid.
#
# SAFETY: `kill -KILL 0` signals the caller's process group, so it is only correct when we lead that
# group. Under $SERVER_LAUNCH_PREFIX (setsid) we do. The pgid==pid guard makes the self-reap a no-op
# if the prefix is ever empty, where our group would be the benchmark's own -- never nuke that.
_ATOM_SUPERVISOR='
"$@" &
_c=$!
trap "kill -TERM $_c 2>/dev/null || true" TERM INT
wait "$_c"
while kill -0 "$_c" 2>/dev/null; do sleep 1; wait "$_c" 2>/dev/null || true; done
if [ "$(ps -o pgid= -p $$ 2>/dev/null | tr -d " ")" = "$$" ]; then
  kill -KILL 0 2>/dev/null || true
fi
true
'

adapter_launch() {
  # Pin GPU_ARCHS so aiter's JIT skips rocm_agent_enumerator/_detect_native (see sglang.sh / vllm.sh).
  local _ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"

  # Engine-internal port MUST differ from the HTTP port (see header). Derived, not fixed at ATOM's
  # 8006 default, so parallel servers on one box don't fight over the engine socket.
  local _eport="${ATOM_ENGINE_PORT:-$(( PORT + 1000 ))}"

  # --max-model-len: mirror the InferenceX fixed_seq_len recipes, which pin 10240 for every shape
  # EXCEPT 1k/1k (where they leave the model default). Explicit MAX_MODEL_LEN always wins. This keeps
  # GEAK numbers comparable to the InferenceX baseline instead of silently sizing KV differently.
  local _mml="${MAX_MODEL_LEN:-}"
  if [ -z "$_mml" ] && ! { [ "${ISL:-}" = "1024" ] && [ "${OSL:-}" = "1024" ]; }; then
    _mml=10240
  fi

  local -a _opt=()
  [ -n "$_mml" ] && _opt+=(--max-model-len "$_mml")
  [ -n "${KV_CACHE_DTYPE:-fp8}" ] && _opt+=(--kv_cache_dtype "${KV_CACHE_DTYPE:-fp8}")
  [ -n "${BLOCK_SIZE:-16}" ] && _opt+=(--block-size "${BLOCK_SIZE:-16}")
  [ -n "${MEM_FRACTION:-}" ] && _opt+=(--gpu-memory-utilization "$MEM_FRACTION")
  # EP / DP-attention follow the recipe's own gating (EP_SIZE>1, DP_ATTENTION=true).
  [ "${EP_SIZE:-1}" -gt 1 ] 2>/dev/null && _opt+=(--enable-expert-parallel)
  [ "${DP_ATTENTION:-false}" = "true" ] && _opt+=(--enable-dp-attention)
  [ "${BENCH_TRUST_REMOTE_CODE:-0}" = "1" ] && _opt+=(--trust-remote-code)

  # Server-side torch profiler. Flag-based, and ATOM asserts the dir already exists.
  # ATOM_PROFILER_MORE is read inside the WORKER process when the profiler is constructed, so it has
  # to be on the launch line — setting it later (e.g. around /start_profile) is too late.
  local -a _prof=() _prof_env=()
  if [ -n "${PROFILE_DIR:-}" ]; then
    mkdir -p "$PROFILE_DIR"
    _prof=(--torch-profiler-dir "$PROFILE_DIR")
    _prof_env=(ATOM_PROFILER_MORE="${ATOM_PROFILER_MORE:-1}")
  fi

  # Launch through $SERVER_LAUNCH_PREFIX (adapter contract): it puts the server in its
  # own session so teardown can prove the process group is ours. Empty when unset.
  # shellcheck disable=SC2086
  ${SERVER_LAUNCH_PREFIX:-} env $EXTRA_ENV \
    ${_ga:+GPU_ARCHS=$_ga} \
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    "${_prof_env[@]}" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    bash -c "$_ATOM_SUPERVISOR" atom-supervisor \
      python3 -m atom.entrypoints.openai_server \
        --model "$MODEL" \
        --host "$HOST" --server-port "$PORT" --port "$_eport" \
        -tp "$TP" \
        "${_opt[@]}" \
        "${_prof[@]}" \
        $EXTRA_SERVER_ARGS \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
}

adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }

# ATOM can answer /health before request-triggered Triton/aiter JIT work has quiesced. The dispatcher
# calls this after its untimed warmup. Isolated replicas intentionally skip that outer warmup, so give
# ATOM one explicit untimed request round here; the native client's own warmups are not a readiness
# proof because they happen inside the later timed invocation.
adapter_prepare_measurement() {
  local stable_sec="${ATOM_READY_STABLE_SEC:-10}"
  local timeout_sec="${ATOM_READY_TIMEOUT_SEC:-600}"
  local poll_sec="${ATOM_READY_POLL_SEC:-2}"
  local start now last_change last_sig sig

  _atom_jit_signature() {
    # Ignore access logs produced by our own /health polling. Only activity that can change the
    # generated kernel/graph state resets the stability window.
    grep -E '\[aiter\] (import|compile)|Using cache directory:.*torch_compile|compiled graph|Capturing bs=|cudagraph capture|warmup_model|Model warmup done' \
      "$LOG" 2>/dev/null | cksum | awk '{print $1 ":" $2}'
  }

  adapter_health || { echo "!!! atom readiness: /health failed after warmup" >&2; return 1; }
  if [ "${GEAK_ISOLATED_REPLICA:-0}" = "1" ]; then
    echo ">>> atom readiness: running an untimed request warmup for the isolated replica ..."
    adapter_bench "${ATOM_READY_WARMUP_PROMPTS:-$CONC}" "$CONC" 0 >/dev/null 2>&1 || {
      echo "!!! atom readiness: request warmup failed" >&2
      return 1
    }
  fi

  # A positive engine-ready marker proves /health belongs to a fully initialized ATOM engine. Then
  # require compile/JIT log activity to remain unchanged for a bounded interval so asynchronous
  # imports/captures triggered by warmup cannot overlap the first timed request.
  if ! grep -Eq 'All( [0-9]+)? EngineCores.*initialized and ready|Server started successfully and ready' "$LOG" 2>/dev/null; then
    echo "!!! atom readiness: no fully-initialized engine marker in $LOG" >&2
    return 1
  fi
  start=$(date +%s); last_change=$start
  last_sig=$(_atom_jit_signature)
  while :; do
    sleep "$poll_sec"
    adapter_health || { echo "!!! atom readiness: /health failed while waiting for JIT" >&2; return 1; }
    now=$(date +%s)
    sig=$(_atom_jit_signature)
    if [ -z "$sig" ]; then
      echo "!!! atom readiness: cannot stat $LOG" >&2
      return 1
    fi
    if [ "$sig" != "$last_sig" ]; then
      last_sig="$sig"; last_change=$now
    elif [ $((now - last_change)) -ge "$stable_sec" ]; then
      echo ">>> atom readiness: engine healthy and compile/JIT activity stable for ${stable_sec}s."
      return 0
    fi
    if [ $((now - start)) -ge "$timeout_sec" ]; then
      echo "!!! atom readiness: compile/JIT activity did not stabilize within ${timeout_sec}s" >&2
      return 1
    fi
  done
}

# adapter_bench NUM_PROMPTS MAX_CONC PROFILE_FLAG
# ATOM ships a vllm-derived client at atom.benchmarks.benchmark_serving with the same flag surface
# (verified via --help), including --num-warmups and --random-range-ratio. This is the NATIVE bench;
# BENCH_CLIENT=inferencex replaces it with InferenceX's client and keeps this as adapter_bench_native.
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"
  local res_dir="${OUT_DIR:-$PROFILE_DIR}/atom_client"
  mkdir -p "$res_dir"
  local res_name="atom_bench_$$_${RANDOM}.json"
  local extra=()
  [ "$PROF" = "1" ] && extra=(--profile)
  [ "${BENCH_TRUST_REMOTE_CODE:-0}" = "1" ] && extra+=(--trust-remote-code)
  [ -n "${REQUEST_RATE:-}" ] && extra+=(--request-rate "$REQUEST_RATE") || extra+=(--request-rate inf)

  python3 -m atom.benchmarks.benchmark_serving \
    --backend vllm --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio "${RANDOM_RANGE_RATIO:-0}" \
    --num-prompts "$NUMP" --max-concurrency "$MAXC" \
    --num-warmups "${NUM_WARMUPS:-$(( MAXC < 8 ? MAXC : 8 ))}" \
    --seed "$SEED" --ignore-eos \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --save-result --result-dir "$res_dir" --result-filename "$res_name" \
    "${extra[@]}" || return $?

  local res_json="$res_dir/$res_name"
  [ -f "$res_json" ] || res_json="$(ls -t "$res_dir"/*.json 2>/dev/null | head -n1)"
  if [ -n "$res_json" ] && [ -f "$res_json" ]; then
    python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$res_json" \
      >> "$RESULT_JSONL" 2>/dev/null || cat "$res_json" >> "$RESULT_JSONL"
    mv "$res_json" "$res_json.consumed" 2>/dev/null || true
  else
    echo "!!! atom client: no result file in $res_dir" >&2
    return 6
  fi
}

# adapter_profile_window — capture a window on the warm, mid-load server via ATOM's HTTP profiler
# (needs --torch-profiler-dir at launch). Like vllm and unlike sglang, /start_profile takes no step
# count: it runs until /stop_profile, so the window is time-controlled (start, sleep, stop).
# Traces land per-rank under $PROFILE_DIR/<rank_*|dp*_tp*>/, hence the recursive find.
#
# TWO ATOM-SPECIFIC HAZARDS, both observed on this box:
#
#  1. FINISHED vs IN-PROGRESS. ATOM may leave `.tmp`, uncompressed `.trace.json`, or a still-growing
#     `.trace.json.gz` while asynchronous finalization runs. A filename alone is not completion; gzip
#     integrity must pass before a rank counts.
#  2. ALL RANKS. /stop_profile fans out to TP workers. One early rank is not a usable distributed trace;
#     wait for every expected rank before returning and allowing teardown.
adapter_profile_window() {
  local wsec marker expected manifest
  marker="$PROFILE_DIR/.atom_profile_started.$$"
  manifest="$PROFILE_DIR/atom_profile_manifest.json"
  : > "$marker"

  # ATOM_PROFILER_MORE enables record_shapes together with expensive stack/memory capture. Keep the
  # backend-specific size guard: a long TP trace can consume tens of GiB and take longer to finalize
  # than the benchmark itself. Operators may raise the cap after checking local disk/RAM headroom.
  wsec="${ATOM_PROFILE_WINDOW_SEC:-${PROFILE_WINDOW_SEC:-20}}"
  local wmax="${ATOM_PROFILE_WINDOW_MAX_SEC:-6}"
  if [ "$wsec" -gt "$wmax" ] 2>/dev/null; then
    echo ">>> atom: clamping profile window ${wsec}->${wmax}s (set ATOM_PROFILE_WINDOW_MAX_SEC to override)"
    wsec="$wmax"
  fi
  expected="${ATOM_PROFILE_EXPECTED_RANKS:-${TP:-1}}"
  case "$expected" in ''|*[!0-9]*|0) echo "!!! atom: invalid expected profiler rank count '$expected'" >&2; return 1 ;; esac

  _atom_write_complete_ranks() {
    : > "$marker.ranks"
    while IFS= read -r -d '' _trace; do
      if gzip -t "$_trace" >/dev/null 2>&1; then
        dirname "$_trace"
      fi
    done < <(find "$PROFILE_DIR" -type f -newer "$marker" -name '*.trace.json.gz' -print0 2>/dev/null) \
      | sed -nE 's#^.*/(rank_[0-9]+|dp[0-9]+_tp[0-9]+)$#\1#p' | sort -u > "$marker.ranks"
  }

  if ! curl -sf -X POST "${BASE_URL}/start_profile" >/dev/null 2>&1; then
    echo "!!! /start_profile request failed (ATOM torch profiler not enabled at launch? needs --torch-profiler-dir)" >&2
    return 1
  fi
  sleep "$wsec"
  # /stop_profile asks every rank to flush; the gzip itself continues asynchronously afterwards.
  curl -s --max-time "${PROFILE_WINDOW_TIMEOUT:-180}" -X POST "${BASE_URL}/stop_profile" \
    >/dev/null 2>&1 || echo "!!! /stop_profile request errored (checking for a trace anyway)" >&2

  # Wait for one NEW finalized gzip per expected rank. Returning after rank 0 lets teardown truncate
  # the other TP workers, which leaves an unusable and biased profile.
  local hard=$(( $(date +%s) + ${ATOM_PROFILE_FINALIZE_TIMEOUT:-900} ))
  local quiet=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
  local last_sz=-1 sz count
  while [ "$(date +%s)" -lt "$hard" ]; do
    _atom_write_complete_ranks
    count=$(wc -l < "$marker.ranks")
    if [ "$count" -ge "$expected" ]; then
      python3 - "$marker.ranks" "$manifest" "$expected" "$wsec" <<'PY'
import json, sys
ranks_path, out_path, expected, window = sys.argv[1:]
ranks = [line.strip() for line in open(ranks_path) if line.strip()]
with open(out_path, "w") as fh:
    json.dump({"status": "complete", "expected_ranks": int(expected),
               "completed_ranks": ranks, "window_sec": float(window)}, fh, indent=2)
PY
      rm -f "$marker" "$marker.ranks"
      return 0
    fi
    if find "$PROFILE_DIR" -type f -newer "$marker" \( -name '*.trace.json.tmp' -o -name '*.trace.json' -o -name '*.trace.json.gz' \) \
         -print -quit 2>/dev/null | grep -q .; then
      # Any trace artifact exists: extend the quiet deadline while the largest one keeps growing.
      sz=$(find "$PROFILE_DIR" -type f -newer "$marker" \( -name '*.trace.json.tmp' -o -name '*.trace.json' -o -name '*.trace.json.gz' \) \
        -printf '%s\n' 2>/dev/null | sort -rn | head -1)
      if [ "${sz:-0}" != "$last_sz" ]; then
        last_sz="${sz:-0}"; quiet=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
      fi
    fi
    [ "$(date +%s)" -ge "$quiet" ] && break
    sleep 5
  done
  _atom_write_complete_ranks
  count=$(wc -l < "$marker.ranks")
  echo "!!! atom: profiler finalized ${count}/${expected} expected rank traces; refusing partial profile." >&2
  find "$PROFILE_DIR" -type f -newer "$marker" \( -name '*.trace.json.tmp' -o -name '*.trace.json' \) \
    -print 2>/dev/null | sed 's/^/    unfinished: /' >&2
  python3 - "$marker.ranks" "$manifest" "$expected" "$wsec" <<'PY'
import json, sys
ranks_path, out_path, expected, window = sys.argv[1:]
ranks = [line.strip() for line in open(ranks_path) if line.strip()]
with open(out_path, "w") as fh:
    json.dump({"status": "incomplete", "expected_ranks": int(expected),
               "completed_ranks": ranks, "window_sec": float(window)}, fh, indent=2)
PY
  # Preserve partial evidence without leaving a normal *.trace.json.gz that the Profiler could pick
  # up as if the distributed capture had succeeded.
  while IFS= read -r -d '' _trace; do
    mv "$_trace" "${_trace}.PARTIAL_missing_ranks" 2>/dev/null || true
  done < <(find "$PROFILE_DIR" -type f -newer "$marker" -name '*.trace.json.gz' -print0 2>/dev/null)
  rm -f "$marker" "$marker.ranks"
  return 1
}
