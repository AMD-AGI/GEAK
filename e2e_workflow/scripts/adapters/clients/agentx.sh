#!/usr/bin/env bash
# AgentX trace-replay CLIENT adapter for bench_e2e.sh.  Sourced (not executed).
#
# Unlike Hyperloom's aiperf_client.sh, this adapter does NOT boot the server:
# GEAK's magpie launcher already did that under MAGPIE_RUN_PHASE=server.  This
# file ONLY redefines adapter_bench to drive aiperf's inferencex-agentx-mvp
# scenario against the warm server at $BASE_URL, then map the export into the
# canonical jsonl line bench_e2e.sh aggregates.
#
# Enable with:  BENCH_CLIENT=agentx  (run_e2e.py selects this automatically
# when handoff.workload_spec.kind == agentx_trace_replay).
#
# Requires: aiperf on PATH (or AIPERF_BIN), INFERENCEX_PATH with map_aiperf.py
# deployed under benchmarks/ (Hyperloom's AgentX runtime copies it there).

adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"

  # NUMP/ISL/OSL are owned by the trace corpus, not the synthetic sweep knobs
  # bench_e2e.sh also exports for the inferencex/native clients.

  if [ "$PROF" = "1" ] && declare -F adapter_bench_native >/dev/null; then
    adapter_bench_native "$NUMP" "$MAXC" 1
    return $?
  fi

  local py="${PYTHON_BIN:-python3}"
  local ix_root="${INFERENCEX_PATH:-}"
  local mapper=""
  for cand in \
    "${ix_root}/benchmarks/map_aiperf.py" \
    "${ix_root}/assets/agentx/map_aiperf.py"; do
    [ -n "$ix_root" ] && [ -f "$cand" ] && { mapper="$cand"; break; }
  done
  if [ -z "$mapper" ]; then
    echo "!!! agentx client: map_aiperf.py not found under INFERENCEX_PATH=${ix_root:-<unset>}." >&2
    return 5
  fi

  local port="${PORT:-8000}"
  case "${BASE_URL:-}" in
    *://*:*/*|*://*:*)
      port="${BASE_URL##*:}"
      port="${port%%/*}"
      ;;
  esac

  local art_dir="${OUT_DIR:-${PROFILE_DIR:-$(pwd)}}/agentx_client_$$_${RANDOM}"
  mkdir -p "$art_dir"
  rm -rf "$art_dir"/*
  mkdir -p "$art_dir"

  local scenario="${GEAK_AGENTX_SCENARIO:-inferencex-agentx-mvp}"
  local corpus="${AGENTX_DATASET:-${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126_256k}}"
  local nent="${AGENTX_NUM_ENTRIES:-393}"
  local conc="${CONC:-${MAXC:-8}}"
  local full_dur="${GEAK_AGENTX_DURATION_S:-3600}"
  local loop_dur="${GEAK_AGENTX_LOOP_DURATION_S:-900}"
  local purpose="${MEASUREMENT_PURPOSE:-search}"
  local duration="$loop_dur"
  if [ "$purpose" = "parity" ] || [ "$purpose" = "validation" ]; then
    duration="$full_dur"
  fi

  # ── Non-canonical workloads may run, but may never look submittable ────────
  # Mirrors aiperf_client.sh: the SCENARIO cannot police this for us. It has no
  # concept of corpus size, and its allowlist admits every dated weka variant,
  # so a 50-entry or wrong-corpus replay still returns submission_valid=true.
  # --unsafe-override does not help either: aiperf only flips the flag when the
  # override actually suppressed a violation, so at 3600s there is nothing to
  # suppress. GEAK's inner search loop runs the 900s scenario floor by design,
  # which means WITHOUT this stamp every search-leg measurement would come back
  # looking leaderboard-valid. So the client states the deviations itself and
  # map_aiperf.py forces submission_valid=false with them attached.
  local canon_entries=393
  local canon_duration="${AGENTX_CANONICAL_DURATION:-3600}"
  local canon_ds="${AGENTX_CANONICAL_DATASET:-$corpus}"
  local -a noncanon=()
  [ "$corpus" != "$canon_ds" ] && noncanon+=("corpus=${corpus}(canonical ${canon_ds})")
  [ "$nent" != "$canon_entries" ] && noncanon+=("entries=${nent}(canonical ${canon_entries})")
  [ "$duration" != "$canon_duration" ] && noncanon+=("duration=${duration}s(canonical ${canon_duration}s)")
  [ -n "${AGENTX_MAX_CTX:-}" ] && noncanon+=("client_context_cap=${AGENTX_MAX_CTX}")
  [ "${AGENTX_UNSAFE_OVERRIDE:-false}" = "true" ] && noncanon+=("unsafe_override_forced")

  local smoke_args=()
  if [ "$duration" -lt "$canon_duration" ] || [ "${AGENTX_UNSAFE_OVERRIDE:-false}" = "true" ]; then
    # Below the scenario's 900s floor aiperf aborts outright; at or above it the
    # flag is harmless and keeps the canonical and smoke paths uniform.
    smoke_args+=(--unsafe-override)
  fi

  # Always exported, empty included: a value inherited from the orchestrator's
  # environment or a previous leg must never survive into a canonical run.
  local noncanon_reasons=""
  if [ ${#noncanon[@]} -gt 0 ]; then
    noncanon_reasons="$(IFS=,; echo "${noncanon[*]}")"
    echo ">>> agentx client: NON-CANONICAL workload [${noncanon_reasons}] -- result will be stamped submission_valid=false and cannot KEEP" >&2
  fi

  local warm_lane="${AGENTX_WARMUP_REQUESTS_PER_LANE:-10}"
  local warm_grace="${AGENTX_WARMUP_GRACE_PERIOD:-1800}"
  local fail_thresh="${AGENTX_FAILED_REQUEST_THRESHOLD:-0.10}"
  local idle_gap="${AGENTX_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
  local aiperf="${AIPERF_BIN:-aiperf}"

  # Resolve the served model id (a reused server may expose a different name).
  local serve_model="$MODEL"
  local served=""
  served="$(curl -sf "${BASE_URL:-http://127.0.0.1:${port}}/v1/models" 2>/dev/null \
    | "$py" -c 'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || true)"
  [ -n "$served" ] && serve_model="$served"

  # Scrub stray AIPERF_* env (aiperf_client.sh does the same).
  local _k _v
  while IFS='=' read -r _k _v; do
    case "$_k" in
      AIPERF_BIN) : ;;
      AIPERF_*) unset "$_k" 2>/dev/null || true ;;
    esac
  done < <(env)

  export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AGENTX_DATASET_CONFIG_TIMEOUT:-1800}"
  export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AGENTX_DATASET_CONFIG_TIMEOUT:-1800}"
  export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AGENTX_LIVE_ASSISTANT:-0}"
  export AIPERF_UI_REALTIME_METRICS_ENABLED="${AGENTX_REALTIME_METRICS:-true}"
  local _mmap_default="${HF_HUB_CACHE:-${HOME:-/tmp}/.cache/huggingface/hub}/aiperf_dataset_mmap"
  export AIPERF_DATASET_MMAP_CACHE_DIR="${AGENTX_MMAP_CACHE_DIR:-$_mmap_default}"

  echo ">>> agentx client: scenario=${scenario} corpus=${corpus} conc=${conc} duration=${duration}s purpose=${purpose}"

  local -a ctx_args=()
  if [ -n "${AGENTX_MAX_CTX:-}" ]; then
    ctx_args+=(--max-context-length "$AGENTX_MAX_CTX")
  fi

  if ! command -v "$aiperf" >/dev/null 2>&1; then
    echo "!!! agentx client: aiperf not found (set AIPERF_BIN)." >&2
    return 5
  fi

  "$aiperf" profile \
    --scenario "$scenario" \
    --url "${BASE_URL:-http://127.0.0.1:${port}}" \
    --endpoint /v1/chat/completions \
    --endpoint-type chat --streaming --use-server-token-count \
    --model "$serve_model" \
    --tokenizer "$MODEL" --tokenizer-trust-remote-code \
    --public-dataset "$corpus" \
    --num-dataset-entries "$nent" \
    --concurrency "$conc" \
    --benchmark-duration "$duration" \
    --random-seed 42 \
    --trajectory-start-min-ratio 0.25 \
    --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane "$warm_lane" \
    --warmup-grace-period "$warm_grace" \
    --trace-idle-gap-cap-seconds "$idle_gap" \
    --failed-request-threshold "$fail_thresh" \
    --stats-interval 30 \
    --slice-duration 1.0 \
    --no-gpu-telemetry \
    ${ctx_args[@]+"${ctx_args[@]}"} \
    ${smoke_args[@]+"${smoke_args[@]}"} \
    --artifact-dir "$art_dir" --ui simple || return $?

  local pj=""
  pj="$(find "$art_dir" -name 'profile_export_aiperf.json' -print -quit)"
  if [ -z "$pj" ]; then
    echo "!!! agentx client: no profile_export_aiperf.json produced in $art_dir" >&2
    return 6
  fi

  local mapped="${art_dir}/inferencex_result.json"
  AGENTX_NONCANONICAL_REASONS="$noncanon_reasons" \
    "$py" "$mapper" "$pj" "$mapped" || return $?

  if [ ! -f "$mapped" ]; then
    echo "!!! agentx client: mapper produced no $mapped" >&2
    return 6
  fi

  "$py" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$mapped" \
    >> "$RESULT_JSONL" 2>/dev/null || cat "$mapped" >> "$RESULT_JSONL"
  mv "$mapped" "${mapped}.consumed" 2>/dev/null || true
}
