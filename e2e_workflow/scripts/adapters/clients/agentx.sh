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

# _agentx_duration — the measured window for THIS leg.
# Shared by the bench and by the profile-window placement below so the two can
# never disagree about how long the replay actually runs.
_agentx_duration() {
  local purpose="${MEASUREMENT_PURPOSE:-search}"
  case "$purpose" in
    parity|validation) echo "${GEAK_AGENTX_DURATION_S:-3600}" ;;
    *)                 echo "${GEAK_AGENTX_LOOP_DURATION_S:-900}" ;;
  esac
}

# adapter_profile_warmup_s — OPTIONAL hook read by bench_e2e.sh: how long to wait
# after load start before opening the profiler window.
#
# bench_e2e.sh defaults PROFILE_WARMUP_SEC to 0, which is right for a synthetic
# sweep -- arming at load start keeps the initial prefill burst in the trace. It
# is wrong here. At load start aiperf has not begun serving the measured window
# at all: it is still resolving the corpus (minutes on an mmap cache miss),
# replaying per-lane cache warmups, and draining them. A window opened then
# records the RAMP (decode batch of 1, no steady-state MoE/attention mix) and
# every kernel decision downstream is made on a shape the graded workload never
# runs. That is not hypothetical -- it is how a full campaign came to route its
# whole kernel budget off 12 seconds of ramp-up trace.
#
# So the window is placed inside steady state, scaled to the leg it runs in:
# 75% of the replay duration, capped at the 2700s the InferenceX reference client
# uses for the canonical 3600s window, and pulled back far enough that the window
# itself still fits before the replay ends. Override with AGENTX_PROFILE_WARMUP_S.
adapter_profile_warmup_s() {
  if [ -n "${AGENTX_PROFILE_WARMUP_S:-}" ]; then
    echo "$AGENTX_PROFILE_WARMUP_S"
    return 0
  fi
  local dur win
  dur="$(_agentx_duration)"
  # Leave room for the capture plus the stop_profile flush.
  win=$(( ${PROFILE_WINDOW_SEC:-20} + 60 ))
  "${PYTHON_BIN:-python3}" - "$dur" "$win" <<'PY'
import sys
dur, reserve = int(float(sys.argv[1])), int(float(sys.argv[2]))
# 75% into the replay, never past 2700s (the reference client's canonical delay),
# and never so late that the window cannot complete before the replay stops.
print(max(0, min(int(dur * 0.75), 2700, dur - reserve)))
PY
}

adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"

  # NUMP/ISL/OSL are owned by the trace corpus, not the synthetic sweep knobs
  # bench_e2e.sh also exports for the inferencex/native clients.

  # PROF=1 asks the CLIENT to profile its own run. The synthetic client can do
  # that; this one deliberately does not delegate to it. bench_e2e.sh's windowed
  # path (adapter_profile_window) drives the load at PROF=0 and brackets the
  # capture itself, which is the faithful path and the one that runs for
  # vllm/sglang. Reaching here with PROF=1 means either a backend with no HTTP
  # profiler hook or an explicit profiled call -- and answering it with a
  # synthetic ISL/OSL sweep would write a trace of a DIFFERENT workload into this
  # run's profile dir, where every consumer would read it as the AgentX trace.
  # Run the real replay instead, and say plainly that the trace depends on
  # someone else opening the window.
  if [ "$PROF" = "1" ]; then
    if [ "${AGENTX_PROFILE_VIA_SYNTHETIC:-0}" = "1" ] && declare -F adapter_bench_native >/dev/null; then
      echo "!!! agentx client: AGENTX_PROFILE_VIA_SYNTHETIC=1 -- profiling a SYNTHETIC ISL/OSL sweep, " \
           "NOT the trace replay. The resulting trace describes a different workload; debugging only." >&2
      adapter_bench_native "$NUMP" "$MAXC" 1
      return $?
    fi
    echo ">>> agentx client: PROF=1 replays the trace as usual; the server-side profiler must be" \
         "bracketed by adapter_profile_window (no client-side /start_profile in aiperf)." >&2
  fi

  local py="${PYTHON_BIN:-python3}"
  local ix_root="${INFERENCEX_PATH:-}"
  local mapper=""
  # A real InferenceX checkout wins: under Hyperloom its AgentX runtime deploys
  # the mapper itself, and an orchestrated run must map its result with exactly
  # the file that runtime placed. GEAK's vendored copy is the LAST resort, so a
  # standalone run needs no checkout instead of failing here -- which it would
  # only do AFTER launching and warming a server.
  for cand in \
    "${ix_root:+${ix_root}/benchmarks/map_aiperf.py}" \
    "${ix_root:+${ix_root}/assets/agentx/map_aiperf.py}" \
    "${BASH_SOURCE[0]%/*}/map_aiperf.py"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { mapper="$cand"; break; }
  done
  if [ -z "$mapper" ]; then
    echo "!!! agentx client: no map_aiperf.py (INFERENCEX_PATH=${ix_root:-<unset>}, and the copy" \
         "vendored beside this adapter is missing)." >&2
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
  # The campaign corpus is the full-context parent: 393 traces, 56.8k main turns,
  # 98.8k total requests, per-request input capped at 990,016 tokens. It matches
  # the campaign server's --max-model-len 1048576, and launch_agentx.sh names it
  # as canonical. The _256k sibling drops every request over 256k tokens (98.8k
  # requests -> 68.3k) for ~256k-context servers; replaying that here measures a
  # lighter workload and yields a baseline that cannot be compared to the
  # campaign's.
  local canon_ds="${AGENTX_CANONICAL_DATASET:-semianalysis_cc_traces_weka_062126}"
  local corpus="${AGENTX_DATASET:-${WEKA_LOADER_OVERRIDE:-$canon_ds}}"
  local nent="${AGENTX_NUM_ENTRIES:-393}"
  local conc="${CONC:-${MAXC:-8}}"
  local purpose="${MEASUREMENT_PURPOSE:-search}"
  local duration
  duration="$(_agentx_duration)"

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
  local -a noncanon=()
  [ "$corpus" != "$canon_ds" ] && noncanon+=("corpus=${corpus}(canonical ${canon_ds})")
  # launch_agentx.sh:65 -- pinning the loader through WEKA_LOADER_OVERRIDE is a
  # deviation in its own right, even when the name it pins matches the canonical
  # one, because it bypasses the scenario's own corpus resolution.
  [ -n "${WEKA_LOADER_OVERRIDE:-}" ] && noncanon+=("weka_loader_override_pinned")
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
  # ── Failed-request tolerance, set by what the measurement is FOR ──────────
  # aiperf enforces this itself: on breach it logs "exceeding the
  # --failed-request-threshold limit" and broadcasts ProfileCancelCommand, and
  # the profile call above propagates that non-zero exit, so this is the gate
  # that decides whether a crashed leg can come back as a clean number.
  #
  # One flat number cannot serve both kinds of leg. A search leg is exploring
  # and a bad tail costs only that probe. A leg whose number will be COMPARED --
  # parity, validation, canonical -- cannot absorb failures at all, because the
  # requests that fail are the long/large trajectories, so dropping them
  # flatters whichever leg did the crashing. On the 20260912 run a candidate
  # killed EngineCore and failed 7.368% of its requests, stayed under the flat
  # 10%, and was accepted as usable against a clean reference.
  #
  # Values below the grace floor are equivalent: aiperf only cancels once at
  # least 10 requests have failed AS WELL as the rate being exceeded, and at
  # these volumes (~200-900 requests/leg) 10 failures is already 1-5%. So 0.01
  # means "cancel at the grace floor", not "cancel at 1%".
  local fail_thresh_default=0.10
  case "$purpose" in
    parity|validation|canonical) fail_thresh_default=0.01 ;;
  esac
  local fail_thresh="${AGENTX_FAILED_REQUEST_THRESHOLD:-$fail_thresh_default}"
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
