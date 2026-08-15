#!/usr/bin/env bash
# Magpie server LAUNCHER adapter for bench_e2e.sh.  Sourced (not executed).
# BACKEND-AGNOSTIC: one adapter serves every backend Magpie ships a script for
# (sglang, vllm, ...), because Magpie standardised the server-phase contract:
#   MAGPIE_RUN_PHASE=server  ->  launch server, wait ready, write pid to
#   MAGPIE_SERVER_PID_FILE, disown, exit 0.  Reads MODEL/TP/PORT/RESULT_DIR/
#   SERVER_LOG/PROFILE.  The only per-backend differences follow a REGULAR naming
#   rule, so they are derived from $BACKEND (never hard-coded per backend):
#     * extra server flags var : EXTRA_<BACKEND_UPPER>_ARGS  (EXTRA_SGLANG_ARGS / EXTRA_VLLM_ARGS)
#     * torch-profiler dir  var : <BACKEND_UPPER>_TORCH_PROFILER_DIR
#
# Redefining ONLY adapter_launch makes the served stack BYTE-IDENTICAL to the
# orchestrator's baseline (mem-fraction, --disable-radix-cache / gpu-mem-util,
# --trust-remote-code, *_USE_AITER, firmware-gated HSA_NO_SCRATCH_RECLAIM, ... all
# owned by that one script). The authored-kernel OVERLAY is prepended to
# PYTHONPATH HERE (Magpie's own path never honors OVERLAY_PYTHONPATH), so
# recipe-parity AND overlay application coexist.
#
# Script resolution (general): $MAGPIE_LAUNCH_SCRIPT, else the per-backend
# $MAGPIE_<BACKEND_UPPER>_SCRIPT. Its sibling benchmark_lib.sh / server_cleanup.sh
# must be present next to it. When no script is resolvable it DELEGATES to the
# native backend launch (adapter_launch_native), so a misconfigured run degrades
# instead of failing hard.
#
# bench_e2e.sh contract: sets global SERVER_PID; writes $LOG. Reads env:
#   BACKEND MODEL TP PORT GPU EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH
#   PROFILE PROFILE_DIR LOG OUT_DIR MAX_MODEL_LEN PROFILE_MAX_ITERS
#   PROFILE_DELAY_ITERS.
#
# TWO logs, deliberately: Magpie's script redirects the server with a
# TRUNCATING '> $SERVER_LOG', so anything this adapter appended to the same file
# would be destroyed the moment the server starts, and the script's own set -x
# trace would interleave with server output in whatever survived. The script's
# stdout/stderr therefore goes to magpie_launch.log and $LOG stays exclusively
# the server's, which is also what makes $LOG parseable for the kernel-selection
# fingerprint in result.json.serving_stack.
# adapter_health is inherited from the BACKEND adapter (curl $BASE_URL/health),
# which works regardless of who launched the server, so it is NOT redefined.

adapter_launch() {
  local backend_uc script var_script
  backend_uc="$(printf '%s' "${BACKEND:-sglang}" | tr '[:lower:]' '[:upper:]')"

  # generic path first, then per-backend MAGPIE_<BACKEND>_SCRIPT (indirection).
  script="${MAGPIE_LAUNCH_SCRIPT:-}"
  if [ -z "$script" ]; then
    var_script="MAGPIE_${backend_uc}_SCRIPT"
    script="${!var_script:-}"
  fi

  if [ -z "$script" ] || [ ! -f "$script" ]; then
    echo "!!! magpie launcher: no Magpie script for BACKEND='$BACKEND'" \
         "(set MAGPIE_LAUNCH_SCRIPT or MAGPIE_${backend_uc}_SCRIPT; got '$script');" \
         "falling back to native backend launch." >&2
    if declare -F adapter_launch_native >/dev/null; then
      adapter_launch_native
      return $?
    fi
    echo "!!! magpie launcher: no native launch to fall back to." >&2
    return 2
  fi

  local _out_dir="${OUT_DIR:-${PROFILE_DIR:-$(pwd)}}"
  local _pidfile="$_out_dir/magpie_server.pid"
  local _launchlog="$_out_dir/magpie_launch.log"
  rm -f "$_pidfile" 2>/dev/null || true

  # Per-backend var NAMES (regular rule), passed to the script via env NAME=VALUE.
  local _args_var="EXTRA_${backend_uc}_ARGS"
  local _prof_var="${backend_uc}_TORCH_PROFILER_DIR"

  # The orchestrator's RECORDED launch environment, replayed as the BASE layer.
  # Without it the two servers agree only where their ${X:-default} expansions
  # happen to agree -- true today only because both run in the same image, and
  # false the moment the orchestrator sets anything explicitly (PATH selecting a
  # different venv is the one that would silently serve a different vLLM build
  # entirely). run_e2e.py has already removed the run-scoped names GEAK must own,
  # so everything left here is safe to apply verbatim. NUL-delimited because the
  # recipe records PATH and word-splitting a value would corrupt it.
  local _recipe_env=()
  if [ -n "${RECIPE_ENV_FILE:-}" ] && [ -f "${RECIPE_ENV_FILE}" ]; then
    mapfile -d '' -t _recipe_env < "$RECIPE_ENV_FILE"
    echo ">>> magpie launcher: replaying ${#_recipe_env[@]} recorded env var(s) from the recipe."
  fi

  # Magpie's vllm script builds its own --profiler-config.* block but sets no step
  # bound, so the host-side event buffer grows for as long as the load runs --
  # the unbounded growth #398 fixed on the native path by passing max_iterations
  # (vllm 0.26+ self-stops the profiler after N worker steps). Its buffer fills
  # FASTER than the native one, too: it asks for with_memory and with_flops, and
  # leaves with_stack at its default, all of which the native adapter declines.
  # The script appends $EXTRA_<BE>_ARGS AFTER its own profiler args, so the bound
  # reaches vllm without editing the recipe's script. ProfilerConfig is strict
  # (extra=forbid) and an unknown key aborts the server, so emit only what this
  # build declares -- the same probe the native adapter runs. It runs under the
  # replayed recipe env because that PATH may select a different venv: the
  # interpreter that answers has to be the one that will serve.
  local _extra_args="${EXTRA_SERVER_ARGS:-}" _bound=""
  if [ "${PROFILE:-0}" = "1" ] && [ "$backend_uc" = "VLLM" ]; then
    local _prof_fields
    _prof_fields="$(env ${_recipe_env[@]+"${_recipe_env[@]}"} python3 - <<'PY' 2>/dev/null
names = set()
try:
    import dataclasses
    from vllm.config import ProfilerConfig
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
    case " $_prof_fields " in
      *" max_iterations "*) _bound="$_bound --profiler-config.max_iterations ${PROFILE_MAX_ITERS:-64}" ;;
    esac
    case " $_prof_fields " in
      *" delay_iterations "*) _bound="$_bound --profiler-config.delay_iterations ${PROFILE_DELAY_ITERS:-0}" ;;
    esac
    if [ -n "$_bound" ]; then
      echo ">>> magpie launcher: bounding the profiler buffer:$_bound"
      _extra_args="$_extra_args$_bound"
    else
      echo ">>> magpie launcher: this vllm build declares no profiler step bound;" \
           "the bench time window is the only cap on the trace." >&2
    fi
  fi

  # Map GEAK's env onto Magpie's server-phase env. Ordering IS the precedence
  # policy: recipe replay first, then the accepted env under test, then the
  # run-scoped names GEAK owns -- so a later layer knowingly overrides an
  # earlier one and nothing GEAK sets can be silently displaced by the recipe.
  # Overlay is prepended so the launch_server child imports the patched subtree
  # first. EXTRA_<BE>_ARGS carries the accepted extra flags; Magpie dedupes them
  # against its own DEFAULT_ARGS.
  #
  # GPU pinning follows the orchestrator's own convention: ROCR alone. ROCR
  # re-indexes the devices it exposes to 0..N-1, so a HIP_VISIBLE_DEVICES
  # carrying the same PHYSICAL ids indexes past the end of that list for any
  # allocation not starting at 0 -- GPU=2 exposes a single device numbered 0 and
  # then asks HIP for device 2. Magpie's script derives the logical HIP range
  # itself, but only while HIP is unset, so the inherited value is CLEARED
  # rather than merely left unassigned. ROCR alone pins correctly either way: a
  # script that derives nothing still sees exactly the allocated devices.
  # shellcheck disable=SC2086
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    ${_recipe_env[@]+"${_recipe_env[@]}"} $EXTRA_ENV \
    ROCR_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    MAGPIE_RUN_PHASE=server \
    MAGPIE_SERVER_PID_FILE="$_pidfile" \
    MODEL="$MODEL" \
    TP="$TP" \
    PORT="$PORT" \
    RESULT_DIR="$_out_dir" \
    SERVER_LOG="$LOG" \
    PROFILE="${PROFILE:-0}" \
    ${MAX_MODEL_LEN:+MAX_MODEL_LEN="$MAX_MODEL_LEN"} \
    ${PROFILE_DIR:+"${_prof_var}=$PROFILE_DIR"} \
    "${_args_var}=${_extra_args}" \
    bash "$script" >> "$_launchlog" 2>&1
  local rc=$?

  if [ "$rc" -ne 0 ]; then
    # Both logs: a script that died before starting the server (missing env,
    # failed download) left nothing in $LOG at all.
    echo "!!! magpie launcher: server-phase script exited $rc. Last launch log:" >&2
    tail -n 40 "$_launchlog" 2>/dev/null || true
    echo "!!! magpie launcher: last server log:" >&2
    tail -n 40 "$LOG" 2>/dev/null || true
    return 2
  fi
  if [ -f "$_pidfile" ]; then
    SERVER_PID="$(cat "$_pidfile" 2>/dev/null)"
  fi
  if [ -z "${SERVER_PID:-}" ]; then
    echo "!!! magpie launcher: no server pid in $_pidfile (server may not have started)." >&2
    tail -n 40 "$_launchlog" 2>/dev/null || true
    tail -n 40 "$LOG" 2>/dev/null || true
    return 2
  fi
  # This pid came from an EXTERNAL script's pid file, so we cannot assume it leads its
  # own group, and a stale file can name a pid that now belongs to someone else
  # entirely (worst case: the caller's orchestrator). Either case must DISABLE group
  # teardown here, at launch, rather than be discovered by a kill that already fired.
  local _mp_pgid _mp_args
  _mp_pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
  _mp_args="$(ps -o args= -p "$SERVER_PID" 2>/dev/null)"
  if [ "$_mp_pgid" != "$SERVER_PID" ]; then
    SERVER_GROUP_UNVERIFIED=1
    echo "!!! magpie launcher: pid $SERVER_PID does not lead its own group (pgid=${_mp_pgid:-?});" \
         "group teardown disabled for this launch." >&2
  fi
  # The "is this actually our server?" test is DERIVED from this run's own config
  # ($BACKEND, $PORT) — never a hard-coded backend list, which would silently
  # mis-judge every future Magpie backend the same way it mis-judges a stale pid.
  case "$_mp_args" in
    *"$BACKEND"*|*"$PORT"*) : ;;
    *)
      SERVER_GROUP_UNVERIFIED=1
      echo "!!! magpie launcher: pid $SERVER_PID matches neither BACKEND='$BACKEND' nor" \
           "PORT='$PORT' (args='$(printf '%s' "$_mp_args" | cut -c1-120)') — stale pid file?" \
           "group teardown disabled for this launch." >&2 ;;
  esac
  export SERVER_GROUP_UNVERIFIED
  echo ">>> magpie launcher: $BACKEND server up (pid $SERVER_PID, pgid=${_mp_pgid:-?}) via $(basename "$script")."
}
