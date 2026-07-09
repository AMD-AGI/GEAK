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
#   PROFILE PROFILE_DIR LOG OUT_DIR.
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
  rm -f "$_pidfile" 2>/dev/null || true

  # Per-backend var NAMES (regular rule), passed to the script via env NAME=VALUE.
  local _args_var="EXTRA_${backend_uc}_ARGS"
  local _prof_var="${backend_uc}_TORCH_PROFILER_DIR"

  # Map GEAK's env onto Magpie's server-phase env. Overlay is prepended so the
  # launch_server child imports the patched subtree first. EXTRA_<BE>_ARGS carries
  # the accepted extra flags; Magpie dedupes them against its own DEFAULT_ARGS.
  # shellcheck disable=SC2086
  env $EXTRA_ENV \
    HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    MAGPIE_RUN_PHASE=server \
    MAGPIE_SERVER_PID_FILE="$_pidfile" \
    MODEL="$MODEL" \
    TP="$TP" \
    PORT="$PORT" \
    RESULT_DIR="$_out_dir" \
    SERVER_LOG="$LOG" \
    PROFILE="${PROFILE:-0}" \
    ${PROFILE_DIR:+"${_prof_var}=$PROFILE_DIR"} \
    "${_args_var}=${EXTRA_SERVER_ARGS:-}" \
    bash "$script" >> "$LOG" 2>&1
  local rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "!!! magpie launcher: server-phase script exited $rc. Last log:" >&2
    tail -n 60 "$LOG" 2>/dev/null || true
    return 2
  fi
  if [ -f "$_pidfile" ]; then
    SERVER_PID="$(cat "$_pidfile" 2>/dev/null)"
  fi
  if [ -z "${SERVER_PID:-}" ]; then
    echo "!!! magpie launcher: no server pid in $_pidfile (server may not have started)." >&2
    tail -n 60 "$LOG" 2>/dev/null || true
    return 2
  fi
  echo ">>> magpie launcher: $BACKEND server up (pid $SERVER_PID) via $(basename "$script")."
}
