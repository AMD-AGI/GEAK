#!/usr/bin/env bash
# Remote server LAUNCHER adapter for bench_e2e.sh.  Sourced, never executed.
#
# One server instance on 1..N pinned GPU machines (tp_span: one TP group, a single
# machine included; pd_disagg: a prefill and a decode engine behind a load balancer).
# The whole leg -- health, profiler, launch, and the client unless it is the backend's
# own (below) -- runs on the driver, where GEAK runs; it reaches every rank over ssh and
# the server over the network. Only WHO starts the server changes; metric and
# acceptance are the single-host ones. Each rank either
# sources adapters/$BACKEND.sh (the copy beside this file) on its own node or runs the
# caller's provided launch script.
#
# Requires the pin the workflow's Setup writes from args.multinode (mn.py check ->
# mn_state.json, as GEAK_MN_STATE; bench_e2e.sh fills it in from `mn.py active-pin`),
# so the baseline and candidate legs land on the same hardware.
#
# HOST/PORT and the serving lock are set at SOURCE time: bench_e2e.sh reads them
# between sourcing this file and calling adapter_launch.

_MN_STATE="${GEAK_MN_STATE:-}"
if [ -z "$_MN_STATE" ] || [ ! -f "$_MN_STATE" ]; then
  echo "!!! multinode launcher: no pinned node set at '${_MN_STATE}' (GEAK_MN_STATE)." >&2
  echo "    The workflow's Setup writes it from args.multinode (mn.py check)." >&2
  exit 3
fi
export GEAK_MN_STATE="$_MN_STATE"
# bench_e2e.sh is copied into $EVAL_DIR without scripts/multi_node; the pin names it.
_MN_DIR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("mn_dir") or "")' \
            "$_MN_STATE" 2>/dev/null || echo "")"
if [ ! -f "$_MN_DIR/mn.py" ]; then
  echo "!!! multinode launcher: the pin $_MN_STATE names no mn_dir with mn.py ('$_MN_DIR'); re-run mn.py check." >&2
  exit 3
fi

# The pin's facts (_MN_HOST/PORT/TOPOLOGY/NNODES/SHARED_FS/LOCK_KEY/PD_ENGINES/LEASES).
# The client dials the master's cluster IP. pd_disagg: one "role=url" line per engine;
# HOST:PORT is the LB, which proxies neither the profiler nor per-engine health. On a
# pin it cannot serve from, leg-env exits non-zero before any lease or lock is taken.
_mn_leg_env="$(python3 "$_MN_DIR/mn.py" leg-env)" || exit 3
eval "$_mn_leg_env"
HOST="$_MN_HOST"
PORT="$_MN_PORT"
export HOST PORT
# Every rank sources its adapter from here (the eval_dir copy on the shared mount).
GEAK_MN_ADAPTERS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GEAK_MN_ADAPTERS

# The bench client runs where its dependencies are. The client adapters (inferencex, agentx)
# import no serving framework and run here, as Hyperloom's rebench does from its sandbox.
# The backend's own client (BENCH_CLIENT=native) imports sglang/vllm/atom, which do not
# import on a machine without a GPU, so it runs on rank0 with this shell's variables and
# functions; mn.py exec drops this host's paths and credentials. RESULT_JSONL and
# PROFILE_DIR are on the shared mount.
eval "_mn_adapter_bench() $(declare -f adapter_bench | sed '1d')"
if [ "${BENCH_CLIENT:-native}" = "native" ]; then
  echo ">>> multinode launcher: BENCH_CLIENT=native runs on rank0; it is not the client or the" \
       "vantage point Hyperloom measures with" >&2
  _mn_client() {
    (
      local _v _scalar='^declare -[^ raA]* '
      for _v in $(compgen -v); do
        [[ $_v =~ ^[A-Z_][A-Z0-9_]*$ && $(declare -p "$_v" 2>/dev/null) =~ $_scalar ]] && export "$_v"
      done
      exec python3 "$_MN_DIR/mn.py" exec --target rank0 --env --script - \
        <<< "$(declare -f)
_mn_adapter_bench $(printf '%q ' "$@")"
    )
  }
else
  _mn_client() { _mn_adapter_bench "$@"; }
fi
# PROF=1 (bench_e2e.sh's fallback when the windowed load ended early) never reaches a client's
# --profile: it is a fresh load with the window opened from here, which spans every rank.
adapter_bench() {
  [ "${3:-0}" = "1" ] || { _mn_client "$@"; return; }
  _mn_client "$1" "$2" 0 >/dev/null 2>&1 &
  local _load=$! _rc=1
  sleep "${PROFILE_WARMUP_SEC:-0}"
  if kill -0 "$_load" 2>/dev/null; then
    adapter_profile_window; _rc=$?
  else
    echo "!!! multinode: the profiling load ended before the window opened" >&2
  fi
  kill "$_load" 2>/dev/null || true; wait "$_load" 2>/dev/null || true
  return "$_rc"
}

[ "$_MN_TOPOLOGY" = "pd_disagg" ] && echo ">>> multinode launcher: pd_disagg topology; client" \
  "endpoint is the load balancer at $HOST:$PORT, engines:" $(echo "$_MN_PD_ENGINES" | tr '\n' ' ')
# A clamped port is one no rank binds.
if [ "$PORT" -lt "${PORT_BASE:-30000}" ] || [ "$PORT" -ge "$(( ${PORT_BASE:-30000} + ${PORT_SPAN:-1000} ))" ] 2>/dev/null; then
  echo ">>> multinode launcher: pinned port $PORT is outside ${PORT_BASE:-30000}..$(( ${PORT_BASE:-30000} + ${PORT_SPAN:-1000} - 1 ));" \
       "disabling the window policy so the pinned port is not clamped away." >&2
  PORT_ENFORCE_RANGE=0
  export PORT_ENFORCE_RANGE
fi

# On the shared mount (the /tmp default is per-container), keyed on the pinned
# node set so runs on disjoint pairs do not serialise.
if [ -z "${SERVING_GPU_LOCK:-}" ]; then
  SERVING_GPU_LOCK="${_MN_SHARED_FS}/geak_serving_mn${_MN_LOCK_KEY:+.$_MN_LOCK_KEY}.lock"
  export SERVING_GPU_LOCK
  echo ">>> multinode launcher: serving lock for this node set at $SERVING_GPU_LOCK"
fi

# The TERM handler reaches every rank over ssh; a 10s KILL would orphan them.
SERVER_STOP_GRACE_S="${SERVER_STOP_GRACE_S:-90}"
export SERVER_STOP_GRACE_S

# Trace wait: done once every expected trace is non-empty and the byte total has been
# quiet for SETTLE; quiet for QUIET with traces missing means a rank died.
_MN_TRACE_SETTLE_S=30
_MN_TRACE_QUIET_S=180
_MN_TRACE_TIMEOUT_S=1800
_MN_STOP_PROFILE_TIMEOUT_S=180
_MN_PD_HEALTH_TIMEOUT_S=10

# pd_disagg window: both engines are opened first and closed
# together, so their GPU-time totals describe the same interval. Prints role=url
# for every engine that opened one. Payload knobs read the backend adapter's env names.
_mn_pd_window() {
  local _acts="${SGLANG_PROFILE_ACTIVITIES:-[\"CPU\",\"GPU\"]}"
  local _stack="${SGLANG_PROFILE_WITH_STACK:-false}" _role _url _windowed="" _stop_rc
  # One directory per engine: both number their ranks TP-0..N.
  while IFS='=' read -r _role _url; do
    [ -n "$_url" ] || continue
    mkdir -p "$PROFILE_DIR/$_role"
    if curl -sf -X POST "${_url}/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\":\"${PROFILE_DIR}/${_role}\",\"num_steps\":${PROFILE_NUM_STEPS},\"record_shapes\":true,\"with_stack\":${_stack},\"activities\":${_acts}}" \
        >/dev/null 2>&1; then
      _windowed="${_windowed}${_role}=${_url}"$'\n'
    else
      echo "!!! multinode: /start_profile failed on the $_role engine ($_url)" >&2
    fi
  done <<< "$_MN_PD_ENGINES"
  [ -n "$_windowed" ] || return 1
  # Bounded by TIME and closed explicitly: num_steps counts forward steps, and
  # decode runs ~10x more of them than prefill, so prefill's window would still
  # be open at teardown and write nothing.
  sleep "${PROFILE_WINDOW_SEC:-20}"
  # Bounded: a wedged scheduler never answers /stop_profile.
  while IFS='=' read -r _role _url; do
    [ -n "$_url" ] || continue
    _stop_rc=0
    curl -sf -m "$_MN_STOP_PROFILE_TIMEOUT_S" -X POST "${_url}/stop_profile" \
      >/dev/null 2>&1 || _stop_rc=$?
    if [ "$_stop_rc" -eq 28 ]; then
      echo "!!! multinode: the $_role engine did not answer /stop_profile within" \
           "${_MN_STOP_PROFILE_TIMEOUT_S}s; its traces will be missing" >&2
    elif [ "$_stop_rc" -ne 0 ]; then
      echo ">>> multinode: the $_role engine had already closed its window" >&2
    fi
  done <<< "$_windowed"
  printf '%s' "$_windowed"
}

# adapter_profile_window: the backend's version returns on ONE new trace; here
# every rank's trace must finish flushing before the gate runs.
adapter_profile_window() {
  local _expect="$TP"
  local _before _after _deadline _depth=1 _windowed=""
  [ "$_MN_TOPOLOGY" = "pd_disagg" ] && _depth=2
  _mn_traces() { find "$PROFILE_DIR" -maxdepth "$_depth" -name '*.trace.json*' "$@" 2>/dev/null; }
  _before=$(_mn_traces | wc -l)

  if [ "$_MN_TOPOLOGY" = "pd_disagg" ]; then
    _windowed="$(_mn_pd_window)"
    if [ -z "$_windowed" ]; then
      echo "!!! multinode: no engine accepted a profiler window" >&2
      return 1
    fi
    local _etp
    # Each engine has its own TP (launch records it per engine); count only the engines
    # that opened a window.
    _etp="$(python3 -c 'import json,sys;e=(json.load(open(sys.argv[1])).get("pd") or {}).get("engines") or [];print(sum(int(x.get("tp") or 0) for x in e if x.get("role") in sys.argv[2:] and not x.get("replica")))' \
            "${OUT_DIR:-.}/mn_launch.json" $(echo "$_windowed" | cut -d= -f1) 2>/dev/null || echo 0)"
    [ "${_etp:-0}" -gt 0 ] && _expect="$_etp"
    echo ">>> multinode: profiler window closed on $(grep -c = <<< "$_windowed") engine(s);" \
         "expecting $_expect rank trace(s)"
  else
    adapter_profile_window_native || {
      echo "!!! multinode: the profiler window failed on the master rank" >&2
      return 1
    }
  fi

  # Done = every expected file non-empty AND the byte total quiet for
  # _MN_TRACE_SETTLE_S: torch creates each file before streaming into it.
  local _settle="$_MN_TRACE_SETTLE_S" _quiet="$_MN_TRACE_QUIET_S"
  local _interval=5 _bytes _last_bytes=-1 _stable=0 _nochange=0
  _deadline=$(( $(date +%s) + _MN_TRACE_TIMEOUT_S ))
  while [ "$(date +%s)" -lt "$_deadline" ]; do
    _after=$(_mn_traces -size +1c | wc -l)
    _bytes=$(_mn_traces -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    if [ "$_bytes" != "$_last_bytes" ]; then
      _last_bytes="$_bytes"; _stable=0; _nochange=0
    else
      _stable=$(( _stable + _interval )); _nochange=$(( _nochange + _interval ))
    fi
    if [ "$(( _after - _before ))" -ge "$_expect" ] && [ "$_stable" -ge "$_settle" ]; then
      break
    fi
    # Quiet but incomplete: a rank died. Let the gate describe what is there.
    if [ "$_nochange" -ge "$_quiet" ]; then
      echo "!!! multinode: $(( _after - _before ))/$_expect trace(s) and no new" \
           "bytes for ${_quiet}s ($_bytes bytes total); giving up on the rest" >&2
      break
    fi
    sleep "$_interval"
  done
  _after=$(_mn_traces -size +1c | wc -l)
  echo ">>> multinode: $(( _after - _before ))/$_expect rank trace(s) landed in" \
       "$PROFILE_DIR ($(( ${_last_bytes:-0} / 1048576 )) MiB total," \
       "quiet for ${_stable}s)"

  local _launch_json="${OUT_DIR:-.}/mn_launch.json"
  [ "$_MN_TOPOLOGY" = "pd_disagg" ] && [ -f "$_launch_json" ] || _launch_json=""
  PROFILE_DIR="$PROFILE_DIR" TP="$TP" \
    python3 "$_MN_DIR/mn.py" trace-gate \
      --profile-dir "$PROFILE_DIR" --expect-ranks "$_expect" \
      --nodes "$_MN_NNODES" \
      ${_launch_json:+--launch-json "$_launch_json"}
  local _rc=$?
  # Non-fatal: a partial set still ranks; the verdict is on disk.
  if [ "$_rc" -ne 0 ]; then
    echo "!!! multinode: the trace set did NOT pass the gate; see" \
         "$PROFILE_DIR/ranks/mn_trace_report.json before trusting the Top-N." >&2
  fi
  [ "$(( _after - _before ))" -gt 0 ]
}

# pd_disagg: the router answers /health before any worker is reachable, so the LB
# AND every engine replica must answer.
if [ "$_MN_TOPOLOGY" = "pd_disagg" ]; then
  adapter_health() {
    local _url
    curl -sf "${BASE_URL}/health" >/dev/null 2>&1 || return 1
    printf '%s\n' "${_MN_PD_HEALTH:-}" | while read -r _url; do
      [ -n "$_url" ] || continue
      curl -sf --max-time "$_MN_PD_HEALTH_TIMEOUT_S" "${_url}/health" \
        >/dev/null 2>&1 || exit 1
    done
  }
fi

# GEAK_MN_ROCM_RUNTIME=<dir> (opt-in): engines load HIP/ROCr/roctracer from <dir>
# via LD_LIBRARY_PATH (the 7.2.0 image's profiler drops hipGraph-replayed kernels).
# <dir> must exist at the same path on every pinned node. Prints the EXTRA_ENV.
_mn_rocm_runtime_env() {
  local _rt="${GEAK_MN_ROCM_RUNTIME:-}" _tok _rest="" _base="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
  if [ -z "$_rt" ]; then printf '%s' "${EXTRA_ENV:-}"; return 0; fi
  if [ ! -e "$_rt/libamdhip64.so.7" ]; then
    echo "!!! multinode launcher: GEAK_MN_ROCM_RUNTIME=$_rt has no libamdhip64.so.7" >&2
    return 1
  fi
  for _tok in ${EXTRA_ENV:-}; do
    case "$_tok" in LD_LIBRARY_PATH=*) _base="${_tok#LD_LIBRARY_PATH=}" ;; *) _rest="$_rest $_tok" ;; esac
  done
  case ":$_base:" in *":$_rt:"*) ;; *) _base="$_rt:$_base" ;; esac
  printf '%s' "LD_LIBRARY_PATH=$_base$_rest"
}

adapter_launch() {
  local _out_dir="${OUT_DIR:-$(pwd)}"
  # Both logs stay OUT of $LOG: bench_e2e.sh greps and size-watches it.
  local _launchlog="$_out_dir/mn_launch.log"
  local _suplog="$_out_dir/mn_supervisor.log"
  local _env
  _env="$(_mn_rocm_runtime_env)" || return 2
  [ -n "${GEAK_MN_ROCM_RUNTIME:-}" ] && echo ">>> multinode launcher: engines use the ROCm runtime in" \
    "$GEAK_MN_ROCM_RUNTIME ($(basename "$(readlink -f "$GEAK_MN_ROCM_RUNTIME/libamdhip64.so.7")"))"
  # Lease every card of every pinned node (a provided script picks its own cards) through
  # the files `mn.py exec --gpu` leases, so no GEAK kernel task runs on a card this leg serves
  # from. The fds stay open in this shell (and the supervisor) until the leg ends.
  local _lease _fd
  if [ -z "$_MN_LEASES" ]; then
    echo "!!! multinode launcher: the pin $_MN_STATE lists no cards; re-run mn.py check." >&2
    return 2
  fi
  while IFS= read -r _lease; do
    exec {_fd}>"$_lease"
    flock -n "$_fd" && continue
    echo ">>> multinode launcher: waiting for $(basename "$_lease" .lock), leased by a GEAK kernel task"
    flock -w "${SERVING_LOCK_WAIT:-7200}" "$_fd" || {
      echo "!!! multinode launcher: $(basename "$_lease" .lock) still leased after ${SERVING_LOCK_WAIT:-7200}s" >&2
      return 2
    }
  done <<< "$_MN_LEASES"
  echo ">>> multinode launcher: starting $(basename "$_MN_STATE") topology; launch log: $_launchlog"
  ( cd "$_MN_DIR" && EXTRA_ENV="$_env" python3 mn.py launch ) >> "$_launchlog" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! multinode launcher: launch failed (rc=$rc). Last launch log:" >&2
    tail -n 60 "$_launchlog" >&2 2>/dev/null || true
    echo "!!! multinode launcher: last rank0 log:" >&2
    tail -n 40 "$LOG" >&2 2>/dev/null || true
    return 2
  fi
  grep -E '^>>> multinode (rank|:)' "$_launchlog" 2>/dev/null | tail -n 8 || true

  # The supervisor is the pid bench_e2e.sh owns; its TERM handler reaps the ranks.
  ${SERVER_LAUNCH_PREFIX:-} env \
    OUT_DIR="$_out_dir" \
    PYTHONPATH="$_MN_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    python3 "$_MN_DIR/mn.py" supervise >> "$_suplog" 2>&1 &
  SERVER_PID=$!

  # A supervisor that died on import must not be handed over as a live server.
  sleep 2
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!!! multinode launcher: the supervisor exited immediately; see $_suplog" >&2
    tail -n 30 "$_suplog" >&2 2>/dev/null || true
    echo "!!! The ranks may be RUNNING but unsupervised. Reaping them now." >&2
    python3 "$_MN_DIR/mn.py" stop "$_out_dir" >&2 2>&1 || true
    SERVER_PID=""
    return 2
  fi
  echo ">>> multinode launcher: ranks spawned, supervisor pid $SERVER_PID," \
       "endpoint $BASE_URL (cold start is minutes; the health loop decides)."
}
