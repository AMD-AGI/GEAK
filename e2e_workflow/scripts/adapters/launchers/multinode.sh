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
