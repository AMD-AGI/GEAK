#!/usr/bin/env bash
# Backend-agnostic e2e serving benchmark dispatcher for e2e_workflow.
#
# ONE script the Director, Profiler, Config Tuner, and e2e Integrator all share so every throughput
# number is measured the SAME way (warm server, fixed ISL/OSL/conc, repeated, median reported). The
# serving STACK (sglang / vllm / ...) is NOT baked in here — it lives in scripts/adapters/<backend>.sh.
# This dispatcher owns only the stack-INDEPENDENT parts:
#   * server lifecycle (launch / health-wait / cleanup), or reuse of a warm server (REUSE_SERVER=1),
#   * warmup (never timed) + N timed repeats + optional bounded profiling trace,
#   * median throughput + spread summary (one machine-readable line + JSON).
# It is config-driven by env so an agent can vary ONE axis at a time. Nothing is model-specific.
#
# The adapter contract (each scripts/adapters/<BACKEND>.sh must define):
#   adapter_default_port            -> echo a sensible default port for this stack
#   adapter_launch                  -> launch the server in background; set global SERVER_PID; write $LOG.
#                                      Reads: MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS
#                                             EXTRA_ENV OVERLAY_PYTHONPATH PROFILE PROFILE_DIR
#                                      MUST launch through the shared prefix:
#                                        ${SERVER_LAUNCH_PREFIX:-} env ... <server> ... & SERVER_PID=$!
#                                      That prefix (server_teardown.sh) puts the server in its OWN
#                                      session, which is the ONLY thing that lets teardown PROVE the
#                                      process group belongs to this launch and reap the whole tree.
#                                      An adapter that launches without it still works, but its
#                                      teardown degrades to pid+descendants. A launcher that cannot
#                                      control the launch (it delegates to an external script) must
#                                      instead set SERVER_GROUP_UNVERIFIED=1 unless it can show the
#                                      pid leads its own group.
#   adapter_health                  -> return 0 iff $BASE_URL is serving (e.g. curl /health)
#   adapter_bench  NUMP MAXC PROF   -> run ONE bench (random ISL/OSL), append a result JSON line to
#                                      $RESULT_JSONL with canonical keys (output_throughput,
#                                      median_ttft_ms, median_tpot_ms). PROF=1 => also emit a trace
#                                      into $PROFILE_DIR. Honors optional REQUEST_RATE (req/s; empty=inf)
#                                      to stagger arrivals.
#   adapter_profile_window          -> OPTIONAL. Capture a profiler window (record_shapes) on the
#                                      ALREADY-RUNNING, warm, mid-load server, so the trace is the real
#                                      steady-state prefill+decode MIX rather than a cold prefill ramp.
#                                      The window is sized per-backend: sglang by PROFILE_NUM_STEPS (its
#                                      /start_profile takes num_steps); vllm by PROFILE_WINDOW_SEC (its
#                                      /start_profile has no step count, so start->sleep->stop). If
#                                      undefined, the PROFILE step falls back to a (less faithful)
#                                      saturated PROF=1 bench.
#
# KEY OUTPUTS (written to $OUT_DIR):
#   server_start.json      {status, reason, phase_hint, wait_sec, ceiling_sec, ...} — ALWAYS
#                          written when we launch, so a failed cold start is a readable REASON
#                          downstream instead of a silently empty output dir
#   bench_runs.jsonl       one bench result object per repeat
#   bench_summary.json     {throughput_tok_s_median (metric-neutral; see metric_basis), metric_basis,
#                           ttft_ms_median, tpot_ms_median, spread, runs}  (E2E_METRIC=output default)
#   SUMMARY line on stdout: "E2E_SUMMARY <metric_basis>=<median> spread=<pct> ttft_ms=<med> tpot_ms=<med>"
#   profile/                trace (if PROFILE=1)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- isolated-server measurement protocol ----
# Keep the existing body below as the single-server implementation.  In isolated
# mode this process is only a scheduler: each attempt runs bench_replica.sh,
# which re-enters this script in legacy mode for exactly one fresh-server
# warmup+measurement lifecycle.
# Profiling is not a throughput replica: it intentionally needs one warm server
# and a sustained load/window.  An aligned run exports isolated mode globally,
# so profile invocations opt back into the existing single-server body here.
if [ "${GEAK_REPEAT_MODE:-legacy}" = "isolated_server" ] && [ "${PROFILE:-0}" = "1" ]; then
  echo ">>> PROFILE=1: using the single-server profiling lifecycle (not a timed replica)."
  GEAK_REPEAT_MODE=legacy
fi
if [ "${GEAK_REPEAT_MODE:-legacy}" = "isolated_server" ]; then
  _replica_runner="$HERE/bench_replica.sh"
  if [ ! -f "$_replica_runner" ]; then
    echo "!!! GEAK_REPEAT_MODE=isolated_server requires $_replica_runner" >&2
    exit 3
  fi
  if [ "${REUSE_SERVER:-0}" = "1" ]; then
    echo "!!! REUSE_SERVER=1 is incompatible with GEAK_REPEAT_MODE=isolated_server; timed replicas must fresh-launch." >&2
    exit 4
  fi
  _purpose="${MEASUREMENT_PURPOSE:-search}"
  if [ -n "${REPEATS+x}" ]; then
    _requested="$REPEATS"
  elif [ -n "${REPLICAS+x}" ]; then
    _requested="$REPLICAS"
  else
    case "$_purpose" in
      validation) _requested=3 ;;
      parity|search|"") _requested=1 ;;
      *)
        echo ">>> Unknown MEASUREMENT_PURPOSE='$_purpose'; using 1 isolated replica." >&2
        _requested=1 ;;
    esac
  fi
  case "$_requested" in
    ''|*[!0-9]*|0)
      echo "!!! REPEATS must be a positive integer in isolated-server mode (got '$_requested')." >&2
      exit 4 ;;
  esac

  _aggregate_out="${OUT_DIR:-$(pwd)/e2e_bench_out}"
  mkdir -p "$_aggregate_out"
  : > "$_aggregate_out/bench_runs.jsonl"
  echo "Measurement: isolated_server  purpose=$_purpose  requested_replicas=$_requested"
  _successful=0
  for ((_replica=1; _replica<=_requested; _replica++)); do
    _replica_dir="$_aggregate_out/replica_$(printf '%03d' "$_replica")"
    mkdir -p "$_replica_dir"
    rm -f "$_replica_dir/selected_summary.json" "$_replica_dir/selected_attempt"
    _replica_ok=0
    for _attempt in 1 2; do
      _attempt_dir="$_replica_dir/attempt_$_attempt"
      rm -f "$_attempt_dir/bench_summary.json"
      echo ">>> Isolated replica $_replica/$_requested (attempt $_attempt/2) ..."
      OUT_DIR="$_attempt_dir" REPLICA_INDEX="$_replica" REPLICA_ATTEMPT="$_attempt" \
        bash "$_replica_runner"
      _rc=$?
      if [ "$_rc" -eq 0 ] && python3 - "$_attempt_dir/bench_summary.json" "${EFFECTIVE_CONFIG_DIGEST:-}" <<'PY'
import json, sys
try:
    summary = json.load(open(sys.argv[1]))
    value = summary.get("throughput_tok_s_median")
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    ok = ok and int(summary.get("runs", 0)) == 1
    expected_digest = sys.argv[2]
    if expected_digest:
        ok = ok and summary.get("effective_config_digest") == expected_digest
except (OSError, ValueError, TypeError):
    ok = False
raise SystemExit(0 if ok else 1)
PY
      then
        cp "$_attempt_dir/bench_summary.json" "$_replica_dir/selected_summary.json"
        if [ -f "$_attempt_dir/bench_runs.jsonl" ]; then
          cat "$_attempt_dir/bench_runs.jsonl" >> "$_aggregate_out/bench_runs.jsonl"
        fi
        printf '%s\n' "$_attempt" > "$_replica_dir/selected_attempt"
        _replica_ok=1
        _successful=$((_successful + 1))
        break
      fi
      echo "!!! Isolated replica $_replica attempt $_attempt failed (rc=$_rc)." >&2
    done
    if [ "$_replica_ok" != "1" ]; then
      echo "!!! Isolated replica $_replica failed after one retry; continuing without same-server fallback." >&2
    fi
  done

  python3 - "$_aggregate_out" "$_requested" "$_successful" "$_purpose" "${EFFECTIVE_CONFIG_DIGEST:-}" <<'PY'
import glob
import json
import os
import statistics
import sys

out_dir, requested_s, successful_s, purpose, effective_digest = sys.argv[1:]
requested, successful = int(requested_s), int(successful_s)
selected = sorted(glob.glob(os.path.join(out_dir, "replica_*", "selected_summary.json")))
summaries = []
replicas = []
for path in selected:
    replica_dir = os.path.dirname(path)
    replica_index = int(os.path.basename(replica_dir).split("_")[-1])
    if replica_index > requested:
        continue
    with open(path) as fh:
        item = json.load(fh)
    summaries.append(item)
    try:
        attempt = int(open(os.path.join(replica_dir, "selected_attempt")).read().strip())
    except (OSError, ValueError):
        attempt = None
    replicas.append({
        "replica": replica_index,
        "attempt": attempt,
        "throughput_tok_s": item.get("throughput_tok_s_median"),
    })

def numbers(key):
    return [
        float(item[key]) for item in summaries
        if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool)
    ]

def median(key):
    values = numbers(key)
    return round(statistics.median(values), 3) if values else None

tputs = numbers("throughput_tok_s_median")
observed = round(statistics.median(tputs), 3) if tputs else None
if len(tputs) < 2 or not observed:
    spread = 0.0
else:
    spread = round(100.0 * (max(tputs) - min(tputs)) / observed, 2)
complete = successful == requested
metric_bases = {item.get("metric_basis") for item in summaries if item.get("metric_basis")}
metric_basis = next(iter(metric_bases)) if len(metric_bases) == 1 else None
summary = {
    "requested": requested,
    "successful": successful,
    "requested_replicas": requested,
    "successful_replicas": successful,
    "status": "complete" if complete else "incomplete",
    "usable_for_acceptance": complete and observed is not None,
    "measurement_mode": "isolated_server",
    "measurement_purpose": purpose,
    "effective_config_digest": effective_digest or None,
    "observed_median": observed,
    "throughput_tok_s_median": observed,
    "throughput_tok_s_spread_pct": spread,
    "output_throughput_tok_s_median": (
        observed if metric_basis == "aggregate_output_tok_s" else None
    ),
    "output_throughput_tok_s_spread_pct": (
        spread if metric_basis == "aggregate_output_tok_s" else None
    ),
    "ttft_ms_median": median("ttft_ms_median"),
    "tpot_ms_median": median("tpot_ms_median"),
    "runs": successful,
    "all_throughput": tputs,
    "metric_basis": metric_basis,
    "replicas": replicas,
}
with open(os.path.join(out_dir, "bench_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(
    f"E2E_SUMMARY {metric_basis or 'unknown'}={observed} spread={spread}% "
    f"requested={requested} successful={successful} status={summary['status']} "
    f"usable_for_acceptance={str(summary['usable_for_acceptance']).lower()} "
    "measurement_mode=isolated_server"
)
PY
  echo ">>> Done. Summary: $_aggregate_out/bench_summary.json"
  # A degraded leg remains observable: callers consume status=incomplete and
  # the successful-replica median.  Only a leg with no measurement at all is a
  # process-level failure.
  [ "$_successful" -gt 0 ] && exit 0
  exit 2
fi

# ---- backend selection (the only thing that picks the stack) ----
BACKEND=${BACKEND:-sglang}
ADAPTER="${ADAPTER:-$HERE/adapters/${BACKEND}.sh}"
if [ ! -f "$ADAPTER" ]; then
  echo "!!! No adapter for BACKEND='$BACKEND' at $ADAPTER" >&2
  echo "    Available: $(ls "$HERE"/adapters/*.sh 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' ')" >&2
  exit 3
fi
# shellcheck disable=SC1090
source "$ADAPTER"
for fn in adapter_launch adapter_health adapter_bench; do
  if ! declare -F "$fn" >/dev/null; then
    echo "!!! Adapter $ADAPTER does not define $fn()" >&2; exit 3
  fi
done

# ---- optional bench-CLIENT override (server stack stays the BACKEND above) ----
# The serving server is always launched by the backend adapter (sglang/vllm).
# BENCH_CLIENT swaps ONLY the client that drives the benchmark, so a run can use
# the EXACT same client as another harness. BENCH_CLIENT=inferencex => Hyperloom/
# Magpie's own InferenceX benchmark_serving.py (measurement-protocol-identical client). Default
# 'native' keeps each backend's built-in bench (sglang.bench_serving / vllm).
BENCH_CLIENT=${BENCH_CLIENT:-native}
copy_function() {  # copy_function SRC DST — clone a shell function under a new name
  declare -F "$1" >/dev/null || return 1
  eval "${2}() $(declare -f "$1" | sed '1d')"
}
if [ "$BENCH_CLIENT" != "native" ]; then
  CLIENT_ADAPTER="${CLIENT_ADAPTER:-$HERE/adapters/clients/${BENCH_CLIENT}.sh}"
  if [ ! -f "$CLIENT_ADAPTER" ]; then
    echo "!!! No bench client '$BENCH_CLIENT' at $CLIENT_ADAPTER" >&2
    echo "    Available: $(ls "$HERE/adapters/clients" 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    exit 3
  fi
  # Preserve the backend's native bench so the client can delegate profiling
  # (server-side trace hooks live in the native bench, not the portable client).
  copy_function adapter_bench adapter_bench_native
  # shellcheck disable=SC1090
  source "$CLIENT_ADAPTER"   # MUST redefine adapter_bench (the timed client)
  if ! declare -F adapter_bench >/dev/null; then
    echo "!!! Client adapter $CLIENT_ADAPTER must define adapter_bench()" >&2; exit 3
  fi
fi

# ---- optional server-LAUNCHER override (align the SERVER launch RECIPE with an
# external harness, e.g. Hyperloom/Magpie, so the served stack is byte-identical:
# same --mem-fraction-static / --disable-radix-cache / --trust-remote-code /
# SGLANG_USE_AITER / firmware-gated envs). The serving STACK is STILL the BACKEND
# above; this hook only changes WHO runs launch_server. Default 'native' keeps each
# backend adapter's own adapter_launch (byte-identical to before). BENCH_LAUNCHER=
# <name> sources adapters/launchers/<name>.sh which MUST redefine adapter_launch;
# the native launch/health are preserved as adapter_launch_native / adapter_health_native
# so a launcher adapter can DELEGATE or FALL BACK. The authored-kernel OVERLAY
# (OVERLAY_PYTHONPATH) is applied BY the launcher (an external harness usually
# cannot), so overlay + recipe-parity coexist. Only affects a FRESH launch
# (REUSE_SERVER=0); nothing else in the measurement changes.
BENCH_LAUNCHER=${BENCH_LAUNCHER:-native}
if [ "$BENCH_LAUNCHER" != "native" ]; then
  LAUNCHER_ADAPTER="${LAUNCHER_ADAPTER:-$HERE/adapters/launchers/${BENCH_LAUNCHER}.sh}"
  if [ ! -f "$LAUNCHER_ADAPTER" ]; then
    echo "!!! No server launcher '$BENCH_LAUNCHER' at $LAUNCHER_ADAPTER" >&2
    echo "    Available: $(ls "$HERE/adapters/launchers" 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    exit 3
  fi
  # Preserve the backend's native launch/health so the launcher can delegate to
  # them (e.g. fall back when the external recipe/script is unavailable).
  copy_function adapter_launch adapter_launch_native
  copy_function adapter_health adapter_health_native
  # shellcheck disable=SC1090
  source "$LAUNCHER_ADAPTER"   # MUST redefine adapter_launch (server lifecycle)
  if ! declare -F adapter_launch >/dev/null; then
    echo "!!! Launcher adapter $LAUNCHER_ADAPTER must define adapter_launch()" >&2; exit 3
  fi
fi

# ---- model / server ----
# MODEL is REQUIRED. No rig-specific default — a wrong-but-silent default benches the wrong target.
MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  echo "!!! MODEL is required (path or HF id). e.g. MODEL=/path/to/model bash bench_e2e.sh" >&2
  exit 4
fi
HOST=${HOST:-127.0.0.1}
TP=${TP:-1}
GPU=${GPU:-0}
MEM_FRACTION=${MEM_FRACTION:-0.9}    # match infer.sh (no --gpu-memory-utilization => vllm default 0.9)
# GPU allow-list (only enforced when ALLOWED_GPUS is set → default behavior unchanged): refuse to launch
# on any GPU id not in the comma-separated list, so a run pinned to GPUs 4-7 can't spill onto others.
if [ -n "${ALLOWED_GPUS:-}" ] && [ "${REUSE_SERVER:-0}" != "1" ]; then
  _allow=",$(echo "$ALLOWED_GPUS" | tr -d ' '),"
  for _g in $(echo "$GPU" | tr ',' ' '); do
    case "$_allow" in
      *",$_g,"*) : ;;
      *) echo "!!! GPU '$_g' not in ALLOWED_GPUS='$ALLOWED_GPUS' — refusing to launch (resource allow-list)." >&2; exit 5 ;;
    esac
  done
fi
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}    # e.g. "--attention-backend triton"
# EXTRA_ENV is applied to the SERVER launch line, space-separated KEY=VAL pairs:
#   EXTRA_ENV="SGLANG_USE_AITER=1 HIPBLASLT_TUNING_FILE=/path/tune.dat"
EXTRA_ENV=${EXTRA_ENV:-}
# OVERLAY_PYTHONPATH: prepend an overlay dir so a patched subtree / monkeypatch loads first.
OVERLAY_PYTHONPATH=${OVERLAY_PYTHONPATH:-}

# ---- port: auto-allocate a free one if not pinned (avoids 30000 collisions on shared boxes) ----
# Constrained auto-allocation: pick a free port inside [PORT_BASE, PORT_BASE+PORT_SPAN) so a run can be
# pinned to a required window (policy: "ports must start with 40"). Default base 40000. An explicit PORT
# OUTSIDE the window is clamped (ignored + re-allocated) unless PORT_ENFORCE_RANGE=0. Port number does not
# affect throughput, so this never changes optimization results.
# RIG CONSTRAINT (deep_mode M3 run): every port MUST start with 30 -> window 30000..30999.
PORT_BASE=${PORT_BASE:-30000}
PORT_SPAN=${PORT_SPAN:-1000}
PORT_ENFORCE_RANGE=${PORT_ENFORCE_RANGE:-1}
PORT=${PORT:-}
if [ -n "$PORT" ] && [ "$PORT_ENFORCE_RANGE" = "1" ]; then
  if [ "$PORT" -lt "$PORT_BASE" ] || [ "$PORT" -ge "$((PORT_BASE+PORT_SPAN))" ] 2>/dev/null; then
    echo "!!! PORT=$PORT outside required window ${PORT_BASE}..$((PORT_BASE+PORT_SPAN-1)); ignoring + auto-allocating in range."
    PORT=""
  fi
fi
if [ -z "$PORT" ]; then
  # RIG CONSTRAINT (M3 run): scan [PORT_BASE, PORT_BASE+PORT_SPAN) = 2000..2099 (every port starts with
  # 20). PORT+10000=12099 << 65535, so also safe for sglang's gRPC-port derivation that upstream guards.
  FREE_PORT=$(PORT_BASE="$PORT_BASE" PORT_SPAN="$PORT_SPAN" python3 - <<'PY' 2>/dev/null || true
import os, socket, random
base=int(os.environ.get("PORT_BASE","40000")); span=int(os.environ.get("PORT_SPAN","1000"))
order=list(range(span)); random.shuffle(order)
for off in order:
    p=base+off
    s=socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close(); continue
PY
)
  if [ -z "$FREE_PORT" ]; then
    echo "!!! No free port in ${PORT_BASE}..$((PORT_BASE+PORT_SPAN-1)); falling back to OS-assigned (may violate range)."
    FREE_PORT=$(python3 - <<'PY' 2>/dev/null || true
import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
)
  fi
  [ -n "$FREE_PORT" ] && PORT="$FREE_PORT"
fi

# ---- workload ----
ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONC=${CONC:-64}
# NUM_PROMPTS default.
#  * native client (standalone GEAK default): keep the original CONC*5 default so
#    standalone behaviour is byte-identical to before the inferencex integration.
#  * inferencex client (Hyperloom/Magpie measurement-protocol alignment): default to
#    Magpie's FIXED CONC*10 (its run_benchmark_serving default is
#    `--num-prompts $((CONC*10))`), so a GEAK measurement matches the Magpie baseline
#    prompt count exactly — a differing prompt count changes the saturation regime and
#    hence the tok/s, so this is a real alignment knob, not cosmetic.
#    Opt-out: NUM_PROMPTS_ADAPTIVE=1 restores the cost-bounded ADAPTIVE factor that
#    scales DOWN as per-request seq cost (ISL+OSL) grows {<=1024:10,<=4096:5,<=16384:3,else 2},
#    for long-sequence standalone runs where CONC*10 is too expensive.
# An explicit NUM_PROMPTS (e.g. Hyperloom's apply_bench_protocol forwarding its own
# measured count) ALWAYS wins over both defaults.
if [ -z "${NUM_PROMPTS:-}" ]; then
  if [ "$BENCH_CLIENT" = "inferencex" ]; then
    if [ "${NUM_PROMPTS_ADAPTIVE:-0}" = "1" ]; then
      _seq_cost=$((ISL + OSL))
      if   [ "$_seq_cost" -le 1024 ];  then _factor=10
      elif [ "$_seq_cost" -le 4096 ];  then _factor=5
      elif [ "$_seq_cost" -le 16384 ]; then _factor=3
      else _factor=2; fi
      NUM_PROMPTS=$(( CONC * _factor > CONC ? CONC * _factor : CONC ))
    else
      NUM_PROMPTS=$(( CONC * 10 ))   # Magpie parity (fixed)
    fi
  else
    NUM_PROMPTS=$((CONC * 5))
  fi
fi
# Client-side warmup prompts (measurement-protocol alignment with Hyperloom's materialize default
# NUM_WARMUPS=min(CONC,8)). Consumed by the inferencex client adapter; the native
# adapters use their own warmup round instead.
NUM_WARMUPS=${NUM_WARMUPS:-$(( CONC < 8 ? CONC : 8 ))}
# RANDOM_RANGE_RATIO / NUM_PROMPTS / NUM_WARMUPS / SEED are the measurement protocol.
# These are STANDALONE defaults: when an external orchestrator (Hyperloom) drives
# the run it exports its own values (interface/run_e2e.py:apply_bench_protocol from
# handoff.bench_protocol) and they override these via the env. Do NOT hard-code a
# value assuming the caller's measurement protocol — ratio=0 is fixed-length, ratio>0 is variable
# (lengths sampled in [(1-ratio)*len, (1+ratio)*len]), and the caller may use
# either. Standalone default = fixed-length (matches infer.sh --random-range-ratio 0).
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0}
REPEATS=${REPEATS:-3}                 # repeat the bench this many times; report median + spread
SEED=${SEED:-0}                       # fixed seed for reproducibility / parity

# ---- client trust-remote-code (general, model-agnostic) ----
# The benchmark CLIENT loads the model's tokenizer; for custom-tokenizer models
# transformers raises ValueError unless trust_remote_code is allowed. Mirror the
# SERVER's effective trust setting across recipe args and EXTRA_SERVER_ARGS,
# including a later --no-trust-remote-code override. The client measuring it
# must use the same state. Stays OFF (no implicit remote-code execution) for
# models that don't need it. An explicit caller value always wins.
_args_trust_state() {
  # Print the effective state contributed by this argument string: "1" for
  # enable, "0" for disable, or nothing when it carries no trust option. vLLM
  # exposes argparse.BooleanOptionalAction spellings --trust-remote-code and
  # --no-trust-remote-code; it rejects `--trust-remote-code=true/false`, so those
  # lookalikes must not influence the benchmark client. SGLang/Magpie recipes
  # also use the underscore spelling, whose matching --no_ form is accepted here.
  #
  # Keep scanning after a match: argparse applies repeated store-style boolean
  # options in argv order, so the LAST enable/disable token is authoritative.
  # `read -ra` splits without pathname expansion.
  local _args=${1//$'\n'/ } _tok _state=""
  local -a _toks=()
  read -ra _toks <<< "$_args"
  for _tok in "${_toks[@]}"; do
    case "$_tok" in
      --trust-remote-code|--trust_remote_code)
        _state=1 ;;
      --no-trust-remote-code|--no_trust_remote_code)
        _state=0 ;;
    esac
  done
  printf '%s' "$_state"
}
if [ -z "${BENCH_TRUST_REMOTE_CODE:-}" ]; then
  BENCH_TRUST_REMOTE_CODE=0
  # The server may inherit --trust-remote-code from the REPLAYED recipe env
  # (EXTRA_<BE>_ARGS recorded by the orchestrator) rather than from
  # EXTRA_SERVER_ARGS. The client that measures such a server has to trust the
  # same remote code, so also honor a trust setting recorded in the recipe env
  # file (NUL-delimited). Parse it BY KEY: a value-blind substring match fails
  # OPEN -- e.g. `DO_NOT_TRUST_REMOTE_CODE=1` or a disabling `HF_HUB_TRUST_REMOTE_CODE=0`
  # would both flip trust ON. Enable only when a KNOWN trust control is set to a
  # truthy value, or when the CURRENT backend's recorded EXTRA_<BE>_ARGS value
  # carries an actual enable/disable token. Apply layers in the same order as
  # the launcher: recipe controls -> recipe args -> GEAK EXTRA_SERVER_ARGS.
  _recipe_trust_control=0
  _recipe_trust_state=""
  _trust_extra_name="EXTRA_${BACKEND^^}_ARGS"
  if [ -n "${RECIPE_ENV_FILE:-}" ] && [ -f "${RECIPE_ENV_FILE}" ]; then
    while IFS= read -r -d '' _trust_kv; do
      _trust_name=${_trust_kv%%=*}
      _trust_val=${_trust_kv#*=}
      case "$_trust_name" in
        BENCH_TRUST_REMOTE_CODE|HF_HUB_TRUST_REMOTE_CODE|MAGPIE_TRUST_REMOTE_CODE|TRANSFORMERS_TRUST_REMOTE_CODE)
          case "${_trust_val,,}" in
            1|true|yes|on) _recipe_trust_control=1 ;;
          esac ;;
        "$_trust_extra_name")
          _state=$(_args_trust_state "$_trust_val")
          [ -n "$_state" ] && _recipe_trust_state="$_state" ;;
      esac
    done < "$RECIPE_ENV_FILE"
  fi
  [ "$_recipe_trust_control" = "1" ] && BENCH_TRUST_REMOTE_CODE=1
  [ -n "$_recipe_trust_state" ] && BENCH_TRUST_REMOTE_CODE="$_recipe_trust_state"
  _geak_trust_state=$(_args_trust_state "${EXTRA_SERVER_ARGS:-}")
  [ -n "$_geak_trust_state" ] && BENCH_TRUST_REMOTE_CODE="$_geak_trust_state"
  unset _trust_kv _trust_name _trust_val _trust_extra_name _state
  unset _recipe_trust_control _recipe_trust_state _geak_trust_state
fi
# transformers / HF hub honor HF_HUB_TRUST_REMOTE_CODE for tokenizer auto-load.
[ "$BENCH_TRUST_REMOTE_CODE" = "1" ] && HF_HUB_TRUST_REMOTE_CODE=${HF_HUB_TRUST_REMOTE_CODE:-1}

# ---- modes ----
REUSE_SERVER=${REUSE_SERVER:-0}       # 1 = a warm server is already up at HOST:PORT; don't launch/kill
PROFILE=${PROFILE:-0}                 # 1 = also capture a profiler trace
# Profile a window mid-load so the trace holds the real prefill+decode steady state, not a cold burst.
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-40}          # sglang step count (floor; auto-raised to target below)
PROFILE_NUM_STEPS_MAX=${PROFILE_NUM_STEPS_MAX:-64}  # step cap: sglang trace is ~MBs/step and its flush blocks the server
# Step target from the workload: RAMP=ceil(CONC*ISL/chunk) prefill passes + STEADY=max(30,5*ceil(OSL/CONC))
# decode steps + margin, so a capture bounded to it always spans a decode steady sample. Drives the sglang
# step count, the vllm time window, and the vllm 0.26+ step cap.
PROFILE_TARGET_STEPS=$(python3 -c "import math;print(math.ceil($CONC*$ISL/max(${PREFILL_CHUNK:-$ISL},1))+max(30,5*math.ceil($OSL/max($CONC,1)))+10)" 2>/dev/null || echo "$PROFILE_NUM_STEPS_MAX")
# vllm 0.26+ ProfilerConfig knobs (adapters/vllm.sh), fixed at server launch. max_iterations self-stops the
# profiler after N worker steps; default to the workload target clamped to the step cap (bounds the buffer).
PROFILE_MAX_ITERS=${PROFILE_MAX_ITERS:-$(( PROFILE_TARGET_STEPS < PROFILE_NUM_STEPS_MAX ? PROFILE_TARGET_STEPS : PROFILE_NUM_STEPS_MAX ))}
PROFILE_DELAY_ITERS=${PROFILE_DELAY_ITERS:-0}       # steps to skip before arming; 0 keeps the prefill burst
PROFILE_WARMUP_SEC=${PROFILE_WARMUP_SEC:-0}         # 0 = arm at load start so prefill is captured too
PROFILE_NUM_PROMPTS=${PROFILE_NUM_PROMPTS:-$((CONC * 4))}   # >CONC so the queue stays saturated
PROFILE_REQUEST_RATE=${PROFILE_REQUEST_RATE:-}      # optional req/s to stagger arrivals; empty = inf
PROFILE_WINDOW_TIMEOUT=${PROFILE_WINDOW_TIMEOUT:-180}      # max wait for the trace file to land
PROFILE_WINDOW_SEC=${PROFILE_WINDOW_SEC:-20}        # vllm time window (floor; auto-scaled below). Sole bound on <0.26
PROFILE_WINDOW_SEC_MAX=${PROFILE_WINDOW_SEC_MAX:-30}      # cap for the auto-scaled window (bounds trace size)
OUT_DIR=${OUT_DIR:-$(pwd)/e2e_bench_out}
LOG=${LOG:-$OUT_DIR/server.log}

mkdir -p "$OUT_DIR"
PROFILE_DIR="$OUT_DIR/profile"
BASE_URL="http://${HOST}:${PORT}"
RESULT_JSONL="$OUT_DIR/bench_runs.jsonl"
: > "$RESULT_JSONL"
# Separate sink for the optional COLD full-round (BENCH_COLD_FINAL=1); kept apart
# from the timed(hot) repeats so it never pollutes the hot median.
COLD_JSONL="$OUT_DIR/bench_runs.cold.jsonl"
: > "$COLD_JSONL"

# export everything the adapter reads
export MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH
export ISL OSL CONC SEED PROFILE PROFILE_DIR PROFILE_NUM_STEPS BASE_URL RESULT_JSONL LOG
export PROFILE_WARMUP_SEC PROFILE_NUM_PROMPTS PROFILE_REQUEST_RATE PROFILE_WINDOW_TIMEOUT PROFILE_WINDOW_SEC
export PROFILE_MAX_ITERS PROFILE_DELAY_ITERS
export NUM_PROMPTS NUM_WARMUPS RANDOM_RANGE_RATIO BENCH_CLIENT
export BENCH_TRUST_REMOTE_CODE HF_HUB_TRUST_REMOTE_CODE

echo "Backend:      $BACKEND  (adapter: $ADAPTER)"
echo "Model:        $MODEL"
echo "Endpoint:     $BASE_URL  (TP=$TP, GPU=$GPU, mem-fraction=$MEM_FRACTION)"
echo "ISL/OSL/conc: $ISL / $OSL / $CONC   num-prompts=$NUM_PROMPTS   repeats=$REPEATS"
echo "Extra args:   ${EXTRA_SERVER_ARGS:-<none>}"
echo "Extra env:    ${EXTRA_ENV:-<none>}"
echo "Overlay PP:   ${OVERLAY_PYTHONPATH:-<none>}"
echo "Reuse server: $REUSE_SERVER   Profile: $PROFILE"
echo "Out dir:      $OUT_DIR"
echo

SERVER_PID=""
# Server lifecycle: the teardown contract lives in server_teardown.sh so the SAME
# identity-verified kill is used by this dispatcher and by any role-authored capture
# script (which previously hand-rolled its own kill). The old cleanup resolved the
# server's pgid AT KILL TIME and group-killed whenever it differed from ours — a pid
# that had exited and been recycled resolved to a stranger's group, which is how a
# teardown can reach the caller's orchestrator / PID 1.
#
# This script is COPIED into $EVAL_DIR (roles/director.md) and run from there, so the
# library has to be found next to the copy. If it is not, the teardown silently becomes
# a no-op: `source` fails, the EXIT trap resolves to a missing function, and the served
# model is left running with its VRAM and port held while the serving-GPU lock is
# released — the next launch then OOMs. So look next to us, then in the ORIGINAL scripts
# dir when the caller told us where that is, and REFUSE to run otherwise. A benchmark
# that cannot stop what it starts must not start it.
TEARDOWN_LIB=""
for _cand in "$HERE/server_teardown.sh" "${SKILL_DIR:-}/scripts/server_teardown.sh" \
             "${WORKFLOW_DIR:-}/scripts/server_teardown.sh"; do
  case "$_cand" in /scripts/server_teardown.sh) continue ;; esac   # unset SKILL_DIR/WORKFLOW_DIR
  [ -f "$_cand" ] && { TEARDOWN_LIB="$_cand"; break; }
done
if [ -z "$TEARDOWN_LIB" ]; then
  echo "!!! server_teardown.sh not found next to this script ($HERE) or under SKILL_DIR/WORKFLOW_DIR." >&2
  echo "    It carries the server-kill contract; without it the EXIT trap is a no-op and the" >&2
  echo "    launched server would be LEAKED (VRAM + port held, serving-GPU lock released)." >&2
  echo "    Stage it alongside bench_e2e.sh: cp \"\$SKILL_DIR/scripts/server_teardown.sh\" \"\$EVAL_DIR/\"" >&2
  exit 3
fi
[ "$TEARDOWN_LIB" = "$HERE/server_teardown.sh" ] || echo ">>> teardown contract: $TEARDOWN_LIB (not staged next to this copy)"
# shellcheck disable=SC1090
source "$TEARDOWN_LIB"
trap server_teardown EXIT

# ---- serving-GPU mutex ----
# TP=N on an N-GPU box means SERVING_GPU = ALL gpus = a SINGLE serving slot.
# Profiler / config-sweep / integrate ref·cand / validation all share it, so
# without a lock a reprofile can be starved indefinitely by a concurrent
# integrate benchmark. Serialize every serving launch behind a per-GPU-set lock.
# (Isolated op-bench uses the SEPARATE GPU_IDS pool and is unaffected.)
if [ "${SERVING_GPU_LOCK_DISABLE:-0}" != "1" ] && [ "${REUSE_SERVER:-0}" != "1" ]; then
  _gpu_key="${GPU:-0}"; _gpu_key="${_gpu_key//,/_}"
  SERVING_LOCK="${SERVING_GPU_LOCK:-/tmp/geak_serving_gpu_${_gpu_key}.lock}"
  exec {SERVING_LOCK_FD}>"$SERVING_LOCK"
  echo ">>> Acquiring serving-GPU lock ($SERVING_LOCK) for GPU=$GPU ..."
  if ! flock -w "${SERVING_LOCK_WAIT:-7200}" "$SERVING_LOCK_FD"; then
    echo "!!! serving-GPU lock timeout (${SERVING_LOCK_WAIT:-7200}s) on GPU=$GPU" >&2
    exit 4
  fi
  echo ">>> serving-GPU lock acquired."
fi

# ---- launch (unless reusing a warm server) ----
# A server is declared dead when it stops MAKING PROGRESS, not when it has taken "too long": wedged
# shows up as SILENCE, while a legitimately slow cold start keeps printing. CEILING is only a backstop
# against a server that spins printing forever, not a per-purpose budget.
if [ "$REUSE_SERVER" != "1" ]; then
  mkdir -p "$PROFILE_DIR"
  STALL_WINDOW_SEC=${STALL_WINDOW_SEC:-600}
  case "${SERVER_STARTUP_TIMEOUT_SEC:-}" in ''|*[!0-9]*) CEILING=7200 ;; *) CEILING="$SERVER_STARTUP_TIMEOUT_SEC" ;; esac

  _up=0; _reason=""
  echo ">>> Launching $BACKEND server (log: $LOG) ..."
  adapter_launch
  if [ -z "${SERVER_PID:-}" ]; then echo "!!! adapter_launch did not set SERVER_PID"; exit 2; fi
  # Freeze the server's process identity NOW (pid, pgid, /proc start time) so the
  # EXIT teardown never has to ask "who owns this pid?" after the pid may be gone.
  server_record_identity "$SERVER_PID"

  echo ">>> Waiting for server health (stall window ${STALL_WINDOW_SEC}s, backstop ${CEILING}s) ..."
  _t0=$SECONDS; _last_tok=""; _last_change=$SECONDS
  while :; do
    if adapter_health >/dev/null 2>&1; then _up=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then _reason="died_early"; break; fi
    if grep -Eq 'CUDA out of memory|HIP out of memory' "$LOG" 2>/dev/null; then _reason="oom"; break; fi
    # FATAL must look like an EMITTED record ('[FATAL]' or a 'FATAL:' prefix), not the bare word:
    # unanchored, it also matches a --log-level legend or a help line and kills a healthy start.
    if grep -Eq 'watchdog timeout|Capturing cuda graph failed|\[FATAL\]|(^|[[:space:]])FATAL:' "$LOG" 2>/dev/null; then
      _reason="fatal_marker"; break
    fi
    _tok=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$_tok" != "$_last_tok" ]; then
      _last_tok="$_tok"; _last_change=$SECONDS
    elif [ $((SECONDS-_last_change)) -ge "$STALL_WINDOW_SEC" ]; then
      _reason="stalled"; break
    fi
    if [ $((SECONDS-_t0)) -ge "$CEILING" ]; then _reason="ceiling_exceeded"; break; fi
    sleep 5
  done
  _waited=$((SECONDS-_t0))
  if [ "$_up" = "1" ]; then
    echo ">>> Server up after ~${_waited}s."
  else
    case "$_reason" in
      died_early|ceiling_exceeded|stalled)
        if grep -Eq 'ngine ?[Cc]ore.*([Tt]imed out|TimeoutError)|frontend.*handshake.*timed out' "$LOG" 2>/dev/null; then
          _reason="engine_core_timeout"
        elif grep -Eq 'NCCL error|RCCL error|rendezvous|Timed out initializing process group|ProcessGroup.*[Tt]imeout' "$LOG" 2>/dev/null; then
          _reason="dist_init_fail"
        fi ;;
    esac
    echo "!!! Server did not come up (reason=$_reason) after ~${_waited}s. Last log:"; tail -n 60 "$LOG"
    server_teardown; SERVER_PID=""
  fi
  # Structured outcome, ALWAYS written (success too), so a failed start is a REASON downstream can
  # read rather than an empty task dir that looks like "authored and found no gain".
  _phase=$(tail -n 1 "$LOG" 2>/dev/null | tr -d '\r\\' | tr '"' "'" | cut -c1-200)
  printf '{"status":"%s","reason":"%s","phase_hint":"%s","wait_sec":%d,"ceiling_sec":%d,"stall_window_sec":%d,"port":"%s","backend":"%s","log":"%s"}\n' \
    "$([ "$_up" = "1" ] && echo ok || echo failed)" "${_reason:-none}" "$_phase" \
    "$_waited" "$CEILING" "$STALL_WINDOW_SEC" "$PORT" "$BACKEND" "$LOG" \
    > "$OUT_DIR/server_start.json"
  if [ "$_up" != "1" ]; then
    echo "!!! Server start FAILED (reason=$_reason) — see $OUT_DIR/server_start.json" >&2
    [ "$_reason" = "ceiling_exceeded" ] && \
      echo "    Still progressing at the ${CEILING}s backstop; raise it with SERVER_STARTUP_TIMEOUT_SEC." >&2
    exit 2
  fi
else
  echo ">>> Reusing warm server at $BASE_URL"
  adapter_health >/dev/null 2>&1 || { echo "!!! No healthy server at $BASE_URL"; exit 2; }
fi

# ---- overlay resident-memory parity guard (only when an overlay is active) ----
# An authored kernel that builds a PERSISTENT dequant/shuffle cache inflates resident VRAM beyond the
# baseline, so a "win" measured with less memory headroom is unfair (and usually OOMs under load anyway).
# Reject such a candidate BEFORE the timed legs instead of after a full A/B. The integrator records the
# free-VRAM floor the baseline leg cleared into MEM_HEADROOM_MIN_MB; a candidate below it fails parity.
# Fail-OPEN on any parse error (missing rocm-smi / unexpected schema) so non-AMD or partial rigs are
# unaffected — this only ever rejects when it can POSITIVELY prove the headroom regressed.
if [ -n "$OVERLAY_PYTHONPATH" ] && [ -n "${MEM_HEADROOM_MIN_MB:-}" ]; then
  _free_mb=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    vals = [int(v["VRAM Total Free Memory (B)"]) // (1024*1024)
            for v in d.values()
            if isinstance(v, dict) and "VRAM Total Free Memory (B)" in v]
    print(min(vals) if vals else "")
except Exception:
    print("")
' 2>/dev/null || echo "")
  if [ -n "$_free_mb" ] && [ "$_free_mb" -lt "$MEM_HEADROOM_MIN_MB" ] 2>/dev/null; then
    echo "!!! Overlay resident VRAM headroom ${_free_mb}MB < baseline floor ${MEM_HEADROOM_MIN_MB}MB"
    echo "    -> memory-parity FAIL; rejecting candidate before timed legs."
    tail -n 30 "$LOG" 2>/dev/null || true
    exit 2
  fi
  echo ">>> overlay memory-parity OK (free ${_free_mb:-?}MB >= floor ${MEM_HEADROOM_MIN_MB}MB)"
fi

# ---- optional COLD full-round (DIAGNOSTIC ONLY, off by default) ----
# One full round (NUM_PROMPTS, no preceding warmup) on the fresh server, recorded
# separately from the timed(hot) repeats.
#
# "Cold" here means only "no warmup round preceded it in THIS bench" — it does
# not mean a cold machine. Only the first bench of a session sees a genuinely
# cold box; every later bench (the final leg included) starts with the JIT/HIP
# kernel caches, torch.compile artifacts and page cache already populated by the
# benches before it. So the baseline's cold round pays the full cache-fill cost
# and the final's pays almost none, and a ratio of the two reports that
# asymmetry as speedup. Never compare cold rounds taken at different points in a
# session, and never promote one as the headline number.
# Default OFF; set BENCH_COLD_FINAL=1 to measure it as a diagnostic (costs one
# extra full round per bench). Only meaningful on a fresh launch (a reused warm
# server has no cold state to measure at all).
if [ "${BENCH_COLD_FINAL:-0}" = "1" ] && [ "$REUSE_SERVER" != "1" ]; then
  echo ">>> Cold full round (NUM_PROMPTS=$NUM_PROMPTS, no warmup; cold-baseline parity) ..."
  # adapter_bench is a shell FUNCTION that reads $RESULT_JSONL — a prefix var
  # assignment on a function has ambiguous persistence in bash, so point
  # RESULT_JSONL at the cold sink explicitly and restore it afterwards. The
  # warmup below re-clears the (restored) hot RESULT_JSONL, so the cold round
  # never touches the timed(hot) results.
  _saved_result_jsonl="$RESULT_JSONL"
  RESULT_JSONL="$COLD_JSONL"; export RESULT_JSONL
  adapter_bench "$NUM_PROMPTS" "$CONC" 0 || echo "!!! cold round failed (continuing)"
  RESULT_JSONL="$_saved_result_jsonl"; export RESULT_JSONL
fi

# ---- optional outer warmup (never timed) ----
# Cache-cold isolated replicas skip this invocation: their benchmark client still
# performs its own internal warmup, but the timed prompt set is not pre-populated
# by an earlier full replay. Legacy/default runs retain the historical short
# CONC-prompt warmup, and explicit full-round callers retain the full warmup.
if [ "${GEAK_ISOLATED_REPLICA:-0}" = "1" ] \
   && [ "${BENCH_OUTER_WARMUP_FULL_ROUND:-0}" != "1" ]; then
  echo ">>> Skipping outer warmup (compute-warm/cache-cold isolated replica) ..."
else
  _warmup_prompts="$CONC"
  if [ "${BENCH_OUTER_WARMUP_FULL_ROUND:-0}" = "1" ]; then
    _warmup_prompts="$NUM_PROMPTS"
    echo ">>> Warmup full round (prompts=$_warmup_prompts) ..."
  else
    echo ">>> Warmup round ..."
  fi
  if ! adapter_bench "$_warmup_prompts" "$CONC" 0 >/dev/null 2>&1; then
    if [ "${BENCH_OUTER_WARMUP_FULL_ROUND:-0}" = "1" ]; then
      echo "!!! Full outer warmup failed; isolated replica is invalid." >&2
      exit 2
    fi
  fi
fi
# the warmup line should not pollute the timed results
: > "$RESULT_JSONL"

# ---- timed repeats ----
_bench_failed=0
for r in $(seq 1 "$REPEATS"); do
  echo ">>> Bench repeat $r/$REPEATS ..."
  if ! adapter_bench "$NUM_PROMPTS" "$CONC" 0; then
    echo "!!! bench repeat $r failed (continuing)"
    _bench_failed=1
  fi
done
if [ "${BENCH_REQUIRE_SUCCESS:-0}" = "1" ] && [ "$_bench_failed" = "1" ]; then
  echo "!!! Required isolated measurement round failed." >&2
  exit 2
fi

# ---- optional profile trace (STEADY-STATE MIX, not a cold prefill burst) ----
# Real serving is continuous batching: at any instant some sequences are prefilling (chunks) and others
# decoding, interleaved by the scheduler. A cold burst profiled from step 0 captures only the prefill
# ramp (TTFT) and misses decode entirely (see knowledge/profile_parse.md). So we instead drive a
# sustained, saturated load and profile a WINDOW once it has reached the mixed steady state.
if [ "$PROFILE" = "1" ]; then
  mkdir -p "$PROFILE_DIR"
  # fix #1: if the caller didn't pin TPOT_MS, derive it from the timed bench we JUST ran (RESULT_JSONL
  # holds one result object per repeat, each with median_tpot_ms). This lets the vLLM time-window auto-
  # scale to the REAL per-decode-step time of THIS workload (below) instead of sitting at the flat floor.
  if [ -z "${TPOT_MS:-}" ]; then
    TPOT_MS=$(python3 - "$RESULT_JSONL" <<'PY' 2>/dev/null || true
import json, sys, statistics
vals=[]
try:
    for line in open(sys.argv[1]):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except Exception: continue
        for k in ("median_tpot_ms","mean_tpot_ms","p50_tpot_ms"):
            if isinstance(d.get(k),(int,float)): vals.append(float(d[k])); break
print(round(statistics.median(vals),3) if vals else "")
PY
)
    case "${TPOT_MS:-}" in ''|*[!0-9.]*) TPOT_MS="" ;; esac   # keep only a clean number
    [ -n "${TPOT_MS:-}" ] && echo ">>> steady-state sizing: derived TPOT_MS=${TPOT_MS}ms from timed bench (vllm window auto-scale)"
  fi
  # Size the single capture to PROFILE_TARGET_STEPS (computed at launch): raise the sglang step count
  # (clamped to the cap) and scale the vllm window to target*TPOT*1.5. vllm 0.26+ is already step-bounded
  # by PROFILE_MAX_ITERS; the window is a safety cap there, the sole bound on <0.26.
  if [ "${PROFILE_NUM_STEPS:-0}" -lt "$PROFILE_TARGET_STEPS" ]; then
    echo ">>> sizing: PROFILE_NUM_STEPS ${PROFILE_NUM_STEPS}->${PROFILE_TARGET_STEPS}"
    PROFILE_NUM_STEPS=$PROFILE_TARGET_STEPS
  fi
  if [ -n "${PROFILE_NUM_STEPS_MAX:-}" ] && [ "$PROFILE_NUM_STEPS" -gt "$PROFILE_NUM_STEPS_MAX" ]; then
    echo ">>> sizing: PROFILE_NUM_STEPS capped ${PROFILE_NUM_STEPS}->${PROFILE_NUM_STEPS_MAX}"
    PROFILE_NUM_STEPS=$PROFILE_NUM_STEPS_MAX
  fi
  _NEED_PROMPTS=$(python3 -c "import math;print($CONC + math.ceil($CONC*$PROFILE_NUM_STEPS/max($OSL,1)) + $CONC)" 2>/dev/null || echo "$PROFILE_NUM_PROMPTS")
  if [ "${PROFILE_NUM_PROMPTS:-0}" -lt "$_NEED_PROMPTS" ]; then
    echo ">>> sizing: PROFILE_NUM_PROMPTS ${PROFILE_NUM_PROMPTS}->${_NEED_PROMPTS}"
    PROFILE_NUM_PROMPTS=$_NEED_PROMPTS
  fi
  if [ -n "${TPOT_MS:-}" ]; then
    _WMAX="${PROFILE_WINDOW_SEC_MAX:-30}"
    _WSEC=$(python3 -c "import math;print(min($_WMAX, max(${PROFILE_WINDOW_SEC:-20}, math.ceil($PROFILE_TARGET_STEPS*$TPOT_MS/1000.0*1.5))))" 2>/dev/null || echo "${PROFILE_WINDOW_SEC:-20}")
    if [ "$_WSEC" != "${PROFILE_WINDOW_SEC}" ]; then
      echo ">>> sizing: PROFILE_WINDOW_SEC ${PROFILE_WINDOW_SEC}->${_WSEC}s"
      PROFILE_WINDOW_SEC=$_WSEC
    fi
  fi
  export PROFILE_NUM_STEPS PROFILE_NUM_PROMPTS PROFILE_WINDOW_SEC
  if declare -F adapter_profile_window >/dev/null; then
    echo ">>> Profiling from load start (warmup ${PROFILE_WARMUP_SEC}s) on a saturated load " \
         "(${PROFILE_NUM_PROMPTS} prompts, conc ${CONC}${PROFILE_REQUEST_RATE:+, rate ${PROFILE_REQUEST_RATE}/s}); " \
         "SINGLE capture of ${PROFILE_NUM_STEPS} steps / ${PROFILE_WINDOW_SEC}s (adaptive re-capture OFF) ..."
    # SINGLE deterministic capture (adaptive re-capture is off — see the sizing note above). Start the
    # sustained, replenishing background load (>CONC prompts, realistic prefill+decode mix; NOT timed, NOT
    # profiled). With PROFILE_WARMUP_SEC=0 the profiler is armed at load start so the capture includes the
    # initial prefill burst (prefill shapes stay visible for head selection).
    REQUEST_RATE="${PROFILE_REQUEST_RATE}" \
      adapter_bench "$PROFILE_NUM_PROMPTS" "$CONC" 0 >/dev/null 2>&1 &
    _bg_load=$!
    sleep "$PROFILE_WARMUP_SEC"
    if kill -0 "$_bg_load" 2>/dev/null; then
      adapter_profile_window || echo "!!! profile window failed"
    else
      echo "!!! background load exited before the profile window (load too short?) — falling back"
      adapter_bench "$PROFILE_NUM_PROMPTS" "$CONC" 1 || echo "!!! profile run failed"
    fi
    kill "$_bg_load" 2>/dev/null || true; wait "$_bg_load" 2>/dev/null || true
  else
    # Backend without an HTTP profiler hook: can't profile a mid-stream window, but at least avoid the
    # pure cold burst — send more prompts so the queue stays full past the prefill ramp and the captured
    # steps include some decode. (Still less faithful than the windowed path; note it.)
    echo ">>> Profiling (no window hook for $BACKEND; ${PROFILE_NUM_PROMPTS} prompts, ${PROFILE_NUM_STEPS} steps) ..."
    REQUEST_RATE="${PROFILE_REQUEST_RATE}" \
      adapter_bench "$PROFILE_NUM_PROMPTS" "$CONC" 1 || echo "!!! profile run failed"
  fi
  echo ">>> Trace(s) in $PROFILE_DIR"
fi

# ---- summarize (median throughput across repeats) — backend-independent ----
python3 - "$RESULT_JSONL" "$OUT_DIR/bench_summary.json" "$COLD_JSONL" <<'PY'
import json, os, sys, statistics
runs_path, out_path = sys.argv[1], sys.argv[2]
cold_path = sys.argv[3] if len(sys.argv) > 3 else None
def pick(d, *keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)): return float(d[k])
    return None
# metric selection: default = OUTPUT-only token throughput (output/s), to match the Hyperloom
# orchestrator's baseline/explore basis (see collectors/_common.py + collectors/explore.py, both read
# output_throughput). Set E2E_METRIC=total for total (input+output)/s. Same key is read for
# baseline+cand so the accept RATIO is consistent; metric_basis records which was used.
_metric = (os.environ.get("E2E_METRIC") or "output").strip().lower()
_is_total = _metric in ("total", "total_token", "total_throughput")
_TPUT_KEYS = (("total_token_throughput", "total_throughput", "total_token_throughput_tok_s")
              if _is_total else
              ("output_throughput", "output_token_throughput", "output_throughput_tok_s"))
def read_tps(path):
    xs = []
    if not path: return xs
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except Exception: continue
                v = pick(d, *_TPUT_KEYS)
                if v is not None: xs.append(v)
    except FileNotFoundError:
        pass
    return xs
tps, ttft, tpot = [], [], []
with open(runs_path) as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        v = pick(d, *_TPUT_KEYS)
        if v is not None: tps.append(v)
        t = pick(d, "median_ttft_ms", "mean_ttft_ms");   ttft.append(t) if t is not None else None
        p = pick(d, "median_tpot_ms", "mean_tpot_ms");   tpot.append(p) if p is not None else None
cold_tps = read_tps(cold_path)
def med(xs): return statistics.median(xs) if xs else None
def spread(xs):
    if len(xs) < 2: return 0.0
    m = med(xs); return round(100.0 * (max(xs)-min(xs)) / m, 2) if m else 0.0
_tput_med = round(med(tps), 3) if tps else None
_tput_spread = spread(tps)
summ = {
    # Canonical, metric-neutral throughput of the SELECTED basis (see metric_basis). Downstream should
    # read this + metric_basis; the accept RATIO is basis-consistent (baseline+cand use the same metric).
    "throughput_tok_s_median": _tput_med,
    "throughput_tok_s_spread_pct": _tput_spread,
    # Legacy output-named alias: populated ONLY in output mode (its literal meaning). In total mode it is
    # None so nobody silently reads total throughput under an "output" name — read throughput_tok_s_median.
    "output_throughput_tok_s_median": _tput_med if not _is_total else None,
    "output_throughput_tok_s_spread_pct": _tput_spread if not _is_total else None,
    "ttft_ms_median": round(med(ttft), 3) if ttft else None,
    "tpot_ms_median": round(med(tpot), 3) if tpot else None,
    "runs": len(tps),
    "all_throughput": tps,
    # Optional COLD full-round (BENCH_COLD_FINAL=1): a single fresh-server round with
    # JIT/graph-capture costs included, for cold-to-cold parity vs Hyperloom's
    # baseline_tput. None when the cold round was not run (default). The hot median
    # above stays the primary metric so existing consumers are unaffected. Uses the
    # SAME metric basis (E2E_METRIC) as the hot median for a consistent comparison.
    "cold_output_throughput_tok_s": round(med(cold_tps), 3) if cold_tps else None,
    "cold_runs": len(cold_tps),
    # Aggregate tok/s (NOT divided by TP). Default matches Hyperloom/Magpie output_throughput protocol;
    # E2E_METRIC=total switches to total (input+output) token throughput.
    "metric_basis": ("aggregate_total_token_tok_s" if _is_total else "aggregate_output_tok_s"),
    "measurement_mode": (
        "isolated_server_replica"
        if os.environ.get("GEAK_ISOLATED_REPLICA") == "1"
        else "legacy_same_server"
    ),
    "effective_config_digest": os.environ.get("EFFECTIVE_CONFIG_DIGEST") or None,
}
with open(out_path, "w") as fh: json.dump(summ, fh, indent=2)
print(f"E2E_SUMMARY {summ['metric_basis']}={summ['throughput_tok_s_median']} "
      f"spread={summ['throughput_tok_s_spread_pct']}% "
      f"ttft_ms={summ['ttft_ms_median']} tpot_ms={summ['tpot_ms_median']} runs={summ['runs']}")
PY

echo ">>> Done. Summary: $OUT_DIR/bench_summary.json"
