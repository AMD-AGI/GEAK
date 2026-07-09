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
#   bench_runs.jsonl       one bench result object per repeat
#   bench_summary.json     {throughput_tok_s_median (metric-neutral; see metric_basis), metric_basis,
#                           ttft_ms_median, tpot_ms_median, spread, runs}  (E2E_METRIC=total default)
#   SUMMARY line on stdout: "E2E_SUMMARY <metric_basis>=<median> spread=<pct> ttft_ms=<med> tpot_ms=<med>"
#   profile/                trace (if PROFILE=1)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# SERVER's trust setting: if the server is launched with --trust-remote-code
# (via EXTRA_SERVER_ARGS), the client measuring it must trust the same remote
# code. Stays OFF (no implicit remote-code execution) for models that don't need
# it, so standalone behaviour is unchanged. An explicit caller value always wins.
if [ -z "${BENCH_TRUST_REMOTE_CODE:-}" ]; then
  case "$EXTRA_SERVER_ARGS" in
    *trust-remote-code*|*trust_remote_code*) BENCH_TRUST_REMOTE_CODE=1 ;;
    *) BENCH_TRUST_REMOTE_CODE=0 ;;
  esac
fi
# transformers / HF hub honor HF_HUB_TRUST_REMOTE_CODE for tokenizer auto-load.
[ "$BENCH_TRUST_REMOTE_CODE" = "1" ] && HF_HUB_TRUST_REMOTE_CODE=${HF_HUB_TRUST_REMOTE_CODE:-1}

# ---- modes ----
REUSE_SERVER=${REUSE_SERVER:-0}       # 1 = a warm server is already up at HOST:PORT; don't launch/kill
PROFILE=${PROFILE:-0}                 # 1 = also capture a profiler trace
# Profiling is meant to capture the REAL continuous-batching steady state — prefill chunks and decode
# steps interleaved as the scheduler actually runs them — NOT a cold prefill burst. So we profile a
# WINDOW in the middle of a sustained, saturated load (see the PROFILE block below). Tunables:
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-40}   # forward steps to capture. 5 couldn't even clear prefill
                                             # (prefill ~ceil(tot_in/chunk) steps); ~40 spans into the
                                             # mixed decode steady state. Decode is steady so a few tens
                                             # is plenty; >100 just bloats the trace.
PROFILE_WARMUP_SEC=${PROFILE_WARMUP_SEC:-15} # let the load pass the initial synchronized prefill ramp
                                             # (≈TTFT) so the profiled window lands in the real mix.
PROFILE_NUM_PROMPTS=${PROFILE_NUM_PROMPTS:-$((CONC * 4))}  # >CONC so the queue stays full and short
                                             # (range-ratio>0) requests get replaced -> phases de-sync.
PROFILE_REQUEST_RATE=${PROFILE_REQUEST_RATE:-}            # optional req/s to stagger arrivals; empty
                                             # = inf (max_concurrency still caps in-flight at CONC).
PROFILE_WINDOW_TIMEOUT=${PROFILE_WINDOW_TIMEOUT:-180}     # max wait for the trace file to appear.
PROFILE_WINDOW_SEC=${PROFILE_WINDOW_SEC:-40}             # capture DURATION for time-windowed backends
                                             # (vllm: /start_profile has no num_steps, so the window is
                                             # controlled by start -> sleep this long -> /stop_profile).
                                             # sglang ignores this (it uses PROFILE_NUM_STEPS instead).
                                             # ~40s spans many decode steps + enough prefill events even
                                             # at long OSL + low concurrency (prefills are sparse: rate
                                             # ~ CONC/(OSL*step_time), e.g. only ~1 every few sec at low
                                             # CONC). Longer is NOT free: a torch trace grows ~linearly
                                             # with the window and flush time with it (must stay <
                                             # PROFILE_WINDOW_TIMEOUT, and can OOM the profiler buffer) —
                                             # lower it if flush fails / the trace is huge.
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
cleanup() {
  [ -n "${SERVER_PID:-}" ] || return 0
  echo ">>> Shutting down server (pid $SERVER_PID) ..."
  # A launcher that starts the server in its OWN process group / session (e.g.
  # the Magpie launcher uses `setsid`) leaves the worker/child procs OUTSIDE
  # $SERVER_PID, so a bare `kill $SERVER_PID` orphans them (leaked VRAM, ghost
  # listeners on the port). When the server's process group differs from OURS,
  # reap the WHOLE group (TERM, then KILL after a grace window). The own-group
  # guard is critical: for a NATIVE launch the server shares our group, so we
  # must NOT group-kill (that would kill bench_e2e.sh itself) — fall back to the
  # single-pid kill, byte-identical to before.
  local _pgid _self
  _pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
  _self="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
  if [ -n "$_pgid" ] && [ "$_pgid" != "$_self" ]; then
    kill -TERM "-$_pgid" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _i in $(seq 1 "${SERVER_STOP_GRACE_S:-10}"); do
      kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
    done
    kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL "-$_pgid" 2>/dev/null || true
  else
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

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
if [ "$REUSE_SERVER" != "1" ]; then
  mkdir -p "$PROFILE_DIR"
  echo ">>> Launching $BACKEND server (log: $LOG) ..."
  adapter_launch
  if [ -z "${SERVER_PID:-}" ]; then echo "!!! adapter_launch did not set SERVER_PID"; exit 2; fi

  echo ">>> Waiting for server health ..."
  # An overlaid candidate can wedge: process stays alive but /health 503s forever (JIT deadlock /
  # cuda-graph capture failure). Don't burn the whole window while holding the serving-GPU lock —
  # fail fast on a fatal server-log marker, and use a TIGHTER budget when an overlay is active so a
  # broken candidate is rejected quickly instead of starving the box. Non-overlay runs keep 180*5s.
  HEALTH_TRIES=${HEALTH_TRIES:-180}
  [ -n "$OVERLAY_PYTHONPATH" ] && HEALTH_TRIES=${OVERLAY_HEALTH_TRIES:-72}   # ~6min for overlays
  _up=0
  for i in $(seq 1 "$HEALTH_TRIES"); do
    if adapter_health >/dev/null 2>&1; then echo ">>> Server up after ~$((i*5))s."; _up=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!!! Server died. Last log:"; tail -n 60 "$LOG"; exit 2; fi
    if grep -Eq 'CUDA out of memory|HIP out of memory|watchdog timeout|Capturing cuda graph failed|FATAL' "$LOG" 2>/dev/null; then
      echo "!!! Fatal server-log marker before health; aborting wait. Last log:"; tail -n 60 "$LOG"; exit 2
    fi
    sleep 5
  done
  [ "$_up" = "1" ] || { echo "!!! Server not healthy within $((HEALTH_TRIES*5))s."; tail -n 60 "$LOG"; exit 2; }
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

# ---- optional COLD full-round (align with Hyperloom's COLD baseline_tput) ----
# Hyperloom's leaderboard denominator baseline_tput is a COLD single fresh-server
# round (first-token / JIT / cuda-graph capture costs INCLUDED, no prior warmup).
# GEAK's own final is a HOT median (warmup discarded). Comparing GEAK's hot final
# to Hyperloom's cold baseline mixes thermal states. When BENCH_COLD_FINAL=1 we
# also measure ONE cold full round (NUM_PROMPTS, no preceding warmup) on the fresh
# server BEFORE the warmup+timed(hot) rounds, and record it separately, so the
# caller can compute a fair cold-to-cold speedup (and keep the hot median as a
# double-check). Default ON (BENCH_COLD_FINAL=1) — set BENCH_COLD_FINAL=0 to skip
# the cold round (e.g. to save the one extra full round per bench). Only meaningful
# on a fresh launch (a reused warm server has no cold state to measure).
if [ "${BENCH_COLD_FINAL:-1}" = "1" ] && [ "$REUSE_SERVER" != "1" ]; then
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

# ---- warmup (one short round; never timed) ----
echo ">>> Warmup round ..."
adapter_bench "$CONC" "$CONC" 0 >/dev/null 2>&1 || true
# the warmup line should not pollute the timed results
: > "$RESULT_JSONL"

# ---- timed repeats ----
for r in $(seq 1 "$REPEATS"); do
  echo ">>> Bench repeat $r/$REPEATS ..."
  adapter_bench "$NUM_PROMPTS" "$CONC" 0 || echo "!!! bench repeat $r failed (continuing)"
done

# ---- optional profile trace (STEADY-STATE MIX, not a cold prefill burst) ----
# Real serving is continuous batching: at any instant some sequences are prefilling (chunks) and others
# decoding, interleaved by the scheduler. A cold burst profiled from step 0 captures only the prefill
# ramp (TTFT) and misses decode entirely (see knowledge/profile_parse.md). So we instead drive a
# sustained, saturated load and profile a WINDOW once it has reached the mixed steady state.
if [ "$PROFILE" = "1" ]; then
  mkdir -p "$PROFILE_DIR"
  # ---- workload-aware steady-state window sizing (don't rely ONLY on the reactive re-capture gate) ----
  # Reaching batch≈CONC = clear the prefill ramp, then sample steady decode:
  #   RAMP   = ceil(CONC*ISL / chunk)     forward passes to prefill all CONC in-flight requests
  #   STEADY = max(30, 5*ceil(OSL/CONC))  decode steps for a stable, representative sample
  # TARGET = RAMP + STEADY + margin. Why per-backend:
  #   - sglang is STEP-controlled and its trace lacks the execute_* step annotations decode_steps.py
  #     needs, so the reactive gate is only a COARSE proxy there — size PROFILE_NUM_STEPS to TARGET up
  #     front (deterministic; the ISL/OSL/CONC/prompts math is the guarantee, not the gate).
  #   - vLLM is TIME-controlled; with detailed_trace_annotation its trace DOES carry step spans, so the
  #     gate converges — we still raise PROFILE_WINDOW_SEC from TPOT (when known) so the first window
  #     usually lands steady without a re-capture.
  # Assumes a saturated queue + KV headroom for CONC concurrent decodes (else batch can't reach CONC).
  _CHUNK="${PREFILL_CHUNK:-$ISL}"
  _RAMP=$(python3 -c "import math;print(math.ceil($CONC*$ISL/max($_CHUNK,1)))" 2>/dev/null || echo "$CONC")
  _STEADYN=$(python3 -c "import math;print(max(30,5*math.ceil($OSL/max($CONC,1))))" 2>/dev/null || echo 30)
  _TARGET_STEPS=$(( _RAMP + _STEADYN + 10 ))
  if [ "${PROFILE_NUM_STEPS:-0}" -lt "$_TARGET_STEPS" ]; then
    echo ">>> steady-state sizing: RAMP=${_RAMP}+STEADY=${_STEADYN}+10 -> PROFILE_NUM_STEPS ${PROFILE_NUM_STEPS}->${_TARGET_STEPS}"
    PROFILE_NUM_STEPS=$_TARGET_STEPS
  fi
  _NEED_PROMPTS=$(python3 -c "import math;print($CONC + math.ceil($CONC*$PROFILE_NUM_STEPS/max($OSL,1)) + $CONC)" 2>/dev/null || echo "$PROFILE_NUM_PROMPTS")
  if [ "${PROFILE_NUM_PROMPTS:-0}" -lt "$_NEED_PROMPTS" ]; then
    echo ">>> steady-state sizing: PROFILE_NUM_PROMPTS ${PROFILE_NUM_PROMPTS}->${_NEED_PROMPTS} (keep the queue full through the window)"
    PROFILE_NUM_PROMPTS=$_NEED_PROMPTS
  fi
  if [ -n "${TPOT_MS:-}" ]; then
    _WSEC=$(python3 -c "import math;print(max(${PROFILE_WINDOW_SEC:-40}, math.ceil($_TARGET_STEPS*$TPOT_MS/1000.0*1.5)))" 2>/dev/null || echo "${PROFILE_WINDOW_SEC:-40}")
    if [ "$_WSEC" -gt "${PROFILE_WINDOW_SEC:-40}" ]; then
      echo ">>> steady-state sizing: PROFILE_WINDOW_SEC ${PROFILE_WINDOW_SEC}->${_WSEC}s (TPOT=${TPOT_MS}ms x ${_TARGET_STEPS} steps x1.5)"
      PROFILE_WINDOW_SEC=$_WSEC
    fi
  fi
  export PROFILE_NUM_STEPS PROFILE_NUM_PROMPTS PROFILE_WINDOW_SEC
  if declare -F adapter_profile_window >/dev/null; then
    echo ">>> Profiling steady-state mix: warm ${PROFILE_WARMUP_SEC}s on a saturated load " \
         "(${PROFILE_NUM_PROMPTS} prompts, conc ${CONC}${PROFILE_REQUEST_RATE:+, rate ${PROFILE_REQUEST_RATE}/s}), " \
         "then capture ${PROFILE_NUM_STEPS} steps ..."
    # Representativeness gate: the profiled window must capture >= N DECODE forward steps, else BOTH head
    # selection (raw %GPU) and the decode weight-share are biased by an under-captured window (e.g. one
    # that landed in the prefill ramp). N scales with the workload:
    #   N = max(30, 5*ceil(OSL/CONC))  — a statistical floor (>=30 stable decode samples) PLUS ~5
    #   prefill-admission cycles (a slot frees every ~OSL/CONC decode steps) so the phase ratio is
    #   representative. If short, ENLARGE the window and re-capture, up to PROFILE_DECODE_TMAX seconds
    #   total; if still short, proceed and FLAG low-confidence (never hang). Reaching N is cheap at steady
    #   state (~N*TPOT), so a good first capture is a single pass — only pathological windows re-capture.
    _N_DECODE=$(python3 -c "import math;print(max(30,5*math.ceil($OSL/max($CONC,1))))" 2>/dev/null || echo 30)
    _PROF_TMAX="${PROFILE_DECODE_TMAX:-120}"
    _prof_t0=$SECONDS; _dsteps=0; _attempt=0
    while : ; do
      _attempt=$((_attempt+1))
      # (re)start the sustained, replenishing background load for THIS attempt (>CONC prompts, varied OSL
      # -> realistic prefill+decode mix; NOT timed, NOT profiled). Restarted per attempt (finite prompts).
      # The representativeness gate below guarantees the window still captures enough decode, so we keep
      # the realistic mixed load (prefill shapes stay visible for head selection) rather than a decode-only
      # load that would bias head selection toward decode-only kernels.
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
      # count decode (graph-replay / shape-hidden) forward steps in the newest trace
      _dsteps=$(python3 "$HERE/decode_steps.py" "$PROFILE_DIR" 2>/dev/null || echo 0)
      case "$_dsteps" in ''|*[!0-9]*) _dsteps=0 ;; esac
      if [ "$_dsteps" -ge "$_N_DECODE" ]; then
        echo ">>> decode-capture OK: ${_dsteps} >= ${_N_DECODE} decode steps (attempt ${_attempt})"
        break
      fi
      if [ $((SECONDS - _prof_t0)) -ge "$_PROF_TMAX" ]; then
        echo "!!! decode-capture LOW-CONFIDENCE: ${_dsteps} < ${_N_DECODE} decode steps after ${_PROF_TMAX}s — proceeding + flagging"
        break
      fi
      PROFILE_WINDOW_SEC=$((PROFILE_WINDOW_SEC * 2)); PROFILE_NUM_STEPS=$((PROFILE_NUM_STEPS * 2))
      echo ">>> decode-capture short (${_dsteps} < ${_N_DECODE}); enlarging window -> ${PROFILE_WINDOW_SEC}s / ${PROFILE_NUM_STEPS} steps; re-capturing ..."
    done
    python3 - "$PROFILE_DIR/decode_capture.json" "$_dsteps" "$_N_DECODE" "$_attempt" <<'PY' 2>/dev/null || true
import json, sys
p, ds, n, att = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
json.dump({"decode_steps": ds, "N": n, "attempts": att, "low_confidence": ds < n}, open(p, "w"))
PY
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
# metric selection: default = TOTAL token throughput ((input+output)/s); set E2E_METRIC=output for
# output-only tok/s (Magpie-aligned). Same key is read for baseline+cand so the accept RATIO is
# consistent; metric_basis records which was used.
_metric = (os.environ.get("E2E_METRIC") or "total").strip().lower()
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
}
with open(out_path, "w") as fh: json.dump(summ, fh, indent=2)
print(f"E2E_SUMMARY {summ['metric_basis']}={summ['throughput_tok_s_median']} "
      f"spread={summ['throughput_tok_s_spread_pct']}% "
      f"ttft_ms={summ['ttft_ms_median']} tpot_ms={summ['tpot_ms_median']} runs={summ['runs']}")
PY

echo ">>> Done. Summary: $OUT_DIR/bench_summary.json"
