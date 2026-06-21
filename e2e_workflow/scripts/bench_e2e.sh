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
#                                      into $PROFILE_DIR.
#
# KEY OUTPUTS (written to $OUT_DIR):
#   bench_runs.jsonl       one bench result object per repeat
#   bench_summary.json     {output_throughput_tok_s_median, ttft_ms_median, tpot_ms_median, spread, runs}
#   SUMMARY line on stdout: "E2E_SUMMARY output_tok_s=<median> spread=<pct> ttft_ms=<med> tpot_ms=<med>"
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
# Magpie's own InferenceX benchmark_serving.py (口径-identical client). Default
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
MEM_FRACTION=${MEM_FRACTION:-0.85}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}    # e.g. "--attention-backend triton"
# EXTRA_ENV is applied to the SERVER launch line, space-separated KEY=VAL pairs:
#   EXTRA_ENV="SGLANG_USE_AITER=1 HIPBLASLT_TUNING_FILE=/path/tune.dat"
EXTRA_ENV=${EXTRA_ENV:-}
# OVERLAY_PYTHONPATH: prepend an overlay dir so a patched subtree / monkeypatch loads first.
OVERLAY_PYTHONPATH=${OVERLAY_PYTHONPATH:-}

# ---- port: auto-allocate a free one if not pinned (avoids 30000 collisions on shared boxes) ----
PORT=${PORT:-}
if [ -z "$PORT" ]; then
  if declare -F adapter_default_port >/dev/null; then PORT="$(adapter_default_port)"; fi
  PORT=${PORT:-0}
  # 0 (or a busy port) -> ask the OS for a free one. MUST stay <= 55535: sglang derives a gRPC
  # port = PORT + 10000 and rejects it if > 65535 (an OS ephemeral bind() can return >55535 and
  # crash the server at launch). So pick a random free port in a bounded safe range, not bind(0).
  FREE_PORT=$(python3 - <<'PY' 2>/dev/null || true
import socket, random
cands = list(range(20000, 55001)); random.shuffle(cands)  # PORT+10000 <= 65001 < 65535
for p in cands:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); print(p); s.close(); break
    except OSError:
        s.close()
PY
)
  [ -n "$FREE_PORT" ] && PORT="$FREE_PORT"
fi

# ---- workload ----
ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONC=${CONC:-64}
# NUM_PROMPTS default.
#  * native client (standalone GEAK default): keep the original CONC*5 default so
#    standalone behaviour is byte-identical to before the inferencex integration.
#  * inferencex client (Hyperloom口径 alignment): mirror Hyperloom's ADAPTIVE
#    factor — the number of timed prompts scales DOWN as the per-request sequence
#    cost (ISL+OSL) grows so each repeat stays bounded.
#    factor = {<=1024:10, <=4096:5, <=16384:3, else 2}.
# Override NUM_PROMPTS to pin a fixed count regardless of client.
if [ -z "${NUM_PROMPTS:-}" ]; then
  if [ "$BENCH_CLIENT" = "inferencex" ]; then
    _seq_cost=$((ISL + OSL))
    if   [ "$_seq_cost" -le 1024 ];  then _factor=10
    elif [ "$_seq_cost" -le 4096 ];  then _factor=5
    elif [ "$_seq_cost" -le 16384 ]; then _factor=3
    else _factor=2; fi
    NUM_PROMPTS=$(( CONC * _factor > CONC ? CONC * _factor : CONC ))
  else
    NUM_PROMPTS=$((CONC * 5))
  fi
fi
# Client-side warmup prompts (口径 alignment with Hyperloom's materialize default
# NUM_WARMUPS=min(CONC,8)). Consumed by the inferencex client adapter; the native
# adapters use their own warmup round instead.
NUM_WARMUPS=${NUM_WARMUPS:-$(( CONC < 8 ? CONC : 8 ))}
# RANDOM_RANGE_RATIO / NUM_PROMPTS / NUM_WARMUPS / SEED are the measurement 口径.
# These are STANDALONE defaults: when an external orchestrator (Hyperloom) drives
# the run it exports its own values (interface/run_e2e.py:apply_bench_protocol from
# handoff.bench_protocol) and they override these via the env. Do NOT hard-code a
# value assuming the caller's 口径 — e.g. ratio=1 is fixed-length, ratio=0 is
# variable-length, and the caller may use either. Standalone default = fixed-length.
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
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
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-5}
OUT_DIR=${OUT_DIR:-$(pwd)/e2e_bench_out}
LOG=${LOG:-$OUT_DIR/server.log}

mkdir -p "$OUT_DIR"
PROFILE_DIR="$OUT_DIR/profile"
BASE_URL="http://${HOST}:${PORT}"
RESULT_JSONL="$OUT_DIR/bench_runs.jsonl"
: > "$RESULT_JSONL"

# export everything the adapter reads
export MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH
export ISL OSL CONC SEED PROFILE PROFILE_DIR PROFILE_NUM_STEPS BASE_URL RESULT_JSONL LOG
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
  if [ -n "$SERVER_PID" ]; then
    echo ">>> Shutting down server (pid $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
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
  for i in $(seq 1 180); do
    if adapter_health >/dev/null 2>&1; then echo ">>> Server up after ~$((i*5))s."; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!!! Server died. Last log:"; tail -n 60 "$LOG"; exit 2; fi
    sleep 5
  done
  adapter_health >/dev/null 2>&1 || { echo "!!! Server not healthy."; tail -n 60 "$LOG"; exit 2; }
else
  echo ">>> Reusing warm server at $BASE_URL"
  adapter_health >/dev/null 2>&1 || { echo "!!! No healthy server at $BASE_URL"; exit 2; }
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

# ---- optional profile trace ----
if [ "$PROFILE" = "1" ]; then
  echo ">>> Profiling bench ($PROFILE_NUM_STEPS steps) ..."
  mkdir -p "$PROFILE_DIR"
  adapter_bench "$CONC" "$CONC" 1 || echo "!!! profile run failed"
  echo ">>> Trace(s) in $PROFILE_DIR"
fi

# ---- summarize (median throughput across repeats) — backend-independent ----
python3 - "$RESULT_JSONL" "$OUT_DIR/bench_summary.json" <<'PY'
import json, sys, statistics
runs_path, out_path = sys.argv[1], sys.argv[2]
def pick(d, *keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)): return float(d[k])
    return None
tps, ttft, tpot = [], [], []
with open(runs_path) as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        v = pick(d, "output_throughput", "output_token_throughput", "output_throughput_tok_s")
        if v is not None: tps.append(v)
        t = pick(d, "median_ttft_ms", "mean_ttft_ms");   ttft.append(t) if t is not None else None
        p = pick(d, "median_tpot_ms", "mean_tpot_ms");   tpot.append(p) if p is not None else None
def med(xs): return statistics.median(xs) if xs else None
def spread(xs):
    if len(xs) < 2: return 0.0
    m = med(xs); return round(100.0 * (max(xs)-min(xs)) / m, 2) if m else 0.0
summ = {
    "output_throughput_tok_s_median": round(med(tps), 3) if tps else None,
    "output_throughput_tok_s_spread_pct": spread(tps),
    "ttft_ms_median": round(med(ttft), 3) if ttft else None,
    "tpot_ms_median": round(med(tpot), 3) if tpot else None,
    "runs": len(tps),
    "all_throughput": tps,
    # Aggregate output tok/s (NOT divided by TP) — matches Hyperloom/Magpie output_throughput 口径.
    "metric_basis": "aggregate_output_tok_s",
}
with open(out_path, "w") as fh: json.dump(summ, fh, indent=2)
print(f"E2E_SUMMARY output_tok_s={summ['output_throughput_tok_s_median']} "
      f"spread={summ['output_throughput_tok_s_spread_pct']}% "
      f"ttft_ms={summ['ttft_ms_median']} tpot_ms={summ['tpot_ms_median']} runs={summ['runs']}")
PY

echo ">>> Done. Summary: $OUT_DIR/bench_summary.json"
