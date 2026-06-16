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
  echo "    Available: $(ls "$HERE/adapters" 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' ')" >&2
  exit 3
fi
# shellcheck disable=SC1090
source "$ADAPTER"
for fn in adapter_launch adapter_health adapter_bench; do
  if ! declare -F "$fn" >/dev/null; then
    echo "!!! Adapter $ADAPTER does not define $fn()" >&2; exit 3
  fi
done

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
  # 0 (or a busy port) -> ask the OS for a free one.
  FREE_PORT=$(python3 - <<'PY' 2>/dev/null || true
import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
)
  [ -n "$FREE_PORT" ] && PORT="$FREE_PORT"
fi

# ---- workload ----
ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONC=${CONC:-64}
NUM_PROMPTS=${NUM_PROMPTS:-$((CONC * 5))}
REPEATS=${REPEATS:-3}                 # repeat the bench this many times; report median + spread
SEED=${SEED:-0}                       # fixed seed for reproducibility / parity

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
}
with open(out_path, "w") as fh: json.dump(summ, fh, indent=2)
print(f"E2E_SUMMARY output_tok_s={summ['output_throughput_tok_s_median']} "
      f"spread={summ['output_throughput_tok_s_spread_pct']}% "
      f"ttft_ms={summ['ttft_ms_median']} tpot_ms={summ['tpot_ms_median']} runs={summ['runs']}")
PY

echo ">>> Done. Summary: $OUT_DIR/bench_summary.json"
