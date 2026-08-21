#!/usr/bin/env bash
# Launch the SGLang server for this experiment's baseline configuration. INSIDE the container.
#
#   ./launch_server.sh            # start, wait for /health, verify the live config
#   ./launch_server.sh --stop     # kill it
#
# Every flag is taken from the ServerArgs of the reference measurement
# (../reference/results/baseline_warmup/server.log.gz), including the ones that do not look like
# tuning knobs -- --mem-fraction-static, --chunked-prefill-size, --disable-radix-cache,
# --attention-backend, --page-size. The harness set them, so dropping any of them changes the
# number you are comparing against.
set -uo pipefail

MODEL="${MODEL:-/shared_nfs/hyperloom/models/Qwen3-14B-FP8}"
PORT="${PORT:-43102}"
LOG="${LOG:-/tmp/sglang_server_qwen3_14b_fp8.log}"
PIDFILE=/tmp/sglang_server_qwen3_14b_fp8.pid

# The scheduler and detokenizer workers rename themselves to "sglang::scheduler" and
# "sglang::detokenizer". The scheduler holds the KV cache, i.e. essentially all of the VRAM, so
# killing the launcher alone can leave the GPU occupied.
PATTERNS=('sglang.launch_server' 'sglang::')

if [ "${1:-}" = "--stop" ]; then
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null
    for p in "${PATTERNS[@]}"; do pkill -f "$p" 2>/dev/null; done
    for _ in $(seq 1 20); do
        pgrep -f 'sglang.launch_server|sglang::' >/dev/null 2>&1 || break
        sleep 1
    done
    for p in "${PATTERNS[@]}"; do pkill -9 -f "$p" 2>/dev/null; done
    sleep 2
    rm -f "$PIDFILE"
    if pgrep -f 'sglang.launch_server|sglang::' >/dev/null 2>&1; then
        echo "[server] WARNING: processes survived SIGKILL:" >&2
        ps -eo pid,etimes,cmd | grep -E '[s]glang.launch_server|[s]glang::' >&2
        exit 1
    fi
    echo "[server] stopped"
    exit 0
fi

# A server already on this port is almost always a leftover, possibly with a different config.
# Attaching to it silently would measure that config instead of this one, so refuse rather than guess.
if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[server] ERROR: something is already serving on port ${PORT}." >&2
    echo "[server] It may be a stale server with a different config. Run --stop first." >&2
    ps -eo pid,etimes,cmd | grep '[s]glang.launch_server' >&2 || true
    exit 1
fi

ARGS=(
    --model-path "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tp-size 1
    --context-length 11264
    --watchdog-timeout 1800
    --mem-fraction-static 0.68
    --chunked-prefill-size 16384
    --page-size 1
    --disable-radix-cache
    --attention-backend aiter
)

echo "[server] port=$PORT log=$LOG tp=1"
nohup python3 -m sglang.launch_server "${ARGS[@]}" >"$LOG" 2>&1 &
echo "$!" > "$PIDFILE"

echo "[server] waiting for /health (up to 900s; a first start also JIT-compiles aiter kernels)"
READY=0
for i in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[server] ready after ${i}s"; READY=1; break
    fi
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "[server] process died -- tail of $LOG:" >&2; tail -40 "$LOG" >&2; exit 1
    fi
    sleep 1
done
if [ "$READY" != 1 ]; then
    echo "[server] TIMEOUT -- tail of $LOG:" >&2; tail -40 "$LOG" >&2; exit 1
fi

# Confirm the live server is the configuration we intended rather than trusting that the flags were
# honoured. A silent mismatch here invalidates every number measured after it.
INFO="$(curl -sf -m 10 "http://127.0.0.1:${PORT}/get_server_info" 2>/dev/null)"
if [ -z "$INFO" ]; then
    echo "[server] WARNING: /get_server_info unavailable; could not verify config" >&2
else
    python3 - "$INFO" <<'PY'
import json, sys
want = {
        "context_length": 11264,
        "tp_size": 1,
        "attention_backend": "aiter",
        "chunked_prefill_size": 16384,
        "disable_radix_cache": True,
        "page_size": 1
}
info = json.loads(sys.argv[1])
args = info.get("server_args", info)
bad = {k: (v, args.get(k)) for k, v in want.items() if k in args and args[k] != v}

# SGLang rescales mem_fraction_static by 0.85 on builds that combine aiter with a context length
# above 8192, so both the requested and the rescaled value are legitimate here.
mfs = args.get("mem_fraction_static")
if mfs is not None and not any(abs(mfs - e) < 1e-6 for e in (0.68, 0.68 * 0.85)):
    bad["mem_fraction_static"] = ("0.68 or 0.578", mfs)

for k, (exp, got) in bad.items():
    print(f"[server] MISMATCH {k}: expected {exp!r}, server reports {got!r}")
missing = [k for k in want if k not in args]
if missing:
    print(f"[server] note: not reported by this endpoint: {', '.join(missing)}")
print("[server] config verified" if not bad else "[server] CONFIG MISMATCH -- do not measure")
sys.exit(1 if bad else 0)
PY
    [ $? -ne 0 ] && exit 1
fi
exit 0
