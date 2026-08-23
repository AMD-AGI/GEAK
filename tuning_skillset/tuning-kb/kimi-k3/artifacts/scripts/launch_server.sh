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

MODEL="${MODEL:-/shared_nfs/hyperloom/models/Kimi-K3}"
PORT="${PORT:-43113}"
LOG="${LOG:-/tmp/sglang_server_kimi_k3.log}"
PIDFILE=/tmp/sglang_server_kimi_k3.pid

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

# The reference environment is NOT fully described by the run's own config.yaml. That file records
# only SGLANG_USE_AITER=1, but the harness process that launched the server carried an ambient
# environment on top of it, and the session lost most of a day to exactly this: subprocesses that
# inherited none of it produced a server that OOMed at --mem-fraction-static 0.8, which was
# misdiagnosed as a host fault ("weight footprint drifted 194.38 -> 249.29 GB/rank") before being
# traced to the missing vars. SGLANG_AITER_K3_OPT alone accounts for +54.9 GB/rank: mxfp4.py:148 and
# :440 make it pick a 128-byte routed-expert intermediate alignment instead of 256, and at TP=8 this
# model's intermediate_size_per_partition is 384, which 256-align rounds up to 512. kimi_k3.py:121
# branches on it too, so it is a code-path choice and not only a memory one.
#
# These are therefore part of the frozen configuration, restored, not tuning knobs to play with. The
# fingerprint check after startup is what proves they took effect.
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_SITUV2_A8W4=1
export SGLANG_MOE_PADDING=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_AITER_MLA_PERSIST=1
export AITER_FLYDSL_FORCE=1
export HSA_NO_SCRATCH_RECLAIM=1

# Serve this model only from the K3 container (see scripts/start_container.sh): sglang
# 0.5.15.post1.dev20260723+g6c9fd0adc5, which is the reference build itself. The *released*
# 0.5.15.post1 shares the version prefix but is a different tree and cannot serve Kimi-K3 at all --
# no KimiK3ForConditionalGeneration in the registry, no sglang.srt.configs.kimi_k3, and its
# kimi_linear.py has no counterpart for most of the checkpoint's weight groups
# (block_sparse_moe experts' weight_packed/weight_scale, routed_expert_up_proj/down_proj,
# self_attention_res_proj, mlp_res_proj, self_attn.g_proj). Evidence:
# analysis/stock_0515_cannot_serve_kimi_k3.txt. So a rejected 'kimi_k3' parser flag means you are
# on the wrong image, not that the flag is optional -- fix the image rather than the flags.
ARGS=(
    --model-path "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tp-size 8
    --context-length 11264
    --watchdog-timeout 1800
    --attention-backend triton
    --dtype bfloat16
    --cuda-graph-max-bs 256
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    # trust_remote_code=True in the reference ServerArgs, and the checkpoint's
    # configuration_kimi_k3.py needs it. Passed explicitly because the flag defaults to False.
    --trust-remote-code
    --moe-runner-backend aiter
    --mem-fraction-static 0.8
    --chunked-prefill-size 16384
    --disable-radix-cache
    --max-running-requests 64
)

echo "[server] port=$PORT log=$LOG tp=8"
nohup python3 -m sglang.launch_server "${ARGS[@]}" >"$LOG" 2>&1 &
echo "$!" > "$PIDFILE"

echo "[server] waiting for /health (up to 5400s; a first start also JIT-compiles aiter kernels)"
READY=0
for i in $(seq 1 5400); do
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

# Startup fingerprint. The reference's eight server starts agree to the byte on all three of these,
# and each one catches a different way of being subtly wrong:
#   mem usage=194.38 GB       -- the 128-aligned routed-expert footprint, i.e. SGLANG_AITER_K3_OPT
#                                took effect. 249.29 GB means the env above was lost.
#   max_mamba_cache_size: 64  -- the explicit --max-running-requests branch in kv_cache_configurator
#                                was taken, rather than the auto-fit that lands on 244 and spends
#                                ~9.5 GB/rank of hybrid state cache on slots this workload never uses.
#   #tokens: 922585           -- the resulting full-attention token pool. A pool near 556,885 is the
#                                pre-baseline sizing; anything much smaller will queue at
#                                concurrency 64 and "regress" for reasons that have nothing to do
#                                with whatever is being tested.
# A number measured on a server that fails this is not comparable to 804.190 tok/s.
FP_OK=1
grep -q 'mem usage=194.38 GB' "$LOG" || { echo "[server] FINGERPRINT: weight footprint is not 194.38 GB/rank:"; grep -o 'Load weight end.*' "$LOG" | tail -1; FP_OK=0; }
grep -q 'max_mamba_cache_size: 64' "$LOG" || { echo "[server] FINGERPRINT: mamba cache is not 64 slots:"; grep -o 'Mamba Cache is allocated.*' "$LOG" | tail -1; FP_OK=0; }
grep -q '#tokens: 922585' "$LOG" || { echo "[server] FINGERPRINT: token pool differs from the reference:"; grep -o 'KV Cache is allocated.*' "$LOG" | tail -1; FP_OK=0; }
if [ "$FP_OK" = 1 ]; then
    echo "[server] fingerprint matches the reference (194.38 GB/rank, mamba 64, pool 922585)"
else
    echo "[server] WARNING: this server is NOT the reference configuration -- see above." >&2
    echo "[server]          Do not measure against 804.190 tok/s until it matches." >&2
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
        "tp_size": 8,
        "attention_backend": "triton",
        "max_running_requests": 64,
        "cuda_graph_max_bs": 256,
        "page_size": 1,
        "chunked_prefill_size": 16384,
        "disable_radix_cache": True
}
info = json.loads(sys.argv[1])
args = info.get("server_args", info)
bad = {k: (v, args.get(k)) for k, v in want.items() if k in args and args[k] != v}

# SGLang rescales mem_fraction_static by 0.85 on builds that combine aiter with a context length
# above 8192, so both the requested and the rescaled value are legitimate here.
mfs = args.get("mem_fraction_static")
if mfs is not None and not any(abs(mfs - e) < 1e-6 for e in (0.8, 0.8 * 0.85)):
    bad["mem_fraction_static"] = ("0.8 or 0.68", mfs)

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
