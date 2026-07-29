#!/usr/bin/env bash
# parity_run.sh <out_json> <extra_server_args...>
set -uo pipefail
EVAL=/wekafs/test_results/Qwen3_14B_20260728/e2e_Qwen3-14B-FP8_20260727_081246_1641958_1684
OUT="$1"; shift
EXTRA="$*"
pkill -f "sglang.launch_server" 2>/dev/null; sleep 3
GA=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
env SGLANG_USE_AITER=1 GPU_ARCHS=$GA HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/sgl-workspace/sglang/python \
  python -m sglang.launch_server --model-path /models/Qwen3-14B-FP8 \
    --host 127.0.0.1 --port 30500 --tp-size 1 --mem-fraction-static 0.9 \
    --watchdog-timeout 600 --fp8-gemm-backend aiter $EXTRA \
    > "${OUT%.json}_server.log" 2>&1 &
SPID=$!
echo "server pid $SPID extra=[$EXTRA]"
for i in $(seq 1 120); do
  curl -sf -m 3 http://127.0.0.1:30500/health >/dev/null 2>&1 && { echo "READY ${i}0s"; break; }
  sleep 10
done
python3 /wekafs/GEAK/e2e_workflow/parity_probe.py http://127.0.0.1:30500 "$OUT"
kill $SPID 2>/dev/null; sleep 4; pkill -f "sglang.launch_server" 2>/dev/null
echo "PARITY_DONE $OUT"
