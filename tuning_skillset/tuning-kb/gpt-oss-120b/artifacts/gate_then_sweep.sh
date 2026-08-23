#!/usr/bin/env bash
# 1) gsm8k gate on the patch-01 arm (server up, patch applied)
# 2) stop the server, sweep the extend-kernel launch config ON TOP of patch 01
set -u
SGL=/sgl-workspace/sglang
F=python/sglang/kernels/ops/attention/extend_attention.py

cd "$SGL" || exit 1
git checkout -- "$F"
git apply --whitespace=nowarn /work/patches/01-prefill-swa-loop-bound.patch || {
  echo "APPLY FAILED"; exit 1; }
echo "== patch 01 applied =="
git diff --stat -- "$F"

cd /work || exit 1
./scripts/launch_server.sh --stop >/dev/null 2>&1
sleep 8
if ! ./scripts/launch_server.sh > /tmp/gate01_launch.log 2>&1; then
  echo "LAUNCH FAILED"; tail -5 /tmp/gate01_launch.log; exit 1
fi
grep -q "config verified" /tmp/gate01_launch.log || {
  echo "CONFIG NOT VERIFIED"; tail -5 /tmp/gate01_launch.log; exit 1; }
echo "== server up, config verified =="

TAG=patch01 ./scripts/run_eval.sh 2>&1 | tail -30
echo "== eval done =="

./scripts/launch_server.sh --stop >/dev/null 2>&1
sleep 10
echo "== server stopped, starting extend-config sweep (patch 01 still applied) =="
python3 /work/analysis/bench_extend_cfg.py --rounds 5 --iters 12 2>&1

cd "$SGL" && git checkout -- "$F"
echo "GATE+SWEEP DONE (tree restored)"
