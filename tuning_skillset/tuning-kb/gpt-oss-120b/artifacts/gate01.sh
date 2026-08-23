#!/usr/bin/env bash
# gsm8k gate for patch 01.
#
# run_eval.sh needs the lm-eval venv on PATH, but launch_server.sh must NOT see
# it -- the venv has no sglang, so `python3 -m sglang.launch_server` dies with
# ModuleNotFoundError. So the PATH edit is scoped to the eval invocation only.
set -u
SGL=/sgl-workspace/sglang
F=python/sglang/kernels/ops/attention/extend_attention.py

cd "$SGL" || exit 1
git checkout -- "$F"
git apply --whitespace=nowarn /work/patches/01-prefill-swa-loop-bound.patch || {
  echo "APPLY FAILED"; exit 1; }
echo "== patch 01 applied =="

cd /work || exit 1
./scripts/launch_server.sh --stop >/dev/null 2>&1
sleep 8
if ! ./scripts/launch_server.sh > /tmp/gate01b_launch.log 2>&1; then
  echo "LAUNCH FAILED"; tail -6 /tmp/gate01b_launch.log
  cd "$SGL" && git checkout -- "$F"; exit 1
fi
grep -q "config verified" /tmp/gate01b_launch.log || {
  echo "CONFIG NOT VERIFIED"; tail -6 /tmp/gate01b_launch.log
  cd "$SGL" && git checkout -- "$F"; exit 1; }
echo "== server up, config verified =="

env PATH=/tmp/lmeval_venv/bin:$PATH TAG=patch01 ./scripts/run_eval.sh 2>&1 | tail -40
echo "== eval done =="

./scripts/launch_server.sh --stop >/dev/null 2>&1
cd "$SGL" && git checkout -- "$F"
echo "GATE01 DONE (tree restored)"
