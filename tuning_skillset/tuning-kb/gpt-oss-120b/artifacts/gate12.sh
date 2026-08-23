#!/usr/bin/env bash
# gsm8k gate for patches 01 + 02 stacked.
#
# PATH note: run_eval.sh needs the lm-eval venv, launch_server.sh must not see it
# (the venv has no sglang), so the PATH edit is scoped to the eval invocation.
set -u
SGL=/sgl-workspace/sglang
F=python/sglang/kernels/ops/attention/extend_attention.py

cd "$SGL" || exit 1
git checkout -- "$F"
for p in /work/patches/01-prefill-swa-loop-bound.patch \
         /work/patches/02-prefill-extend-launch-config.patch; do
  git apply --whitespace=nowarn "$p" || { echo "APPLY FAILED $p"; exit 1; }
done
echo "== patches 01+02 applied =="
git diff --stat -- "$F"

cd /work || exit 1
./scripts/launch_server.sh --stop >/dev/null 2>&1
sleep 8
if ! ./scripts/launch_server.sh > /tmp/gate12_launch.log 2>&1; then
  echo "LAUNCH FAILED"; tail -6 /tmp/gate12_launch.log
  cd "$SGL" && git checkout -- "$F"; exit 1
fi
grep -q "config verified" /tmp/gate12_launch.log || {
  echo "CONFIG NOT VERIFIED"; tail -6 /tmp/gate12_launch.log
  cd "$SGL" && git checkout -- "$F"; exit 1; }
echo "== server up, config verified =="

env PATH=/tmp/lmeval_venv/bin:$PATH TAG=patch0102 ./scripts/run_eval.sh 2>&1 | tail -12
echo "== eval done =="

./scripts/launch_server.sh --stop >/dev/null 2>&1
cd "$SGL" && git checkout -- "$F"
echo "GATE12 DONE (tree restored)"
