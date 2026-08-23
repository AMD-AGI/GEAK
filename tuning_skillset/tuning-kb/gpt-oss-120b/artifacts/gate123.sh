#!/usr/bin/env bash
# gsm8k gate for patches 01 + 02 + 03 stacked.
#
# 03 lives in a different repo (aiter) from 01/02 (sglang), so this restores and
# applies per-repo rather than assuming one file.
#
# PATH note: run_eval.sh needs the lm-eval venv, launch_server.sh must not see it
# (the venv has no sglang), so the PATH edit is scoped to the eval invocation.
set -u
SGL=/sgl-workspace/sglang
SGLF=python/sglang/kernels/ops/attention/extend_attention.py
AIT=/sgl-workspace/aiter
AITF=aiter/ops/triton/attention/unified_attention.py

restore() {
  ( cd "$SGL" && git checkout -- "$SGLF" )
  ( cd "$AIT" && git checkout -- "$AITF" )
}

restore
( cd "$SGL" && for p in /work/patches/01-prefill-swa-loop-bound.patch \
                        /work/patches/02-prefill-extend-launch-config.patch; do
    git apply --whitespace=nowarn "$p" || { echo "APPLY FAILED $p"; exit 1; }
  done ) || { restore; exit 1; }
( cd "$AIT" && git apply --whitespace=nowarn \
    /work/patches/03-decode-attn-segments.patch ) \
  || { echo "APPLY FAILED 03"; restore; exit 1; }

echo "== patches 01+02+03 applied =="
( cd "$SGL" && git diff --stat -- "$SGLF" )
( cd "$AIT" && git diff --stat -- "$AITF" )

cd /work || exit 1
./scripts/launch_server.sh --stop >/dev/null 2>&1
sleep 8
if ! ./scripts/launch_server.sh > /tmp/gate123_launch.log 2>&1; then
  echo "LAUNCH FAILED"; tail -6 /tmp/gate123_launch.log; restore; exit 1
fi
grep -q "config verified" /tmp/gate123_launch.log || {
  echo "CONFIG NOT VERIFIED"; tail -6 /tmp/gate123_launch.log; restore; exit 1; }
echo "== server up, config verified =="

env PATH=/tmp/lmeval_venv/bin:$PATH TAG=patch010203 ./scripts/run_eval.sh 2>&1 | tail -12
echo "== eval done =="

./scripts/launch_server.sh --stop >/dev/null 2>&1
restore
echo "GATE123 DONE (trees restored)"
