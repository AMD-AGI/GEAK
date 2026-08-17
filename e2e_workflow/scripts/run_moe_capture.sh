#!/usr/bin/env bash
# Capture boot for a grouped fused-MoE oracle: runs bench_e2e.sh once with a capture overlay
# installed, so the target apply() records its real inputs/outputs to a replayable oracle.
#
# --enforce-eager (added to EXTRA_SERVER_ARGS below) is a CAPTURE-ONLY accommodation: the decode-path
# apply() calls are otherwise CUDA-graph-replayed and never re-enter Python, so nothing would be
# recorded. It does NOT change apply()'s math, and the deployment regime (cuda_graph=true) is what
# meta.regime records and the unittest honors.
#
# Required env:
#   TASK        kernel task dir (holds _capture_overlay/ and receives e2e_bench_out/)
#   MODEL       model path or HF id
# Optional env:
#   BACKEND=vllm  TP=8  GPU=0,..,TP-1  MEM_FRACTION=0.9
#   ISL=1024  OSL=1024  CONC=64  REPEATS=1
#   CAPTURE_EXTRA_SERVER_ARGS   appended to the capture server args
#   CAPTURE_EXTRA_ENV           appended to EXTRA_ENV (e.g. backend-specific AITER toggles)
#   CACHE_ROOT=/dev/shm/moe_cap_caches
#
# Example:
#   TASK=$EVAL/kernels/matmul_ogs_mxfp4_grouped_fused_moe_task \
#   MODEL=/models/SomeMoE TP=8 \
#   CAPTURE_EXTRA_SERVER_ARGS="--kv-cache-dtype fp8_ds_mla --max-model-len 6144" \
#   bash e2e_workflow/scripts/run_moe_capture.sh
set -uo pipefail

: "${TASK:?set TASK to the kernel task dir}"
: "${MODEL:?set MODEL to the model path or HF id}"

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP="$TASK/_capture_overlay"

export BACKEND="${BACKEND:-vllm}"
export MODEL
export TP="${TP:-8}"
export GPU="${GPU:-$(seq -s, 0 $((TP - 1)))}"
export MEM_FRACTION="${MEM_FRACTION:-0.9}"
export EXTRA_SERVER_ARGS="${CAPTURE_EXTRA_SERVER_ARGS:-} --enforce-eager"

# Redirect ALL transient caches onto a RAM-backed volume. A shared /home can oscillate to 100% as
# sibling e2e runs write multi-GB oracles, and a cache write landing in that window kills the boot
# (sitecustomize OSError Errno 122). None of these caches affect the captured math — they are
# JIT/compile/tokenizer scratch.
_CACHE="${CACHE_ROOT:-/dev/shm/moe_cap_caches}"
mkdir -p "$_CACHE"/{triton,inductor,xdg,vllm,hf,tmp,ptk,run}

# PYTHONSAFEPATH=1 stops Python auto-prepending CWD/script-dir to sys.path, so the task dir's
# unittest.py can NEVER shadow stdlib `unittest` (a vllm import chain does `from unittest import
# mock`). We ALSO cd to a neutral scratch dir below so CWD is not the task dir at all. Belt and
# suspenders.
export EXTRA_ENV="PYTHONSAFEPATH=1 \
TRITON_CACHE_DIR=$_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$_CACHE/inductor \
XDG_CACHE_HOME=$_CACHE/xdg VLLM_CACHE_ROOT=$_CACHE/vllm HF_HOME=$_CACHE/hf \
PYTORCH_KERNEL_CACHE_PATH=$_CACHE/ptk TMPDIR=$_CACHE/tmp ${CAPTURE_EXTRA_ENV:-}"

export OUT_DIR="$TASK/e2e_bench_out"   # pin explicitly: we cd to a neutral scratch dir, not $TASK
export OVERLAY_PYTHONPATH="$CAP"
export OVERLAY_KIND=capture
export ISL="${ISL:-1024}" OSL="${OSL:-1024}" CONC="${CONC:-64}"
export REPEATS="${REPEATS:-1}"
export PROFILE=0

# Run from a NEUTRAL scratch dir (not $TASK) so CWD never lands the task's unittest.py on sys.path.
# bench_e2e.sh uses absolute out-dir + lock paths, so CWD is irrelevant to its bookkeeping.
cd "$_CACHE/run" || exit 1
bash "$SKILL/scripts/bench_e2e.sh"
echo "BENCH_E2E_EXIT=$?"

echo "=== oracle shards ==="
ls -la "${CAPTURE_OUT:-$TASK}"/reference_io.pt* "${CAPTURE_OUT:-$TASK}"/capture_meta.json* 2>/dev/null \
  || echo "NO SHARDS WRITTEN"
