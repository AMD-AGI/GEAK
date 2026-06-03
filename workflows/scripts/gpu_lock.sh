#!/bin/bash
# GPU lock + per-workspace build isolation wrapper.
# Usage:  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>
#
# Run EVERY kernel command (compile / correctness / benchmark / profile) through this wrapper,
# invoked from inside the workspace directory. It does three generic things — none kernel-specific:
#
#  1. flock per GPU id  -> multiple engineers can share GPUs safely (exclusive during the command).
#  2. TORCH_EXTENSIONS_DIR = <workspace>/.torch_ext  -> isolates the torch cpp_extension build cache
#     PER WORKSPACE. Without this, torch.utils.cpp_extension.load(name=...) compiles every engineer's
#     DIFFERENT source into ONE global cache (~/.cache/torch_extensions/...), which both serializes
#     all parallel compiles on a single global lock AND lets one engineer benchmark another's .so.
#     Deriving it from $PWD makes each isolated workspace get its own cache. (Honors a caller-set
#     TORCH_EXTENSIONS_DIR if already exported.)
#  3. PYTORCH_ROCM_ARCH = the local GPU's gfx arch only -> avoids compiling for ~9 architectures
#     (huge compile speedup). Runtime perf and correctness are unaffected (the kernel runs on the
#     local arch either way). Honors a caller-set PYTORCH_ROCM_ARCH if already exported.

set -euo pipefail

GPU_ID="${1:?Usage: gpu_lock.sh <gpu_id> <command...>}"
shift

LOCK_DIR="/tmp/team_gpu_locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/gpu_${GPU_ID}.lock"

# (2) Per-workspace torch extension build cache (default: a hidden dir in the current workspace).
: "${TORCH_EXTENSIONS_DIR:=$PWD/.torch_ext}"
export TORCH_EXTENSIONS_DIR
mkdir -p "$TORCH_EXTENSIONS_DIR" 2>/dev/null || true

# (3) Compile for the local GPU arch only. The environment's default PYTORCH_ROCM_ARCH is often a
# long multi-arch list (~9 targets) → ~9x slower compiles for no benefit on a single-arch box. We
# OVERRIDE it to the detected local arch. Set KERNEL_ENV_KEEP_ARCH=1 to opt out (multi-arch boxes).
if [ "${KERNEL_ENV_KEEP_ARCH:-0}" != "1" ]; then
    _ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
    [ -n "${_ARCH:-}" ] && export PYTORCH_ROCM_ARCH="$_ARCH"
fi

(
    flock -x -w 1200 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 1200s"; exit 1; }
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    "$@"
) 200>"$LOCK_FILE"
