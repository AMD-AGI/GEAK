#!/bin/bash
# GPU lock script - ensures exclusive GPU access for benchmarking
# Usage: bash gpu_lock.sh <gpu_id> <command...>
# Uses flock for exclusive access - multiple engineers can share GPUs safely

set -euo pipefail

GPU_ID="${1:?Usage: gpu_lock.sh <gpu_id> <command...>}"
shift

LOCK_DIR="/tmp/team_gpu_locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/gpu_${GPU_ID}.lock"

# Reap ORPHANED hung rocm_agent_enumerator procs (aiter import spawns one per Python proc; they hang
# under GPU/KFD contention and pile up -> task-count box-hang). Kill only ppid==1 + >60s old = safe.
if [ "${KERNEL_ENV_SKIP_ENUM_REAP:-0}" != "1" ]; then
    for _p in $(pgrep -f rocm_agent_enumerator 2>/dev/null || true); do
        _pp="$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        _et="$(ps -o etimes= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        if [ "${_pp:-0}" = "1" ] && [ -n "${_et:-}" ] && [ "${_et:-0}" -gt 60 ] 2>/dev/null; then
            kill -9 "$_p" 2>/dev/null || true
        fi
    done
fi

# Pin GPU_ARCHS so aiter's JIT (chip_info.get_gfx_list) uses the env instead of _detect_native(),
# which shells to rocm_agent_enumerator -> rocminfo per cold-build worker (~77 per cold import) and
# storms the box under contention. Detect once; honor a caller-set value.
if [ -z "${GPU_ARCHS:-}" ]; then
    _ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
    [ -n "${_ARCH:-}" ] && export GPU_ARCHS="$_ARCH"
fi

(
    flock -x -w 600 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 600s"; exit 1; }
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    "$@"
) 200>"$LOCK_FILE"
