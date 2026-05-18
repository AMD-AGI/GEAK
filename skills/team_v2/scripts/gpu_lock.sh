#!/bin/bash
# GPU lock script - ensures exclusive GPU access for benchmarking
# Usage: bash gpu_lock.sh <gpu_id> <command...>
# Uses flock for exclusive access - multiple engineers can share GPUs safely

set -euo pipefail

GPU_ID="${1:?Usage: gpu_lock.sh <gpu_id> <command...>}"
shift

LOCK_DIR="/tmp/team_v2_gpu_locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/gpu_${GPU_ID}.lock"

(
    flock -x -w 600 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 600s"; exit 1; }
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    "$@"
) 200>"$LOCK_FILE"
