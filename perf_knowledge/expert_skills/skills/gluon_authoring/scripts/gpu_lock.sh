#!/usr/bin/env bash
# Serialize benchmark access to one GPU (borrowed from perf-team).
# Usage: bash gpu_lock.sh <gpu_id> <command...>

set -euo pipefail

GPU_ID="${1:?Usage: gpu_lock.sh <gpu_id> <command...>}"
shift

LOCK_DIR="${PERF_GLUON_TILE_GPU_LOCK_DIR:-/tmp/perf_gluon_tile_gpu_locks}"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/gpu_${GPU_ID}.lock"

(
    flock -x -w 600 200 || {
        echo "ERROR: failed to acquire GPU ${GPU_ID} lock after 600s" >&2
        exit 1
    }
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    "$@"
) 200>"$LOCK_FILE"
