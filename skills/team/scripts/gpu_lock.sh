#!/bin/bash
# gpu_lock.sh — Execute a command with exclusive GPU access via flock.
# Ensures only one benchmark/profile runs on a given GPU at a time,
# so multiple engineers can share GPUs without timing interference.
#
# Usage: gpu_lock.sh <gpu_id> <command...>
# Example: gpu_lock.sh 0 python3 scripts/task_runner.py performance
#          gpu_lock.sh 2 rocprof-compute profile --no-roof -- python3 harness.py --profile

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <command...>" >&2
    exit 1
fi

GPU_ID="$1"; shift
LOCK_DIR="/tmp/team_gpu_locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/gpu_${GPU_ID}.lock"

(
    flock -x 200
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    "$@"
) 200>"$LOCK_FILE"
