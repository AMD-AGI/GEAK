#!/usr/bin/env bash
# Serialize benchmark access to one GPU (borrowed from perf-team).
# Usage: bash gpu_lock.sh <gpu_id> <command...>

set -euo pipefail

GPU_ID="${1:?Usage: gpu_lock.sh <gpu_id> <command...>}"
shift

# Multinode run (GEAK_MN_STATE) or an <engine>:<id> card: kernel_workflow's wrapper locks the
# shared-mount card leases GEAK's serving legs hold, and forwards an engine's card to its machine.
_KW_LOCK="$(dirname "$(readlink -f "$0")")/../../../../../kernel_workflow/scripts/gpu_lock.sh"
_MN_PY="$(dirname "$_KW_LOCK")/../../e2e_workflow/scripts/multi_node/mn.py"
if [[ "$GPU_ID" == *:* ]] || { [ -f "$_MN_PY" ] && [ -n "${GEAK_MN_STATE:-}" ]; }; then
    exec bash "$_KW_LOCK" "$GPU_ID" "$@"
fi

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
