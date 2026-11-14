#!/bin/bash
set -euo pipefail

target_rank="${GEAK_TRACE_RANK:-0}"
local_rank="${LOCAL_RANK:-${RANK:-}}"
if [ -z "$local_rank" ]; then
    echo "GEAK kernel-trace wrapper requires LOCAL_RANK or RANK" >&2
    exit 2
fi
if [ "$local_rank" != "$target_rank" ]; then
    exec "$@"
fi

: "${GEAK_TRACE_OUTPUT_DIR:?GEAK_TRACE_OUTPUT_DIR is required}"

exec rocprofv3 \
    --kernel-trace \
    --stats \
    --output-directory "$GEAK_TRACE_OUTPUT_DIR" \
    --output-file "${GEAK_TRACE_OUTPUT_NAME:-kernel_trace}" \
    --output-format csv \
    -- "$@"
