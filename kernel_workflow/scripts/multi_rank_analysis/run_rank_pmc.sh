#!/bin/bash
set -euo pipefail

target_rank="${GEAK_PMC_RANK:-0}"
local_rank="${LOCAL_RANK:-${RANK:-}}"
if [ -z "$local_rank" ]; then
    echo "GEAK PMC wrapper requires LOCAL_RANK or RANK" >&2
    exit 2
fi
if [ "$local_rank" != "$target_rank" ]; then
    exec "$@"
fi

: "${GEAK_PMC_OUTPUT_DIR:?GEAK_PMC_OUTPUT_DIR is required}"
: "${GEAK_PMC_COUNTERS:?GEAK_PMC_COUNTERS is required}"

read -r -a counters <<< "$GEAK_PMC_COUNTERS"
if [ "${#counters[@]}" -eq 0 ]; then
    echo "GEAK_PMC_COUNTERS must contain at least one counter" >&2
    exit 2
fi

extra_args=()
if [ -n "${GEAK_PMC_KERNEL_REGEX:-}" ]; then
    extra_args+=(--kernel-include-regex "$GEAK_PMC_KERNEL_REGEX")
fi

exec rocprofv3 \
    --pmc "${counters[@]}" \
    "${extra_args[@]}" \
    --output-directory "$GEAK_PMC_OUTPUT_DIR" \
    --output-file "${GEAK_PMC_OUTPUT_NAME:-pmc}" \
    --output-format csv \
    -- "$@"
