#!/usr/bin/env bash
# Shim: rocprofv3_safe.sh now lives in kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh (a GEAK shared kernel tool).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh" "$@"
