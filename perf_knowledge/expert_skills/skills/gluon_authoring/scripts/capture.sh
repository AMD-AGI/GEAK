#!/usr/bin/env bash
# Shim: capture.sh now lives in kernel_workflow/scripts/kernel_tools/capture.sh (a GEAK shared kernel tool).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/capture.sh" "$@"
