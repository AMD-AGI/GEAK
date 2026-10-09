#!/usr/bin/env bash
# Shim: hw_sources.sh now lives in kernel_workflow/scripts/kernel_tools/hw_sources.sh (a GEAK shared kernel tool).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/hw_sources.sh" "$@"
