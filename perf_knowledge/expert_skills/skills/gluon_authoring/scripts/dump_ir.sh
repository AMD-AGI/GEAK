#!/usr/bin/env bash
# Shim: dump_ir.sh now lives in kernel_workflow/scripts/kernel_tools/dump_ir.sh (a GEAK shared kernel tool).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/dump_ir.sh" "$@"
