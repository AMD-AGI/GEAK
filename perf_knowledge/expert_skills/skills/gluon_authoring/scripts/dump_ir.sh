#!/usr/bin/env bash
# Compatibility entry point; shared tools are maintained in kernel_workflow.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/dump_ir.sh" "$@"
