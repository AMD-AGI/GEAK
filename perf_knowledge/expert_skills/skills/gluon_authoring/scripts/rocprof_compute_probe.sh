#!/usr/bin/env bash
# Shim: rocprof_compute_probe.sh now lives in kernel_workflow/scripts/kernel_tools/rocprof_compute_probe.sh (a GEAK shared kernel tool).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/rocprof_compute_probe.sh" "$@"
