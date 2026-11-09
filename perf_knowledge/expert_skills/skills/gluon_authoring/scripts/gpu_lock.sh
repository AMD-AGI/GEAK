#!/usr/bin/env bash
# Compatibility entry point: Gluon and other GEAK commands share the public GPU lock namespace.
# Single-card locks and multi-GPU leases remain owned by kernel_workflow/scripts.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/gpu_lock.sh" "$@"
