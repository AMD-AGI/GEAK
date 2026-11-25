#!/bin/bash
# Forwarder: AVO reuses GEAK's GPU lock (kernel_workflow/scripts/gpu_lock.sh).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../../kernel_workflow/scripts" && pwd)/gpu_lock.sh" "$@"
