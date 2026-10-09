#!/usr/bin/env bash
# Shim: gpu_lock.sh now lives in kernel_workflow/scripts/gpu_lock.sh (GEAK's single GPU lock).
#
# There is ONE lock implementation in this repo. This shim execs it, so a pack command and a GEAK
# role command on the same GPU id flock the same /tmp/team_gpu_locks/gpu_<id>.lock and serialize.
# Everything the pack's former copy did is reached through it: the flock + idleness check, the
# per-workspace TORCH_EXTENSIONS_DIR, the arch pin (ROCR-scoped, mixed-ISA refusal), the enumerator
# reap, --help / --selftest, the use-log `mode` field, and the OPTIONAL fleet broker seam
# (../scheduler, enabled only with GEAK_GPU_BROKER=1). See `bash gpu_lock.sh --help`.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/gpu_lock.sh" "$@"
