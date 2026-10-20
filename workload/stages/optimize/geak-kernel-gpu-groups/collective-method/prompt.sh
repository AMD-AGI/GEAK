#!/usr/bin/env bash
# Preserve the whole device set for true multi-GPU tasks.
set -euo pipefail
case "${GPU_IDS:?}" in *,*) ;; *) echo 'collective method requires multiple GPUs' >&2; exit 2;; esac
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../native-task" && pwd)/prompt.sh"
