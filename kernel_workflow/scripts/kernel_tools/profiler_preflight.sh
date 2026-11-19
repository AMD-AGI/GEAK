#!/usr/bin/env bash
# Report the dependencies in the same execution locus as capture.sh. No GPU work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = --help ]; then
  echo 'Usage: profiler_preflight.sh [--att-only] (JSON; respects TILE_KERNEL_CONTAINER)'
  exit 0
fi
source "$HERE/_kt_common.sh"
locus_preflight compile-only || exit 4
script="$(locus_path "$HERE/profiler_deps.py")"
locus_run python3 "$script" "$@"
