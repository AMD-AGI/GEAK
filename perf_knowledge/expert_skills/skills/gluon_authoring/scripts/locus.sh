#!/usr/bin/env bash
# Source-compatible entry point for GEAK shared host/container helpers.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/kernel_workflow/scripts/kernel_tools/locus.sh"
if [ "${BASH_SOURCE[0]}" = "${0}" ] && [ "${1:-}" = "--selftest" ]; then
  _locus_selftest
fi
