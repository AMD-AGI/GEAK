#!/usr/bin/env bash
# Legacy Gluon entry. Full GEAK checkout required; GEAK_ROOT supports relocation.
set -euo pipefail
if [ -n "${GEAK_ROOT:-}" ]; then
  root="$GEAK_ROOT"
else
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$root" != / ] && [ ! -f "$root/kernel_workflow/scripts/kernel_tools/dump_ir.sh" ]; do
    root="$(dirname "$root")"
  done
fi
shared="$root/kernel_workflow/scripts/kernel_tools"
if [ ! -f "$shared/dump_ir.sh" ] || [ ! -f "$shared/collect_ir.py" ]; then
  echo 'ERROR: Gluon dump_ir requires GEAK shared kernel tools. Use a full GEAK checkout or set GEAK_ROOT=/path/to/GEAK (inside the container when applicable).' >&2
  exit 3
fi
exec bash "$shared/dump_ir.sh" "$@"
