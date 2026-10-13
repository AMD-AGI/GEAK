#!/usr/bin/env bash
# Same as geak-kernel-workflow: reuse the slot-level orchestration; the only difference
# is prompt.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/orchestrate.sh"
