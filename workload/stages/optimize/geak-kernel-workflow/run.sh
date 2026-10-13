#!/usr/bin/env bash
# Scheduling, timing, fault tolerance and re-verification are the same for any
# optimization method, so they live at slot level. This skill provides only prompt.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/orchestrate.sh"
