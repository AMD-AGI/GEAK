#!/usr/bin/env bash
# script-mode entry point (contract §2 A). The logic lives in adopt_ci.py.
set -uo pipefail
exec python3 -I "$SKILL_ROOT/adopt_ci.py" "$@"
