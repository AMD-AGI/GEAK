#!/usr/bin/env bash
# script-mode entry point (contract §2 A). The logic lives in adopt.py; this only
# forwards to it.
set -uo pipefail
exec python3 "$SKILL_ROOT/adopt.py" "$@"
