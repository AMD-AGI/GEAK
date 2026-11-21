#!/usr/bin/env bash
# Run the avo-workflow Layer-1 deterministic test suite (no GPU, no LLM).
# Python tests always run; node tests run if `node` is available.
# Exit non-zero if any suite fails. See TESTING.md.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0

echo "== avo-workflow Layer-1 deterministic tests =="

# --- Python: lineage writer (geomean / commit gate refusal / context / messages)
if [ -n "${GEAK_PYTHON_BIN:-}" ]; then
  PY="$GEAK_PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
echo; echo "-- pytest ($PY) --"
"$PY" -m pytest "$HERE/tests/" -q || rc=1

# --- JS: every deterministic module/evidence test.
# Discover instead of maintaining a hand-written list, which previously omitted
# ASM screening/formal artifact/session-guard coverage.
NODE_BIN="${GEAK_NODE_BIN:-}"
if [ -z "$NODE_BIN" ] && command -v node >/dev/null 2>&1; then
  NODE_BIN="$(command -v node)"
fi
if [ -z "$NODE_BIN" ] && [ -x "$HERE/../../../tooling/node/bin/node" ]; then
  NODE_BIN="$HERE/../../../tooling/node/bin/node"
fi
if [ -n "$NODE_BIN" ] && [ "$("$NODE_BIN" -p 'Number(process.versions.node.split(".")[0])')" -ge 18 ]; then
  for test_path in "$HERE"/tests/*.test.mjs; do
    [ -e "$test_path" ] || continue
    echo; echo "-- node: $(basename "$test_path") --"
    "$NODE_BIN" --test "$test_path" || rc=1
  done
else
  echo; echo "-- Node.js >= 18 not found: CANNOT run tests/*.test.mjs --"
  echo "   install Node.js >= 18; skipping deterministic gate/evidence tests is not a full pass."
  rc=1
fi

echo
if [ "$rc" -eq 0 ]; then echo "ALL LAYER-1 TESTS PASSED"; else echo "LAYER-1 FAILURES (rc=$rc)"; fi
exit "$rc"
