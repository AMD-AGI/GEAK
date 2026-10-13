#!/usr/bin/env bash
# runner: stub — calls no agent at all; used to exercise the scheduler and the contract
# itself.
#
# It exists as evidence for the contract rather than as a testing convenience: the claim
# that the scheduler knows nothing about agents is only proven when an implementation
# that is not an agent at all still carries the whole chain through. tests/ and the
# fault-tolerance drills use it — no tokens, no GPUs.
#
#   bash runners/stub.sh <prompt_file> <workdir> <log_file>
#
# Behaviour is driven by environment variables; by default it simply succeeds:
#   STUB_RC=<n>         always return this code (78 exercises the blocked branch)
#   STUB_SLEEP=<sec>    sleep first, to exercise the outer timeout
#   STUB_SCRIPT=<path>  run this script inside workdir, to fake an attempt's artifacts
set -uo pipefail

PROMPT="${1:?prompt_file}"
WORKDIR="${2:?workdir}"
LOG="${3:?log_file}"

mkdir -p "$(dirname "$LOG")" "$WORKDIR" 2>/dev/null
[ -f "$PROMPT" ] || { echo "[runner:stub] BLOCKED no such prompt $PROMPT" >>"$LOG"; exit 78; }
cd "$WORKDIR" || { echo "[runner:stub] BLOCKED cannot enter $WORKDIR" >>"$LOG"; exit 78; }

{
  echo "[runner:stub] $(date -Is) workdir=$WORKDIR"
  echo "[runner:stub] prompt is $(wc -c <"$PROMPT") bytes, first 5 lines:"
  head -5 "$PROMPT" | sed 's/^/    /'
} >> "$LOG"

[ -n "${STUB_SLEEP:-}" ] && sleep "$STUB_SLEEP"

RC="${STUB_RC:-0}"
if [ -n "${STUB_SCRIPT:-}" ] && [ -f "$STUB_SCRIPT" ]; then
  bash "$STUB_SCRIPT" >> "$LOG" 2>&1
  RC=$?
fi

echo "[runner:stub] $(date -Is) exit=$RC" >> "$LOG"
exit "$RC"
