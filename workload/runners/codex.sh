#!/usr/bin/env bash
# runner: codex. Contract in CONTRACT.md.
#
# Status: **written to the contract, not yet exercised** — the development container has
# no codex binary. Every precondition check returns an honest 78 rather than pretending
# to run. Before the first real use, confirm at least these two things:
#   1. the codex binary is on PATH (or CODEX_BIN points at it);
#   2. the code-mode-host sidecar is present. It is a separate release asset, and without
#      it tool calls **fail closed silently** — the UI shows a single "Started" line, the
#      agent does nothing, and the exit code may still be 0. That kind of "success" costs
#      far more than an error, so it is treated here as a blocker, not a warning.
#
#   bash runners/codex.sh <prompt_file> <workdir> <log_file>
#   Exit codes: 0 finished / 78 unmet precondition (do not retry) / other non-zero
#   session failure (retryable)
set -uo pipefail

PROMPT="${1:?prompt_file}"
WORKDIR="${2:?workdir}"
LOG="${3:?log_file}"

mkdir -p "$(dirname "$LOG")" "$WORKDIR" 2>/dev/null
blocked() { echo "[runner:codex] BLOCKED $*" | tee -a "$LOG" >&2; exit 78; }

# ---- 1. Precondition self-check.
BIN="${CODEX_BIN:-}"
if [ -z "$BIN" ]; then BIN="$(command -v codex 2>/dev/null)"; fi
if [ -z "$BIN" ]; then
  for c in /root/.local/bin/codex "$HOME/.local/bin/codex" /usr/local/bin/codex; do
    [ -x "$c" ] && { BIN="$c"; break; }
  done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || blocked "no codex executable found. Install it, or point CODEX_BIN at it."
[ -f "$PROMPT" ] || blocked "prompt file does not exist: $PROMPT"

# The sidecar. Its absence raises no error and simply gets nothing done, so check for it.
SIDECAR="${CODEX_CODE_MODE_HOST:-}"
if [ -z "$SIDECAR" ]; then SIDECAR="$(command -v code-mode-host 2>/dev/null)"; fi
if [ -z "$SIDECAR" ]; then
  for c in "$(dirname "$BIN")/code-mode-host" "$HOME/.codex/bin/code-mode-host" \
           /usr/local/bin/code-mode-host; do
    [ -x "$c" ] && { SIDECAR="$c"; break; }
  done
fi
[ -n "$SIDECAR" ] && [ -x "$SIDECAR" ] || blocked \
  "no code-mode-host sidecar found. It is a separate release asset, and without it tool calls fail closed silently: the session looks alive, the exit code may be 0, and the agent has done nothing at all. Install it, or point CODEX_CODE_MODE_HOST at it."

if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f "${CODEX_HOME:-$HOME/.codex}/auth.json" ]; then
  blocked "neither OPENAI_API_KEY nor ${CODEX_HOME:-$HOME/.codex}/auth.json is present."
fi

# ---- 2. State directory on /dev/shm, not NFS (same reason as claude.sh: a poisoned
#         sqlite -shm).
STATE_BASE="${WORKLOAD_AGENT_STATE_BASE:-/dev/shm/workload_agent_state}"
STATE_DIR="${WORKLOAD_AGENT_STATE_DIR:-$STATE_BASE/codex.$(basename "$WORKDIR").$$}"
mkdir -p "$STATE_DIR" || blocked "cannot create state directory $STATE_DIR"

SRC_CFG="${CODEX_SOURCE_HOME:-${CODEX_HOME:-$HOME/.codex}}"
if [ -d "$SRC_CFG" ] && [ "$SRC_CFG" != "$STATE_DIR" ]; then
  for f in auth.json config.toml; do
    [ -f "$SRC_CFG/$f" ] && cp -f "$SRC_CFG/$f" "$STATE_DIR/$f" 2>/dev/null
  done
fi
export CODEX_HOME="$STATE_DIR"

# ---- 3. Make the model explicit, for the same reason as claude.sh: the environment may
#         already export a model name.
MODEL_ARGS=()
if [ -n "${WORKLOAD_AGENT_MODEL:-}" ]; then
  MODEL_ARGS=(--model "$WORKLOAD_AGENT_MODEL")
fi

# ---- 4. GPU visibility: not touched at all. Physical isolation is the caller's per-GPU
#         container.

cd "$WORKDIR" || blocked "cannot enter workdir $WORKDIR"

{
  echo "[runner:codex] $(date -Is) bin=$BIN sidecar=$SIDECAR workdir=$WORKDIR state=$STATE_DIR"
  echo "[runner:codex] this runner has not been exercised; on first use, confirm the log actually shows tool calls happening."
} >> "$LOG"

# Non-interactive, allowed to act on its own. No internal timeout: the time budget is
# enforced from the outside by the scheduler with timeout(1).
"$BIN" exec \
  --cd "$WORKDIR" \
  --dangerously-bypass-approvals-and-sandbox \
  "${MODEL_ARGS[@]}" \
  ${WORKLOAD_AGENT_EXTRA_ARGS:-} \
  - < "$PROMPT" >> "$LOG" 2>&1
RC=$?

echo "[runner:codex] $(date -Is) exit=$RC" >> "$LOG"
rm -rf "$STATE_DIR" 2>/dev/null
exit $RC
