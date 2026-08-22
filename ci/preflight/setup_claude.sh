#!/usr/bin/env bash
# Step D — install Claude Code inside the container and probe it.
# All Claude state (install + config + logs) is kept under $CLAUDE_HOME so it
# survives outside the ephemeral container, in the run's timestamped folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

: "${CLAUDE_HOME:?set CLAUDE_HOME (e.g. <out_dir>/claude)}"
mkdir -p "$CLAUDE_HOME"
export HOME="$CLAUDE_HOME"          # redirect ~/.claude, ~/.claude.json, ~/.local/bin here

log "installing Claude Code into HOME=$HOME"
bash "$CLAUDE_SETUP"
export PATH="$HOME/.local/bin:$PATH"

# run_e2e.py prefers the Python SDK path (_invoke_via_sdk), which is the ONLY path
# that survives Claude Code routing the Workflow to a background task. Without the
# SDK it falls back to `claude -p` (_invoke_via_cli), which tears the backgrounded
# workflow down after baseline -> workflow_parse_error. Install it into the same
# python3 that runs run_e2e.
log "installing claude_agent_sdk (python) for run_e2e SDK path"
# Retry: a transient PyPI/network blip here is just as fatal as the claude install
# (it runs before the e2e workflow), so don't let one flake sink a multi-hour job.
_pip_sdk() {
  python3 -m pip install --quiet --no-input --break-system-packages claude-agent-sdk \
    || python3 -m pip install --quiet --no-input claude-agent-sdk
}
_delay=5
for _try in 1 2 3 4 5; do
  if _pip_sdk && python3 -c "import claude_agent_sdk" 2>/dev/null; then
    log "claude_agent_sdk installed (attempt $_try)"
    break
  fi
  [ "$_try" -eq 5 ] && die "pip install claude-agent-sdk failed after 5 attempts"
  log "claude_agent_sdk install attempt $_try failed — retrying in ${_delay}s ..."
  sleep "$_delay"; _delay=$(( _delay * 2 ))
done

log "probing claude (-p)"
if claude -p "Reply with exactly: SETUP OK" \
      --model "${PERFSKILLS_CLAUDE_MODEL:-claude-opus-5}" </dev/null; then
  log "claude probe OK"
else
  die "claude probe failed — refusing to continue to the GPU stage"
fi
