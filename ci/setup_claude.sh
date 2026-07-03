#!/usr/bin/env bash
# Step D — install Claude Code inside the container and probe it.
# All Claude state (install + config + logs) is kept under $CLAUDE_HOME so it
# survives outside the ephemeral container, in the run's timestamped folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib.sh"

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
python3 -m pip install --quiet --no-input --break-system-packages claude-agent-sdk \
  || python3 -m pip install --quiet --no-input claude-agent-sdk \
  || die "pip install claude-agent-sdk failed"
python3 -c "import claude_agent_sdk" || die "claude_agent_sdk import failed after install"

log "probing claude (-p)"
if claude -p "Reply with exactly: SETUP OK" \
      --model "${PERFSKILLS_CLAUDE_MODEL:-claude-opus-4-8}" </dev/null; then
  log "claude probe OK"
else
  die "claude probe failed — refusing to continue to the GPU stage"
fi
