#!/usr/bin/env bash
# One-shot Claude Code install + config for CI (runs inside the ephemeral GPU
# container). Re-run every container start (the container filesystem is ephemeral).
# Usage:  bash claude_setup.sh
#
# Auth: global AMD LiteLLM proxy (NOT tied to a personal key). Override via env:
#   LITELLM_API_KEY   — the LiteLLM virtual key
#   LITELLM_BASE_URL  — the proxy root (Claude Code appends /v1/messages itself)
set -euo pipefail

# SECURITY: never hardcode the key OR the proxy endpoint. Both MUST come from the
# environment (CI secrets / local export). Nothing sensitive lives in this file.
LITELLM_KEY="${LITELLM_API_KEY:?set LITELLM_API_KEY (LiteLLM virtual key; do NOT hardcode it)}"
LITELLM_BASE="${LITELLM_BASE_URL:?set LITELLM_BASE_URL (LiteLLM proxy base URL; do NOT hardcode it)}"
# Agent model, shared with run_geak_e2e.sh / setup_claude.sh via the same env var so
# one override switches the settings.json aliases AND the run_e2e invocation together.
CLAUDE_MODEL="${PERFSKILLS_CLAUDE_MODEL:-claude-opus-5}"

echo "[1/4] Installing Claude Code (native, latest)..."
# Retry the network install: a single transient curl reset (e.g. errno 104
# "Connection reset by peer") otherwise aborts the whole multi-hour job before the
# e2e workflow ever starts (see moonshotai-Kimi-K2.6-int4 job 40794). Backoff
# 5/10/20/40s; treat the install as done only once the claude binary is runnable.
_claude_install_ok() { "$HOME/.local/bin/claude" --version >/dev/null 2>&1; }
_delay=5
for _try in 1 2 3 4 5; do
  echo "  install attempt $_try/5 ..."
  if curl -fsSL https://claude.ai/install.sh | bash && _claude_install_ok; then
    echo "  claude install OK on attempt $_try"
    break
  fi
  if [ "$_try" -eq 5 ]; then
    echo "  ERROR: claude install failed after 5 attempts (network?)" >&2
    exit 1
  fi
  echo "  install attempt $_try failed — retrying in ${_delay}s ..." >&2
  sleep "$_delay"; _delay=$(( _delay * 2 ))
done

echo "[2/4] Writing ~/.claude/settings.json ..."
mkdir -p "$HOME/.claude"
# NOTE: the proxy serves Claude over an internal corporate CA, which the
# base container image does not trust; NODE_TLS_REJECT_UNAUTHORIZED=0 is a stopgap.
# Preferred long-term fix: bake the corporate CA bundle into the image and drop it.
# Only the claude-opus family is served by this proxy, so the haiku/sonnet defaults
# also point at $CLAUDE_MODEL (nonessential traffic is disabled regardless).
cat > "$HOME/.claude/settings.json" <<EOF
{
  "\$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "ANTHROPIC_BASE_URL": "${LITELLM_BASE}",
    "ANTHROPIC_API_KEY": "${LITELLM_KEY}",
    "NODE_TLS_REJECT_UNAUTHORIZED": "0",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "${CLAUDE_MODEL}",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "${CLAUDE_MODEL}",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "${CLAUDE_MODEL}",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "ENABLE_TOOL_SEARCH": "true"
  },
  "model": "opus",
  "theme": "dark"
}
EOF

echo "[3/4] Patching ~/.claude.json (skip onboarding + approve key)..."
# Merge into the file the installer creates, preserving userID/machineID if present.
export _APPROVE_KEY="$LITELLM_KEY"
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/.claude.json")
data = {}
if os.path.exists(p):
    try:
        with open(p) as f: data = json.load(f)
    except Exception:
        data = {}
data["hasCompletedOnboarding"] = True
data.setdefault("installMethod", "native")
key = os.environ.get("_APPROVE_KEY", "")
data["customApiKeyResponses"] = {"approved": [key] if key else [], "rejected": []}
with open(p, "w") as f: json.dump(data, f, indent=2)
print("  .claude.json updated")
PY

echo "[4/4] Ensuring ~/.local/bin on PATH ..."
for rc in "$HOME/.bashrc" "$HOME/.bash_profile"; do
  grep -q '.local/bin' "$rc" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
done
export PATH="$HOME/.local/bin:$PATH"

echo
echo "Done. Version: $("$HOME/.local/bin/claude" --version)"
echo "Quick test:"
"$HOME/.local/bin/claude" -p "Reply with exactly: SETUP OK" --model "$CLAUDE_MODEL" </dev/null || true
echo
echo "Run it with:  IS_SANDBOX=1 claude --dangerously-skip-permissions --model $CLAUDE_MODEL"
