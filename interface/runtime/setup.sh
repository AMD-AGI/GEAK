#!/bin/sh
# Portable env for the GEAK standalone runtime's OpenAI-dialect backends
# (codex now; qwen/kimi share the same gateway pattern). No hardcoded paths:
# node comes from PATH, key/CA come from YOUR environment, CODEX_HOME points at
# the in-repo codex-home next to this script.
#
# SOURCE it (do not execute in a subshell), so the exports reach your shell:
#     . interface/runtime/setup.sh      # bash / sh / zsh
#
# REQUIRED (from your environment — never committed):
#   OPENAI_API_KEY   gateway key (falls back to ANTHROPIC_API_KEY)
#   SSL_CERT_FILE    CA bundle for gateway TLS (if your gateway needs one)
# OPTIONAL:
#   GEAK_GW_BASE     gateway base_url (default: SaFE global)
#   SHIM_PORT        shim listen port (default: 8791)
#
# Requires `node` on PATH. No npm install needed (runtime + shim use only builtins).

# Resolve this script's own directory (works when sourced from bash or zsh; falls
# back to cwd for plain POSIX sh where $0 is the shell).
_geak_here() {
  # bash
  [ -n "$BASH_SOURCE" ] && { cd "$(dirname "$BASH_SOURCE")" 2>/dev/null && pwd; return; }
  # zsh
  [ -n "$ZSH_VERSION" ] && { cd "$(dirname "${(%):-%N}")" 2>/dev/null && pwd; return; }
  # fallback: assume invoked from repo root as `. interface/runtime/setup.sh`
  if [ -f interface/runtime/setup.sh ]; then cd interface/runtime && pwd; else pwd; fi
}
GEAK_RT_DIR=$(_geak_here)

export OPENAI_API_KEY="${OPENAI_API_KEY:-$ANTHROPIC_API_KEY}"
export NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-$SSL_CERT_FILE}"
export CODEX_HOME="$GEAK_RT_DIR/codex-home"
GEAK_GW_BASE="${GEAK_GW_BASE:-https://global.primus-safe.amd.com/api/v1/llm-proxy/v1}"
SHIM_PORT="${SHIM_PORT:-8791}"

# qwen non-interactive auth selection (idempotent; harmless if qwen unused)
mkdir -p "$HOME/.qwen"
[ -f "$HOME/.qwen/settings.json" ] || printf '{"security":{"auth":{"selectedType":"openai"}}}\n' > "$HOME/.qwen/settings.json"
export QWEN_CODE_SUPPRESS_YOLO_WARNING=1

# Preflight checks (warn, do not exit — this file is sourced).
if ! command -v node >/dev/null 2>&1; then
  echo "[setup] WARNING: 'node' not on PATH — install Node.js before running the runtime." >&2
fi
if [ -z "$OPENAI_API_KEY" ]; then
  echo "[setup] WARNING: OPENAI_API_KEY (or ANTHROPIC_API_KEY) is empty — codex/qwen will 401." >&2
fi

# Start the codex responses-shim if not already up (needed for codex + claude).
if command -v node >/dev/null 2>&1; then
  if pgrep -f "$GEAK_RT_DIR/responses_shim.mjs" >/dev/null 2>&1; then
    echo "[setup] shim already running on :$SHIM_PORT"
  else
    GW_BASE="$GEAK_GW_BASE" SHIM_PORT="$SHIM_PORT" OPENAI_API_KEY="$OPENAI_API_KEY" SSL_CERT_FILE="$SSL_CERT_FILE" \
      node "$GEAK_RT_DIR/responses_shim.mjs" > "$GEAK_RT_DIR/shim.log" 2>&1 &
    sleep 3
    echo "[setup] started shim on :$SHIM_PORT (log: $GEAK_RT_DIR/shim.log)"
  fi
fi

echo "[setup] CODEX_HOME=$CODEX_HOME"
echo "[setup] gateway=$GEAK_GW_BASE"
echo "[setup] run: node $GEAK_RT_DIR/run_workflow.mjs <workflow.js> --profile codex-opus48"

# cursor is unrelated to the shim/gateway (it uses Cursor's private cloud).
if command -v cursor-agent >/dev/null 2>&1; then
  echo "[setup] note: cursor-agent found — for --profile cursor, run 'cursor-agent login' (your own Cursor account); it does NOT use this gateway/shim."
fi
