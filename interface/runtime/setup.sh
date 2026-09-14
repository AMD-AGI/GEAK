#!/bin/sh
# Portable env for the GEAK standalone runtime's codex backend. No hardcoded
# paths: node comes from PATH, the key comes from YOUR environment, CODEX_HOME
# points at the in-repo codex-home next to this script.
#
# Optional: on the common paths (AMD gateway, official OpenAI) the runtime
# auto-configures codex and you do not need this file at all -- exporting AMDKEY
# or OPENAI_API_KEY is enough. Source it when you want the in-repo CODEX_HOME or
# the de-streaming shim. See SETUP.md.
#
# SOURCE it (do not execute in a subshell), so the exports reach your shell:
#     . interface/runtime/setup.sh      # bash / sh / zsh
#
# REQUIRED (from your environment — never committed):
#   AMDKEY | OPENAI_API_KEY   provider key; the runtime picks the endpoint from it
# OPTIONAL:
#   SSL_CERT_FILE    CA bundle, only for a private gateway whose CA is not public
#   GEAK_GW_BASE     upstream base_url for the shim. No default; unset means the
#                    shim is not started (the common endpoints do not need it).
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

export NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-$SSL_CERT_FILE}"
export CODEX_HOME="$GEAK_RT_DIR/codex-home"
GEAK_GW_BASE="${GEAK_GW_BASE:-}"
SHIM_PORT="${SHIM_PORT:-8791}"

# Preflight checks (warn, do not exit — this file is sourced).
if ! command -v node >/dev/null 2>&1; then
  echo "[setup] WARNING: 'node' not on PATH — install Node.js before running the runtime." >&2
fi
if [ -z "$AMDKEY" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "[setup] WARNING: no provider key set (AMDKEY / OPENAI_API_KEY) — codex will 401." >&2
fi

# Start the de-streaming shim only when an upstream was named. The common paths
# (AMD gateway, official OpenAI) serve /v1/responses directly and need no shim.
if [ -z "$GEAK_GW_BASE" ]; then
  echo "[setup] GEAK_GW_BASE unset — shim not started (not needed for direct /v1/responses endpoints)"
elif command -v node >/dev/null 2>&1; then
  if pgrep -f "$GEAK_RT_DIR/responses_shim.mjs" >/dev/null 2>&1; then
    echo "[setup] shim already running on :$SHIM_PORT"
  else
    GW_BASE="$GEAK_GW_BASE" SHIM_PORT="$SHIM_PORT" OPENAI_API_KEY="$OPENAI_API_KEY" SSL_CERT_FILE="$SSL_CERT_FILE" \
      node "$GEAK_RT_DIR/responses_shim.mjs" > "$GEAK_RT_DIR/shim.log" 2>&1 &
    sleep 3
    echo "[setup] started shim on :$SHIM_PORT -> $GEAK_GW_BASE (log: $GEAK_RT_DIR/shim.log)"
  fi
fi

echo "[setup] CODEX_HOME=$CODEX_HOME"
echo "[setup] run: node $GEAK_RT_DIR/engine/run_workflow.mjs <workflow.js> --agent codex"
