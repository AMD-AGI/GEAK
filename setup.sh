#!/usr/bin/env bash
# GEAK v4 environment installer.
#
# GEAK v4 is not a Python package — the workflows (e2e_workflow.js /
# kernel_workflow.js) run *inside Claude Code*. This script:
#   1. Installs the Claude Code CLI (>= 2.1.177) via its native, standalone
#      installer (curl https://claude.ai/install.sh | bash) — no Node.js.
#   2. Installs the small pure-Python helper libs the workflow scripts import
#      (pyyaml, requests, datasets).
#   3. Detects — but never installs — the heavy, image-provided ROCm / profiler /
#      serving-backend prerequisites, and warns if any are missing.
#
# Just run it: ./setup.sh   (idempotent — every step skips when already present)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLAUDE_MIN_VERSION="2.1.177"
# Native-installer target: a version string, "stable", or "latest".
CLAUDE_VERSION="${CLAUDE_VERSION:-latest}"
# Where the native installer drops the binary.
CLAUDE_BIN_DIR="${CLAUDE_BIN_DIR:-$HOME/.local/bin}"
GEAK_CLAUDE_LOCALBIN=0
# Minimal Python libs the workflow scripts import at runtime.
PY_DEPS=(pyyaml requests datasets)

log()  { echo "[geak-setup] $*"; }
warn() { echo "[geak-setup WARN] $*" >&2; }
die()  { echo "[geak-setup ERROR] $*" >&2; exit 1; }
run()  { log "$*"; "$@"; }

# ver_ge A B -> true when semver A >= B. Compares dotted fields numerically, so
# it works with BSD sort (macOS) too, not just GNU `sort -V`.
ver_ge() {
  [ "$1" = "$2" ] && return 0
  local first
  first="$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)"
  [ "$first" = "$2" ]
}

# Leading semver from `claude --version` (e.g. "2.1.206 (Claude Code)").
claude_version() {
  claude --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

# --- 0. Resolve PYTHON ---
resolve_python() {
  if [ -n "${PYTHON:-}" ] && [ -x "$(command -v "$PYTHON" 2>/dev/null || true)" ]; then
    PYTHON="$(command -v "$PYTHON")"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    die "no python3 found; install Python 3.8+ (GEAK needs it for the workflow helper scripts)"
  fi
  export PYTHON
  log "PYTHON=${PYTHON} ($("$PYTHON" --version 2>&1))"
}

# --- 1. Claude Code CLI (native, standalone installer — no Node.js) ---

# Add CLAUDE_BIN_DIR to PATH for the rest of this run, and flag it so the printed
# next-steps remind the user to persist it.
ensure_claude_bindir_on_path() {
  case ":${PATH}:" in
    *":${CLAUDE_BIN_DIR}:"*) ;;
    *) export PATH="${CLAUDE_BIN_DIR}:${PATH}"; GEAK_CLAUDE_LOCALBIN=1 ;;
  esac
}

# Install Claude Code. Preferred: the official native installer (standalone
# binary, no Node); if that fails and npm exists, fall back to npm.
# The installer pulls a ~260MB binary via a silent `curl -fsSL`, so on slow links
# it can sit for several minutes with no output — a slow download, NOT a hang. We
# deliberately do not put a --max-time on it (that would kill a valid transfer).
install_claude_native() {
  command -v curl >/dev/null 2>&1 || command -v npm >/dev/null 2>&1 \
    || die "need curl (native installer) or npm. Install one, or install Claude Code manually (https://code.claude.com/docs/en/setup), then re-run."

  if command -v curl >/dev/null 2>&1; then
    log "installing Claude Code (${CLAUDE_VERSION}) via the native installer"
    log "  downloads a ~260MB standalone binary; on slow networks this can take"
    log "  several minutes with no progress output. Please wait."
    if curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | bash -s "${CLAUDE_VERSION}"; then
      ensure_claude_bindir_on_path
      hash -r 2>/dev/null || true
      return 0
    fi
    warn "native installer failed"
  fi

  if command -v npm >/dev/null 2>&1; then
    warn "falling back to npm: npm install -g @anthropic-ai/claude-code"
    npm install -g @anthropic-ai/claude-code
    hash -r 2>/dev/null || true
    return 0
  fi
  die "could not install Claude Code (native installer failed; npm not found). Check network access or install manually: https://code.claude.com/docs/en/setup"
}

ensure_claude_code() {
  # Make a freshly-installed-but-not-yet-on-PATH binary visible first.
  command -v claude >/dev/null 2>&1 || [ ! -x "${CLAUDE_BIN_DIR}/claude" ] || ensure_claude_bindir_on_path

  if command -v claude >/dev/null 2>&1; then
    local cur; cur="$(claude_version)"
    if [ -n "$cur" ] && ver_ge "$cur" "$CLAUDE_MIN_VERSION"; then
      log "Claude Code present (${cur}) >= ${CLAUDE_MIN_VERSION}"
      return 0
    fi
    warn "Claude Code ${cur:-unknown} is older than ${CLAUDE_MIN_VERSION}; updating"
    run claude update || true
    hash -r 2>/dev/null || true
    cur="$(claude_version)"
    if [ -z "$cur" ] || ! ver_ge "$cur" "$CLAUDE_MIN_VERSION"; then
      install_claude_native
    fi
  else
    warn "Claude Code CLI not found"
    install_claude_native
  fi

  command -v claude >/dev/null 2>&1 \
    || warn "claude not on PATH after install; ensure '${CLAUDE_BIN_DIR}' is on your PATH"
  local cur; cur="$(claude_version)"
  [ -n "$cur" ] && ! ver_ge "$cur" "$CLAUDE_MIN_VERSION" \
    && warn "installed Claude Code ${cur} is still < ${CLAUDE_MIN_VERSION}; run 'claude update' or set CLAUDE_VERSION"
  return 0
}

# --- 2. Python helper libs ---
# The workflow scripts only import three small pure-Python libs. On a system
# (non-venv) interpreter, pip 23.0.1+ needs --break-system-packages (PEP 668).
ensure_python_deps() {
  log "ensuring Python helper libs: ${PY_DEPS[*]}"
  local pip_extra=()
  if "$PYTHON" - <<'PY' 2>/dev/null
import sys
raise SystemExit(0 if sys.prefix == sys.base_prefix else 1)
PY
  then
    if "$PYTHON" -m pip install --break-system-packages --help >/dev/null 2>&1; then
      pip_extra=(--break-system-packages)
      log "non-venv PYTHON; pip will use --break-system-packages"
    fi
  fi

  local missing=()
  for pkg in "${PY_DEPS[@]}"; do
    local import_name="$pkg"
    [ "$pkg" = "pyyaml" ] && import_name="yaml"   # pip name -> import name
    "$PYTHON" -c "import ${import_name}" >/dev/null 2>&1 || missing+=("$pkg")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    log "Python helper libs already satisfied"
    return 0
  fi
  log "missing: ${missing[*]}"
  run "$PYTHON" -m pip install "${pip_extra[@]}" "${missing[@]}"
}

# --- 3. Environment prerequisites (detect only) ---
check_environment() {
  log "checking ROCm / profiler / serving-backend prerequisites (detect only)"

  if command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
    log "  ROCm: present"
  else
    warn "  ROCm not detected (rocminfo/rocm-smi missing). GEAK targets AMD Instinct MI GPUs; install ROCm 6+."
  fi

  local profiler=""
  for p in rocprof-compute rocprofv3 rocprof omniperf metrix; do
    if command -v "$p" >/dev/null 2>&1; then profiler="$p"; break; fi
  done
  if [ -n "$profiler" ]; then
    log "  profiler: ${profiler}"
  else
    warn "  no profiler found (rocprof-compute/rocprofv3/rocprof). Profiling steps will be degraded."
  fi

  local backend=""
  "$PYTHON" -c "import sglang" >/dev/null 2>&1 && backend="sglang"
  [ -z "$backend" ] && { "$PYTHON" -c "import vllm" >/dev/null 2>&1 && backend="vllm"; }
  if [ -n "$backend" ]; then
    log "  serving backend: ${backend}"
  else
    warn "  no serving backend (sglang/vllm) importable in ${PYTHON}. Required for e2e_workflow only."
  fi
}

print_next_steps() {
  if [ "$GEAK_CLAUDE_LOCALBIN" -eq 1 ]; then
    cat <<EOF

[geak-setup] NOTE: Claude Code is installed at ${CLAUDE_BIN_DIR}, which is not on
your PATH. Add it (official recommendation; use ~/.zshrc for zsh):
    echo 'export PATH="${CLAUDE_BIN_DIR}:\$PATH"' >> ~/.bashrc && source ~/.bashrc
EOF
  fi

  cat <<EOF

[geak-setup] setup complete.

Next steps — configure Claude Code, then launch it:

1) Give Claude Code API access (pick ONE):

   a. Anthropic API directly:
        export ANTHROPIC_API_KEY=sk-ant-...

   b. A gateway / proxy (OpenAI-compatible or Anthropic-compatible):
        export ANTHROPIC_BASE_URL=https://your-gateway.example.com
        export ANTHROPIC_AUTH_TOKEN=your-token

   c. Interactive login (Claude / Anthropic Console account):
        claude            # then run: /login   and follow the browser flow

   (Persist your choice in ~/.bashrc so future shells inherit it.)

2) Launch Claude Code in auto-approve mode from the repo root, then ask:
     cd ${REPO_ROOT}
     IS_SANDBOX=1 claude --dangerously-skip-permissions

   Example prompts:
     use ${REPO_ROOT}/kernel_workflow to optimize ${REPO_ROOT}/examples/tasks/knn
     use ${REPO_ROOT}/e2e_workflow to optimize inference for /models/<model>, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
EOF
}

main() {
  log "REPO_ROOT=${REPO_ROOT}"
  resolve_python
  ensure_claude_code
  ensure_python_deps
  check_environment
  print_next_steps
}

main
