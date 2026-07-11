#!/usr/bin/env bash
# GEAK v4 environment installer.
#
# GEAK v4 is not a Python package — the workflows (e2e_workflow.js /
# kernel_workflow.js) run *inside Claude Code*. This script:
#   1. Installs the Claude Code CLI (>= 2.1.177) via its native, standalone
#      installer (curl https://claude.ai/install.sh | bash) — no Node.js.
#   2. Installs the Python dependencies listed in requirements.txt (add new
#      packages there — no need to edit this script).
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
# Python dependencies live in requirements.txt (edit that, not this script).
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_ROOT}/requirements.txt}"

# Bold-green styling for the user-facing commands in the printed next-steps. Only
# emit ANSI codes to a real terminal, so piping/logging stays free of escape junk.
if [ -t 1 ]; then
  C_CMD=$'\033[1;32m'
  C_OFF=$'\033[0m'
else
  C_CMD=''
  C_OFF=''
fi

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

# Make sure `$PYTHON -m pip` actually runs. A stale pip in site-packages can be
# incompatible with a newer interpreter (e.g. pip's vendored pkg_resources calls
# pkgutil.ImpImporter, removed in Python 3.12), so every pip invocation crashes.
# When that happens we bootstrap a fresh pip via ensurepip and upgrade the build
# stack. Fatal if pip still won't run afterwards — nothing downstream can install.
ensure_pip_works() {
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    log "pip OK ($("$PYTHON" -m pip --version 2>&1))"
    return 0
  fi
  warn "pip is broken for ${PYTHON}; bootstrapping via ensurepip"
  local boot_extra=()
  local flag; flag="$(pip_break_system_flag)"
  [ -n "$flag" ] && boot_extra=("$flag")
  "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
  run "$PYTHON" -m pip install "${boot_extra[@]}" --upgrade pip setuptools wheel \
    || die "could not repair pip (ensurepip + upgrade failed). Fix the Python install, then re-run."
  "$PYTHON" -m pip --version >/dev/null 2>&1 \
    || die "pip still not runnable after bootstrap. Fix the Python install, then re-run."
  log "pip repaired ($("$PYTHON" -m pip --version 2>&1))"
}

# On a system (non-venv) interpreter, pip 23.0.1+ needs --break-system-packages
# (PEP 668). Prints the flag on stdout when it applies, nothing otherwise.
# Assumes pip runs; call after ensure_pip_works so a broken pip isn't mistaken
# for "flag not supported".
pip_break_system_flag() {
  if "$PYTHON" - <<'PY' 2>/dev/null
import sys
raise SystemExit(0 if sys.prefix == sys.base_prefix else 1)
PY
  then
    if "$PYTHON" -m pip install --break-system-packages --help >/dev/null 2>&1; then
      echo "--break-system-packages"
    fi
  fi
}

# Installs everything in requirements.txt. pip is idempotent — already-satisfied
# packages are skipped — so re-running is cheap.
ensure_python_deps() {
  [ -f "$REQUIREMENTS_FILE" ] || die "requirements file not found: ${REQUIREMENTS_FILE}"
  ensure_pip_works
  # VCS (git+...) requirements need git; fail early with a clear message.
  if grep -qiE '(^|[[:space:]@])git\+' "$REQUIREMENTS_FILE" \
     && ! command -v git >/dev/null 2>&1; then
    die "requirements.txt has a git+ dependency but git is not installed. Install git, then re-run."
  fi
  log "installing Python dependencies from ${REQUIREMENTS_FILE}"
  local pip_extra=()
  local flag; flag="$(pip_break_system_flag)"
  if [ -n "$flag" ]; then
    pip_extra=("$flag")
    log "non-venv PYTHON; pip will use --break-system-packages"
  fi
  run "$PYTHON" -m pip install "${pip_extra[@]}" -r "$REQUIREMENTS_FILE"
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
  for p in rocprof-compute rocprofv3 rocprof metrix; do
    if command -v "$p" >/dev/null 2>&1; then profiler="$p"; break; fi
  done
  if [ -n "$profiler" ]; then
    log "  profiler: ${profiler}"
  else
    warn "  no profiler found (rocprof-compute/rocprofv3/rocprof/metrix). Profiling steps will be degraded."
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
    ${C_CMD}echo 'export PATH="${CLAUDE_BIN_DIR}:\$PATH"' >> ~/.bashrc && source ~/.bashrc${C_OFF}
EOF
  fi

  cat <<EOF

[geak-setup] setup complete.

Next steps — configure Claude Code, then launch it:

1) Give Claude Code API access (pick ONE):

   a. Anthropic API directly:
        ${C_CMD}export ANTHROPIC_API_KEY=sk-ant-...${C_OFF}

   b. A gateway / proxy (OpenAI-compatible or Anthropic-compatible):
        ${C_CMD}export ANTHROPIC_BASE_URL=https://your-gateway.example.com${C_OFF}
        ${C_CMD}export ANTHROPIC_AUTH_TOKEN=your-token${C_OFF}

   c. Interactive login (Claude / Anthropic Console account):
        ${C_CMD}claude${C_OFF}            # then run: /login   and follow the browser flow

   (Persist your choice in ~/.bashrc so future shells inherit it.)

2) Launch Claude Code in auto-approve mode from the repo root:
     ${C_CMD}cd ${REPO_ROOT}${C_OFF}
     ${C_CMD}IS_SANDBOX=1 claude --dangerously-skip-permissions${C_OFF}
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
