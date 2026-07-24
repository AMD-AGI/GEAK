#!/usr/bin/env bash
# Install or validate rocprof-compute for GEAK roofline profiling.
#
# The default mode is read-only. --install attempts an apt installation and
# repairs the tool's Python dependencies, but remains fail-soft unless
# --required is also supplied.

set -u

MODE="check"
REQUIRED=0
JSON_OUT=""

usage() {
  cat <<'EOF'
Usage: install_rocprof_compute.sh [--check|--install] [--required] [--json-out PATH]

Modes:
  --check       Detect and validate rocprof-compute without system changes (default).
  --install     Install rocprofiler-compute when missing and repair Python dependencies.
  --required    Return nonzero unless rocprof-compute is runnable after the operation.

Environment overrides:
  GEAK_ROOFLINE_COMPUTE_PATH   Explicit rocprof-compute executable or containing directory.
  GEAK_ROOFLINE_APT_BIN        apt/apt-get command (default: apt-get).
  GEAK_ROOFLINE_PYTHON         Python used for profiler dependencies (default: python3).
  GEAK_ROOFLINE_REQUIREMENTS   Explicit rocprof-compute requirements.txt.
  GEAK_ROOFLINE_SUDO_BIN       Privilege command for apt (default: sudo when non-root).
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    --install) MODE="install" ;;
    --required) REQUIRED=1 ;;
    --json-out)
      shift
      [ "$#" -gt 0 ] || { echo "--json-out requires a path" >&2; exit 2; }
      JSON_OUT="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PYTHON_BIN="${GEAK_ROOFLINE_PYTHON:-python3}"
APT_BIN="${GEAK_ROOFLINE_APT_BIN:-apt-get}"
SUDO_BIN="${GEAK_ROOFLINE_SUDO_BIN:-sudo}"
TOOL_BIN="${GEAK_ROOFLINE_COMPUTE_BIN:-rocprof-compute}"
EXPLICIT_PATH="${GEAK_ROOFLINE_COMPUTE_PATH:-${GEAK_ROCPROF_COMPUTE_PATH:-${ROCPROF_COMPUTE_PATH:-}}}"
REQUIREMENTS_PATH="${GEAK_ROOFLINE_REQUIREMENTS:-}"

STATUS="missing"
REASON=""
RESOLVED=""
VERSION=""
INSTALLED=0
DEPS_FIXED=0
PANDAS_FIXED=0
REPAIR_FAILED=0

log() {
  echo "[geak-roofline-install] $*" >&2
}

resolve_tool() {
  local candidate=""
  if [ -n "$EXPLICIT_PATH" ]; then
    candidate="$EXPLICIT_PATH"
    [ -d "$candidate" ] && candidate="${candidate%/}/rocprof-compute"
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  if command -v "$TOOL_BIN" >/dev/null 2>&1; then
    command -v "$TOOL_BIN"
    return 0
  fi
  for candidate in \
    /opt/rocm/bin/rocprof-compute \
    /opt/rocm/libexec/rocprofiler-compute/rocprof-compute \
    /opt/rocprofiler-compute/bin/rocprof-compute; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

tool_version() {
  local output=""
  output="$("$1" --version 2>&1)" || return 1
  printf '%s\n' "$output" | awk 'NF { print; exit }'
}

tool_health() {
  "$1" profile --help >/dev/null 2>&1 \
    && "$1" analyze --help >/dev/null 2>&1
}

find_requirements() {
  local candidate=""
  if [ -n "$REQUIREMENTS_PATH" ] && [ -f "$REQUIREMENTS_PATH" ]; then
    printf '%s\n' "$REQUIREMENTS_PATH"
    return 0
  fi
  for candidate in \
    /opt/rocm/libexec/rocprofiler-compute/requirements.txt \
    /opt/rocm-*/libexec/rocprofiler-compute/requirements.txt; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

pip_install() {
  local help_text=""
  local -a extra=()
  help_text="$("$PYTHON_BIN" -m pip install --help 2>/dev/null)" || true
  if [[ "$help_text" == *"--break-system-packages"* ]]; then
    extra+=(--break-system-packages)
  fi
  "$PYTHON_BIN" -m pip install --quiet --no-cache-dir "${extra[@]}" "$@"
}

repair_python_dependencies() {
  local requirements=""
  if tool_health "$RESOLVED"; then
    return 0
  fi
  requirements="$(find_requirements)" || {
    log "rocprof-compute is present but no requirements.txt was found"
    return 1
  }
  log "installing rocprof-compute Python dependencies from $requirements"
  if pip_install -r "$requirements" >/dev/null 2>&1; then
    DEPS_FIXED=1
    return 0
  fi
  log "rocprof-compute Python dependency installation failed"
  return 1
}

repair_pandas_compatibility() {
  if ! "$PYTHON_BIN" -c \
    'import pandas; v=tuple(int(x) for x in pandas.__version__.split(".")[:2]); raise SystemExit(0 if v >= (3, 0) else 1)' \
    >/dev/null 2>&1; then
    return 0
  fi
  log "pandas 3.x detected; installing pandas>=2.1,<3 for rocprof-compute compatibility"
  if pip_install "pandas>=2.1,<3" >/dev/null 2>&1; then
    PANDAS_FIXED=1
    return 0
  fi
  log "pandas compatibility repair failed"
  return 1
}

install_package() {
  local -a command=("$APT_BIN")
  if ! command -v "$APT_BIN" >/dev/null 2>&1; then
    REASON="apt_unavailable"
    return 1
  fi
  case "$(basename "$APT_BIN")" in
    apt|apt-get)
      if [ "$(id -u)" -ne 0 ]; then
        if ! command -v "$SUDO_BIN" >/dev/null 2>&1; then
          REASON="privilege_escalation_unavailable"
          return 1
        fi
        if [ "$(basename "$SUDO_BIN")" = "sudo" ]; then
          command=("$SUDO_BIN" -n "$APT_BIN")
        else
          command=("$SUDO_BIN" "$APT_BIN")
        fi
      fi
      ;;
  esac
  log "installing rocprofiler-compute"
  if "${command[@]}" update -qq >/dev/null 2>&1 \
      && DEBIAN_FRONTEND=noninteractive "${command[@]}" install -y \
        --no-install-recommends rocprofiler-compute >/dev/null 2>&1; then
    INSTALLED=1
    return 0
  fi
  REASON="apt_install_failed"
  return 1
}

emit_result() {
  local output=""
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    output="$(STATUS="$STATUS" REASON="$REASON" RESOLVED="$RESOLVED" VERSION="$VERSION" \
      MODE="$MODE" INSTALLED="$INSTALLED" DEPS_FIXED="$DEPS_FIXED" PANDAS_FIXED="$PANDAS_FIXED" \
      REPAIR_FAILED="$REPAIR_FAILED" \
      "$PYTHON_BIN" - <<'PY'
import json
import os

print(json.dumps({
    "status": os.environ["STATUS"],
    "reason": os.environ["REASON"],
    "mode": os.environ["MODE"],
    "path": os.environ["RESOLVED"] or None,
    "version": os.environ["VERSION"] or None,
    "installed": os.environ["INSTALLED"] == "1",
    "python_dependencies_fixed": os.environ["DEPS_FIXED"] == "1",
    "pandas_compatibility_fixed": os.environ["PANDAS_FIXED"] == "1",
    "dependency_repair_failed": os.environ["REPAIR_FAILED"] == "1",
}, sort_keys=True))
PY
)"
  else
    output="{\"status\":\"$STATUS\",\"reason\":\"$REASON\",\"mode\":\"$MODE\"}"
  fi
  printf '%s\n' "$output"
  if [ -n "$JSON_OUT" ]; then
    mkdir -p "$(dirname "$JSON_OUT")"
    printf '%s\n' "$output" > "${JSON_OUT}.tmp.$$"
    mv "${JSON_OUT}.tmp.$$" "$JSON_OUT"
  fi
}

if RESOLVED="$(resolve_tool)"; then
  log "rocprof-compute found at $RESOLVED"
elif [ "$MODE" = "install" ]; then
  install_package || true
  RESOLVED="$(resolve_tool)" || RESOLVED=""
else
  REASON="rocprof_compute_unavailable"
fi

if [ -n "$RESOLVED" ]; then
  if [ "$MODE" = "install" ]; then
    repair_python_dependencies || REPAIR_FAILED=1
    repair_pandas_compatibility || REPAIR_FAILED=1
  fi
  if VERSION="$(tool_version "$RESOLVED")"; then
    if ! tool_health "$RESOLVED"; then
      STATUS="failed"
      REASON="command_health_check_failed"
    elif [ "$REPAIR_FAILED" -eq 1 ]; then
      STATUS="failed"
      REASON="python_dependency_repair_failed"
    else
      STATUS="$([ "$INSTALLED" -eq 1 ] && echo installed || echo present)"
      REASON=""
    fi
  else
    STATUS="failed"
    REASON="version_check_failed"
    VERSION=""
  fi
elif [ -z "$REASON" ]; then
  REASON="rocprof_compute_unavailable"
fi

emit_result
if [ "$REQUIRED" -eq 1 ] && [ "$STATUS" != "present" ] && [ "$STATUS" != "installed" ]; then
  exit 1
fi
exit 0
