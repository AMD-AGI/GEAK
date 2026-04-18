#!/usr/bin/env bash
# Reverse-knowledge runner: same intent as
#   (cd "$REVERSE_KL_PATH" && geak -c <config-relative-to-ws> -t "<task>" --repo .)
# geak/mini still require --kernel-url for the optimizer pipeline, so this script
# loads mini_reverse_kl.yaml and runs InteractiveAgent (yolo: transcripts on the terminal, no prompts) from the
# workspace directory (no homogeneous runner, no patch selection).

set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  bash scripts/run-reverse_knowledge.sh <unoptimized_path> <optimized_path>
  bash scripts/run-reverse_knowledge.sh <local_git_repository_path>

Environment:
  REVERSE_KL_PATH   Exported for convenience; always set to the canonical AMD user-case path
                    (GEAK_ROOT/mcp_tools/rag-mcp/.../user-case/user). reverse_knowledge.py ignores
                    any different value and always uses that directory for cwd and deliverables.

  API keys (mini_reverse_kl.yaml keeps model.api_key null; export before running — see GEAK/README.md):
                    AMD_LLM_API_KEY or LLM_GATEWAY_KEY (amd_llm / AMD gateway), or MSWEA_MODEL_API_KEY.

Notes:
  - Baseline/optimized/repo paths in the task are relative to that canonical workspace (for relpath).
  - Python prepends the absolute workspace path to the task so the model must write only there.
USAGE
}

_realpath() {
  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

_relpath() {
  # relpath FROM_BASE TO_TARGET
  python3 -c 'import os, sys; b, t = sys.argv[1], sys.argv[2]; print(os.path.relpath(os.path.realpath(t), os.path.realpath(b)))' "$1" "$2"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REL_WS="mcp_tools/rag-mcp/knowledge-base/amd-knowledge-base/layer-6-extended/optimize-guides/user-case/user"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" != 1 && "$#" != 2 ]]; then
  usage
  exit 1
fi

WS_ABS="$(_realpath "${GEAK_ROOT}/${DEFAULT_REL_WS}")"
export REVERSE_KL_PATH="${WS_ABS}"

CONFIG_ABS="$(_realpath "${GEAK_ROOT}/src/minisweagent/config/mini_reverse_kl.yaml")"
CONFIG_REL="$(_relpath "${WS_ABS}" "${CONFIG_ABS}")"

if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "error: config not found: ${CONFIG_ABS}" >&2
  exit 1
fi
if [[ ! -d "${WS_ABS}" ]]; then
  echo "error: workspace is not a directory: ${WS_ABS}" >&2
  exit 1
fi

TASK_FILE="$(mktemp)"
cleanup() { rm -f "${TASK_FILE}"; }
trap cleanup EXIT

if [[ "$#" -eq 2 ]]; then
  U_RAW="$1"
  O_RAW="$2"
  U_ABS="$(_realpath "${U_RAW}")"
  O_ABS="$(_realpath "${O_RAW}")"
  if [[ ! -e "${U_ABS}" ]]; then
    echo "error: unoptimized path does not exist: ${U_RAW}" >&2
    exit 1
  fi
  if [[ ! -e "${O_ABS}" ]]; then
    echo "error: optimized path does not exist: ${O_RAW}" >&2
    exit 1
  fi
  REL_U="$(_relpath "${WS_ABS}" "${U_ABS}")"
  REL_O="$(_relpath "${WS_ABS}" "${O_ABS}")"
  cat >"${TASK_FILE}" <<EOF
Reverse-knowledge analysis (baseline vs optimized):

- Baseline (unoptimized) path, relative to the agent workspace root: ${REL_U}
- Optimized path, relative to the agent workspace root: ${REL_O}

Treat the optimized tree as the latest state and the baseline as the prior state; compare at directory level as appropriate. Follow the workflow in your system instructions (kernel discovery, reports, simplified reports, workspace cleanup to simplified-only).
EOF
else
  REPO_RAW="$1"
  REPO_ABS="$(_realpath "${REPO_RAW}")"
  if [[ ! -d "${REPO_ABS}" ]]; then
    echo "error: not a directory: ${REPO_RAW}" >&2
    exit 1
  fi
  if ! git -C "${REPO_ABS}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: not a git repository (no valid .git): ${REPO_RAW}" >&2
    exit 1
  fi
  REL_R="$(_relpath "${WS_ABS}" "${REPO_ABS}")"
  cat >"${TASK_FILE}" <<EOF
Reverse-knowledge analysis (git repository):

- Repository path, relative to the agent workspace root: ${REL_R}

Analyze optimization history in this repository as described in your system instructions (clone only if the task required a URL; here use this local path). Follow kernel discovery, reports, simplified reports, and workspace cleanup to simplified-only under the workspace.
EOF
fi

export GEAK_ROOT
export GEAK_REVERSE_KL_CONFIG="${CONFIG_ABS}"
export GEAK_REVERSE_KL_TASK_FILE="${TASK_FILE}"

echo "[run-reverse_knowledge] workspace (REVERSE_KL_PATH)=${WS_ABS}" >&2
echo "[run-reverse_knowledge] agent YAML (prompts, model, env): ${CONFIG_ABS}" >&2
echo "[run-reverse_knowledge] config path relative to workspace cwd: ${CONFIG_REL}" >&2
echo "[run-reverse_knowledge] ~/.config/mini-swe-agent/.env loads only API keys (dotenv); it is not the agent YAML above." >&2

export PYTHONPATH="${GEAK_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Hide generic 'Loading global config...' banner (that file is keys only, not mini_reverse_kl.yaml). Set MSWEA_SILENT_STARTUP=0 to show it.
export MSWEA_SILENT_STARTUP="${MSWEA_SILENT_STARTUP:-1}"

# docker exec / CI without a TTY: skip interactive mini-swe-agent global setup (otherwise prompt_toolkit EOFError).
if [[ ! -t 0 ]]; then
  export MSWEA_CONFIGURED="${MSWEA_CONFIGURED:-true}"
fi

cd "${WS_ABS}"

python3 "${SCRIPT_DIR}/reverse_knowledge.py"
