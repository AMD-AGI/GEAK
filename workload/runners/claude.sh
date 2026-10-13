#!/usr/bin/env bash
# runner: claude (exercised in practice). Contract in CONTRACT.md.
#
#   bash runners/claude.sh <prompt_file> <workdir> <log_file>
#   Exit codes: 0 finished / 78 unmet precondition (do not retry) / other non-zero
#   session failure (retryable)
set -uo pipefail

PROMPT="${1:?prompt_file}"
WORKDIR="${2:?workdir}"
LOG="${3:?log_file}"

mkdir -p "$(dirname "$LOG")" "$WORKDIR" 2>/dev/null
blocked() { echo "[runner:claude] BLOCKED $*" | tee -a "$LOG" >&2; exit 78; }

# ---- 0. Credentials/gateway come from **config**, never from the repo.
# This cluster does not authenticate with ~/.claude credentials; it goes through a
# gateway (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN + ANTHROPIC_CUSTOM_HEADERS). That
# block contains a subscription key and must not be committed, so it lives in an env file
# outside the repository, pointed at by runner_env.env_file in the config.
# See runners/claude.env.example for the template.
if [ -n "${WORKLOAD_AGENT_ENV_FILE:-}" ]; then
  [ -f "$WORKLOAD_AGENT_ENV_FILE" ] || blocked "WORKLOAD_AGENT_ENV_FILE points at a file that does not exist: $WORKLOAD_AGENT_ENV_FILE"
  set -a; . "$WORKLOAD_AGENT_ENV_FILE"; set +a
fi

# A CA bundle pointing at a missing file makes every https call **fail silently** (fail
# closed). The login environment on this cluster has exactly that defect. Anything that
# does not resolve to a file is cleared rather than carried along.
for _v in NODE_EXTRA_CA_CERTS SSL_CERT_FILE CURL_CA_BUNDLE REQUESTS_CA_BUNDLE; do
  _p="$(eval "printf '%s' \"\${$_v:-}\"")"
  if [ -n "$_p" ] && [ ! -f "$_p" ]; then
    echo "[runner:claude] $_v points at missing $_p; cleared (it would make https fail silently)" >> "$LOG"
    unset "$_v"
  fi
done

# ---- 1. Precondition self-check. Missing pieces return 78 instead of pretending to run.
BIN="${CLAUDE_BIN:-}"
if [ -z "$BIN" ]; then BIN="$(command -v claude 2>/dev/null)"; fi
if [ -z "$BIN" ]; then
  for c in /root/.local/bin/claude "$HOME/.local/bin/claude" /usr/local/bin/claude; do
    [ -x "$c" ] && { BIN="$c"; break; }
  done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || blocked "no claude executable found. Install it, or point CLAUDE_BIN at it."
[ -f "$PROMPT" ] || blocked "prompt file does not exist: $PROMPT"

# ---- 2. State directory on /dev/shm, never on NFS.
# A hard-killed sqlite on NFS leaves a stale -shm behind, after which every start fails
# with `code 15 locking protocol`. One directory per worker, no cross-talk.
STATE_BASE="${WORKLOAD_AGENT_STATE_BASE:-/dev/shm/workload_agent_state}"
STATE_DIR="${WORKLOAD_AGENT_STATE_DIR:-$STATE_BASE/claude.$(basename "$WORKDIR").$$}"
mkdir -p "$STATE_DIR" || blocked "cannot create state directory $STATE_DIR"

# A fresh state directory is empty, so credentials have to be carried over from the
# original config directory or the session starts unauthenticated.
SRC_CFG="${CLAUDE_SOURCE_CONFIG_DIR:-${CLAUDE_CONFIG_DIR:-$HOME/.claude}}"
if [ -d "$SRC_CFG" ] && [ "$SRC_CFG" != "$STATE_DIR" ]; then
  for f in .credentials.json settings.json; do
    [ -f "$SRC_CFG/$f" ] && cp -f "$SRC_CFG/$f" "$STATE_DIR/$f" 2>/dev/null
  done
fi
if [ ! -f "$STATE_DIR/.credentials.json" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] \
   && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  blocked "no usable credentials: $SRC_CFG/.credentials.json does not exist and neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set. Put the gateway config in an env file outside the repo and point runner_env.env_file at it (template: runners/claude.env.example)."
fi
export CLAUDE_CONFIG_DIR="$STATE_DIR"
# --dangerously-skip-permissions can still raise a one-time confirmation even when
# non-interactive; this line disables it.
[ -f "$STATE_DIR/settings.json" ] || \
  echo '{"skipDangerousModePermissionPrompt": true}' > "$STATE_DIR/settings.json"

# Background tasks are killed at 600 s by default; a single kernel_workflow attempt runs
# far longer than that.
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS="${CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS:-0}"

# The per-GPU container runs as root, and the CLI refuses to skip permissions for root:
#   "--dangerously-skip-permissions cannot be used with root/sudo privileges"
# This variable is its own opt-out. Isolation here does not come from permission prompts
# — it comes from one container per GPU with exactly one renderD node mounted, so
# allowing this does not widen the blast radius. To disable it, set IS_SANDBOX=0 in the
# env file.
export IS_SANDBOX="${IS_SANDBOX:-1}"

# ---- 3. Make the model explicit.
# The environment may already export ANTHROPIC_MODEL (some clusters do this in .bashrc),
# in which case a `${VAR:-default}` idiom silently runs a different model. Only
# WORKLOAD_AGENT_MODEL is honoured here; with nothing configured the inherited value is
# cleared so the CLI falls back to its own default.
MODEL_ARGS=()
if [ -n "${WORKLOAD_AGENT_MODEL:-}" ]; then
  MODEL_ARGS=(--model "$WORKLOAD_AGENT_MODEL")
else
  unset ANTHROPIC_MODEL
fi

# ---- 4. GPU visibility: not touched at all.
# The caller already enforces physical isolation — one container per GPU, with a single
# /dev/dri/renderD<128+8*i> mounted. An env-only pin gets reset by the agent itself, so
# neither ROCR_ nor HIP_VISIBLE_DEVICES is set or overridden here.

cd "$WORKDIR" || blocked "cannot enter workdir $WORKDIR"

{
  echo "[runner:claude] $(date -Is) bin=$BIN workdir=$WORKDIR state=$STATE_DIR"
  echo "[runner:claude] model=${WORKLOAD_AGENT_MODEL:-<CLI default>}"
} >> "$LOG"

# No internal timeout: the time budget is enforced from the outside by the scheduler
# with timeout(1).
"$BIN" -p \
  --permission-mode bypassPermissions \
  --add-dir "$WORKDIR" \
  "${MODEL_ARGS[@]}" \
  ${WORKLOAD_AGENT_EXTRA_ARGS:-} \
  < "$PROMPT" >> "$LOG" 2>&1
RC=$?

echo "[runner:claude] $(date -Is) exit=$RC" >> "$LOG"
rm -rf "$STATE_DIR" 2>/dev/null
exit $RC
