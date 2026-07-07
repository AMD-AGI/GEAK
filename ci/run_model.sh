#!/usr/bin/env bash
# Step E + F — run GEAK_v4 e2e for ONE model into a timestamped folder, then judge.
# Runs INSIDE the container for a real run; runs on the host for --dry-run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib.sh"

MODEL_KEY="${1:?usage: run_model.sh <model_key> [--dry-run]}"
DRY="${2:-}"

MODEL_DIR="$HF_LOGS/$MODEL_KEY"
[ -d "$MODEL_DIR" ]              || die "no model dir: $MODEL_DIR"
[ -f "$MODEL_DIR/handoff.json" ] || die "no handoff.json in $MODEL_DIR"

# exp_root MUST stay <model>/perfskills: run_e2e.py discovers the TraceLens priors
# relative to it (artifacts live one level up at <model>/kernel-agent, <model>/runs).
# The workflow itself auto-timestamps its eval_dir (perfskills/e2e_<model>_<ts>), so
# runs stay distinguishable even though exp_root is fixed.
EXP_ROOT="${EXP_ROOT:-$MODEL_DIR/perfskills}"
# Timestamped run folder for CI-level outputs (result.json + logs + claude state).
RUN_TS="${RUN_TS:-$(new_ts)}"
OUT_DIR="${OUT_DIR:-$MODEL_DIR/ci_runs/$RUN_TS}"
mkdir -p "$OUT_DIR" "$EXP_ROOT"
LOG="$OUT_DIR/run.log"

# Weights: handoff still carries /wekafs/...; override with the real local path.
MODEL_PATH="${MODEL_PATH:-$(model_weights "$MODEL_KEY" 2>/dev/null || true)}"

if [ "$DRY" != "--dry-run" ]; then
  [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ] \
    || die "weights not found for $MODEL_KEY (MODEL_PATH='$MODEL_PATH')"
  # Claude was installed under CLAUDE_HOME by Step D; make it reachable here too.
  if [ -n "${CLAUDE_HOME:-}" ]; then
    export HOME="$CLAUDE_HOME"; export PATH="$HOME/.local/bin:$PATH"
  fi
fi

export EXP_ROOT OUT_DIR MODEL_PATH INFERENCEX_PATH GEAK_ROOT
export PERFSKILLS_E2E_TIMEOUT_S="${PERFSKILLS_E2E_TIMEOUT_S:-1800}"

log "model=$MODEL_KEY dry=${DRY:-no} out=$OUT_DIR exp_root=$EXP_ROOT budget=${PERFSKILLS_E2E_TIMEOUT_S}s"
log "weights=$MODEL_PATH inferencex=$INFERENCEX_PATH"

set +e
bash "$HERE/run_geak_e2e.sh" "$MODEL_DIR" ${DRY:+$DRY} 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

# ---- Step F: deterministic hard judge (exit code + result.json.status) ----
if [ "$DRY" = "--dry-run" ]; then
  [ "$RC" -eq 0 ] && { log "DRY-RUN mapping OK"; exit 0; } || die "dry-run failed rc=$RC" "$RC"
fi

RESULT="$OUT_DIR/result.json"
[ -f "$RESULT" ] || die "no result.json at $RESULT (rc=$RC)"
STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$RESULT" 2>/dev/null || echo "")
ERRCLASS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("error_class",""))' "$RESULT" 2>/dev/null || echo "")
log "run rc=$RC status=$STATUS error_class=${ERRCLASS:-<none>}"

# workflow_parse_error is the fingerprint of run_e2e falling back to the `claude -p`
# CLI path (no background-task keep-alive) — almost always means the Python
# claude_agent_sdk isn't importable in the container. Flag it explicitly so this
# regression is obvious rather than looking like a generic e2e failure.
if [ "$ERRCLASS" = "workflow_parse_error" ]; then
  log "HINT: workflow_parse_error usually means the Python 'claude_agent_sdk' is"
  log "      missing in the container -> run_e2e used the CLI fallback. Check that"
  log "      ci/setup_claude.sh installed claude-agent-sdk (import must succeed)."
fi

# A trustworthy ok/no_gain REQUIRES a real measured baseline (>0). A zero/absent
# baseline_throughput_tok_s means NOTHING was actually measured — GPU unusable,
# the serving path never came up, etc. — which perfskills reports as a graceful
# no_gain. Treat that as a HARD failure (error_class=gpu_unusable) so the CI goes
# red instead of a false-green no_gain. (baseline_throughput_tok_s is always
# present on the ok/no_gain result path; error/timeout are caught by the * case.)
measured_baseline() {
  python3 -c 'import json,sys
try:
    b = json.load(open(sys.argv[1])).get("baseline_throughput_tok_s") or 0
    print("1" if float(b) > 0 else "0")
except Exception:
    print("0")' "$RESULT" 2>/dev/null || echo "0"
}

case "$STATUS" in
  ok|no_gain)
    if [ "$(measured_baseline)" = "1" ]; then
      log "PASS ($STATUS)"; exit 0
    fi
    die "FAIL status='$STATUS' but baseline_throughput_tok_s<=0 (unmeasured — GPU unusable / serving never healthy) error_class=gpu_unusable rc=$RC"
    ;;
  *) die "FAIL status='$STATUS' error_class=${ERRCLASS:-<none>} rc=$RC" ;;
esac
