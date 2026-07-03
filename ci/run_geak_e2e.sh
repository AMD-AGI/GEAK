#!/usr/bin/env bash
# =============================================================================
# run_geak_e2e.sh — reproduce ONE geakv4 (GEAK@GEAK_v4, aka "perfskills") e2e run
# EXACTLY the way Hyperloom's KERNEL_AGENT phase launches it.
#
# Grounded in source (read, not guessed):
#   Hyperloom: inference_optimizer/orchestrator/coordinator.py::_run_perfskills_kernel_phase
#       cmd = ["python3", run_e2e.py, <handoff.json>, <out_dir>, "--timeout-s", <runner_timeout>]
#       subprocess.Popen(cmd, env=dict(os.environ), start_new_session=True)
#   GEAK:      interface/run_e2e.py::main
#       - positional args:  args[0]=handoff.json   args[1]=result.json   (only --dry-run flag is read)
#       - BUDGET is read from env PERFSKILLS_E2E_TIMEOUT_S (default 43200s=12h).
#         The CLI "--timeout-s" value is DISCARDED by run_e2e (it lands in an ignored positional).
#       - PERFSKILLS_ROOT is derived from run_e2e.py's own location (interface/..), so calling the
#         real path is enough; it maps the handoff onto e2e_workflow/e2e_workflow.js and drives it
#         via the Claude SDK (model claude-opus-4-8, effort ultracode).
#
# Usage:   ./run_geak_e2e.sh <model_dir> [--dry-run]
#   <model_dir> is one of the per-model folders here (contains handoff.json [+ baseline_config...]).
#   Start with --dry-run: it prints the mapped e2e_workflow.js args + prompt and does NO GPU work.
#
# Optional env overrides:
#   GEAK_ROOT                 default: the GEAK repo two levels up from this script (ci/..)
#   PERFSKILLS_E2E_TIMEOUT_S  geak's REAL wall-clock budget in seconds (default 17100 ≈ 4.75h)
#   EXP_ROOT                  writable run root; patches handoff.exp_root (default: <model_dir>/repro_out/exp)
#   MODEL_PATH                real served model dir; patches handoff.model_path (default: keep handoff value)
#   INFERENCEX_PATH           InferenceX checkout  -> bench_client=inferencex (else geak falls back to native)
#   OUT_DIR                   where result.json is written (default <model_dir>/repro_out)
#   PERFSKILLS_CLAUDE_MODEL / PERFSKILLS_CLAUDE_EFFORT / PERFSKILLS_CLAUDE_BIN  (defaults match run_e2e.py)
#
# HARD external deps for a REAL (non --dry-run) run:
#   * Claude credentials in the environment (ANTHROPIC_API_KEY / CURSOR_API_KEY / `claude` login) —
#     geak IS a Claude-SDK workflow; without creds the workflow cannot run.
#   * A GPU box with the framework (vllm/sglang) + the actual model weights at handoff.model_path.
#   * (optional) InferenceX checkout for byte-identical bench client vs Hyperloom.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # <ws>/GEAK/ci
MODEL_DIR="${1:?usage: run_geak_e2e.sh <model_dir> [--dry-run]}"
DRY="${2:-}"

GEAK_ROOT="${GEAK_ROOT:-$(dirname "$HERE")}"                   # <ws>/GEAK
RUNNER="$GEAK_ROOT/interface/run_e2e.py"
[ -f "$RUNNER" ] || { echo "run_e2e.py not found at $RUNNER (set GEAK_ROOT)"; exit 2; }

HANDOFF_SRC="$MODEL_DIR/handoff.json"
[ -f "$HANDOFF_SRC" ] || { echo "no handoff.json in $MODEL_DIR"; exit 2; }

OUT_DIR="${OUT_DIR:-$MODEL_DIR/repro_out}"
mkdir -p "$OUT_DIR"
RESULT="$OUT_DIR/result.json"
EXP_ROOT="${EXP_ROOT:-$OUT_DIR/exp}"
mkdir -p "$EXP_ROOT"

# ---- Patch exp_root (always, to a writable dir) and model_path (only if MODEL_PATH given) ----
# The dataset handoff.exp_root points at the original /hyperloom/... path; rewrite it so geak
# writes under a local writable location. model_path is left as-is unless you pass MODEL_PATH.
# InferenceX = Hyperloom/Magpie's bench CLIENT (utils/bench_serving/benchmark_serving.py). The
# handoff carries a stale /tmp/hyperloom/... inferencex_path that run_e2e prefers over $INFERENCEX_PATH,
# so we repoint it here. Default to the local checkout; set INFERENCEX_PATH="" to force native bench.
INFERENCEX_PATH="${INFERENCEX_PATH-$(dirname "$GEAK_ROOT")/InferenceX}"
export INFERENCEX_PATH

# If the local recipe was shipped alongside, repoint launch_recipe at it (the dataset value is a
# stale /hyperloom/... path that won't exist on your box).
LOCAL_RECIPE="$MODEL_DIR/baseline_config.with_envs.yaml"
HANDOFF="$OUT_DIR/handoff.patched.json"
python3 - "$HANDOFF_SRC" "$HANDOFF" "$EXP_ROOT" "${MODEL_PATH:-}" "$LOCAL_RECIPE" "${INFERENCEX_PATH:-}" <<'PY'
import json, os, sys
src, dst, exp_root, model_path, local_recipe, ix = sys.argv[1:7]
h = json.load(open(src))
h["exp_root"] = exp_root
if model_path:
    h["model_path"] = model_path
if os.path.isfile(local_recipe):
    h["launch_recipe"] = os.path.abspath(local_recipe)
# Repoint inferencex_path to the local checkout (or drop it so $INFERENCEX_PATH / native applies).
if ix and os.path.isdir(ix):
    h["inferencex_path"] = os.path.abspath(ix)
else:
    h.pop("inferencex_path", None)
json.dump(h, open(dst, "w"), indent=2)
print(f"patched handoff -> {dst}\n  exp_root={h['exp_root']}\n  model_path={h.get('model_path')}\n  launch_recipe={h.get('launch_recipe')}\n  inferencex_path={h.get('inferencex_path')}")
PY

# ---- Budget: run_e2e reads PERFSKILLS_E2E_TIMEOUT_S (NOT the CLI flag). Export it. ----
export PERFSKILLS_E2E_TIMEOUT_S="${PERFSKILLS_E2E_TIMEOUT_S:-17100}"

# ---- Claude workflow knobs (defaults already match run_e2e.py) ----
export PERFSKILLS_CLAUDE_MODEL="${PERFSKILLS_CLAUDE_MODEL:-claude-opus-4-8}"
export PERFSKILLS_CLAUDE_EFFORT="${PERFSKILLS_CLAUDE_EFFORT:-ultracode}"

# (INFERENCEX_PATH already exported above; run_e2e exports BENCH_CLIENT from it.)

echo "=============================================================="
echo " GEAK e2e reproduction"
echo "   runner   = $RUNNER"
echo "   handoff  = $HANDOFF"
echo "   result   = $RESULT"
echo "   budget   = PERFSKILLS_E2E_TIMEOUT_S=$PERFSKILLS_E2E_TIMEOUT_S s"
echo "   claude   = $PERFSKILLS_CLAUDE_MODEL / effort=$PERFSKILLS_CLAUDE_EFFORT"
echo "   inferencex_path = ${INFERENCEX_PATH:-<unset -> native bench>}"
echo "   dry_run  = ${DRY:-<no>}"
echo "=============================================================="

# We pass --timeout-s too, purely to mirror Hyperloom's exact argv (run_e2e ignores its value;
# the effective budget is the PERFSKILLS_E2E_TIMEOUT_S env above).
exec python3 "$RUNNER" "$HANDOFF" "$RESULT" --timeout-s "$PERFSKILLS_E2E_TIMEOUT_S" ${DRY:+$DRY}
