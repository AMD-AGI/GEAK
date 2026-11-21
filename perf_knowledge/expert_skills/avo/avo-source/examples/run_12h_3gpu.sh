#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# <GEAK>/perf_knowledge/expert_skills/avo/avo-source/examples -> <GEAK>
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SITE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
OUTPUT_ROOT="${AVO_EXAMPLE_RUNS_ROOT:-${REPO_ROOT}-runs/avo-examples}"

CASE="${1:-all}"
GPU_IDS="${2:-${GPU_IDS:-0,1,2}}"
BUDGET_S="${BUDGET_S:-43200}"
DRY_RUN="${DRY_RUN:-0}"
CLAUDE_ACTIVATE="${CLAUDE_ACTIVATE:-$SITE_ROOT/claude-portable/activate.sh}"
FLYDSL_ENV="${AVO_FLYDSL_ENV:-$SITE_ROOT/flydsl_rocm_env.sh}"

if [[ ! "$GPU_IDS" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "GPU_IDS must contain exactly three comma-separated IDs" >&2
  exit 2
fi
IFS=, read -r GPU_A GPU_B GPU_C <<<"$GPU_IDS"
if [[ "$GPU_A" == "$GPU_B" || "$GPU_A" == "$GPU_C" || "$GPU_B" == "$GPU_C" ]]; then
  echo "GPU_IDS must contain three distinct IDs" >&2
  exit 2
fi
if [[ ! "$BUDGET_S" =~ ^[0-9]+$ ]] || (( BUDGET_S < 1 )); then
  echo "BUDGET_S must be a positive integer" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ ! -f "$CLAUDE_ACTIVATE" ]]; then
  echo "Claude activation script not found: $CLAUDE_ACTIVATE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CLAUDE_ACTIVATE"

run_case() {
  local case_name="$1"
  local case_output="$OUTPUT_ROOT/$case_name"
  local timestamp log
  local -a dry_args=() command=()

  if [[ "$case_name" == flydsl-* ]]; then
    if [[ ! -f "$FLYDSL_ENV" ]]; then
      echo "FlyDSL environment script not found: $FLYDSL_ENV" >&2
      return 2
    fi
    export AVO_FLYDSL_ENV="$FLYDSL_ENV"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    dry_args+=(--dry-run)
  fi

  echo "Starting $case_name: budget=${BUDGET_S}s GPUs=$GPU_IDS"
  command=(python3 "$SCRIPT_DIR/run_example.py" "$case_name" \
    --budget-s "$BUDGET_S" \
    --final-phase-reserve-s 900 \
    --gpu-ids "$GPU_IDS" \
    --operator-directions 3 \
    --start-policy diverse \
    --start-k 3 \
    --escalate-workers 3 \
    --max-steps 1000 \
    --verify-repeats 5 \
    --use-profiling true \
    "${dry_args[@]}")
  if [[ "$DRY_RUN" == "1" ]]; then
    "${command[@]}"
  else
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$case_output/.avo"
    log="$case_output/.avo/launch_${timestamp}.log"
    "${command[@]}" 2>&1 | tee "$log"
  fi
}

case "$CASE" in
  triton)
    run_case "triton-w8a8-block-scaled-mm"
    ;;
  flydsl)
    run_case "flydsl-per-token-group-quant-fp8"
    ;;
  flydsl-gemm)
    run_case "flydsl-gemm-a16-w16"
    ;;
  all)
    # Sequential by design: all campaigns use the same three GPUs.
    run_case "triton-w8a8-block-scaled-mm"
    run_case "flydsl-per-token-group-quant-fp8"
    run_case "flydsl-gemm-a16-w16"
    ;;
  *)
    echo "usage: $0 [triton|flydsl|flydsl-gemm|all] [gpu0,gpu1,gpu2]" >&2
    exit 2
    ;;
esac
