#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_benchmark.sh
#
# Generic benchmark runner: preprocess a kernel, then orchestrate in
# homogeneous mode, heterogeneous mode, or both sequentially.
#
# Usage:
#   ./scripts/run_benchmark.sh <kernel_url> --gpu-ids 0,1
#   ./scripts/run_benchmark.sh <kernel_url> --gpu-ids 0,1 --mode homo
#   ./scripts/run_benchmark.sh <kernel_url> --gpu-ids 0,1 --mode hetero
#   ./scripts/run_benchmark.sh <kernel_url> --gpu-ids 0,1 --output-dir my_bench --max-rounds 3
#
# Designed to run inside the Docker container where geak-preprocess and
# geak-orchestrate are on PATH.
# ============================================================================

usage() {
  cat <<EOF
Usage: $0 <kernel_url> --gpu-ids <ids> [options]

Positional:
  kernel_url          GitHub URL or local path to the kernel

Required:
  --gpu-ids IDS       Comma-separated GPU device IDs (e.g. 0,1,2,3)

Options:
  --output-dir DIR    Output directory (default: benchmark_output)
  --mode MODE         homo | hetero | both (default: both)
  --max-rounds N      Maximum optimisation rounds per mode (default: 2)
  --model NAME        Model name forwarded to geak-preprocess and geak-orchestrate
  -h, --help          Show this help
EOF
  exit "${1:-0}"
}

# ── Arg parsing ──────────────────────────────────────────────────────────────

KERNEL_URL=""
GPU_IDS=""
OUTPUT_DIR="benchmark_output"
MODE="both"
MAX_ROUNDS=2
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)     usage 0 ;;
    --gpu-ids)     GPU_IDS="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --mode)        MODE="$2"; shift 2 ;;
    --max-rounds)  MAX_ROUNDS="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    -*)            echo "Unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -z "$KERNEL_URL" ]]; then
        KERNEL_URL="$1"; shift
      else
        echo "Unexpected argument: $1" >&2; usage 1
      fi
      ;;
  esac
done

[[ -z "$KERNEL_URL" ]] && { echo "Error: kernel_url is required" >&2; usage 1; }
[[ -z "$GPU_IDS" ]]    && { echo "Error: --gpu-ids is required" >&2; usage 1; }

case "$MODE" in
  homo|hetero|both) ;;
  *) echo "Error: --mode must be homo, hetero, or both (got: $MODE)" >&2; exit 1 ;;
esac

FIRST_GPU="${GPU_IDS%%,*}"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

PREPROCESS_DIR="${OUTPUT_DIR}/preprocess"
SUMMARY_LOG="${OUTPUT_DIR}/summary.log"
: > "$SUMMARY_LOG"

# Preprocess artifacts expected by the orchestrator
ARTIFACT_FILES=(
  resolved.json
  discovery.json
  harness_path.txt
  profile.json
  baseline_metrics.json
  benchmark_baseline.txt
  full_benchmark_baseline.txt
  COMMANDMENT.md
  CODEBASE_CONTEXT.md
  harness_results.json
)

echo "============================================================"
echo " GEAK Benchmark Runner"
echo " Kernel:     ${KERNEL_URL}"
echo " Output:     ${OUTPUT_DIR}"
echo " GPU IDs:    ${GPU_IDS}"
echo " Mode:       ${MODE}"
echo " Max rounds: ${MAX_ROUNDS}"
echo "============================================================"
echo ""

# ── Step 1: Preprocess ──────────────────────────────────────────────────────

echo "--- Step 1: Preprocessing ---"

PREPROCESS_CMD=(geak-preprocess "$KERNEL_URL" -o "$PREPROCESS_DIR" --gpu "$FIRST_GPU")
[[ -n "$MODEL" ]] && PREPROCESS_CMD+=(-m "$MODEL")

echo "  Running: ${PREPROCESS_CMD[*]}"
PREPROCESS_LOG="${OUTPUT_DIR}/preprocess.log"
START_TS=$(date +%s)

if ! "${PREPROCESS_CMD[@]}" > "$PREPROCESS_LOG" 2>&1; then
  ELAPSED=$(( $(date +%s) - START_TS ))
  echo "  FAIL  (${ELAPSED}s) — see ${PREPROCESS_LOG}"
  echo "preprocess  FAIL  ${ELAPSED}s" >> "$SUMMARY_LOG"
  echo ""
  echo "Preprocessing failed. Cannot continue."
  exit 1
fi

ELAPSED=$(( $(date +%s) - START_TS ))
echo "  PASS  (${ELAPSED}s)"
echo "preprocess  PASS  ${ELAPSED}s" >> "$SUMMARY_LOG"
echo ""

# ── Helper: copy artifacts and run orchestrator ─────────────────────────────

run_orchestrate() {
  local mode_name="$1"
  local hetero_flag="$2"
  local mode_dir="${OUTPUT_DIR}/${mode_name}"

  echo "--- ${mode_name} orchestration ---"

  # Copy preprocess artifacts into mode-specific directory
  mkdir -p "$mode_dir"
  for f in "${ARTIFACT_FILES[@]}"; do
    if [[ -f "${PREPROCESS_DIR}/${f}" ]]; then
      cp "${PREPROCESS_DIR}/${f}" "${mode_dir}/${f}"
    fi
  done
  echo "  Copied artifacts to ${mode_dir}"

  # Build orchestrator command
  local cmd=(
    geak-orchestrate
    --preprocess-dir "$mode_dir"
    --gpu-ids "$GPU_IDS"
    --max-rounds "$MAX_ROUNDS"
  )
  [[ -n "$hetero_flag" ]] && cmd+=(--heterogeneous)
  [[ -n "$MODEL" ]] && cmd+=(--model "$MODEL")

  echo "  Running: ${cmd[*]}"
  local log_file="${mode_dir}/orchestrate.log"
  local start_ts
  start_ts=$(date +%s)

  set +e
  "${cmd[@]}" > "$log_file" 2>&1
  local exit_code=$?
  set -e

  local elapsed=$(( $(date +%s) - start_ts ))

  if [[ $exit_code -eq 0 ]]; then
    echo "  PASS  (${elapsed}s)"
    echo "${mode_name}  PASS  ${elapsed}s" >> "$SUMMARY_LOG"
  else
    echo "  FAIL  exit=${exit_code} (${elapsed}s) — see ${log_file}"
    echo "${mode_name}  FAIL(${exit_code})  ${elapsed}s" >> "$SUMMARY_LOG"
  fi

  # Show final report path if it exists
  local report="${mode_dir}/final_report.json"
  if [[ -f "$report" ]]; then
    echo "  Report: ${report}"
  fi
  echo ""
}

# ── Step 2 & 3: Orchestrate ────────────────────────────────────────────────

if [[ "$MODE" == "both" || "$MODE" == "homo" ]]; then
  run_orchestrate "homo" ""
fi

if [[ "$MODE" == "both" || "$MODE" == "hetero" ]]; then
  run_orchestrate "hetero" "--heterogeneous"
fi

# ── Step 4: Summary ─────────────────────────────────────────────────────────

echo "============================================================"
echo " Summary"
echo "============================================================"
column -t "$SUMMARY_LOG" 2>/dev/null || cat "$SUMMARY_LOG"
echo ""
echo "Results in: ${OUTPUT_DIR}"
