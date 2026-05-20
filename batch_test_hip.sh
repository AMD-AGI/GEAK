#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
AMD_LLM_API_KEY="${AMD_LLM_API_KEY:?Set AMD_LLM_API_KEY}"
MODEL="${GEAK_MODEL:-claude-opus-4.6}"
HIP_CONTAINER="${HIP_CONTAINER:-geak_gwiab_hip}"
BATCH_DIR="${BATCH_DIR:-/data/sapmajum/parity_test/runs_$(date +%Y%m%d)}"
AKA_HIP="${AKA_HIP:-$BATCH_DIR/AKA_hip/tasks/hip2hip/others}"
GEAK_DIR="/workspace"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_PARALLEL="${NUM_PARALLEL:-4}"
RUN_MODE="${RUN_MODE:-full}"
PIPELINE_MODE="fixed"
RESULTS_CSV="$BATCH_DIR/hip_results.csv"

DEFAULT_HIP_TEST="python3 scripts/task_runner.py compile && \
python3 scripts/task_runner.py correctness && \
python3 scripts/task_runner.py performance"

# All 12 HIP kernels
HIP_KERNELS=(
  assign_score_withk
  ball_query
  furthest_point_sample
  gather_points
  knn
  matrix_multiplication
  points_in_boxes
  roiaware_pool3d
  roipoint_pool3d
  silu
  three_interpolate
  three_nn
)

# ── Pre-flight checks ────────────────────────────────────────────
echo "=== Pre-flight checks ==="

# Verify container is running
if ! docker inspect "$HIP_CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
  echo "ERROR: Container $HIP_CONTAINER is not running."
  exit 1
fi

# Verify geak is installed
if ! docker exec "$HIP_CONTAINER" geak --help &>/dev/null; then
  echo "ERROR: geak not found in $HIP_CONTAINER. Run: docker exec $HIP_CONTAINER bash -c 'cd /workspace && make install-dev'"
  exit 1
fi

# Verify GPUs are visible
echo "Checking GPUs..."
docker exec "$HIP_CONTAINER" rocm-smi --showid 2>/dev/null | head -5 || true

# Kill stale geak/python processes
echo "Cleaning stale processes..."
docker exec "$HIP_CONTAINER" bash -c "pkill -f geak 2>/dev/null; pkill -f 'python.*mini' 2>/dev/null" || true
docker exec "$HIP_CONTAINER" bash -c \
  'for p in $(ps aux | grep defunct | awk "{print \$2}"); do kill -9 $p 2>/dev/null; done' || true

# Clear pycache
docker exec "$HIP_CONTAINER" find /workspace/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Pre-flight OK"
echo ""

# ── Initialize results CSV ───────────────────────────────────────
mkdir -p "$BATCH_DIR"
[[ -f "$RESULTS_CSV" ]] || echo "name,mode,gpu_ids,num_parallel,verified_speedups,final_speedup,status,elapsed_s" > "$RESULTS_CSV"

# ── Helper: find .hip kernel file ────────────────────────────────
resolve_hip_kernel() {
  local REPO_DIR=$1
  find "$REPO_DIR" \( -name '*.hip' \) \
    ! -name '*_ref.hip' ! -path '*/build/*' ! -path '*/_geak_*' \
    -print 2>/dev/null | head -1
}

# ── Helper: extract results ─────────────────────────────────────
extract_results() {
  local OUT_DIR=$1 LOG_FILE=$2
  local FINAL="" VS="" STATUS="done"

  if [[ -f "$OUT_DIR/out/final_report.json" ]]; then
    FINAL=$(python3 -c "
import json
d = json.load(open('$OUT_DIR/out/final_report.json'))
print(d.get('total_speedup') or d.get('best_speedup') or '')
" 2>/dev/null || true)
  else
    STATUS="no_final_report"
  fi

  VS=$(grep -oE 'Verified speedup: [0-9.]+x' "$LOG_FILE" 2>/dev/null \
    | tr '\n' ' ' | sed 's/ $//') || true

  echo "${FINAL}|${VS}|${STATUS}"
}

# ── Helper: clean worktrees ──────────────────────────────────────
cleanup_worktrees() {
  local OUT_DIR=$1
  docker exec "$HIP_CONTAINER" bash -c \
    "find $OUT_DIR/out/results -type d -name worktrees \
     -prune -exec rm -rf {} + 2>/dev/null || true" 2>/dev/null || true
}

# ── Run one HIP kernel ──────────────────────────────────────────
run_hip_one() {
  local NAME=$1
  local REPO_DIR="${AKA_HIP}/${NAME}"
  local OUT_DIR="$BATCH_DIR/hip_${NAME}"

  echo "=========================================="
  echo "Kernel: $NAME"
  echo "=========================================="

  # Skip if already completed
  if [[ -f "$OUT_DIR/out/final_report.json" ]]; then
    echo "  SKIP: final_report.json exists, already completed."
    return 0
  fi

  # Verify kernel directory exists
  if [[ ! -d "$REPO_DIR" ]]; then
    echo "  SKIP: $REPO_DIR does not exist."
    echo "${NAME},${PIPELINE_MODE},${GPU_IDS},${NUM_PARALLEL},,,,0" >> "$RESULTS_CSV"
    return 0
  fi

  # Resolve .hip kernel file
  local KERNEL_PATH
  KERNEL_PATH=$(resolve_hip_kernel "$REPO_DIR")
  if [[ -z "$KERNEL_PATH" ]]; then
    echo "  SKIP: No .hip file found in $REPO_DIR."
    echo "${NAME},${PIPELINE_MODE},${GPU_IDS},${NUM_PARALLEL},,,,0" >> "$RESULTS_CSV"
    return 0
  fi

  rm -rf "$OUT_DIR" 2>/dev/null || true
  mkdir -p "$OUT_DIR"

  local PROMPT="Optimize the HIP kernel at ${KERNEL_PATH} in the repo ${REPO_DIR}. \
Test command is: ${DEFAULT_HIP_TEST}
Pipeline mode: fixed — run parallel workers with the same task body \
(identical strategy per worker, best-of-N style). \
Use GPUs ${GPU_IDS}. \
The output directory should be ${OUT_DIR}/out."

  echo "  Kernel file: $KERNEL_PATH"
  echo "  Output: $OUT_DIR"
  echo "  GPUs: $GPU_IDS"
  echo "  Started at: $(date)"

  local T0
  T0=$(date +%s)

  docker exec \
    -e "GEAK_MODEL=$MODEL" \
    -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
    -e "GEAK_PIPELINE_MODE=$PIPELINE_MODE" \
    "$HIP_CONTAINER" bash -c \
      "cd $GEAK_DIR && geak -t $(printf %q "$PROMPT") \
       --num-parallel $NUM_PARALLEL --gpu-ids $GPU_IDS --mode $RUN_MODE \
       -y --exit-immediately" \
    > "$OUT_DIR/run.log" 2>&1 || true

  local ELAPSED=$(( $(date +%s) - T0 ))
  local ELAPSED_MIN=$(( ELAPSED / 60 ))

  # Extract results
  IFS='|' read -r FINAL VS STATUS <<< "$(extract_results "$OUT_DIR" "$OUT_DIR/run.log")"
  echo "${NAME},${PIPELINE_MODE},${GPU_IDS},${NUM_PARALLEL},\"${VS}\",${FINAL},${STATUS},${ELAPSED}" \
    >> "$RESULTS_CSV"

  echo "  Finished in ${ELAPSED_MIN}m (${ELAPSED}s)"
  echo "  Status: $STATUS"
  echo "  Final speedup: ${FINAL:-N/A}"
  echo "  Verified: ${VS:-none}"
  echo ""

  # Clean worktrees to save disk
  cleanup_worktrees "$OUT_DIR"
}

# ── Main ─────────────────────────────────────────────────────────
echo "=== HIP Batch Test ==="
echo "Container:  $HIP_CONTAINER"
echo "Model:      $MODEL"
echo "GPU IDs:    $GPU_IDS"
echo "Parallel:   $NUM_PARALLEL"
echo "Mode:       $RUN_MODE"
echo "Pipeline:   $PIPELINE_MODE"
echo "Batch dir:  $BATCH_DIR"
echo "Kernels:    ${#HIP_KERNELS[@]}"
echo ""

BATCH_T0=$(date +%s)

for NAME in "${HIP_KERNELS[@]}"; do
  run_hip_one "$NAME"
done

BATCH_ELAPSED=$(( $(date +%s) - BATCH_T0 ))
BATCH_HOURS=$(( BATCH_ELAPSED / 3600 ))
BATCH_MINS=$(( (BATCH_ELAPSED % 3600) / 60 ))

echo "=========================================="
echo "=== All HIP kernels done ==="
echo "Total time: ${BATCH_HOURS}h ${BATCH_MINS}m"
echo "=========================================="
echo ""
echo "=== Results ==="
column -t -s, "$RESULTS_CSV"
