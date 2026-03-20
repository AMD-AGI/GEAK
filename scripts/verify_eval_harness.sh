#!/usr/bin/env bash
# Rigorous independent verification of eval_harness speedups.
#
# For each aiter-based kernel:
#   1. Create a FRESH worktree from the clean repo HEAD (no leftover artifacts)
#   2. Run harness --benchmark 3 times (50 warmup, 200 iterations = harness defaults)
#   3. Take the median GEAK_RESULT_LATENCY_MS across 3 runs -> baseline_ms
#   4. Remove worktree, create a NEW fresh worktree, apply patch
#   5. Run harness --benchmark 3 times
#   6. Take the median GEAK_RESULT_LATENCY_MS across 3 runs -> patched_ms
#   7. Speedup = baseline_ms / patched_ms
#
# The harness internally: per shape, takes median of 200 iterations,
# then reports geometric mean across all shapes as GEAK_RESULT_LATENCY_MS.
#
# Usage (inside geak-agent-sdubagun):
#   bash /workspace/scripts/verify_eval_harness.sh
#   bash /workspace/scripts/verify_eval_harness.sh topk

set -uo pipefail

GPU="${GEAK_VERIFY_GPU:-4}"
RUNS=3
REPO="/home/sdubagun/work/repos/aiter"
BASE="/workspace/outputs/eval_harness"
VERIFY_DIR="$BASE/_verification_v2"
mkdir -p "$VERIFY_DIR"

declare -A PP_DIRS=(
  [topk]="preprocess"
  [fused_qkv_rope]="preprocess"
  [fused_rms_fp8]="preprocess"
  [lean_atten_paged]="preprocess"
  [moe_routing_sigmoid_top1]="preprocess"
)

KERNELS=(topk fused_qkv_rope fused_rms_fp8 lean_atten_paged moe_routing_sigmoid_top1)

if [[ $# -gt 0 ]]; then
  KERNELS=("$@")
fi

extract_latency() {
  grep -oP 'GEAK_RESULT_LATENCY_MS=\K[\d.]+' "$1" 2>/dev/null | tail -1
}

median_of_three() {
  python3 -c "
import sys
vals = sorted([float(x) for x in sys.argv[1:]])
print(vals[len(vals)//2])
" "$@"
}

create_clean_worktree() {
  local wt_path="$1"
  rm -rf "$wt_path"
  # Verify main repo is clean
  local dirty
  dirty=$(git -C "$REPO" status --porcelain 2>/dev/null | head -1)
  if [[ -n "$dirty" ]]; then
    echo "  WARNING: repo has uncommitted changes, resetting..."
    git -C "$REPO" checkout -- . 2>/dev/null
  fi
  git -C "$REPO" worktree add --detach "$wt_path" HEAD 2>/dev/null
  # Double-check worktree is clean
  local wt_dirty
  wt_dirty=$(git -C "$wt_path" status --porcelain 2>/dev/null | head -1)
  if [[ -n "$wt_dirty" ]]; then
    echo "  ERROR: worktree not clean after creation!"
    return 1
  fi
}

remove_worktree() {
  local wt_path="$1"
  git -C "$REPO" worktree remove --force "$wt_path" 2>/dev/null || true
  rm -rf "$wt_path" 2>/dev/null || true
}

run_benchmark() {
  local harness="$1"
  local wt_path="$2"
  local out_file="$3"

  HIP_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$wt_path:${ORIG_PYTHONPATH:-}" \
    python3 "$harness" --benchmark --warmup 50 --iterations 200 \
    > "$out_file" 2>/dev/null
}

ORIG_PYTHONPATH="${PYTHONPATH:-}"

echo "================================================================"
echo "  RIGOROUS VERIFICATION"
echo "  GPU=$GPU, runs=$RUNS, warmup=50, iterations=200"
echo "  Repo: $REPO"
echo "================================================================"
echo ""

RESULTS_FILE="$VERIFY_DIR/results.csv"
echo "kernel,baseline_r1,baseline_r2,baseline_r3,baseline_median,patched_r1,patched_r2,patched_r3,patched_median,speedup,reported" > "$RESULTS_FILE"

for kernel in "${KERNELS[@]}"; do
  pp="${PP_DIRS[$kernel]}"
  pp_dir="$BASE/$kernel/$pp"
  report="$pp_dir/final_report.json"

  if [[ ! -f "$report" ]]; then
    echo "SKIP $kernel: no final_report.json"
    continue
  fi

  harness=$(cat "$pp_dir/harness_path.txt")
  patch=$(python3 -c "import json; print(json.load(open('$report')).get('best_patch','') or '')")
  reported=$(python3 -c "import json; print(json.load(open('$report')).get('best_speedup','?'))")

  echo "============================================================"
  echo "  $kernel (reported: ${reported}x)"
  echo "  harness: $harness"
  echo "  patch: $(basename "$patch")"
  echo "============================================================"

  work="$VERIFY_DIR/$kernel"
  mkdir -p "$work"

  # ── Baseline: 3 runs on clean worktree ─────────────────────────
  echo "  [Baseline] Creating clean worktree..."
  wt_bl="$work/wt_baseline"
  create_clean_worktree "$wt_bl"

  bl_lats=()
  for run in $(seq 1 $RUNS); do
    out="$work/baseline_run${run}.txt"
    echo -n "  [Baseline] Run $run/$RUNS... "
    run_benchmark "$harness" "$wt_bl" "$out"
    lat=$(extract_latency "$out")
    echo "${lat:-FAILED} ms"
    bl_lats+=("${lat:-0}")
  done

  remove_worktree "$wt_bl"
  baseline_median=$(median_of_three "${bl_lats[@]}")
  echo "  [Baseline] Median: $baseline_median ms"

  # ── Patched: 3 runs on clean worktree + patch ──────────────────
  patched_median="0"
  p_lats=()
  if [[ -n "$patch" && -f "$patch" ]]; then
    echo ""
    echo "  [Patched] Creating clean worktree..."
    wt_p="$work/wt_patched"
    create_clean_worktree "$wt_p"

    echo "  [Patched] Applying patch..."
    apply_out=$(git -C "$wt_p" apply --whitespace=nowarn "$patch" 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "  ERROR: git apply failed: $apply_out"
      remove_worktree "$wt_p"
      echo "$kernel,${bl_lats[0]},${bl_lats[1]},${bl_lats[2]},$baseline_median,FAIL,FAIL,FAIL,FAIL,FAIL,$reported" >> "$RESULTS_FILE"
      echo ""
      continue
    fi

    # Verify patch was applied (show changed files)
    changed=$(git -C "$wt_p" diff --stat HEAD 2>/dev/null | tail -1)
    echo "  [Patched] Changes: $changed"

    for run in $(seq 1 $RUNS); do
      out="$work/patched_run${run}.txt"
      echo -n "  [Patched] Run $run/$RUNS... "
      run_benchmark "$harness" "$wt_p" "$out"
      lat=$(extract_latency "$out")
      echo "${lat:-FAILED} ms"
      p_lats+=("${lat:-0}")
    done

    remove_worktree "$wt_p"
    patched_median=$(median_of_three "${p_lats[@]}")
    echo "  [Patched] Median: $patched_median ms"
  else
    echo "  SKIP patched: no patch file"
    p_lats=("0" "0" "0")
  fi

  # ── Compute speedup ────────────────────────────────────────────
  if python3 -c "exit(0 if float('$baseline_median') > 0 and float('$patched_median') > 0 else 1)" 2>/dev/null; then
    speedup=$(python3 -c "print(round(float('$baseline_median') / float('$patched_median'), 4))")
    echo ""
    echo "  >>> SPEEDUP: ${speedup}x (reported: ${reported}x)"
  else
    speedup="FAIL"
    echo "  >>> Could not compute speedup"
  fi

  echo "$kernel,${bl_lats[0]},${bl_lats[1]},${bl_lats[2]},$baseline_median,${p_lats[0]:-0},${p_lats[1]:-0},${p_lats[2]:-0},$patched_median,$speedup,$reported" >> "$RESULTS_FILE"

  rm -rf "$work"
  echo ""
done

export PYTHONPATH="$ORIG_PYTHONPATH"

echo ""
echo "================================================================"
echo "  FINAL RESULTS"
echo "================================================================"
printf "%-25s %14s %14s %10s %10s\n" "Kernel" "Baseline (ms)" "Patched (ms)" "Verified" "Reported"
echo "------------------------------------------------------------------------"
tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r k br1 br2 br3 bm pr1 pr2 pr3 pm spd rep; do
  printf "%-25s %14s %14s %10s %10s\n" "$k" "$bm" "$pm" "${spd}x" "${rep}x"
done
echo ""
echo "Raw data: $RESULTS_FILE"
