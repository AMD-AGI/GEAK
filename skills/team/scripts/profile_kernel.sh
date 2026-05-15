#!/bin/bash
# profile_kernel.sh — Profile a kernel with exclusive GPU access.
# Uses gpu_lock.sh to ensure the profiler has uncontested GPU access.
#
# Usage: profile_kernel.sh <harness_command> <output_dir> <gpu_id> [num_warmup]
# Example: profile_kernel.sh "python3 scripts/task_runner.py performance" ./profile_out 0 3

set -euo pipefail

HARNESS_CMD="$1"
OUTPUT_DIR="$2"
GPU_ID="${3:-0}"
NUM_WARMUP="${4:-3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_LOCK="$SCRIPT_DIR/gpu_lock.sh"

mkdir -p "$OUTPUT_DIR"

echo "[profile] Warming up ($NUM_WARMUP runs)..."
for i in $(seq 1 "$NUM_WARMUP"); do
    bash "$GPU_LOCK" "$GPU_ID" bash -c "$HARNESS_CMD" > /dev/null 2>&1 || true
done

echo "[profile] Profiling..."

PROFILER=""
if command -v rocprof-compute &> /dev/null; then
    PROFILER="rocprof-compute"
elif command -v omniperf &> /dev/null; then
    PROFILER="omniperf"
fi

if [ -n "$PROFILER" ]; then
    WORKLOAD_BASE="/tmp/workloads"
    rm -rf "$WORKLOAD_BASE" 2>/dev/null || true

    bash "$GPU_LOCK" "$GPU_ID" \
        "$PROFILER" profile --no-roof -- $HARNESS_CMD \
        > "$OUTPUT_DIR/profile_raw.txt" 2>&1 || true

    WORKLOAD_DIR=$(ls -td "$WORKLOAD_BASE"/*/ 2>/dev/null | head -1)
    if [ -n "$WORKLOAD_DIR" ]; then
        echo "[profile] Analyzing..."
        "$PROFILER" analyze -p "$WORKLOAD_DIR" -o "$OUTPUT_DIR/analysis/" \
            > "$OUTPUT_DIR/analysis_output.txt" 2>&1 || true

        if [ -f "$OUTPUT_DIR/analysis/log.txt" ]; then
            cp "$OUTPUT_DIR/analysis/log.txt" "$OUTPUT_DIR/profile_report.txt"
        fi
    else
        echo "[profile] WARNING: No workload directory found in $WORKLOAD_BASE" >&2
    fi
elif command -v rocprof &> /dev/null; then
    echo "[profile] Falling back to rocprof --stats"
    bash "$GPU_LOCK" "$GPU_ID" \
        rocprof --stats $HARNESS_CMD \
        > "$OUTPUT_DIR/profile_raw.txt" 2>&1 || true
else
    echo "[profile] WARNING: No profiling tool found. Running benchmark only." >&2
    bash "$GPU_LOCK" "$GPU_ID" bash -c "$HARNESS_CMD" \
        > "$OUTPUT_DIR/benchmark_only.txt" 2>&1
fi

echo "[profile] Done. Output in $OUTPUT_DIR/"
