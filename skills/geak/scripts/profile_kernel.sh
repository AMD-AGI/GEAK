#!/usr/bin/env bash
# GEAK Kernel Profiling Wrapper
# Usage: profile_kernel.sh <harness_path> <output_dir> [gpu_id]
#
# Profiles a kernel using rocprof-compute (formerly omniperf) with warmup runs.
# Generates both raw profiling data and analyzed reports.

set -euo pipefail

HARNESS_PATH="${1:?Usage: profile_kernel.sh <harness_path> <output_dir> [gpu_id]}"
OUTPUT_DIR="${2:?Usage: profile_kernel.sh <harness_path> <output_dir> [gpu_id]}"
GPU_ID="${3:-0}"
NUM_WARMUP="${4:-3}"

mkdir -p "$OUTPUT_DIR"

echo "=== GEAK Kernel Profiler ==="
echo "Harness: $HARNESS_PATH"
echo "Output:  $OUTPUT_DIR"
echo "GPU ID:  $GPU_ID"

# Step 1: Warmup runs (stabilize GPU clocks and caches)
echo ""
echo "--- Step 1: Warmup ($NUM_WARMUP runs) ---"
for i in $(seq 1 "$NUM_WARMUP"); do
    echo "  Warmup run $i/$NUM_WARMUP..."
    python3 "$HARNESS_PATH" --benchmark > /dev/null 2>&1 || true
done

# Step 2: Profile with rocprof-compute
echo ""
echo "--- Step 2: Profiling with rocprof-compute ---"

PROFILE_CMD="python3 $HARNESS_PATH --profile"

# Check if rocprof-compute is available
if command -v rocprof-compute &> /dev/null; then
    PROFILER="rocprof-compute"
elif command -v omniperf &> /dev/null; then
    PROFILER="omniperf"
else
    echo "WARNING: Neither rocprof-compute nor omniperf found."
    echo "Falling back to basic timing-only profiling."

    # Basic timing profile using rocprof
    if command -v rocprof &> /dev/null; then
        echo "Using rocprof for basic profiling..."
        rocprof --stats python3 "$HARNESS_PATH" --profile 2>&1 | tee "$OUTPUT_DIR/rocprof_output.txt"
    else
        echo "No profiling tools available. Running benchmark only."
        python3 "$HARNESS_PATH" --benchmark 2>&1 | tee "$OUTPUT_DIR/benchmark_output.txt"
    fi
    exit 0
fi

echo "Using profiler: $PROFILER"

# Run profiling (no roofline to save time)
$PROFILER profile --no-roof -- $PROFILE_CMD 2>&1 | tee "$OUTPUT_DIR/profile_raw.txt"

# Find the workload directory (most recent)
WORKLOAD_DIR=$(ls -td /tmp/workloads/*/ 2>/dev/null | head -1)

if [ -z "$WORKLOAD_DIR" ]; then
    echo "WARNING: No workload directory found in /tmp/workloads/"
    echo "Profiling may have failed. Check $OUTPUT_DIR/profile_raw.txt"
    exit 1
fi

echo "Workload directory: $WORKLOAD_DIR"

# Step 3: Analyze
echo ""
echo "--- Step 3: Analyzing profiling data ---"
$PROFILER analyze -p "$WORKLOAD_DIR" -o "$OUTPUT_DIR/analysis/" 2>&1 | tee "$OUTPUT_DIR/analysis_output.txt"

# Step 4: Extract key metrics
echo ""
echo "--- Step 4: Extracting key metrics ---"

# Copy the full analysis report
if [ -f "$OUTPUT_DIR/analysis/log.txt" ]; then
    cp "$OUTPUT_DIR/analysis/log.txt" "$OUTPUT_DIR/profile_report.txt"
    echo "Full report saved to: $OUTPUT_DIR/profile_report.txt"
fi

# Try to generate a summary
if [ -f "$OUTPUT_DIR/profile_report.txt" ]; then
    echo ""
    echo "--- Profile Report Summary ---"
    # Extract key sections
    grep -A 20 "System Speed-of-Light" "$OUTPUT_DIR/profile_report.txt" 2>/dev/null | head -25 || true
    echo "..."
    grep -A 10 "Wavefront" "$OUTPUT_DIR/profile_report.txt" 2>/dev/null | head -15 || true
fi

echo ""
echo "=== Profiling Complete ==="
echo "Output directory: $OUTPUT_DIR"
echo "Full report: $OUTPUT_DIR/profile_report.txt"
echo "Raw output: $OUTPUT_DIR/profile_raw.txt"
