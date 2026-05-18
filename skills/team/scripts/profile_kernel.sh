#!/bin/bash
# Kernel profiling wrapper with warmup and fallback chain
# Usage: bash profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir>
# Profiles the kernel using rocprof-compute (preferred) with fallbacks

set -euo pipefail

GPU_ID="${1:?Usage: profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir>}"
BENCHMARK_CMD="${2:?Missing benchmark command}"
OUTPUT_DIR="${3:?Missing output directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LOCK="$SCRIPT_DIR/gpu_lock.sh"

mkdir -p "$OUTPUT_DIR"

echo "=== Profiling Setup ==="
echo "GPU: $GPU_ID"
echo "Command: $BENCHMARK_CMD"
echo "Output: $OUTPUT_DIR"

# Step 1: Warmup runs to stabilize GPU clocks
echo ""
echo "=== Warmup (3 runs) ==="
for i in 1 2 3; do
    echo "Warmup run $i/3..."
    bash "$GPU_LOCK" "$GPU_ID" bash -c "$BENCHMARK_CMD" > /dev/null 2>&1 || true
done

# Step 2: Try profilers in order of preference
PROFILER=""
PROFILE_SUCCESS=false

# Try rocprof-compute (formerly omniperf)
if command -v rocprof-compute &> /dev/null; then
    PROFILER="rocprof-compute"
elif command -v omniperf &> /dev/null; then
    PROFILER="omniperf"
fi

if [ -n "$PROFILER" ]; then
    echo ""
    echo "=== Profiling with $PROFILER ==="

    WORKLOAD_DIR="/tmp/team_v2_workloads/profile_$(date +%s)"
    mkdir -p "$(dirname "$WORKLOAD_DIR")"

    # Profile (with GPU lock)
    bash "$GPU_LOCK" "$GPU_ID" \
        $PROFILER profile --no-roof -n "$WORKLOAD_DIR" -- bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/profile_raw.log" 2>&1

    if [ $? -eq 0 ] && [ -d "$WORKLOAD_DIR" ]; then
        echo "Profile data collected at $WORKLOAD_DIR"

        # Analyze
        echo ""
        echo "=== Analyzing profile data ==="
        $PROFILER analyze -p "$WORKLOAD_DIR" \
            > "$OUTPUT_DIR/profile_report.txt" 2>&1 || true

        if [ -f "$OUTPUT_DIR/profile_report.txt" ] && [ -s "$OUTPUT_DIR/profile_report.txt" ]; then
            PROFILE_SUCCESS=true
            echo "Profile report saved to $OUTPUT_DIR/profile_report.txt"

            # Extract key sections
            echo ""
            echo "=== Key Metrics ==="
            grep -A 50 "System Speed-of-Light" "$OUTPUT_DIR/profile_report.txt" 2>/dev/null | head -60 || true
            echo "---"
            grep -A 30 "Wavefront" "$OUTPUT_DIR/profile_report.txt" 2>/dev/null | head -40 || true
        fi

        # Cleanup workload directory
        rm -rf "$WORKLOAD_DIR" 2>/dev/null || true
    fi
fi

# Fallback: rocprof --stats
if [ "$PROFILE_SUCCESS" = false ] && command -v rocprof &> /dev/null; then
    echo ""
    echo "=== Fallback: rocprof --stats ==="
    PROFILER="rocprof"

    bash "$GPU_LOCK" "$GPU_ID" \
        rocprof --stats bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/profile_report.txt" 2>&1 || true

    if [ -f "$OUTPUT_DIR/profile_report.txt" ] && [ -s "$OUTPUT_DIR/profile_report.txt" ]; then
        PROFILE_SUCCESS=true
        echo "rocprof stats saved"
    fi
fi

# Final fallback: benchmark-only
if [ "$PROFILE_SUCCESS" = false ]; then
    echo ""
    echo "=== Fallback: benchmark-only (no profiler available) ==="
    PROFILER="benchmark-only"

    bash "$GPU_LOCK" "$GPU_ID" bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/profile_report.txt" 2>&1

    echo "Benchmark output saved (no profiler data)"
fi

echo ""
echo "=== Profiling Complete ==="
echo "Profiler used: $PROFILER"
echo "Report: $OUTPUT_DIR/profile_report.txt"
echo "Success: $PROFILE_SUCCESS"
