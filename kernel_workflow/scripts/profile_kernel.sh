#!/bin/bash
# Thin profiling wrapper: warmup + gpu_lock + detect best available profiler + run it + dump RAW output.
#
# It deliberately does NOT parse or interpret the profiler output. The profile_engineer reads the raw
# artifacts written here and classifies the bottleneck itself, following knowledge/profiling_guide.md
# (which documents how to extract the key metrics + dispatch counts from EACH profiler's format and how
# to degrade gracefully when a field is absent). Keeping the parsing out of this script is what makes it
# portable: it never greps for profiler-/version-specific section names ("System Speed-of-Light",
# "Wavefront", …) or assumes a particular CSV layout, so it keeps working when the toolchain changes.
#
# Usage: bash profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir>
#
# Optional env overrides (all have sensible defaults; nothing kernel-specific is hard-coded):
#   PROFILER_PRIORITY  space-separated profiler order to try
#                      (default: "rocprof-compute omniperf rocprofv3 rocprof")
#   WARMUP_RUNS        number of warmup runs before profiling (default: 3)
#   RPC_PROFILE_ARGS   extra args passed to rocprof-compute/omniperf `profile` (default: "--no-roof")
#   RPV3_TRACE_ARGS    args passed to rocprofv3 (default: "--kernel-trace --stats --output-format csv")
#   RPROF_ARGS         args passed to legacy rocprof (default: "--stats")
#   METRIX_ARGS        args passed to metrix (default: ""; override per `metrix --help` for this toolchain)
#
# Fault tolerance: a profiler that fails (e.g. a flag was renamed across toolchain versions) no longer
# degrades silently. The failure + a self-heal pointer (which env var to override, where the recipe is)
# is written into profile_report.txt so the engineer can re-run with a corrected arg. See the
# "Profiler failed? — fault-tolerance ladder" section of knowledge/profiling_guide.md.
#
# Output: everything lands under <output_dir>. The single entry point for the profile_engineer is
#   <output_dir>/profile_report.txt   (raw, human/agent-readable; the chosen profiler's full output)
# plus any profiler-native artifacts (e.g. rocprofv3 CSVs) left in <output_dir>/ for deeper parsing.

set -euo pipefail

GPU_ID="${1:?Usage: profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir>}"
BENCHMARK_CMD="${2:?Missing benchmark command}"
OUTPUT_DIR="${3:?Missing output directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LOCK="$SCRIPT_DIR/gpu_lock.sh"

WARMUP_RUNS="${WARMUP_RUNS:-3}"
PROFILER_PRIORITY="${PROFILER_PRIORITY:-rocprof-compute omniperf rocprofv3 rocprof metrix}"
RPC_PROFILE_ARGS="${RPC_PROFILE_ARGS:---no-roof}"
RPV3_TRACE_ARGS="${RPV3_TRACE_ARGS:---kernel-trace --stats --output-format csv}"
RPROF_ARGS="${RPROF_ARGS:---stats}"
METRIX_ARGS="${METRIX_ARGS:-}"

mkdir -p "$OUTPUT_DIR"
REPORT="$OUTPUT_DIR/profile_report.txt"
: > "$REPORT"

# Compile for the local GPU arch only so profiling does not trigger a fresh ~9-arch global rebuild
# (the BENCHMARK_CMD is expected to `cd` into the workspace whose isolated .torch_ext already holds the
# built .so). Generic; honors a caller-set PYTORCH_ROCM_ARCH, and KERNEL_ENV_KEEP_ARCH=1 opts out.
if [ "${KERNEL_ENV_KEEP_ARCH:-0}" != "1" ] && [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
    _ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
    [ -n "${_ARCH:-}" ] && export PYTORCH_ROCM_ARCH="$_ARCH"
fi

echo "=== Profiling setup ==="
echo "GPU: $GPU_ID"
echo "Command: $BENCHMARK_CMD"
echo "Output: $OUTPUT_DIR"
echo "Profiler priority: $PROFILER_PRIORITY"

# Step 1: Warmup to stabilize GPU clocks (all GPU work goes through gpu_lock).
echo ""
echo "=== Warmup ($WARMUP_RUNS runs) ==="
for i in $(seq 1 "$WARMUP_RUNS"); do
    echo "Warmup run $i/$WARMUP_RUNS..."
    bash "$GPU_LOCK" "$GPU_ID" bash -c "$BENCHMARK_CMD" > /dev/null 2>&1 || true
done

# Step 2: Pick the first available profiler from the priority list.
PROFILER=""
for p in $PROFILER_PRIORITY; do
    if command -v "$p" &> /dev/null; then PROFILER="$p"; break; fi
done

PROFILE_SUCCESS=false

# Surface a profiler failure (instead of silently degrading) + tell the engineer how to self-heal:
# which env var to override and where the per-profiler recovery recipe lives.
emit_profiler_failure() {  # <tool> <exit_code> <override_env_var> <raw_log>
    local tool="$1" code="$2" envvar="$3" log="$4"
    {
        echo ""
        echo "!!! PROFILER FAILED: $tool exited $code — its output may be unusable; degrading."
        echo ">>> Most likely a CLI/version mismatch (a flag was renamed or removed in this toolchain)."
        echo ">>> Self-heal: run \`$tool --help\` to find the current flag, then re-run this script with"
        echo ">>>   an override, e.g.   $envvar=\"<corrected args>\" bash profile_kernel.sh <gpu> <cmd> <out>"
        echo ">>> Recipe: knowledge/profiling_guide.md  →  \"Profiler failed? — fault-tolerance ladder\"  →  $tool"
        if [ -n "$log" ] && [ -s "$log" ]; then
            echo ">>> Last error lines from $(basename "$log"):"
            tail -n 15 "$log" 2>/dev/null | sed 's/^/    /'
        fi
        echo ""
    } >> "$REPORT"
}

run_rocprof_compute() {  # rocprof-compute / omniperf: profile -> analyze, dump the FULL analyze text.
    local tool="$1"
    local workload="$OUTPUT_DIR/${tool}_workload"
    # NO `rm` (prompts + blocks autonomous runs): move any stale profiler dir aside, then make fresh.
    [ -e "$workload" ] && mv "$workload" "${workload}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$workload"
    echo "=== Profiling with $tool (profile $RPC_PROFILE_ARGS) ==="
    local rc=0
    bash "$GPU_LOCK" "$GPU_ID" \
        "$tool" profile $RPC_PROFILE_ARGS -n "$workload" -- bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/${tool}_profile_raw.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then emit_profiler_failure "$tool" "$rc" RPC_PROFILE_ARGS "$OUTPUT_DIR/${tool}_profile_raw.log"; fi
    if [ -d "$workload" ]; then
        echo "=== $tool analyze (full, unparsed) ===" >> "$REPORT"
        bash "$GPU_LOCK" "$GPU_ID" "$tool" analyze -p "$workload" >> "$REPORT" 2>&1 || true
    fi
    [ -s "$REPORT" ] && PROFILE_SUCCESS=true
}

run_rocprofv3() {        # modern profiler: kernel trace + stats CSVs (per-kernel dispatch counts + durations).
    local dir="$OUTPUT_DIR/rocprofv3"
    # NO `rm` (prompts + blocks autonomous runs): move any stale dir aside, then make fresh.
    [ -e "$dir" ] && mv "$dir" "${dir}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$dir"
    echo "=== Profiling with rocprofv3 ($RPV3_TRACE_ARGS) ==="
    local rc=0
    bash "$GPU_LOCK" "$GPU_ID" \
        rocprofv3 $RPV3_TRACE_ARGS -d "$dir" -- bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/rocprofv3_run.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then emit_profiler_failure rocprofv3 "$rc" RPV3_TRACE_ARGS "$OUTPUT_DIR/rocprofv3_run.log"; fi
    # Surface every artifact rocprofv3 produced into the report (generic: no fixed filename glob).
    { cat "$OUTPUT_DIR/rocprofv3_run.log"; echo ""; } >> "$REPORT" 2>/dev/null || true
    while IFS= read -r f; do
        { echo ""; echo "=== rocprofv3 artifact: $f ==="; cat "$f"; } >> "$REPORT" 2>/dev/null || true
    done < <(find "$dir" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.txt' \) 2>/dev/null | sort)
    [ -s "$REPORT" ] && PROFILE_SUCCESS=true
}

run_rocprof() {          # legacy: rocprof --stats (HIP dispatch stats).
    echo "=== Profiling with rocprof ($RPROF_ARGS) ==="
    local rc=0
    bash "$GPU_LOCK" "$GPU_ID" rocprof $RPROF_ARGS bash -c "$BENCHMARK_CMD" >> "$REPORT" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then emit_profiler_failure rocprof "$rc" RPROF_ARGS "$REPORT"; fi
    [ -s "$REPORT" ] && PROFILE_SUCCESS=true
}

run_metrix() {           # generic/extensible profiler: env-driven (METRIX_ARGS), harvest any csv/json/txt.
    local dir="$OUTPUT_DIR/metrix"
    # NO `rm` (prompts + blocks autonomous runs): move any stale dir aside, then make fresh.
    [ -e "$dir" ] && mv "$dir" "${dir}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$dir"
    echo "=== Profiling with metrix ($METRIX_ARGS) ==="
    local rc=0
    # No hardcoded flags: pass METRIX_ARGS through and hint the output dir via env (ignored if unused).
    bash "$GPU_LOCK" "$GPU_ID" env METRIX_OUTPUT_DIR="$dir" \
        metrix $METRIX_ARGS bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/metrix_run.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then emit_profiler_failure metrix "$rc" METRIX_ARGS "$OUTPUT_DIR/metrix_run.log"; fi
    { cat "$OUTPUT_DIR/metrix_run.log"; echo ""; } >> "$REPORT" 2>/dev/null || true
    while IFS= read -r f; do
        { echo ""; echo "=== metrix artifact: $f ==="; cat "$f"; } >> "$REPORT" 2>/dev/null || true
    done < <(find "$dir" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.txt' \) 2>/dev/null | sort)
    [ -s "$REPORT" ] && PROFILE_SUCCESS=true
}

case "$PROFILER" in
    rocprof-compute|omniperf) run_rocprof_compute "$PROFILER" ;;
    rocprofv3)                run_rocprofv3 ;;
    rocprof)                  run_rocprof ;;
    metrix)                   run_metrix ;;
    "")                       echo "No profiler found in priority list; benchmark-only." ;;
esac

# Final fallback: no profiler available, or the chosen one produced nothing -> benchmark-only.
if [ "$PROFILE_SUCCESS" = false ]; then
    echo ""
    echo "=== Fallback: benchmark-only (no usable profiler output) ==="
    PROFILER="benchmark-only"
    bash "$GPU_LOCK" "$GPU_ID" bash -c "$BENCHMARK_CMD" >> "$REPORT" 2>&1 || true
fi

echo ""
echo "=== Profiling complete ==="
echo "Profiler used: ${PROFILER:-benchmark-only}"
echo "Report: $REPORT"
echo "Artifacts:"
find "$OUTPUT_DIR" -maxdepth 2 -type f 2>/dev/null | sort | sed 's/^/  /' || true
