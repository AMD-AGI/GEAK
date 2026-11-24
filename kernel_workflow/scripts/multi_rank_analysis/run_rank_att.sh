#!/bin/bash
# Per-rank ATT (SQTT thread-trace) wrapper. Insert between `torchrun --no-python` and the real
# command; only the rank named by GEAK_ATT_RANK is traced, every other rank runs untouched.
#
# The resulting `ui_output_agent_*_dispatch_*/` directory loads directly in rocprof-compute-viewer
# (Menu -> Import -> "Rocprofv3 UI Output", or pass the directory as argv[1]).
#
# Required:
#   GEAK_ATT_OUTPUT_DIR     where to write
#   GEAK_ATT_KERNEL_REGEX   MUST match exactly ONE kernel. A regex matching two kernels writes both
#                           into the same directory and their code-object dumps collide.
# Optional:
#   GEAK_ATT_RANK           traced rank (default 0)
#   GEAK_ATT_OUTPUT_NAME    filename stem (default "att")
#   GEAK_ATT_GPU_INDEX      (default 0)
#   GEAK_ATT_TARGET_CU      (default 1)
#   GEAK_ATT_SIMD_SELECT    (default 0xF)
#   GEAK_ATT_LIBRARY_PATH   decoder search path. Bundled with rocprofv3 from ROCm 7.13; on older
#                           ROCm point this at the decoder or you get raw .att and no ui_output.
#   GEAK_ATT_PERFCOUNTERS   space-separated SQ counters, e.g.
#                             "SQ_WAIT_ANY SQ_INSTS_VALU SQ_BUSY_CYCLES SQ_WAVE_CYCLES"
#                           Max 8, SQ domain only, 4 recommended. Per-counter SIMD mask via
#                           `NAME:0xMask`. WITHOUT this, filenames.json carries
#                           "counter_names":[] and the viewer's counter panes are EMPTY -- you get
#                           the instruction timeline but nothing to correlate it against.
#   GEAK_ATT_PERFCOUNTER_CTRL  collection period, integer 1..32 (default 3 when counters are set).
#   GEAK_ATT_DEBUG_INFO=1   build the FlyDSL kernels WITH debug info so the viewer can map ISA back
#                           to the Python kernel source. Not free -- see below.
#   GEAK_ATT_JIT_DIR        writable JIT cache, REQUIRED when GEAK_ATT_DEBUG_INFO=1.
#
# ---------------------------------------------------------------------------------------------
# On GEAK_ATT_DEBUG_INFO: without it the trace has NO source correlation. Measured on a real
# capture: the `Source` column was populated on 0 of 7631 rows, no snapshots.json was emitted, and
# the code-object ELF carried no .debug_* sections. The viewer still shows ISA plus
# hitcount/latency/stall, but you cannot tell which line of the kernel a stall belongs to.
#
# Setting it to 1 exports FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1, which turns on three things at once:
# per-statement source locations pointing at the real .py file (ast_rewriter), a LineTablesOnly DI
# pass, and `-g` to the ROCm backend.
#
# Two consequences to plan for:
#   1. It needs a REBUILD. A shared prebuilt AITER_JIT_DIR holds no-debug objects and would be
#      reused as-is, silently giving you a no-debug trace. Hence GEAK_ATT_JIT_DIR is mandatory in
#      this mode and must not be the shared cache.
#   2. Its TIMINGS ARE NOT COMPARABLE to a no-debug build. LineTablesOnly should not change
#      instruction selection, but that is not guaranteed. Use a debug capture for ATTRIBUTION
#      ("which line waits"), and a separate no-debug capture for any number you report.
# ---------------------------------------------------------------------------------------------
set -euo pipefail

target_rank="${GEAK_ATT_RANK:-0}"
local_rank="${LOCAL_RANK:-${RANK:-}}"
if [ -z "$local_rank" ]; then
    echo "GEAK ATT wrapper requires LOCAL_RANK or RANK" >&2
    exit 2
fi

if [ "$local_rank" != "$target_rank" ]; then
    exec "$@"
fi

: "${GEAK_ATT_OUTPUT_DIR:?GEAK_ATT_OUTPUT_DIR is required}"
: "${GEAK_ATT_KERNEL_REGEX:?GEAK_ATT_KERNEL_REGEX is required}"

decoder_args=()
if [ -n "${GEAK_ATT_LIBRARY_PATH:-}" ]; then
    decoder_args=(--att-library-path "$GEAK_ATT_LIBRARY_PATH")
fi

# Perf counters. rocprofv3 takes the whole list as ONE argument, so keep it quoted as a unit.
counter_args=()
if [ -n "${GEAK_ATT_PERFCOUNTERS:-}" ]; then
    n_counters=$(wc -w <<<"$GEAK_ATT_PERFCOUNTERS")
    if [ "$n_counters" -gt 8 ]; then
        echo "GEAK ATT: $n_counters counters requested; the hardware limit is 8 (4 recommended)" >&2
        exit 2
    fi
    counter_args=(
        --att-perfcounter-ctrl "${GEAK_ATT_PERFCOUNTER_CTRL:-3}"
        --att-perfcounters "$GEAK_ATT_PERFCOUNTERS"
    )
elif [ -n "${GEAK_ATT_PERFCOUNTER_CTRL:-}" ]; then
    echo "GEAK ATT: GEAK_ATT_PERFCOUNTER_CTRL set without GEAK_ATT_PERFCOUNTERS -- ignoring" >&2
fi

# Debug build, for ISA <-> source correlation in the viewer.
if [ "${GEAK_ATT_DEBUG_INFO:-0}" = "1" ]; then
    if [ -z "${GEAK_ATT_JIT_DIR:-}" ]; then
        echo "GEAK ATT: GEAK_ATT_DEBUG_INFO=1 requires GEAK_ATT_JIT_DIR (a WRITABLE cache). A shared" >&2
        echo "          prebuilt cache holds no-debug objects and would be reused, silently" >&2
        echo "          defeating this flag and yielding a trace with no source correlation." >&2
        exit 2
    fi
    if [ -n "${AITER_JIT_DIR:-}" ] && [ "$GEAK_ATT_JIT_DIR" = "$AITER_JIT_DIR" ]; then
        echo "GEAK ATT: GEAK_ATT_JIT_DIR must differ from the inherited AITER_JIT_DIR" >&2
        exit 2
    fi
    mkdir -p "$GEAK_ATT_JIT_DIR"
    export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
    export AITER_JIT_DIR="$GEAK_ATT_JIT_DIR"
    echo "GEAK ATT: debug-info build ON (JIT dir $GEAK_ATT_JIT_DIR). Expect a full rebuild." >&2
    echo "GEAK ATT: timings from this capture are for ATTRIBUTION ONLY -- do not report them." >&2
fi

exec rocprofv3 \
    --att \
    "${decoder_args[@]}" \
    "${counter_args[@]}" \
    --att-gpu-index "${GEAK_ATT_GPU_INDEX:-0}" \
    --att-target-cu "${GEAK_ATT_TARGET_CU:-1}" \
    --att-simd-select "${GEAK_ATT_SIMD_SELECT:-0xF}" \
    --kernel-include-regex "$GEAK_ATT_KERNEL_REGEX" \
    --output-directory "$GEAK_ATT_OUTPUT_DIR" \
    --output-file "${GEAK_ATT_OUTPUT_NAME:-att}" \
    --output-format csv \
    -- "$@"
