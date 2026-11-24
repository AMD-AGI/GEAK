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
# Usage: bash profile_kernel.sh <gpu_spec> <benchmark_cmd> <output_dir>   (gpu_spec: any gpu_lock.sh form)
#
# Optional env overrides (all have sensible defaults; nothing kernel-specific is hard-coded):
#   PROFILER_PRIORITY  space-separated profiler order to try (default is arch-aware:
#                      gfx1201/gfx1100/gfx115x -> rocprofv3 first; CDNA/other -> rocprof-compute first)
#   WARMUP_RUNS        number of warmup runs before profiling (default: 3)
#   RPC_PROFILE_ARGS   extra args passed to rocprof-compute/omniperf `profile` (default: "--no-roof")
#   RPV3_TRACE_ARGS    args passed to rocprofv3 (default: "--kernel-trace --stats --output-format csv",
#                      plus "--disable-signal-handlers true" for multi-GPU specs)
#   RPROF_ARGS         args passed to legacy rocprof (default: "--stats")
#   METRIX_ARGS        args passed to metrix (default: ""; override per `metrix --help` for this toolchain)
#   KERNEL_TOOLS_PROFILE  1 = shared kernel_tools extras (rocprof-compute output flag by version, then a
#                      rocprofv3 counter pass when only the trace survives); default 0 = unchanged.
#                      kernel_workflow offers it through the role prompts only when use_expert_skills is on.
#   ROCPROF_SIGNAL_HANDLERS  rocprofv3's own signal handlers in the profiled process (default: 0, off)
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

GPU_ID="${1:?Usage: profile_kernel.sh <gpu_spec> <benchmark_cmd> <output_dir>}"
BENCHMARK_CMD="${2:?Missing benchmark command}"
OUTPUT_DIR="${3:?Missing output directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Multinode: the driver sees no GPU, and this wrapper picks the arch and the profiler from the
# machine it runs on. So all of it goes where gpu_lock.sh would send the command ("<engine>:<spec>"
# to that engine's machine, a bare spec to the master; group: and pool: are specs, not engines),
# as the run directory's copy, whose gpu_lock.sh is its sibling there. On a node (GEAK_MN_HOP) it
# runs as is.
_MN_PY="$SCRIPT_DIR/../../e2e_workflow/scripts/multi_node/mn.py"
if [ -z "${GEAK_MN_HOP:-}" ] && [ -f "$_MN_PY" ] && [ -n "${GEAK_MN_STATE:-}" ]; then
    _MN_PIN="" _MN_LEASE_DIR="" _MN_TOOLS=""
    IFS=$'\t' read -r _MN_PIN _MN_LEASE_DIR _MN_TOOLS < <(python3 "$_MN_PY" active-pin --lease-dir 2>/dev/null) || true
    if [ -n "$_MN_PIN" ]; then
        if [ -z "$_MN_TOOLS" ]; then
            echo "ERROR: profile_kernel.sh: the multinode pin $_MN_PIN has no tools_dir; re-run mn.py check with a work_dir." >&2
            exit 2
        fi
        case "$GPU_ID" in
            group:*|pool:*) _MN_TARGET=master ;;
            *:*) _MN_TARGET="${GPU_ID%%:*}" GPU_ID="${GPU_ID#*:}" ;;
            *)   _MN_TARGET=master ;;
        esac
        exec env GEAK_MN_STATE="$_MN_PIN" GEAK_MN_LEASE_DIR="$_MN_LEASE_DIR" python3 "$_MN_PY" exec --env \
            --target "$_MN_TARGET" -- bash "$_MN_TOOLS/profile_kernel.sh" "$GPU_ID" "$BENCHMARK_CMD" "$OUTPUT_DIR"
    fi
fi
GPU_LOCK="$SCRIPT_DIR/gpu_lock.sh"
source "$SCRIPT_DIR/profile_policy.sh"

WARMUP_RUNS="${WARMUP_RUNS:-3}"
# Arch detection masks rocminfo to the requested GPUs, so it needs plain ids, not a group spec.
case "$GPU_ID" in
    group:*) _ARCH_IDS="${GPU_ID#group:}" ;;
    pool:*:*) _ARCH_IDS="${GPU_ID#pool:*:}" ;;
    *) _ARCH_IDS="$GPU_ID" ;;
esac
DETECTED_ARCH="$(GPU_ID="$_ARCH_IDS" profile_detect_arch)"
PROFILER_PRIORITY="${PROFILER_PRIORITY:-$(profiler_priority_for_arch "$DETECTED_ARCH")}"
RPC_PROFILE_ARGS="${RPC_PROFILE_ARGS:---no-roof}"
RPV3_TRACE_ARGS_DEFAULT="--kernel-trace --stats --output-format csv"
# Multi-GPU jobs are multi-rank: rocprofv3's own SIGABRT handler can deadlock a launcher whose
# teardown aborts, so leave signal handling to the application.
case "$GPU_ID" in
    group:*|pool:*:*) RPV3_TRACE_ARGS_DEFAULT="$RPV3_TRACE_ARGS_DEFAULT --disable-signal-handlers true" ;;
esac
RPV3_TRACE_ARGS="${RPV3_TRACE_ARGS:-$RPV3_TRACE_ARGS_DEFAULT}"
# The same deadlock hits a single process that aborts (e.g. while it loads), and the tools that run
# rocprofv3 underneath (rocprof-compute, metrix) take no such flag. The flag only sets this variable
# for the profiled process, and a toolchain that does not read it ignores it.
export ROCPROF_SIGNAL_HANDLERS="${ROCPROF_SIGNAL_HANDLERS:-0}"
RPROF_ARGS="${RPROF_ARGS:---stats}"
METRIX_ARGS="${METRIX_ARGS:-}"

mkdir -p "$OUTPUT_DIR"
REPORT="$OUTPUT_DIR/profile_report.txt"
: > "$REPORT"

# Compile for the local GPU arch only so profiling does not trigger a fresh ~9-arch global rebuild
# (the BENCHMARK_CMD is expected to `cd` into the workspace whose isolated .torch_ext already holds the
# built .so). Generic; honors a caller-set PYTORCH_ROCM_ARCH, and KERNEL_ENV_KEEP_ARCH=1 opts out.
if [ "${KERNEL_ENV_KEEP_ARCH:-0}" != "1" ] && [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
    [ -n "$DETECTED_ARCH" ] && export PYTORCH_ROCM_ARCH="$DETECTED_ARCH"
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

# Step 2: Try each installed profiler until one produces real artifacts (csv/json/txt
# from the tool, not just a failure log stuffed into profile_report.txt).
PROFILE_SUCCESS=false
PROFILER=""

# Surface a profiler failure (instead of silently degrading) + tell the engineer how to self-heal:
# which env var to override and where the per-profiler recovery recipe lives.
emit_profiler_failure() {  # <tool> <exit_code> <override_env_var> <raw_log>
    local tool="$1" code="$2" envvar="$3" log="$4"
    {
        echo ""
        echo "!!! PROFILER FAILED: $tool exited $code — its output may be unusable; trying next tool."
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

profiler_artifacts_ok() {  # true when the tool left a non-empty csv/json (not just the report)
    local dir="${1:-$OUTPUT_DIR}"
    find "$dir" -type f \( -name '*.csv' -o -name '*.json' \) -size +0 2>/dev/null | grep -q .
}

run_rocprof_compute() {  # rocprof-compute / omniperf: profile -> analyze, dump the FULL analyze text.
    local tool="$1"
    # -n is only a label: rocprof-compute refuses "/" in it, and omniperf also refuses "-", "." and
    # more than 35 characters. An explicit -p is written to as given, and a path with ".." is refused.
    local workload
    workload="$(cd "$OUTPUT_DIR" && pwd)/${tool}_workload"
    # NO `rm` (prompts + blocks autonomous runs): move any stale profiler dir aside, then make fresh.
    [ -e "$workload" ] && mv "$workload" "${workload}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$workload"
    echo "=== Profiling with $tool (profile $RPC_PROFILE_ARGS) ==="
    local rc=0
    # rocprof-compute >= 3.x replaced `profile -p <dir>` with `--output-directory`/`-d` (`-p` is now
    # rejected: exit 2, no counters). `analyze -p` is unchanged. Pick the flag this install accepts.
    local outflag=(-p "$workload")
    # Opt-in (KERNEL_TOOLS_PROFILE=1, set by the roles only when use_expert_skills is on): default runs
    # keep main's invocation unchanged.
    if [ "${KERNEL_TOOLS_PROFILE:-0}" = "1" ]; then
        "$tool" profile --help 2>/dev/null | grep -q -- '--output-directory' && outflag=(--output-directory "$workload")
    fi
    bash "$GPU_LOCK" "$GPU_ID" \
        "$tool" profile $RPC_PROFILE_ARGS -n workload "${outflag[@]}" -- bash -c "$BENCHMARK_CMD" \
        > "$OUTPUT_DIR/${tool}_profile_raw.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then emit_profiler_failure "$tool" "$rc" RPC_PROFILE_ARGS "$OUTPUT_DIR/${tool}_profile_raw.log"; return 1; fi
    if [ -d "$workload" ]; then
        echo "=== $tool analyze (full, unparsed) ===" >> "$REPORT"
        bash "$GPU_LOCK" "$GPU_ID" "$tool" analyze -p "$workload" >> "$REPORT" 2>&1 || true
    fi
    if profiler_artifacts_ok "$workload"; then
        PROFILE_SUCCESS=true
        return 0
    fi
    return 1
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
    if [ "$rc" -eq 0 ] && profiler_artifacts_ok "$dir"; then
        PROFILE_SUCCESS=true
        return 0
    fi
    return 1
}

run_rocprof() {          # legacy: rocprof --stats (HIP dispatch stats).
    local dir="$OUTPUT_DIR/rocprof"
    [ -e "$dir" ] && mv "$dir" "${dir}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$dir"
    local output="$dir/results.csv"
    local log="$OUTPUT_DIR/rocprof_run.log"
    echo "=== Profiling with rocprof ($RPROF_ARGS) ==="
    local rc=0
    bash "$GPU_LOCK" "$GPU_ID" \
        rocprof $RPROF_ARGS -o "$output" bash -c "$BENCHMARK_CMD" \
        > "$log" 2>&1 || rc=$?
    { cat "$log"; echo ""; } >> "$REPORT" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then emit_profiler_failure rocprof "$rc" RPROF_ARGS "$log"; return 1; fi
    if profiler_artifacts_ok "$dir"; then
        PROFILE_SUCCESS=true
        return 0
    fi
    return 1
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
    if [ "$rc" -eq 0 ] && profiler_artifacts_ok "$dir"; then
        PROFILE_SUCCESS=true
        return 0
    fi
    return 1
}

for p in $PROFILER_PRIORITY; do
    command -v "$p" &> /dev/null || continue
    case "$p" in
        rocprof-compute|omniperf) run_rocprof_compute "$p" && { PROFILER="$p"; break; } || true ;;
        rocprofv3)                run_rocprofv3 && { PROFILER="$p"; break; } || true ;;
        rocprof)                  run_rocprof && { PROFILER="$p"; break; } || true ;;
        metrix)                   run_metrix && { PROFILER="$p"; break; } || true ;;
    esac
    [ "$PROFILE_SUCCESS" = true ] && break
done

# Counter pass. rocprof-compute aborts on some ROCm/gfx950 installs (a counter it requests cannot be
# collected), which leaves only the rocprofv3 kernel trace: durations and dispatch counts, NO counters.
# Collect a small set of PMC groups for the hottest user kernel with the shared kernel_tools
# (rocprofv3_safe.sh: per-group timeout, kernel filter, abort -> degraded; parse_pmc.py: derived
# metrics), one small group per pass (a bundled group can exceed what one pass may collect).
#   Opt-in: runs only with KERNEL_TOOLS_PROFILE=1 (the role prompts offer it when use_expert_skills is on), so a
#   default run is unchanged. PMC_PASS=0 disables it even then; PMC_KERNEL_REGEX overrides the kernel;
#   PMC_GROUPS overrides the groups (';'-separated counter lists). Single-GPU specs only.
KT_DIR="$SCRIPT_DIR/kernel_tools"
run_pmc_pass() {
    [ "${KERNEL_TOOLS_PROFILE:-0}" = "1" ] || return 0
    [ "${PMC_PASS:-1}" = "0" ] && return 0
    case "$GPU_ID" in group:*|pool:*:*) return 0 ;; esac
    [ -f "$KT_DIR/rocprofv3_safe.sh" ] && [ -f "$KT_DIR/parse_pmc.py" ] || return 0
    command -v rocprofv3 &> /dev/null || return 0
    local kre="${PMC_KERNEL_REGEX:-}" stats seen
    stats="$(find "$OUTPUT_DIR/rocprofv3" -name '*kernel_stats.csv' 2>/dev/null | head -1)"
    [ -n "$kre" ] || [ -n "$stats" ] || return 0
    seen="$(mktemp)"
    # With PMC_KERNEL_REGEX: check it against the trace's kernel names. Without: the hottest kernel by
    # total time that is not a torch/runtime/library helper. HIP user kernels (demangled "void f<..>"
    # or mangled "_ZN...") are candidates; torch's own (at::, _ZN2at, c10) are not. No match: the
    # kernel names the trace saw go to the report and the pass is skipped.
    if [ -n "$stats" ]; then
        kre="$(python3 - "$stats" "$kre" 2>"$seen" <<'PY'
import csv, re, sys
names = [r.get("Name") or "" for r in csv.DictReader(open(sys.argv[1]))]
user = sys.argv[2]
if user:
    ok = any(re.search(user, n) for n in names)
    print(user if ok else "")
else:
    skip = re.compile(r"^(?:void\s+)?(at::|c10::|__amd|Cijk_|ck::|ck_tile::|rocclr|hipblaslt|_ZN2at|_ZN3c10|_ZN2ck)")
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if not skip.match(r.get("Name") or "")]
    rows.sort(key=lambda r: -float(r.get("TotalDurationNs") or 0))
    print(re.escape(rows[0]["Name"][:200]) if rows else "")  # cut before escaping: no lone "\"
    ok = bool(rows)
if not ok:
    sys.stderr.write("; ".join(sorted(set(n[:120] for n in names)))[:2000])
PY
)"
    fi
    if [ -z "$kre" ]; then
        { echo ""; echo "=== PMC counters: skipped, no kernel matched ${PMC_KERNEL_REGEX:+PMC_KERNEL_REGEX='$PMC_KERNEL_REGEX' }(kernels seen: $(cat "$seen" 2>/dev/null)) ==="; } >> "$REPORT"
        rm -f "$seen"
        return 0
    fi
    rm -f "$seen"
    # One pass per parse_pmc.py counter group (its PMC_GROUPS is the single source); PMC_GROUPS in the
    # env ("C1 C2;C3;...") overrides; the literal is a fallback if the parser cannot be run.
    local groups="${PMC_GROUPS:-}"
    [ -n "$groups" ] || groups="$(python3 "$KT_DIR/parse_pmc.py" --print-groups 2>/dev/null \
        | sed -n 's/^[A-Za-z0-9_]*: *//p' | paste -sd';' -)"
    [ -n "$groups" ] || groups="SQ_WAVES SQ_BUSY_CYCLES SQ_WAIT_ANY;VALUBusy MfmaUtil OccupancyPercent;FETCH_SIZE;WRITE_SIZE;TCC_HIT_sum TCC_MISS_sum;SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE"
    local pdir="$OUTPUT_DIR/pmc"
    [ -e "$pdir" ] && mv "$pdir" "${pdir}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$pdir"
    echo "=== Counter pass (rocprofv3 --pmc via kernel_tools/rocprofv3_safe.sh, kernel ~ ${kre:0:80}) ==="
    { echo ""; echo "=== PMC counters (rocprofv3 --pmc, one small group per pass; kernel regex: $kre) ==="; } >> "$REPORT"
    local i=0 g
    IFS=';' read -ra _GL <<< "$groups"
    for g in "${_GL[@]}"; do
        i=$((i+1)); local rc=0
        bash "$GPU_LOCK" "$GPU_ID" bash "$KT_DIR/rocprofv3_safe.sh" --kernel "$kre" --out "$pdir/pmc_g$i" \
            --pmc "$g" --cmd "$BENCHMARK_CMD" > "$pdir/pmc_g$i.log" 2>&1 || rc=$?
        echo "group $i [$g]: rc=$rc$([ "$rc" -ne 0 ] && echo " (not collected; see pmc/pmc_g$i.log)")" >> "$REPORT"
    done
    echo "parse: parse_pmc.py <pmc dir> '<kernel regex>' --regex" >> "$REPORT"
    python3 "$KT_DIR/parse_pmc.py" "$pdir" "$kre" --regex --arch "${DETECTED_ARCH:-}" >> "$REPORT" 2>&1 || \
        echo "(parse_pmc.py could not derive metrics; raw CSVs are under $pdir)" >> "$REPORT"
}
if [ "${PROFILER:-}" = "rocprofv3" ]; then run_pmc_pass || true; fi

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
