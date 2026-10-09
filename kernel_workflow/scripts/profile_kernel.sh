#!/bin/bash
# Thin profiling wrapper: warmup + gpu_lock + detect best available profiler + run it + dump RAW output.
#
# The default run deliberately does NOT parse or interpret the profiler output (the optional layers
# below add a parser summary from kernel_tools/ BESIDE their raw data). The profile_engineer reads the raw
# artifacts written here and classifies the bottleneck itself, following knowledge/profiling_guide.md
# (which documents how to extract the key metrics + dispatch counts from EACH profiler's format and how
# to degrade gracefully when a field is absent). Keeping the parsing out of this script is what makes it
# portable: it never greps for profiler-/version-specific section names ("System Speed-of-Light",
# "Wavefront", …) or assumes a particular CSV layout, so it keeps working when the toolchain changes.
#
# Usage: bash profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir> [--pmc|--derived] [--att] [--spi]
#                                [--kernel <regex>]
#        bash profile_kernel.sh --help
#
# OPTIONAL LAYERS (off by default; the default run is unchanged). Each is collected AFTER the main
# profiler step, writes into its own subdir, appends a section to profile_report.txt, and DEGRADES
# (records why, never fails the run) when the tool, arch, or decoder is missing. All of them go
# through gpu_lock.sh (via kernel_tools/rocprofv3_safe.sh: timeout + kernel filter + HIP/ROCR guard).
#   --pmc       rocprofv3 counter groups from kernel_tools/parse_pmc.py PMC_GROUPS (memory, memory_ea,
#               sol, stall, waitbusy, lds_raw), one pass per group, bisected on abort/timeout ->
#               <out>/pmc/pmc_<group>/ + <out>/pmc/pmc_summary.txt (parse_pmc: busy counters, C1
#               achieved DRAM bandwidth by independent routes). CDNA (gfx9) names; off gfx9 the layer
#               is skipped unless PROFILE_PMC_GROUPS="name:C1 C2;name2:C3" names this build's counters
#               (list them with `rocprofv3 -L` or `rocprofv3-avail list --pmc`).
#   --derived   the cheap subset of --pmc: only the derived busy/stall groups (sol, stall).
#   --att       rocprofv3 ATT (advanced thread trace) -> <out>/att/ + hotspot_analyzer.py summary.
#               Needs the rocprof-trace-decoder (NOT vendored): export ROCPROF_ATT_LIBRARY_PATH=<dir>.
#   --spi       occupancy-limiter evidence: rocprof-compute analyze --block 6.2 2.1.15 on the main
#               step's workload when rocprof-compute ran; else PROFILE_SPI_COUNTERS (build-specific
#               raw SPI counters) through rocprofv3 -> <out>/spi/.
#   --kernel R  kernel-name regex for the optional layers. Default: the dominant non-helper kernel
#               of a kernel-trace pass (recorded in <out>/kernel_select/selected_kernel.txt).
# Interpretation of every layer: knowledge/profiling_guide.md (ratios over Total, busy counters
# VALUBusy/MfmaUtil not the VALUUtilization duty-cycle). Layer states: <out>/profile_layers.json.
#
# Optional env overrides (all have sensible defaults; nothing kernel-specific is hard-coded):
#   PROFILER_PRIORITY  space-separated profiler order to try (default is arch-aware:
#                      gfx1201 -> rocprofv3 first; CDNA/other -> rocprof-compute first)
#   WARMUP_RUNS        number of warmup runs before profiling (default: 3)
#   RPC_PROFILE_ARGS   extra args passed to rocprof-compute/omniperf `profile` (default: "--no-roof")
#   RPV3_TRACE_ARGS    args passed to rocprofv3 (default: "--kernel-trace --stats --output-format csv")
#   RPROF_ARGS         args passed to legacy rocprof (default: "--stats")
#   METRIX_ARGS        args passed to metrix (default: ""; override per `metrix --help` for this toolchain)
#   PROFILE_PMC_GROUPS   explicit counter groups for --pmc/--derived ("name:C1 C2;name2:C3")
#   PROFILE_PMC_TIMEOUT  per-pass timeout in seconds for counter passes (default 150; 0 = none)
#   PROFILE_ATT_TIMEOUT  ATT pass timeout in seconds (default 600; 0 = none)
#   PROFILE_SPI_COUNTERS raw rocprofv3 SPI counters for --spi when rocprof-compute did not run
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

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d'
    exit 0
fi

GPU_ID="${1:?Usage: profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir> [--pmc|--derived] [--att] [--spi] [--kernel <regex>]}"
BENCHMARK_CMD="${2:?Missing benchmark command}"
OUTPUT_DIR="${3:?Missing output directory}"
shift 3
WANT_PMC=""; WANT_ATT=0; WANT_SPI=0; KERNEL_FILTER="${PROFILE_KERNEL:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --pmc)     WANT_PMC="pmc" ;;
        --derived) [ -z "$WANT_PMC" ] && WANT_PMC="derived" ;;
        --att)     WANT_ATT=1 ;;
        --spi)     WANT_SPI=1 ;;
        --kernel)  KERNEL_FILTER="${2:?--kernel needs a regex}"; shift ;;
        *) echo "profile_kernel.sh: unknown option '$1' (see --help)" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LOCK="$SCRIPT_DIR/gpu_lock.sh"
KT_DIR="$SCRIPT_DIR/kernel_tools"
RV3_SAFE="$KT_DIR/rocprofv3_safe.sh"
source "$SCRIPT_DIR/profile_policy.sh"

WARMUP_RUNS="${WARMUP_RUNS:-3}"
DETECTED_ARCH="$(profile_detect_arch)"
PROFILER_PRIORITY="${PROFILER_PRIORITY:-$(profiler_priority_for_arch "$DETECTED_ARCH")}"
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
    local workload="$OUTPUT_DIR/${tool}_workload"
    # NO `rm` (prompts + blocks autonomous runs): move any stale profiler dir aside, then make fresh.
    [ -e "$workload" ] && mv "$workload" "${workload}.old_$(date +%s)_$$" 2>/dev/null || true
    mkdir -p "$workload"
    echo "=== Profiling with $tool (profile $RPC_PROFILE_ARGS) ==="
    local rc=0
    bash "$GPU_LOCK" "$GPU_ID" \
        "$tool" profile $RPC_PROFILE_ARGS -n "$workload" -- bash -c "$BENCHMARK_CMD" \
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

# Final fallback: no profiler available, or the chosen one produced nothing -> benchmark-only.
if [ "$PROFILE_SUCCESS" = false ]; then
    echo ""
    echo "=== Fallback: benchmark-only (no usable profiler output) ==="
    PROFILER="benchmark-only"
    bash "$GPU_LOCK" "$GPU_ID" bash -c "$BENCHMARK_CMD" >> "$REPORT" 2>&1 || true
fi

# Step 3 (optional layers): each one degrades -- records why -- instead of failing the profile.
LAYER_PMC="off"; LAYER_ATT="off"; LAYER_SPI="off"

layer_note() {  # <section title> <text...>: append to the report and echo
    local title="$1"; shift
    { echo ""; echo "=== $title ==="; printf '%s\n' "$@"; } >> "$REPORT"
    printf '%s\n' "[$title] ${1:-} (full text in profile_report.txt)"
}

move_aside() { [ -e "$1" ] && mv "$1" "${1}.old_$(date +%s)_$$" 2>/dev/null || true; }

# The kernel filter every counter/ATT pass needs (rocprofv3_safe refuses an unfiltered collection).
select_kernel() {
    [ -n "$KERNEL_FILTER" ] && return 0
    command -v rocprofv3 &> /dev/null || return 1
    local d="$OUTPUT_DIR/kernel_select" name
    move_aside "$d"
    bash "$RV3_SAFE" --gpu "$GPU_ID" --kernel '.' --out "$d" --kernel-trace \
        --timeout "${PROFILE_PMC_TIMEOUT:-150}" --cmd "$BENCHMARK_CMD" > "$OUTPUT_DIR/kernel_select.log" 2>&1 || true
    name="$(python3 "$KT_DIR/parse_pmc.py" --top-kernel "$d" 2>/dev/null || true)"
    [ -n "$name" ] || return 1
    printf '%s\n' "$name" > "$d/selected_kernel.txt"
    KERNEL_FILTER="$(python3 -c 'import re,sys; print(re.escape(sys.argv[1]))' "$name")"
}

run_pmc_layer() {  # $1 = pmc | derived
    local mode="$1" dir="$OUTPUT_DIR/pmc" groups_spec=""
    if [ -n "${PROFILE_PMC_GROUPS:-}" ]; then
        groups_spec="$PROFILE_PMC_GROUPS"
    elif [ -z "$DETECTED_ARCH" ]; then
        LAYER_PMC="degraded:arch_unknown"
        layer_note "PMC layer DEGRADED" "GPU arch could not be identified; the default counter names are" \
            "arch-specific, so none were guessed. Set PYTORCH_ROCM_ARCH=gfxNNN or PROFILE_PMC_GROUPS."
        return 0
    elif ! profile_pmc_default_groups_ok "$DETECTED_ARCH"; then
        LAYER_PMC="degraded:non_cdna_arch"
        layer_note "PMC layer DEGRADED" "The default groups use CDNA (gfx9) counter names; this GPU is $DETECTED_ARCH." \
            "List this build's counters with \`rocprofv3 -L\` or \`rocprofv3-avail list --pmc\` and pass" \
            "PROFILE_PMC_GROUPS=\"name:C1 C2;name2:C3\". Classify from kernel-trace + roofline meanwhile."
        return 0
    else
        local g
        for g in $(profile_pmc_groups_for_mode "$mode"); do
            groups_spec+="$(python3 "$KT_DIR/parse_pmc.py" --print-groups | awk -v g="$g" -F': ' '$1==g{print $1":"$2}');"
        done
    fi
    if ! command -v rocprofv3 &> /dev/null; then
        LAYER_PMC="degraded:no_rocprofv3"
        layer_note "PMC layer DEGRADED" "rocprofv3 is not installed; counters were not collected."
        return 0
    fi
    if ! select_kernel; then
        LAYER_PMC="degraded:no_kernel"
        layer_note "PMC layer DEGRADED" "No kernel filter: pass --kernel <regex> (the kernel-trace pass found no dispatch; see kernel_select.log)."
        return 0
    fi
    move_aside "$dir"; mkdir -p "$dir"
    GROUPS_SPEC="$groups_spec" python3 - "$KT_DIR" "$RV3_SAFE" "$GPU_ID" "$KERNEL_FILTER" "$dir" \
        "${PROFILE_PMC_TIMEOUT:-150}" "$BENCHMARK_CMD" > "$dir/pmc_collection.log" 2>&1 <<'PY' || true
import json, os, subprocess, sys
kt, safe, gpu, kernel, out, timeout, bench = sys.argv[1:8]
sys.path.insert(0, kt)
import parse_pmc
groups = []
for item in os.environ["GROUPS_SPEC"].split(";"):
    if ":" in item:
        name, counters = item.split(":", 1)
        if counters.split():
            groups.append((name.strip(), counters.split()))
record = {"kernel_regex": kernel, "groups": {}}
for name, counters in groups:
    n = {"i": 0}
    def run_pass(sub, name=name):
        n["i"] += 1
        d = os.path.join(out, f"pmc_{name}" + ("" if n["i"] == 1 else f"_{n['i']}"))
        rc = subprocess.call(["bash", safe, "--gpu", gpu, "--kernel", kernel, "--out", d,
                              "--pmc", " ".join(sub), "--timeout", timeout, "--cmd", bench])
        print(f"pass {name}#{n['i']} {sub} rc={rc}", flush=True)
        return rc
    record["groups"][name] = parse_pmc.collect_pmc_group(counters, run_pass)
json.dump(record, open(os.path.join(out, "pmc_collection.json"), "w"), indent=1)
PY
    python3 "$KT_DIR/parse_pmc.py" "$dir" "" --arch "${DETECTED_ARCH:-}" > "$dir/pmc_summary.txt" 2>&1 || true
    local dropped
    dropped="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(c for g in d["groups"].values() for c in g["dropped"]))' "$dir/pmc_collection.json" 2>/dev/null || echo "?")"
    LAYER_PMC="collected:$mode"
    [ -n "$dropped" ] && LAYER_PMC="partial:$mode"
    layer_note "PMC layer ($mode, kernel~=$KERNEL_FILTER) -- parse_pmc summary" \
        "$(cat "$dir/pmc_summary.txt" 2>/dev/null)" \
        "${dropped:+counters dropped (abort/timeout alone, unavailable on this box): $dropped}"
}

run_att_layer() {
    local dir="$OUTPUT_DIR/att" rc=0 disp=""
    if ! command -v rocprofv3 &> /dev/null; then
        LAYER_ATT="degraded:no_rocprofv3"; layer_note "ATT layer DEGRADED" "rocprofv3 is not installed."; return 0
    fi
    if ! select_kernel; then
        LAYER_ATT="degraded:no_kernel"; layer_note "ATT layer DEGRADED" "No kernel filter: pass --kernel <regex>."; return 0
    fi
    move_aside "$dir"
    bash "$RV3_SAFE" --gpu "$GPU_ID" --kernel "$KERNEL_FILTER" --out "$dir" --att \
        --timeout "${PROFILE_ATT_TIMEOUT:-600}" --cmd "$BENCHMARK_CMD" > "$OUTPUT_DIR/att_run.log" 2>&1 || rc=$?
    if [ "$rc" -eq 3 ]; then
        LAYER_ATT="degraded:decoder_absent"
        layer_note "ATT layer DEGRADED" "rocprof-trace-decoder not found (GEAK does not vendor it); export" \
            "ROCPROF_ATT_LIBRARY_PATH=<dir holding librocprof-trace-decoder.so> to enable --att."
        return 0
    fi
    disp="$(find "$dir" -maxdepth 3 -type d -name 'ui_output_agent_*' 2>/dev/null | head -1 || true)"
    if [ -n "$disp" ] && grep -q '"code": *null' "$disp/code.json" 2>/dev/null; then
        # the traced CU caught no waves: retry once selecting every SIMD
        move_aside "$dir"
        bash "$RV3_SAFE" --gpu "$GPU_ID" --kernel "$KERNEL_FILTER" --out "$dir" --att-wide \
            --timeout "${PROFILE_ATT_TIMEOUT:-600}" --cmd "$BENCHMARK_CMD" >> "$OUTPUT_DIR/att_run.log" 2>&1 || rc=$?
        disp="$(find "$dir" -maxdepth 3 -type d -name 'ui_output_agent_*' 2>/dev/null | head -1 || true)"
    fi
    if [ -z "$disp" ] || [ ! -f "$disp/code.json" ]; then
        LAYER_ATT="degraded:no_trace(rc=$rc)"
        layer_note "ATT layer DEGRADED" "rocprofv3 --att produced no decoded dispatch (rc=$rc); see att_run.log."
        return 0
    fi
    python3 "$KT_DIR/hotspot_analyzer.py" "$disp" --topk 15 --mode both > "$dir/hotspots.txt" 2>&1 || true
    LAYER_ATT="collected"
    layer_note "ATT layer (kernel~=$KERNEL_FILTER) -- hotspot_analyzer" "dispatch: $disp" "$(cat "$dir/hotspots.txt" 2>/dev/null)"
}

run_spi_layer() {
    local dir="$OUTPUT_DIR/spi" workload="$OUTPUT_DIR/${PROFILER}_workload"
    move_aside "$dir"; mkdir -p "$dir"
    case "$PROFILER" in
        rocprof-compute|omniperf)
            if [ -d "$workload" ]; then
                bash "$GPU_LOCK" "$GPU_ID" "$PROFILER" analyze -p "$workload" --block 6.2 2.1.15 \
                    > "$dir/spi_limiter.txt" 2>&1 || true
                LAYER_SPI="collected:rocprof-compute"
                layer_note "SPI occupancy-limiter ($PROFILER analyze --block 6.2 2.1.15)" "$(cat "$dir/spi_limiter.txt")"
                return 0
            fi ;;
    esac
    if [ -n "${PROFILE_SPI_COUNTERS:-}" ] && command -v rocprofv3 &> /dev/null && select_kernel; then
        bash "$RV3_SAFE" --gpu "$GPU_ID" --kernel "$KERNEL_FILTER" --out "$dir/pmc_spi" \
            --pmc "${PROFILE_SPI_COUNTERS//,/ }" --timeout "${PROFILE_PMC_TIMEOUT:-150}" \
            --cmd "$BENCHMARK_CMD" > "$dir/spi_run.log" 2>&1 || true
        LAYER_SPI="collected:rocprofv3"
        layer_note "SPI occupancy-limiter (rocprofv3 --pmc $PROFILE_SPI_COUNTERS)" "raw CSVs under $dir/pmc_spi"
        return 0
    fi
    LAYER_SPI="degraded:no_source"
    layer_note "SPI layer DEGRADED" "Needs rocprof-compute as the main profiler (blocks 6.2 / 2.1.15) or" \
        "PROFILE_SPI_COUNTERS=<this build's raw SPI counter names> (probe per build)."
}

[ -n "$WANT_PMC" ] && run_pmc_layer "$WANT_PMC"
[ "$WANT_ATT" = 1 ] && run_att_layer
[ "$WANT_SPI" = 1 ] && run_spi_layer
printf '{"profiler":"%s","pmc":"%s","att":"%s","spi":"%s"}\n' \
    "${PROFILER:-benchmark-only}" "$LAYER_PMC" "$LAYER_ATT" "$LAYER_SPI" > "$OUTPUT_DIR/profile_layers.json"

echo ""
echo "=== Profiling complete ==="
echo "Profiler used: ${PROFILER:-benchmark-only}"
echo "Optional layers: pmc=$LAYER_PMC att=$LAYER_ATT spi=$LAYER_SPI"
echo "Report: $REPORT"
echo "Artifacts:"
find "$OUTPUT_DIR" -maxdepth 2 -type f 2>/dev/null | sort | sed 's/^/  /' || true
