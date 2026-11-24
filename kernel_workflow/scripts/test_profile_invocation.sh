#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The profiler command lines must be ones the installed tool accepts: rocprof-compute gets a bare
# -n label with the workload directory in -p. Every profiler runs with rocprofv3's own signal
# handlers off, which a process that aborts can otherwise deadlock in.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/work" "$tmp/locks"

cat > "$tmp/bin/rocminfo" <<'EOF'
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 1
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Compute Unit:            304
ROCMINFO
EOF

# Name rules of both releases: rocprof-compute refuses "/", omniperf refuses "-", "." and more than
# 35 characters; both refuse ".." in the path. Counters go to -p and analyze reads them back from it.
cat > "$tmp/bin/rocprof-compute" <<'EOF'
#!/bin/sh
mode="$1"; shift
name=""; path=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--name) name="$2"; shift 2 ;;
    -p|--path) path="$2"; shift 2 ;;
    --) break ;;
    *) shift ;;
  esac
done
case "$mode" in
  profile)
    printf '%s\n' "${ROCPROF_SIGNAL_HANDLERS-unset}" > "$STUB_DIR/rocprof-compute_signal_handlers"
    case "$name" in ""|*/*|*-*|*.*) echo "ERROR profile name not permitted: $name" >&2; exit 1 ;; esac
    [ "${#name}" -le 35 ] || { echo "ERROR profile name exceeds 35 characters" >&2; exit 1; }
    case "$path" in ""|*..*) echo "ERROR cannot access parent directories: $path" >&2; exit 1 ;; esac
    mkdir -p "$path"
    printf 'Dispatch_ID,Kernel_Name\n0,fixture_kernel\n' > "$path/pmc_perf.csv"
    ;;
  analyze)
    cat "$path/pmc_perf.csv"
    ;;
esac
EOF

cat > "$tmp/bin/rocprofv3" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$STUB_DIR/rocprofv3_calls"
printf '%s\n' "${ROCPROF_SIGNAL_HANDLERS-unset}" > "$STUB_DIR/rocprofv3_signal_handlers"
dir=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -d) dir="$2"; shift 2 ;;
    --) break ;;
    *) shift ;;
  esac
done
mkdir -p "$dir"
if [ -n "${STUB_KERNEL_STATS:-}" ]; then cp "$STUB_KERNEL_STATS" "$dir/kernel_stats.csv"
else printf 'Name,Calls\nfixture_kernel,1\n' > "$dir/kernel_stats.csv"; fi
EOF
chmod +x "$tmp/bin/rocminfo" "$tmp/bin/rocprof-compute" "$tmp/bin/rocprofv3"

run_profile() {  # <output_dir> [VAR=value ...]
  local out="$1"
  shift
  (
    cd "$tmp/work"
    env -u ROCPROF_SIGNAL_HANDLERS PATH="$tmp/bin:$PATH" STUB_DIR="$tmp" KERNEL_EXTRACT_LOCK_DIR="$tmp/locks" \
      PYTORCH_ROCM_ARCH= WARMUP_RUNS=0 KERNEL_ENV_KEEP_ARCH=1 KERNEL_ENV_SKIP_ENUM_REAP=1 \
      GEAK_GPU_REQUIRE_IDLE=0 "$@" \
      bash "$SCRIPT_DIR/profile_kernel.sh" 0 "true" "$out" > "$tmp/stdout.log" 2>&1
  )
}

fail() { echo "FAIL: $*" >&2; cat "$tmp/stdout.log" >&2; exit 1; }

# rocprof-compute: a relative output dir with ".." still yields a usable workload path.
run_profile "../work/sub/../out-rpc" PROFILER_PRIORITY="rocprof-compute"
grep -q "Profiler used: rocprof-compute" "$tmp/stdout.log" || fail "rocprof-compute was not used"
grep -q fixture_kernel "$tmp/work/out-rpc/rocprof-compute_workload/pmc_perf.csv" \
  || fail "rocprof-compute did not write into the workload dir"
grep -q fixture_kernel "$tmp/work/out-rpc/profile_report.txt" || fail "analyze output missing from the report"
[ "$(cat "$tmp/rocprof-compute_signal_handlers")" = 0 ] \
  || fail "rocprof-compute ran with rocprofv3's signal handlers: $(cat "$tmp/rocprof-compute_signal_handlers")"

# rocprofv3 on a single GPU: handlers off, and its arguments are what they were.
run_profile "$tmp/out-v3" PROFILER_PRIORITY="rocprofv3"
grep -q "Profiler used: rocprofv3" "$tmp/stdout.log" || fail "rocprofv3 was not used"
[ "$(cat "$tmp/rocprofv3_signal_handlers")" = 0 ] \
  || fail "rocprofv3 kept its own signal handlers: $(cat "$tmp/rocprofv3_signal_handlers")"
grep -q -- "^--kernel-trace --stats --output-format csv -d " "$tmp/rocprofv3_calls" \
  || fail "default rocprofv3 arguments changed: $(cat "$tmp/rocprofv3_calls")"
# Default run: unchanged -- no counter pass.
! grep -q -- "--pmc" "$tmp/rocprofv3_calls" || fail "a default run ran a counter pass: $(cat "$tmp/rocprofv3_calls")"
! grep -q "=== PMC counters" "$tmp/out-v3/profile_report.txt" || fail "a default report carries a counter pass"
# Opt-in (KERNEL_TOOLS_PROFILE=1, set by the roles when use_expert_skills is on): the trace is followed by
# a counter pass on the hottest user kernel (kernel_tools/rocprofv3_safe.sh).
: > "$tmp/rocprofv3_calls"
run_profile "$tmp/out-v3-kt" PROFILER_PRIORITY="rocprofv3" KERNEL_TOOLS_PROFILE=1
grep -q -- "--pmc .*--kernel-include-regex fixture_kernel" "$tmp/rocprofv3_calls" \
  || fail "no counter pass after the kernel trace: $(cat "$tmp/rocprofv3_calls")"
grep -q "=== PMC counters" "$tmp/out-v3-kt/profile_report.txt" || fail "counter pass missing from the report"
grep -q "parse_pmc.py .* --regex" "$tmp/out-v3-kt/profile_report.txt" || fail "parse_pmc.py not told the filter is a regex"
# A HIP user kernel (demangled "void ..." or mangled "_ZN...") is profiled; torch's own kernels are not.
cat > "$tmp/stats_hip.csv" <<'CSV'
Name,Calls,TotalDurationNs
"void at::native::elementwise_kernel<128, 4>(int)",10,900000
_ZN2at6native13reduce_kernelEv,10,800000
"void my_gemm<64>(float*, float const*)",10,500000
_ZN5mykns9small_addEPfS0_,10,100000
CSV
: > "$tmp/rocprofv3_calls"
run_profile "$tmp/out-v3-hip" PROFILER_PRIORITY="rocprofv3" KERNEL_TOOLS_PROFILE=1 STUB_KERNEL_STATS="$tmp/stats_hip.csv"
grep -q -- "--pmc .*--kernel-include-regex void.*my_gemm" "$tmp/rocprofv3_calls" \
  || fail "the hottest HIP user kernel was not profiled: $(cat "$tmp/rocprofv3_calls")"
# A long templated name: the regex is cut to 200 characters of the name before it is escaped, so it
# stays a valid pattern (a cut after escaping can end in a lone backslash) and still matches the kernel.
python3 - "$tmp/stats_long.csv" <<'PY'
import csv, re, sys
name = "void my_kernel__<" + ", ".join(["int"] * 29) + ">(float*)"
assert re.escape(name)[:200].endswith("\\"), "fixture no longer exercises the cut-after-escape case"
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Name", "Calls", "TotalDurationNs"]); w.writerow([name, 10, 500000])
PY
: > "$tmp/rocprofv3_calls"
run_profile "$tmp/out-v3-long" PROFILER_PRIORITY="rocprofv3" KERNEL_TOOLS_PROFILE=1 STUB_KERNEL_STATS="$tmp/stats_long.csv"
grep -q -- "--pmc" "$tmp/rocprofv3_calls" || fail "no counter pass for a long templated kernel name"
long_re="$(sed -n 's/^=== PMC counters (.*kernel regex: \(.*\)) ===$/\1/p' "$tmp/out-v3-long/profile_report.txt")"
[ -n "$long_re" ] || fail "kernel regex for the long name missing from the report"
python3 - "$long_re" "$tmp/stats_long.csv" <<'PY' || fail "long-name kernel regex is invalid or does not match: $long_re"
import csv, re, sys
name = next(csv.DictReader(open(sys.argv[2])))["Name"]
assert re.search(sys.argv[1], name)
PY
# No candidate: no pass, and the report lists the kernel names the trace saw.
cat > "$tmp/stats_torch.csv" <<'CSV'
Name,Calls,TotalDurationNs
"void at::native::elementwise_kernel<128, 4>(int)",10,900000
Cijk_Ailk_Bljk_HHS_BH,10,800000
CSV
: > "$tmp/rocprofv3_calls"
run_profile "$tmp/out-v3-none" PROFILER_PRIORITY="rocprofv3" KERNEL_TOOLS_PROFILE=1 STUB_KERNEL_STATS="$tmp/stats_torch.csv"
! grep -q -- "--pmc" "$tmp/rocprofv3_calls" || fail "a counter pass ran with no candidate kernel"
grep -q "no kernel matched (kernels seen: .*Cijk_Ailk_Bljk_HHS_BH" "$tmp/out-v3-none/profile_report.txt" \
  || fail "no-match report does not list the kernels seen"
# A PMC_KERNEL_REGEX that matches nothing in the trace: skipped, names listed.
: > "$tmp/rocprofv3_calls"
run_profile "$tmp/out-v3-nore" PROFILER_PRIORITY="rocprofv3" KERNEL_TOOLS_PROFILE=1 STUB_KERNEL_STATS="$tmp/stats_hip.csv" \
  PMC_KERNEL_REGEX="no_such_kernel"
! grep -q -- "--pmc" "$tmp/rocprofv3_calls" || fail "a counter pass ran for a regex that matches no kernel"
grep -q "PMC_KERNEL_REGEX='no_such_kernel' (kernels seen: .*my_gemm" "$tmp/out-v3-nore/profile_report.txt" \
  || fail "unmatched PMC_KERNEL_REGEX not reported with the kernels seen"
: > "$tmp/rocprofv3_calls"
run_profile "$tmp/out-v3-nopmc" PROFILER_PRIORITY="rocprofv3" KERNEL_TOOLS_PROFILE=1 PMC_PASS=0
! grep -q -- "--pmc" "$tmp/rocprofv3_calls" || fail "PMC_PASS=0 still ran a counter pass"

# A caller's own setting is kept.
run_profile "$tmp/out-v3-caller" PROFILER_PRIORITY="rocprofv3" ROCPROF_SIGNAL_HANDLERS=1
[ "$(cat "$tmp/rocprofv3_signal_handlers")" = 1 ] || fail "the caller's ROCPROF_SIGNAL_HANDLERS was overridden"

echo "PASS: profiler invocations match what the installed tools accept."
