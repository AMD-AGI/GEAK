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
printf '%s\n' "$*" > "$STUB_DIR/rocprofv3_args"
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
printf 'Name,Calls\nfixture_kernel,1\n' > "$dir/kernel_stats.csv"
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
grep -q -- "^--kernel-trace --stats --output-format csv -d " "$tmp/rocprofv3_args" \
  || fail "default rocprofv3 arguments changed: $(cat "$tmp/rocprofv3_args")"

# A caller's own setting is kept.
run_profile "$tmp/out-v3-caller" PROFILER_PRIORITY="rocprofv3" ROCPROF_SIGNAL_HANDLERS=1
[ "$(cat "$tmp/rocprofv3_signal_handlers")" = 1 ] || fail "the caller's ROCPROF_SIGNAL_HANDLERS was overridden"

echo "PASS: profiler invocations match what the installed tools accept."
