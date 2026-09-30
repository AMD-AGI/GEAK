#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Installed-but-broken profiler must not count as success; try the next tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/out-empty" "$tmp/out-real" "$tmp/work"

cat > "$tmp/bin/rocprofv3" <<'EOF'
#!/bin/sh
echo "stale Kernel Duration text from failed rocprofv3" >&2
echo "unrecognized argument --output-format" >&2
exit 2
EOF
cat > "$tmp/bin/rocprof" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$tmp/bin/rocprofv3" "$tmp/bin/rocprof"

# A failed v3 attempt contaminates the shared report with Kernel/Duration
# strings. An empty successful rocprof process must still fall through.
(
  cd "$tmp/work"
  PATH="$tmp/bin:$PATH" \
  PROFILER_PRIORITY="rocprofv3 rocprof" \
  WARMUP_RUNS=0 \
  KERNEL_ENV_KEEP_ARCH=1 \
  KERNEL_ENV_SKIP_ENUM_REAP=1 \
  GEAK_GPU_REQUIRE_IDLE=0 \
    bash "$SCRIPT_DIR/profile_kernel.sh" 0 "true" "$tmp/out-empty" \
      > "$tmp/stdout-empty.log" 2>&1
)

grep -q "PROFILER FAILED: rocprofv3 exited 2" "$tmp/out-empty/profile_report.txt"
grep -q "stale Kernel Duration text" "$tmp/out-empty/profile_report.txt"
grep -q "Profiler used: benchmark-only" "$tmp/stdout-empty.log"
if grep -q "Profiler used: rocprof$" "$tmp/stdout-empty.log"; then
  echo "FAIL: empty rocprof was treated as successful from stale report text" >&2
  exit 1
fi

# Positive control: legacy rocprof writes the requested CSV artifact.
cat > "$tmp/bin/rocprof" <<'EOF'
#!/bin/sh
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    out="$2"
    shift 2
    continue
  fi
  shift
done
[ -n "$out" ] || exit 3
mkdir -p "$(dirname "$out")"
printf 'Name,Duration,Count\nfixture_kernel,12.0,1\n' > "$out"
echo "Kernel Duration Count"
EOF
chmod +x "$tmp/bin/rocprof"

(
  cd "$tmp/work"
  PATH="$tmp/bin:$PATH" \
  PROFILER_PRIORITY="rocprofv3 rocprof" \
  WARMUP_RUNS=0 \
  KERNEL_ENV_KEEP_ARCH=1 \
  KERNEL_ENV_SKIP_ENUM_REAP=1 \
  GEAK_GPU_REQUIRE_IDLE=0 \
    bash "$SCRIPT_DIR/profile_kernel.sh" 0 "true" "$tmp/out-real" \
      > "$tmp/stdout-real.log" 2>&1
)

grep -q "Profiler used: rocprof" "$tmp/stdout-real.log"
grep -q "fixture_kernel" "$tmp/out-real/rocprof/results.csv"

echo "PASS: profiler fallback requires a real artifact from the successful attempt."
