#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Installed-but-broken profiler must not count as success; try the next tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/out" "$tmp/work"

cat > "$tmp/bin/rocprofv3" <<'EOF'
#!/bin/sh
echo "unrecognized argument --output-format" >&2
exit 2
EOF
cat > "$tmp/bin/rocprof" <<'EOF'
#!/bin/sh
echo "Kernel Duration Count"
echo "fixture_kernel 12.0 1"
EOF
chmod +x "$tmp/bin/rocprofv3" "$tmp/bin/rocprof"

(
  cd "$tmp/work"
  PATH="$tmp/bin:$PATH" \
  PROFILER_PRIORITY="rocprofv3 rocprof" \
  WARMUP_RUNS=0 \
  KERNEL_ENV_KEEP_ARCH=1 \
  KERNEL_ENV_SKIP_ENUM_REAP=1 \
  GEAK_GPU_REQUIRE_IDLE=0 \
    bash "$SCRIPT_DIR/profile_kernel.sh" 0 "true" "$tmp/out" \
      > "$tmp/stdout.log" 2>&1
)

grep -q "PROFILER FAILED: rocprofv3 exited 2" "$tmp/out/profile_report.txt"
grep -q "Kernel Duration Count" "$tmp/out/profile_report.txt"
grep -q "Profiler used: rocprof" "$tmp/stdout.log"
if grep -q "Profiler used: rocprofv3" "$tmp/stdout.log"; then
  echo "FAIL: failed rocprofv3 was treated as successful" >&2
  exit 1
fi

echo "PASS: failed profiler falls through until a real profile is produced."
