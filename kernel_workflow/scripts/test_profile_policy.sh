#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_policy.sh"

rdna="$(profiler_priority_for_arch gfx1201)"
# The RDNA 3.5 APUs take the gfx1201 order: rocprof-compute 3.4.0 (ROCm 7.2) refuses gfx115x, and
# from 3.7.0 its gfx1151 bandwidth metrics need --specs-correction. gfx1100 and gfx1103 keep the default.
gfx1151="$(profiler_priority_for_arch gfx1151)"
gfx1150="$(profiler_priority_for_arch gfx1150)"
gfx1100="$(profiler_priority_for_arch gfx1100)"
gfx1103="$(profiler_priority_for_arch gfx1103)"
gfx950="$(profiler_priority_for_arch gfx950)"
gfx942="$(profiler_priority_for_arch gfx942)"
gfx90a="$(profiler_priority_for_arch gfx90a)"
unknown="$(profiler_priority_for_arch gfx9999)"

for priority in "$rdna" "$gfx1151" "$gfx1150"; do
    [[ "$priority" == "rocprofv3 rocprof metrix rocprof-compute omniperf" ]]
done
for priority in "$gfx950" "$gfx942" "$gfx90a" "$gfx1100" "$gfx1103" "$unknown"; do
    [[ "$priority" == "rocprof-compute omniperf rocprofv3 rocprof metrix" ]]
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 1
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Compute Unit:            64
*******
Agent 2
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Compute Unit:            2
ROCMINFO
EOF
chmod +x "$tmp/rocminfo"
detected="$(PATH="$tmp:$PATH" PYTORCH_ROCM_ARCH= profile_detect_arch)"
[[ "$detected" == "gfx1201" ]]

echo "PASS: gfx1201, gfx115x, gfx950, gfx942, gfx90a, gfx1100, gfx1103, and unknown policies are explicit."
