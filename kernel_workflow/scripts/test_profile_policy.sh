#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_policy.sh"

rdna="$(profiler_priority_for_arch gfx1201)"
gfx950="$(profiler_priority_for_arch gfx950)"
gfx942="$(profiler_priority_for_arch gfx942)"
gfx90a="$(profiler_priority_for_arch gfx90a)"
unknown="$(profiler_priority_for_arch gfx9999)"

[[ "$rdna" == "rocprofv3 rocprof metrix rocprof-compute omniperf" ]]
for priority in "$gfx950" "$gfx942" "$gfx90a" "$unknown"; do
    [[ "$priority" == "rocprof-compute omniperf rocprofv3 rocprof metrix" ]]
done

echo "PASS: gfx1201, gfx950, gfx942, gfx90a, and unknown policies are explicit."
