#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Pure profiler-policy helper. Kept separate so CI can test architecture routing
# without a GPU or profiler installation.

# A caller-pinned PYTORCH_ROCM_ARCH names the local GPU only when it is a single gfx token.
# Framework images (rocm/vllm) export a multi-arch build list that includes both CDNA and
# gfx1201, so a list says nothing about this GPU and rocminfo decides instead.
profile_detect_arch() {
    local pinned="${PYTORCH_ROCM_ARCH:-}"
    if [[ "$pinned" =~ ^gfx[0-9a-f]+$ ]]; then
        printf '%s\n' "$pinned"
        return
    fi
    rocminfo 2>/dev/null | awk '
        /^ *Name: *gfx[0-9a-f]+/ && $2 != "gfx000" { print $2; exit }
    ' || true
}

profiler_priority_for_arch() {
    case "${1:-}" in
        gfx1201) printf '%s\n' "rocprofv3 rocprof metrix rocprof-compute omniperf" ;;
        *)       printf '%s\n' "rocprof-compute omniperf rocprofv3 rocprof metrix" ;;
    esac
}
