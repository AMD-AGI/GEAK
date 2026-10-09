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
    local identity_script="${GPU_IDENTITY_SCRIPT:-${SCRIPT_DIR:-.}/../../scripts/gpu_identity.py}"
    local identity
    if [ -f "$identity_script" ]; then
        if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || [ -z "${GPU_ID:-}" ]; then
            identity="$(python3 "$identity_script")" || return 1
        else
            identity="$(ROCR_VISIBLE_DEVICES="$GPU_ID" python3 "$identity_script")" || return 1
        fi
        python3 -c 'import json,sys; print(json.loads(sys.argv[1])["gfx"])' "$identity"
        return
    fi
    # Standalone/vendor fallback: return an architecture only for a homogeneous
    # visible set. Never select the first agent on a mixed host.
    rocminfo 2>/dev/null | awk '
        /^ *Name: *gfx[0-9a-f]+/ && $2 != "gfx000" { seen[$2] = 1 }
        END {
            for (gfx in seen) { count += 1; only = gfx }
            if (count == 1) print only
            else if (count > 1) exit 1
        }
    '
}

profiler_priority_for_arch() {
    case "${1:-}" in
        gfx1201) printf '%s\n' "rocprofv3 rocprof metrix rocprof-compute omniperf" ;;
        *)       printf '%s\n' "rocprof-compute omniperf rocprofv3 rocprof metrix" ;;
    esac
}

# Optional counter layer (profile_kernel.sh --pmc / --derived). The default counter groups come from
# kernel_tools/parse_pmc.py PMC_GROUPS and use CDNA (gfx9: gfx950 first, gfx942) counter names --
# MfmaUtil, TCC_*, SQ_* -- which do not exist on RDNA (gfx1201 has WMMA and GL2C counters). An
# unresolvable name fails the WHOLE rocprofv3 pass, so off gfx9 the default groups are skipped and
# the layer is reported DEGRADED; list what the build exposes with `rocprofv3 -L` or
# `rocprofv3-avail list --pmc` and pass explicit groups via PROFILE_PMC_GROUPS. An unknown/empty
# arch is refused rather than defaulted.
profile_pmc_default_groups_ok() {
    case "${1:-}" in
        gfx9*) return 0 ;;
        *)     return 1 ;;
    esac
}

# Groups each optional mode collects (names are parse_pmc.PMC_GROUPS keys).
profile_pmc_groups_for_mode() {
    case "${1:-}" in
        pmc)     printf '%s\n' "memory memory_ea sol stall waitbusy lds_raw" ;;
        derived) printf '%s\n' "sol stall" ;;
        *)       return 1 ;;
    esac
}
