# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared GPU architecture detection via rocminfo."""

import functools
import subprocess


@functools.lru_cache(maxsize=1)
def detect_gpu_arch() -> str:
    """Return the GFX architecture string (e.g. 'gfx942', 'gfx1201') or '' on failure.

    Cached: ``rocminfo`` is invoked at most once per process. The arch of the
    host GPU does not change at runtime, so caching is safe and avoids
    spawning rocminfo for every prompt-injection / harness-compile call.
    """
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "gfx" in line.lower() and "name:" in line.lower():
                for p in line.split():
                    if p.startswith("gfx"):
                        return p
    except Exception:
        pass
    return ""


def is_rdna(arch: str) -> bool:
    """True if *arch* is an RDNA architecture (gfx10xx, gfx11xx, gfx12xx)."""
    return arch.startswith(("gfx10", "gfx11", "gfx12"))


def is_wmma_capable(arch: str) -> bool:
    """True if *arch* supports WMMA (Wave Matrix Multiply-Accumulate).

    WMMA was introduced on RDNA3 (gfx11) and is also present on RDNA4
    (gfx12). RDNA1/RDNA2 (gfx10xx) do not have WMMA, so prompts that
    suggest WMMA reformulations should be gated on this check rather
    than on ``is_rdna``.
    """
    return arch.startswith(("gfx11", "gfx12"))


def rdna_arch_context(gpu_info: dict, arch: str) -> list[str] | None:
    """Return RDNA-specific GPU context lines, or None if arch is not RDNA."""
    if not is_rdna(arch):
        return None
    name = gpu_info.get("name", gpu_info.get("model", "AMD GPU"))
    cus = gpu_info.get("compute_units", "?")
    hbm_bw = gpu_info.get("peak_hbm_bandwidth_gbps", gpu_info.get("hbm_bandwidth", "?"))
    lds_per_cu = gpu_info.get("lds_per_cu_kb", 64)
    vgprs = gpu_info.get("vgprs_per_cu", 512)
    wave_size = gpu_info.get("wavefront_size", 32)
    lines = [
        f"## GPU Architecture: {name} ({arch})",
        f"- Architecture: {arch} (RDNA)",
        f"- Compute Units: {cus}",
        f"- Peak VRAM bandwidth: {hbm_bw} GB/s",
        f"- LDS per CU: {lds_per_cu} KB",
        f"- VGPRs per CU: {vgprs}",
        f"- Wavefront size: {wave_size} (RDNA default 32, supports 64)",
    ]
    if is_wmma_capable(arch):
        lines.append("- WMMA (Wave Matrix Multiply-Accumulate) instructions for matrix math (RDNA3+)")
    lines.extend(
        [
            "- Use these specs to guide your kernel optimizations (tile sizes, occupancy, LDS usage).",
            "",
        ]
    )
    return lines


_RDNA_COMPUTE_BOUND_GUIDANCE = (
    "## Optimization Guidance (bottleneck: compute-bound)\n"
    "The kernel is limited by arithmetic throughput. Focus on kernel-body changes:\n"
    "1. REDUCE INSTRUCTION COUNT: Simplify expressions, use hardware intrinsics "
    "(tl.math.rsqrt, fma), eliminate redundant computations.\n"
    "2. USE WMMA INSTRUCTIONS: On RDNA GPUs, restructure computation to use Wave "
    "Matrix Multiply-Accumulate for dense linear algebra.\n"
    "3. STRENGTH REDUCTION: Replace expensive ops (div, mod, pow) with cheaper "
    "equivalents (shifts, masks, lookup tables).\n"
    "4. LOOP UNROLLING: Manually unroll inner loops to help the compiler schedule "
    "instructions more aggressively.\n"
    "5. ALGORITHM CHANGE: Switch to an algorithm with lower computational complexity "
    "(e.g., O(n log n) vs O(n^2), approximate methods).\n"
)


def rdna_compute_bound_guidance() -> str:
    """Return RDNA-specific compute-bound guidance (WMMA instead of MFMA)."""
    return _RDNA_COMPUTE_BOUND_GUIDANCE


def hipcc_offload_arch_flags() -> list[str]:
    """Return ['--offload-arch=<gfx>'] on RDNA, else [].

    Only emitted on RDNA where hipcc's default arch detection can pick the
    wrong target on multi-GPU hosts (gfx1151 vs gfx1201 etc.). On CDNA we
    intentionally leave the compile command unchanged from pre-PR behavior
    and rely on hipcc's default agent enumeration.
    """
    arch = detect_gpu_arch()
    if arch and is_rdna(arch):
        return [f"--offload-arch={arch}"]
    return []


def guard_rocprof_compute(backend: str) -> tuple[str, str]:
    """If *backend* is 'rocprof-compute' on RDNA, return ('metrix', arch). Otherwise (backend, '')."""
    if backend != "rocprof-compute":
        return backend, ""
    arch = detect_gpu_arch()
    if is_rdna(arch):
        return "metrix", arch
    return backend, ""
