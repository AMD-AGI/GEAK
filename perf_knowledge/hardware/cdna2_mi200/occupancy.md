---
title: CDNA2 / MI250X / MI210 (gfx90a) — occupancy math
kind: hardware
gens: [gfx90a]
dtypes: []
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
---

# CDNA2 / MI250X / MI210 (gfx90a) — occupancy math

> Cross-gen model in [../shared/wavefront_simd_vgpr_agpr.md](../shared/wavefront_simd_vgpr_agpr.md).
> CDNA2 uses the **same 512-VGPR / 64-KiB-LDS / 8-wave** model as CDNA3 — so the MI300X occupancy
> formulas apply per GCD; the only differences are CU count (110/104 per GCD) and clock (1.7 GHz).

## TL;DR
> Per GCD, occupancy = `min(floor(512/N) waves/SIMD, floor(65536/L) wg/CU, 8 waves/SIMD)`. Identical
> to MI300X math. Register pressure is the #1 killer; attention/softmax are LDS-limited. Tune per GCD
> (110 CU on MI250X), not per OAM.

## Concepts

### Limits (per GCD)
```
occ_vgpr (waves/SIMD)    = floor(512 / N)            # cap 8
occ_lds  (workgroups/CU) = floor(65536 / L)          # 64 KiB LDS
wave_slots               = 8/SIMD = 32/CU            # hard cap
nW                       = ceil(threads_per_block / 64)
wg_from_vgpr             = floor(occ_vgpr * 4 / nW)  # 4 SIMDs/CU
wg_per_CU                = min(wg_from_vgpr, occ_lds, floor(32 / nW))
waves_per_CU             = wg_per_CU * nW
```
N = VGPRs/wave rounded up to a multiple of 16; L = LDS bytes/workgroup. Same worked examples as
[../cdna3_mi300/occupancy.md](../cdna3_mi300/occupancy.md) — the per-CU/per-SIMD resources are
identical between gfx90a and gfx942.

### Grid sizing (per GCD)
- MI250X: **110 CUs/GCD** → launch **≥ ~440–880 workgroups per GCD** (≥4–8/CU) to fill + hide tails.
  Across both GCDs of an OAM you launch on each device separately (two GPUs).
- MI210: single 104-CU GCD.
- No XCD/8-multiple rule (CDNA2 is not chiplet-XCD); just fill the GCD's CUs.

## The levers
1. **Cut VGPRs** (smaller tile, `waves_per_eu`, AGPR escape hatch) — same knobs as CDNA3.
2. **Cut LDS** when it binds (attention).
3. **Direct global→LDS** to free staging VGPRs.
4. **≥4 waves/CU** for HBM-latency hiding; GEMM can run low-occupancy + double-buffered.
5. **Size grids per GCD**, not per OAM.

## Pitfalls
- **Sizing a grid for "220 CUs"** — it's 2 separate 110-CU GPUs.
- **16-granule VGPR rounding** crossing an occupancy tier.
- **Ignoring the LDS limiter** for attention.

## Verify
- ISA `.vgpr_count`/`.lds_size`; `rocprof-compute` occupancy panel per GCD.

## Sources
- HIP Hardware implementation (wave slots, SIMD, VGPR/AGPR — gfx90a):
  https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- AMD CDNA2 White Paper (512 VGPR/SIMD, 64 KiB LDS, CU model):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
- AMD Instinct MI250 microarchitecture — ROCm Docs (110/104 CU per GCD, 1.7 GHz):
  https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
