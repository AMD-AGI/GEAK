---
title: CDNA3 / MI300X (gfx942) — occupancy math & tuning
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-10-07
sources:
  - perf_knowledge/hardware/data/hw_constants.json
  - https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://rocm.github.io/rocprofiler-compute/performance_model.html
---

# CDNA3 / MI300X (gfx942) — occupancy math & tuning

> Cross-gen model in [../shared/wavefront_simd_vgpr_agpr.md](../shared/wavefront_simd_vgpr_agpr.md).
> This file is the MI300X numbers and worked examples.

## TL;DR
> The static occupancy bound combines register, LDS and wave-slot limits after
> converting them to the same units using the actual launch. The register cap is
> **8 waves/SIMD** (32/CU); LDS is counted in workgroups/CU. Measured occupancy may
> be lower. A pipelined GEMM can perform well with few resident workgroups.

## Concepts

### Inputs
1. **N** = combined register count. AMDHSA `.vgpr_count` and
   `.amdhsa_next_free_vgpr` already include AGPRs; do not add `.agpr_count` again.
   For separate LLVM `NumVgprs` / `NumAgprs`, use `alignTo(arch, 4) + agpr`
   when AGPRs are used. Round the result up to 8 (`vgpr_alloc_granule` in
   [`../data/hw_constants.json`](../data/hw_constants.json), compiler-derived).
2. **L** = LDS bytes/workgroup.
3. **nW** = waves/workgroup = `ceil(actual_threads_per_block / 64)`.
   `.max_flat_workgroup_size` is a compile-time bound, not the observed launch.

### Static bounds (MI300X; balanced waves across four SIMDs)
```
occ_vgpr (waves/SIMD)    = floor(512 / roundup8(N))  # cap 8; unified ArchVGPR+AGPR file
occ_lds  (workgroups/CU) = floor(65536 / L)          # 64 KiB LDS
wave_slots               = 8/SIMD = 32/CU            # hard cap
wg_from_vgpr             = floor(occ_vgpr * 4 / nW)  # 4 SIMDs/CU
wg_per_CU                = min(wg_from_vgpr, occ_lds, floor(32 / nW))
waves_per_CU             = wg_per_CU * nW
```

### Worked examples
**A — VGPR-limited GEMM.** N=176, threads=256 (nW=4), L=32 KiB.
```
occ_vgpr=floor(512/176)=2 ; wg_from_vgpr=floor(2*4/4)=2 ; occ_lds=floor(65536/32768)=2
wg_per_CU=min(2,2,8)=2 -> 8 waves/CU
```
Dropping N to 128 gives occ_vgpr=4 but `occ_lds=2` now binds → still 2 wg/CU. Reduce LDS to gain.

**B — LDS-limited attention/softmax.** N=64, threads=512 (nW=8), L=48 KiB.
```
occ_vgpr=floor(512/64)=8 ; wg_from_vgpr=floor(8*4/8)=4 ; occ_lds=floor(65536/49152)=1  <-- binds
wg_per_CU=min(4,1,4)=1 -> 8 waves/CU
```
Halving LDS → occ_lds=2 → 2 wg/CU → 16 waves/CU.

**C — fully occupied bandwidth kernel.** N=48, threads=256 (nW=4), L=8 KiB.
```
occ_vgpr=floor(512/48)=10->cap 8 ; wg_from_vgpr=floor(8*4/4)=8 ; occ_lds=floor(65536/8192)=8
wave-slot cap floor(32/4)=8 -> wg_per_CU=8 -> 32 waves/CU (max)
```

### The 8-VGPR granule trap
170 VGPRs rounds to 176 → 2 waves/SIMD, where 168 gives 3. The tier boundaries — the last combined
VGPR count that still fits each wave count — are 64/72/80/96/128/168/256 → 8/7/6/5/4/3/2 waves
(`vgpr_wave_steps` in [`../data/hw_constants.json`](../data/hw_constants.json), measured from LLVM's
`; Occupancy:` output; `kernel_workflow/scripts/kernel_tools/amd_occupancy.py --vgpr N --arch gfx942`
computes it). Shaving a few VGPRs across a boundary can jump an occupancy tier. A 16-VGPR granule
(which some guides state) has no 7-wave tier and puts 168 at 2 waves — it does not match the compiler.

## The levers
1. **Cut VGPRs** (smaller tile, `waves_per_eu`, **AGPR escape hatch**
   `-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256` — AGPRs come out of the same
   512-entry file, so they lift the per-wave ceiling past 256 arch VGPRs without spilling but do not
   lower the occupancy count).
2. **Cut LDS** (less buffering, packing) when LDS binds (attention).
3. **`buffer_load_to_lds`** removes staging VGPRs — biggest GEMM occupancy win (see
   [memory_hierarchy.md](memory_hierarchy.md)).
4. **`__launch_bounds__(threads, waves_per_eu)`** to cap VGPRs to a target.
5. **≥4 waves/CU** for HBM-latency hiding; for MFMA GEMM accept 1–2 wg/CU + double-buffered LDS.

## Tuning checklist (from Omniperf symptoms)
| Symptom | Cause | Fix |
|---|---|---|
| Low waves/CU, high VGPR | register pressure | shrink tile, waves_per_eu, AGPR, buffer_load_to_lds |
| Low waves/CU, high LDS | LDS pressure | smaller tiles, less buffering, pack |
| LDS bank-conflict stalls | column access / stride mult of 32 | pad LD (+1/+4), swizzle, ds_*_b128 |
| Low HBM BW | uncoalesced/scalar loads | global_load_dwordx4, 128 B align |
| Memory-latency-bound | too few waves | raise to ≥4 waves/CU; prefetch |
| MFMA underfed | LDS read conflicts / no double-buffer | swizzled LDS, async copy + double buffer |

**Tuned FP16/FP8 GEMM target (MI300X):** 256×256 tile, 2-stage prefetch, **384–448 VGPR** budget,
≥4 waves/CU, `mfma_16x16`, `buffer_load_to_lds`, conflict-free swizzled LDS.

## Pitfalls
- **Maximizing occupancy blindly** — GEMM often wins at low occupancy with deep prefetch.
- **Ignoring the LDS limiter** for attention.
- **Forgetting 8-granule rounding**, or counting arch VGPRs only — AGPRs count toward the same budget.

## Verify
- ISA `.vgpr_count`/`.lds_size`; on-box `occ.sh` (ROCm workload guide) → waves/CU.
- `rocprof-compute` occupancy panel: resident vs theoretical waves, which resource binds.

## Sources
- GEAK hardware constants (`vgpr_alloc_granule` = 8, `vgpr_wave_steps`, compiler-derived on ROCm 7.2.1 /
  LLVM 22; the single source for the granule): `perf_knowledge/hardware/data/hw_constants.json`
- LLVM `AMDGPUBaseInfo.cpp` — `getVGPRAllocGranule` returns 8 for every `FeatureGFX90AInsts` target and
  `getTotalNumVGPRs` = `alignTo(ArchVGPR, 4) + AGPR`:
  https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
- ROCm MI300X workload optimization (512 VGPR/EU, occ.sh, waves_per_eu; its 16-granule statement is
  superseded by the compiler-derived granule above):
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- HIP Hardware implementation (wave slots, SIMD, VGPR/AGPR):
  https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- rocprofiler-compute performance model (occupancy):
  https://rocm.github.io/rocprofiler-compute/performance_model.html
- Optimizing Triton kernels on MI300X (VGPR rounding, LDS padding, occupancy):
  https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
