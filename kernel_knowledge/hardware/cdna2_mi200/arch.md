---
title: CDNA2 / MI250X / MI210 (gfx90a) — architecture overview
kind: hardware
gens: [gfx90a]
dtypes: [fp64, fp32, bf16, fp16, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi250x-datasheet.pdf
---

# CDNA2 / MI250X / MI210 (gfx90a) — architecture overview

> Target: **AMD Instinct MI250 / MI250X / MI210**, CDNA2, ISA **gfx90a**. The key structural fact: the
> MI250(X) OAM is **two GCDs = two separate GPU devices** to software. MI210 is a single GCD.

## TL;DR
> CDNA2 is **multi-die but not unified**: an MI250X package has **2 GCDs × 110 CUs**, each a separate
> device, linked by a 400 GB/s bidirectional GCD↔GCD bridge. Per GCD: wave64, 4 SIMD/CU, **512 VGPR**,
> ≤256 AGPR, 64 KiB LDS (32 banks). No FP8/FP6/FP4 and no chiplet-XCD scheduler — treat each GCD as a
> standalone gfx90a GPU. **Watch subnormal flush** (CDNA2 flushes some, hurting training).

## The one-screen cheat sheet (per GCD)
| Fact | MI250X (per GCD) | MI210 |
|---|---|---|
| Devices per package | **2 GCDs** (separate) | 1 GCD |
| CUs | **110 active** (MI250X) / 104 (MI250) | 104 |
| Wavefront | 64 | 64 |
| SIMD/CU | 4 | 4 |
| Wave slots | 8/SIMD → 32/CU | same |
| Peak clock | **1.7 GHz** | 1.7 GHz |
| VGPR | **512 ×4 B/SIMD** (doubled vs CDNA1) | same |
| AGPR | ≤256 | same |
| LDS | 64 KiB/CU, 32 banks | same |
| L2 | ~8 MiB per GCD | per GCD |
| HBM2e | 64 GB per GCD (128 GB/OAM) | 64 GB |
| HBM2e BW | **1.6 TB/s per GCD** (3.2 TB/s OAM) | 1.6 TB/s |
| GCD↔GCD | 400 GB/s bidir (200/dir) | n/a |
| TDP | 500–560 W (OAM) | 300 W |

## Concepts

### Two GCDs, two devices
The MI250(X) OAM hosts **two Graphics Compute Dies (GCDs)**, each "one GPU device." Software sees **two
GPUs**, not one — `hipGetDeviceCount` returns 2 per OAM. Data shared across the two GCDs crosses a
**400 GB/s bidirectional** (200 GB/s/dir) Infinity Fabric bridge, **not** on-die bandwidth. This is the
opposite of CDNA3's single-logical-GPU chiplet model: there is **no unified address space across the
two GCDs by default**, and no XCD hardware scheduler. Multi-GCD work is a multi-GPU problem (NCCL/RCCL
or explicit peer copies). See [../shared/hbm_infinity_fabric.md](../shared/hbm_infinity_fabric.md).

### Compute Unit
Each gfx90a CU: 4 × SIMD64, Matrix Cores (CDNA2, FP64-capable), **512 VGPR/SIMD** (doubled from
CDNA1's 256), ≤256 AGPR, 64 KiB LDS (32 banks), 32 KiB L1. CDNA2's headline was **full-rate FP64
matrix** (HPC focus) — see [matrix_core.md](matrix_core.md).

### Memory
Per GCD: 64 GB HBM2e @ 1.6 TB/s, ~8 MiB L2. Bank/coalescing rules match the 32-bank,
128 B-line CDNA family — see [../shared/memory_model_lds_bank.md](../shared/memory_model_lds_bank.md)
and [memory.md](memory.md).

## The levers
1. **Treat each GCD as a standalone GPU**; place data to avoid the 400 GB/s GCD↔GCD bridge.
2. **Use MFMA** (FP16/BF16/INT8 + full-rate FP64) — [matrix_core.md](matrix_core.md).
3. **512 VGPR / ≤256 AGPR** occupancy model (same as CDNA3) — [occupancy.md](occupancy.md).
4. **Guard subnormals** if training: CDNA2 flushes some, which hurt stability (fixed in CDNA3).
5. **Coalesce to 128 B**, pad LDS off 32-bank conflicts.

## Pitfalls
- **Treating an MI250X as one 220-CU GPU** — it's two 110-CU devices; cross-GCD is a 400 GB/s link.
- **Expecting FP8/FP6/FP4** — CDNA2 has none (FP16/BF16/INT8/FP32/FP64 only).
- **Ignoring subnormal flush** in training.

## Verify
- `rocminfo` shows **two gfx90a devices** per MI250X OAM; `rocm-smi` lists both.
- `rocprof-compute` for per-GCD occupancy/BW.

## Sources
- AMD Instinct MI250 microarchitecture — ROCm Docs (2 GCDs, 104/110 CU, 1.7 GHz, 1.6/3.2 TB/s, GCD↔GCD
  400 GB/s, peak FLOPS): https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
- AMD Instinct MI250X Data Sheet (HBM2e capacity/BW, TDP):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi250x-datasheet.pdf
- AMD CDNA2 White Paper (512 VGPR, full-rate FP64 matrix, subnormal behavior):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
