---
title: CDNA1 / MI100 (gfx908) — architecture overview
kind: hardware
gens: [gfx908]
dtypes: [fp64, fp32, bf16, fp16, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi100.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-whitepaper.pdf
---

# CDNA1 / MI100 (gfx908) — architecture overview

> Target: **AMD Instinct MI100**, CDNA1, ISA **gfx908** — the **first** Matrix Core GPU. Mostly of
> historical/portability interest now; its constraints (256 VGPR, 10 waves, no FP64 matrix) explain
> why later-gen tile sizes must shrink when back-porting.

## TL;DR
> MI100 is the **first-gen Matrix Core** GPU: **120 CUs** (8 shader engines × 15), wave64, 4 SIMD/CU,
> **only 256 VGPR/CU** (half of CDNA2+), **10 waves/SIMD**, 32 GB HBM2 @ 1.23 TB/s, 1.5 GHz. It
> introduced **MFMA + AGPRs** but has **no FP64 matrix** and no FP8. Monolithic die (no GCD/XCD).

## The one-screen cheat sheet
| Fact | Value | Note |
|---|---|---|
| Compute Units | **120** (8 SE × 15 CU) | monolithic |
| Wavefront | 64 | |
| SIMD/CU | 4 (SIMD16, 4-cycle issue) | |
| Wave slots | **10/SIMD → 40/CU** | higher than CDNA2+ (8/32) |
| Peak clock | **1.5 GHz** | |
| Memory clock | 1.2 GHz | |
| VGPR | **256/CU** | half of CDNA2+ (512) |
| AGPR | ≤256 | **first gen** (with first Matrix Core) |
| SGPR | 800/CU (≤102/wave) | |
| LDS | 64 KiB/CU, 32 banks | |
| I-cache | 32 KiB | |
| HBM2 | **32 GB** (4 stacks) | 4096-bit, 1.228 TB/s |
| Matrix FP16 | 184.6 TF | first Matrix Core |
| Matrix BF16 | 92.3 TF (**half-rate**) | improved in CDNA2 |
| Matrix FP32 | 46.1 TF | |
| FP64 matrix | **none** | vector FP64 only (11.5 TF) |

## Concepts

### First-gen Matrix Core
MI100 introduced the **MFMA** instruction family and the **AGPR** register file (accumulators) in one
step. Supported matrix dtypes: **FP32, FP16, BF16, INT8** — **no FP64 matrix** (FP64 is vector-only at
11.5 TF) and **no FP8/FP6/FP4** (those start at CDNA3/CDNA4). BF16 runs at **half** the FP16 rate
(92.3 vs 184.6 TF) — a CDNA1 quirk fixed in CDNA2. See [matrix_core.md](matrix_core.md).

### CU resources (the back-port constraints)
- **256 VGPR/CU** (vs 512 on CDNA2+) → tile sizes that fit MI300X often **overflow** MI100; halve them.
- **10 waves/SIMD → 40/CU** (vs 8/32 later) → slightly more latency-hiding headroom per CU.
- 4 SIMD16 units, 4-cycle issue for a 64-lane wave; 64 KiB LDS (32 banks), 32 KiB L1.

### Topology
**Monolithic** single die (no GCD, no XCD) → no chiplet locality rules, no per-XCD L2, no 8-multiple
tile rule. Just fill the 120 CUs. 8 shader engines × 15 CUs.

### Memory
32 GB HBM2 @ 1.228 TB/s (4096-bit, 1.2 GHz). 64 KiB LDS, 32 KiB L1, L2 shared device-wide (single
die). Coalesce to 128 B; pad LDS off 32-bank conflicts — same CDNA family rules
([../shared/memory_model_lds_bank.md](../shared/memory_model_lds_bank.md)).

## The levers
1. **Shrink tiles** vs CDNA2+ (256 VGPR ceiling).
2. **Use MFMA** (FP16/BF16/INT8) + AGPR accumulators.
3. **Prefer FP16 over BF16** on MI100 (BF16 is half-rate here, unlike later gens).
4. **Fill 120 CUs** (no XCD/chiplet rules); ≥4 waves/CU for latency hiding.
5. **Coalesce to 128 B**, pad LDS.

## Pitfalls
- **Porting a 512-VGPR tile** to MI100 → spills/occupancy collapse.
- **Assuming FP64 matrix or FP8** — neither exists on CDNA1.
- **Assuming BF16 == FP16 rate** — BF16 is half on CDNA1.

## Verify
- `rocminfo` → single gfx908 device, 120 CU.
- ISA `.vgpr_count` against the 256 ceiling; `rocprof-compute` occupancy.

## Sources
- AMD Instinct MI100 microarchitecture — ROCm Docs (120 CU, 256 VGPR, 10 waves, peak FP16/BF16/FP32,
  HBM2 32 GB/1.228 TB/s, clocks): https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi100.html
- AMD CDNA (1) White Paper (first Matrix Core, AGPR, CU model):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-whitepaper.pdf
