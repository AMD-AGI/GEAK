---
title: CDNA1 / MI100 (gfx908) — Matrix Core / MFMA (first gen)
kind: hardware
gens: [gfx908]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi100.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-whitepaper.pdf
---

# CDNA1 / MI100 (gfx908) — Matrix Core / MFMA (first gen)

> Cross-gen concepts in [../shared/matrix_core_mfma_smfmac.md](../shared/matrix_core_mfma_smfmac.md).
> This file is the **first-generation** Matrix Core: what it has and (mostly) what it lacks.

## TL;DR
> MI100 is where the **MFMA** instruction family and the **AGPR** accumulator file debuted. It supports
> **FP32 / FP16 / BF16 / INT8** matmul with FP32/INT32 accumulate — but **BF16 is half-rate** (92.3 vs
> 184.6 TF FP16), there is **no FP64 matrix**, and **no FP8/FP6/FP4/MXFP or SMFMAC**. Wave64,
> 4 Matrix Cores/CU.

## Concepts

### Supported MFMA dtypes
| Dtype | CDNA1 (gfx908) | Notes |
|---|---|---|
| FP32 | ✓ matrix | 46.1 TF |
| FP16 | ✓ | 184.6 TF, FP32 accumulate |
| BF16 | ✓ **half-rate** | 92.3 TF (fixed to full-rate in CDNA2) |
| INT8 | ✓ | INT32 accumulate |
| FP64 matrix | ✗ | vector FP64 only (11.5 TF); FP64 matrix arrives in CDNA2 |
| FP8 / FP6 / FP4 / MXFP | ✗ | CDNA3 (FP8 FNUZ) / CDNA4 |
| TF32 | ✗ | CDNA3 only |
| SMFMAC (sparse) | ✗ | CDNA3+ |

Instruction naming and lane mapping follow the cross-gen scheme: `v_mfma_f32_16x16x16_f16`,
`v_mfma_f32_32x32x8_bf16` (half-rate here), `v_mfma_i32_*_i8`. AGPRs hold the FP32/INT32 accumulators
(read out via `v_accvgpr_read_b32` in the epilogue) — the first GPU to offer this.

### Per-CU FLOPs/clock & OAM peak
| Computation | FLOPs/clock/CU | MI100 peak (120 CU @ 1.5 GHz) |
|---|---|---|
| Vector FP64 | 64 | 11.5 TFLOP/s |
| Vector FP32 | 128 | 23.1 TFLOP/s |
| Matrix FP32 | 256 | 46.1 TFLOP/s |
| Matrix BF16 | 512 | 92.3 TFLOP/s (half FP16) |
| Matrix FP16 | 1024 | 184.6 TFLOP/s |

### Numerics
- FP32 accumulate for FP16/BF16; INT32 for INT8. Keep accumulators wide through the K-loop.
- BF16's half-rate is a hardware property of CDNA1 — on MI100 **prefer FP16** when accuracy allows, the
  opposite of the BF16-first guidance for CDNA4 (where TF32 was dropped in favor of BF16).

## The levers
1. **Use MFMA** (FP16/BF16/INT8) — first-gen, but mandatory for competitive matmul.
2. **Prefer FP16 over BF16** (BF16 half-rate on CDNA1).
3. **AGPR accumulators** (≤256) — but mind the 256-VGPR CU ceiling
   ([arch.md](arch.md)); shrink tiles vs later gens.
4. **16×16 over 32×32** for register footprint.

## Pitfalls
- **Expecting FP64 matrix / FP8 / SMFMAC** — none on CDNA1.
- **Assuming BF16 == FP16 throughput** — half on MI100.
- **256-VGPR overflow** when porting later-gen tiles.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna --list-instructions` (gfx908 MFMA set).
- `--detail-instruction` for cycles/FLOPs/GPR usage.

## Sources
- AMD Instinct MI100 microarchitecture — ROCm Docs (FLOPs/clock, FP16/BF16/FP32 peaks, 120 CU,
  1.5 GHz): https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi100.html
- AMD matrix cores — GPUOpen lab notes (MFMA across CDNA, first-gen mapping):
  https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
- AMD CDNA (1) White Paper (first Matrix Core + AGPR, no FP64 matrix):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-whitepaper.pdf
