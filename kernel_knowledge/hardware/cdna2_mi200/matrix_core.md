---
title: CDNA2 / MI250X / MI210 (gfx90a) — Matrix Core / MFMA
kind: hardware
gens: [gfx90a]
dtypes: [fp64, fp32, bf16, fp16, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
---

# CDNA2 / MI250X / MI210 (gfx90a) — Matrix Core / MFMA

> Cross-gen concepts in [../shared/matrix_core_mfma_smfmac.md](../shared/matrix_core_mfma_smfmac.md).
> This file is the CDNA2-specific facts: the **full-rate FP64 matrix** headline and supported MFMA.

## TL;DR
> CDNA2's headline is **full-rate FP64 matrix** (HPC) — MI250X reaches **90.5 TF FP64 matrix** per OAM,
> 2× the FP64 vector rate, with `v_mfma_f64_*` instructions. For AI: FP16/BF16/INT8 MFMA at
> **362.1 TF/TOPS** per OAM. **No FP8/FP6/FP4** and **no SMFMAC** (CDNA3+ only).

## Concepts

### Supported MFMA dtypes
| Dtype | CDNA2 (gfx90a) | Notes |
|---|---|---|
| FP64 | ✓ **full-rate matrix** | the CDNA2 differentiator; `v_mfma_f64_16x16x4_f64` |
| FP32 | ✓ | matrix + vector |
| FP16 | ✓ | FP32 accumulate |
| BF16 | ✓ | FP32 accumulate (CDNA2 added improved BF16 throughput vs CDNA1's half-rate) |
| INT8 | ✓ | INT32 accumulate |
| FP8 / FP6 / FP4 | ✗ | CDNA3 (FP8 FNUZ) / CDNA4 (FP6/FP4) only |
| TF32 | ✗ | CDNA3 only (then removed in CDNA4) |
| SMFMAC (sparse) | ✗ | CDNA3+ |

Instruction naming and lane mapping follow the cross-gen scheme
([../shared/matrix_core_mfma_smfmac.md](../shared/matrix_core_mfma_smfmac.md)): `v_mfma_f32_16x16x16_f16`,
`v_mfma_f32_32x32x8_bf16`, `v_mfma_i32_*_i8`, plus the FP64 family.

### Peak (per OAM = both GCDs)
| Computation | MI250X peak |
|---|---|
| Matrix FP64 | 90.5 TFLOP/s |
| Vector FP64 | 45.3 TFLOP/s |
| Matrix FP32 / Packed FP32 | 90.5 TFLOP/s |
| Vector FP32 | 45.3 TFLOP/s |
| Matrix FP16 / BF16 / INT8 | 362.1 TF / TOPS |

Per single GCD: FP64 matrix 45.3 TF, FP64 vector 22.6 TF (half the OAM figures). Engine clock 1.7 GHz.

### Numerics
- FP32 accumulate for FP16/BF16/INT8 (INT32 for INT8). Keep accumulators wide through the K-loop.
- **Subnormal flush**: CDNA2 flushes some subnormals (a training-stability concern that CDNA3 fixed by
  fully supporting subnormals). For training, validate or use the documented workaround. See
  [../shared/dtype_numerics.md](../shared/dtype_numerics.md).

## The levers
1. **Use FP64 MFMA** for HPC double-precision matmul — CDNA2's standout capability.
2. **FP16/BF16/INT8 MFMA** for AI; accumulate in FP32/INT32.
3. **16×16 over 32×32** for register footprint (same rationale as later gens).
4. **AGPR accumulators** (≤256) for large tiles.
5. **Guard subnormals** in training.

## Pitfalls
- **Expecting FP8/FP6/FP4 or SMFMAC** — not on CDNA2.
- **Subnormal flush** silently hurting training accuracy.
- **Down-converting the accumulator** in the K-loop.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna2 --list-instructions` for the gfx90a MFMA set.
- `--detail-instruction` for cycles / FLOPs / GPR usage of any instruction.

## Sources
- AMD Instinct MI250 microarchitecture — ROCm Docs (peak FP64/FP32/FP16/BF16/INT8 per OAM):
  https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
- AMD matrix cores — GPUOpen lab notes (MFMA mapping/intrinsics across CDNA):
  https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
- AMD CDNA2 White Paper (full-rate FP64 matrix, subnormal behavior):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
