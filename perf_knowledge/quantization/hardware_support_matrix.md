---
title: Hardware support matrix — CDNA1–4 MFMA dtype support + rate
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
dtypes: [fp16, bf16, fp8_e4m3_fnuz, fp8_e4m3, fp6_e2m3, fp6_e3m2, fp4_e2m1, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
  - https://rocm.docs.amd.com/en/latest/reference/precision-support.html
  - https://www.koicomputers.com/wp-content/uploads/2025/08/amd-instinct-mi350x-gpu-datasheet.pdf
---

# Hardware support matrix — CDNA1–4 MFMA dtypes

> **TL;DR.** What the **matrix core (MFMA)** actually accelerates per generation. FP8 is **CDNA3+**;
> FP4/FP6 + block-scaled MFMA are **CDNA4-only**; INT8 is everywhere; BF16 native from CDNA2. Anything
> marked ✗ either runs in a higher precision or is *software-simulated* with no speedup. Peak detail:
> [[hardware/cdna4_mi350]] (peak_tables); dialect: [[fnuz_vs_ocp.md]]; MX: [[block_scaling_mxfp.md]].

## Support × generation
| dtype | CDNA1 MI100 (gfx906) | CDNA2 MI200 (gfx90a) | CDNA3 MI300/325 (gfx942) | CDNA4 MI350/355 (gfx950) |
|---|---|---|---|---|
| **FP16** | ✓ MFMA | ✓ MFMA | ✓ MFMA | ✓ MFMA (**2× CDNA3**) |
| **BF16** | ✗ (emulated) | ✓ MFMA | ✓ MFMA | ✓ MFMA (**2× CDNA3**) |
| **FP8 E4M3** | ✗ | ✗ | ✓ MFMA **FNUZ** | ✓ MFMA **OCP** (+FNUZ) |
| **FP8 E5M2** | ✗ | ✗ | ✓ MFMA FNUZ | ✓ MFMA OCP |
| **FP6 (E2M3/E3M2)** | ✗ | ✗ | ✗ (no HW) | ✓ MFMA (**FP4 rate**) |
| **FP4 (E2M1)** | ✗ | ✗ | ✗ (no HW) | ✓ MFMA |
| **MXFP4/6/8 (block-scaled)** | ✗ | ✗ | ✗ (sim only) | ✓ `v_mfma_scale_*` (E8M0) |
| **INT8** | ✓ MFMA | ✓ MFMA | ✓ MFMA | ✓ MFMA (**2× CDNA3**) |

Notes: CDNA1 lacks native BF16 (FP16/INT8 are its matrix dtypes); FP8 first appears on CDNA3 in the
**FNUZ** dialect; FP4/FP6 and the **block-scaled MFMA** are introduced on CDNA4. INT4 has no dedicated
MFMA path on any gen — INT4 weights are dequantized to fp16/fp8 for the GEMM
([[formats_overview.md]], [[calibration_and_quark.md]]).

## Relative MFMA rate (matrix core, vs FP16=1×)
| dtype | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|
| FP16 / BF16 | 1× | 1× |
| FP8 | 2× | 2× |
| FP6 | — (no HW) | **4×** (= FP4) |
| FP4 | — (no HW) | **4×** |
| INT8 | 2× | 2× |
On CDNA4, **FP6 and FP4 share the data path → both 4× FP16** (FP6 marginally slower in practice from
power limits). This is why FP6 is "free accuracy" over FP4 ([[block_scaling_mxfp.md]]).

## CDNA4 absolute dense peaks (MI355X, per OAM)
| dtype | dense peak | vs FP32 |
|---|---|---|
| FP16 / BF16 | **2.5 PFLOP/s** | 16× |
| FP8 (OCP) | **5 PFLOP/s** | 32× |
| FP6 | **10 PFLOP/s** | 64× |
| FP4 | **10 PFLOP/s** | 64× |
| INT8 | **~5 POPS** | 32× |
| FP32 matrix | 157 TFLOP/s | 1× |
Marketing figures of FP8 10 PF / FP6+FP4 20 PF / FP16 5 PF include **2:4 structured sparsity** (2×).
**TF32 is removed on CDNA4.** Full table + memory roofline: [[hardware/cdna4_mi350]] (peak_tables).
All theoretical peak — MI300X sustains ~45%; quote *achieved* ([[index/conventions.md]]).

## Strategy implications
- **CDNA1/2**: no FP8 HW → INT8 w8a8 or stay bf16/fp16 for compute wins; quant is footprint-only.
- **CDNA3**: FP8 is the throughput lever (2×); FP4/FP6 are **sim-only** (footprint + forward-compat,
  no speedup); FP4BMM crashes gfx942 (`VLLM_ROCM_USE_AITER_FP4BMM=0`, [[block_scaling_mxfp.md]]).
- **CDNA4**: FP8 (2×) and MXFP4/MXFP6 (4×) are real compute wins; prefer MXFP6/mixed when MXFP4
  degrades since the rate is identical.

## Pitfalls
- **Expecting FP4/FP6 speedup on MI300** — none; HW absent.
- **Comparing dense vs sparsity peaks** — AMD's 20 PF FP4 includes 2:4 sparsity; dense is ~10 PF.
- **Forgetting the FP8 dialect differs by gen** even though both CDNA3/4 "support FP8"
  ([[fnuz_vs_ocp.md]]).
- **Quoting peak as achievable** — use measured, tagged numbers.

## Verify
- Confirm a dtype path is HW-accelerated (not silently upcast) via ISA dump / the matrix instruction
  calculator; benchmark vs the FP16 baseline to confirm the expected multiple ([[profiling]]).

## Sources
- Matrix Core CDNA3/4 (per-gen dtype MFMA support, FP6=FP4 rate, FP8 dialect): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- CDNA4 white paper (FP16 2.5 PF, FP8 5 PF, FP6/FP4 10 PF dense): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
- ROCm precision support (per-arch dtype table): https://rocm.docs.amd.com/en/latest/reference/precision-support.html
- MI350X datasheet (dense vs sparsity peaks): https://www.koicomputers.com/wp-content/uploads/2025/08/amd-instinct-mi350x-gpu-datasheet.pdf
