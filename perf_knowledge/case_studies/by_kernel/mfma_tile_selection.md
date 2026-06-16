---
title: MFMA tile selection — 16×16 vs 32×32 on MI300X GEMM
kind: case_study
operator: dense_gemm
backend: triton_amd
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, training, both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
---

# MFMA tile selection: 16×16 vs 32×32 on MI300X GEMM

> The recommendation here is **vendor-sourced** (AMD CDNA3 ISA reference + ROCm matrix-cores blog
> + the MI300X workload optimization guide), labelled as such. The actionable rule is a *default
> with an A/B escape hatch* — confirm per shape on your box.

## Context
The CDNA3 matrix cores execute `v_mfma_*` instructions: `D = A·B + C`, reading A/B from VGPRs and
accumulating `C` in **AGPRs**. For bf16/fp16 GEMM the two common shapes are **`mfma_16x16x16`**
and **`mfma_32x32x8`** (fp8 variants pack 2× K). The choice between them is a real GEMM tuning
knob (triton `matrix_instr_nonkdim ∈ {16,32}`) that moves throughput on MI300X. Background:
[`../../optimization/mfma_scheduling.md`](../../optimization/mfma_scheduling.md),
[`../../hardware/cdna3_mi300/matrix_core.md`](../../hardware/cdna3_mi300/matrix_core.md),
[`../../operators/dense_gemm/tuning.md`](../../operators/dense_gemm/tuning.md).

## Baseline / the trade-off
| | `mfma_16x16x16` | `mfma_32x32x8` |
|---|---|---|
| AGPR (accumulator) pressure | **lower** | higher |
| occupancy (waves that fit) | **higher** | lower |
| register-tile flexibility | finer (good for skinny N/K) | coarser |
| best for | **most LLM GEMMs on MI300X** (attention/MLP N/K) | very large square tiles that amortize the larger op |

A `32×32` instruction's accumulator is larger, so it eats more of the **512-register** budget and
drops occupancy. On the N/K shapes of LLM attention/MLP GEMMs, the occupancy loss usually
outweighs the per-op amortization.

## What works / what doesn't
- **Default to 16×16** (`matrix_instr_nonkdim=16`) on MI300X — the validated vendor default for
  hand and library GEMM. This is the SOTA default recorded in
  [`../../operators/dense_gemm/tuning.md`](../../operators/dense_gemm/tuning.md).
- **Switch to 32×32 only if a tune shows it wins** for a specific large square shape — keep the
  A/B escape hatch; don't assume.
- **The CUDA-habit anti-pattern:** defaulting to 32×32 (the NVIDIA mental model) *lowers* MI300X
  throughput via occupancy loss. This is the single most common MFMA-selection mistake.
- **Pair the tile choice with the epilogue fix:** the default CShuffle epilogue routes the
  accumulator→C write through a **512-byte Tagram** staging path that serializes on write-heavy
  epilogues. **`OPTIMIZE_EPILOGUE=1`** bypasses it — a standard MI300X default that compounds
  with the tile choice. See [`../../operators/gemm_epilogue_fused/overview.md`](../../operators/gemm_epilogue_fused/overview.md).
- **Keep MFMAs back-to-back** with multiple accumulator sub-tiles so the systolic pipeline never
  drains (a single accumulator + dependent K-chain exposes MFMA latency every step).

> **Connection to the measured GEMM win:** the aiter DB tune
> ([`gemm_aiter_db_tuning.md`](gemm_aiter_db_tuning.md)) is the *automated* form of this — gradlib
> races solutions (each with its own internal MFMA shape) and banks the fastest per shape, which
> is why a hand MFMA-tile choice rarely beats the tuned DB on the live path. This case study is
> the *why* behind the default that the tuner usually lands on.

## Final result (the rule, vendor-sourced)
- **`mfma_16x16x16` is the MI300X default** for LLM GEMM N/K (lower AGPR pressure → higher
  occupancy). Switch to `32x32x8` only when a per-shape tune proves it on a large square tile.
- **`OPTIMIZE_EPILOGUE=1`** is the standard MI300X epilogue default (avoids the 512B Tagram
  hotspot).
- No standalone e2e percentage is claimed — this is a tuning *default*, and on the live sglang
  path the per-shape winner is chosen by the aiter DB tune, not hand-set.

## Lessons
1. **16×16 beats 32×32 on most MI300X LLM GEMMs** — occupancy (AGPR budget) dominates the
   per-op-amortization argument at LLM N/K.
2. **Don't carry the CUDA 32×32 habit to CDNA3** — it usually costs throughput.
3. **Tile choice + `OPTIMIZE_EPILOGUE=1` are paired levers** — set both.
4. **Verify with the ISA dump and Omniperf** — confirm `v_mfma_*_16x16x16` fired, count AGPRs,
   check the K-loop is unrolled with multiple accumulators; target high MFMA-busy with low
   `ds_read`/mem stall.
5. **The DB tune is the production form of this knob** — hand-tile only when authoring a kernel;
   on the serving path let the tuner pick.

## Cross-links
- MFMA scheduling (the technique): [`../../optimization/mfma_scheduling.md`](../../optimization/mfma_scheduling.md)
- Occupancy/registers: [`../../optimization/occupancy_and_registers.md`](../../optimization/occupancy_and_registers.md) · LDS: [`../../optimization/lds_and_bank_conflicts.md`](../../optimization/lds_and_bank_conflicts.md) · pipelining: [`../../optimization/memory_pipelining.md`](../../optimization/memory_pipelining.md)
- Hardware: [`../../hardware/cdna3_mi300/matrix_core.md`](../../hardware/cdna3_mi300/matrix_core.md) · [`../../hardware/shared/matrix_core_mfma_smfmac.md`](../../hardware/shared/matrix_core_mfma_smfmac.md)
- GEMM tuning: [`../../operators/dense_gemm/tuning.md`](../../operators/dense_gemm/tuning.md) · epilogue: [`../../operators/gemm_epilogue_fused/overview.md`](../../operators/gemm_epilogue_fused/overview.md)
- The automated form (DB tune): [`gemm_aiter_db_tuning.md`](gemm_aiter_db_tuning.md)
- ISA verify: [`../../languages/triton_amd/isa_verify.md`](../../languages/triton_amd/isa_verify.md) · asm: [`../../languages/asm_mfma/`](../../languages/asm_mfma/)

## Sources
- `mfma_16x16 > 32x32` on MI300X, ≥1024 WGs, 8-multiple tiles, `OPTIMIZE_EPILOGUE`, 512B Tagram (vendor): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- MFMA semantics, `v_mfma` shapes, AGPR accumulators (vendor): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html and the AMD CDNA3 ISA reference PDF.

<!-- MANIFEST: MFMA tile selection on MI300X GEMM — vendor rule: default mfma_16x16x16 (lower AGPR, higher occupancy) over 32x32x8 (CUDA-habit anti-pattern), pair with OPTIMIZE_EPILOGUE=1; aiter DB tune is the automated production form. -->
