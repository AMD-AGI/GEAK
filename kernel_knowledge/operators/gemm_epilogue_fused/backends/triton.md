---
title: gemm_epilogue_fused on triton — SOTA card
kind: sota_card
operator: gemm_epilogue_fused
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# gemm_epilogue_fused × triton

## TL;DR
> Triton is the **easiest** way to author an arbitrary fused epilogue (gated silu·mul, residual-add,
> custom act, fp8 output quant) — you just write the elementwise after the MFMA accumulation, before the
> store. Competitive on MI300X and far more flexible than hipBLASLt's fixed enum; the GEMM core may
> trail tuned hipBLASLt, but for a *custom* fused op it's often the pragmatic SOTA. For standard
> bias+gelu / scaled-fp8 on the live path, prefer [aiter.md](aiter.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton fused-epilogue matmul (inline act/bias/residual/quant) | Triton matmul tutorial + AMD autotune | gfx942/950; bf16/fp16/fp8 | core competitive (loses to tuned hipBLASLt on plain GEMM, see [[operators/dense_gemm/backends/triton.md]]); epilogue ≈ free, so net often best for *custom* fusion | gated act / residual / custom quant epilogue |

## Config space / knobs
- `BLOCK_M/N/K` (8-multiples), `matrix_instr_nonkdim=16`, `num_stages`, `num_warps`, `waves_per_eu`,
  `GROUP_SIZE_M`, `SPLIT_K` (decode).
- **`OPTIMIZE_EPILOGUE=1`** to keep the fused C write off the 512B Tagram hotspot.
- Epilogue inline: apply `α`, bias, residual, act **in fp32** on the accumulator, then down-cast/quant
  on store (see [../numerics.md](../numerics.md)).
- Autotune over the config grid per (M,N,K,flags).

## Numerics / parity
fp32 epilogue before down-cast → equal-or-better than unfused; fp8 output task-gated. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
JIT kernel callable from Python — wire into your op or register as an aiter candidate. No env-overlay
into the library GEMM path.

## Pitfalls & anti-patterns
- Untuned configs → core well below hipBLASLt; always autotune.
- Down-casting before act → accuracy loss.
- Forgetting `OPTIMIZE_EPILOGUE` → store-bound on the fused write.

## How to verify
Bench the fused kernel vs (hipBLASLt GEMM + separate elementwise) on the same shape; adopt on a net win;
gate quant on task eval.

## Alternatives / cross-links
[ck.md](ck.md) · [aiter.md](aiter.md) · [hipblaslt.md](hipblaslt.md) · [hip.md](hip.md) ·
[../overview.md](../overview.md) · language ref [[languages/triton_amd/...]].

## Sources
- Triton matmul tutorial: triton-lang.org.
- mfma/tile/OPTIMIZE_EPILOGUE/Tagram levers: ROCm workload guide.
