---
title: all_to_all_dispatch_combine — numerics
kind: technique
operator: all_to_all_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# all_to_all_dispatch_combine — numerics

## Two error sources, neither is byte-parity
1. **combine reduction order.** Combine reduces `topk` expert partials back into each token *and* multiplies
   the per-expert router weight in. Across ranks the reduction order is **non-deterministic** and differs from
   a dense reference MoE → small fp delta. **Gate task accuracy** (greedy/temp=0 eval), not byte parity.
2. **FP8 dispatch quant.** The standard config sends tokens as **FP8 (dispatch) and accumulates combine in
   BF16**. The FP8 round-trip on the token payload introduces quant error on top of the reduction-order
   delta. Validate end-to-end, not per-kernel tolerance.

## FNUZ vs OCP on the wire
gfx942 = **FNUZ** fp8, gfx950 = **OCP** fp8 (exponent bias differs by 1). A homogeneous EP group is fine, but
the dispatch payload dtype must match what the **grouped-GEMM consumer** expects — a wrong-dialect read is a
silent **2×** error, not a crash. Confirm the dispatch quant dtype and the FusedMoE input dtype agree
([[hardware/shared/dtype_numerics.md]], [[languages/triton_amd/pitfalls.md]]).

## Shared-expert fusion preserves math (by design)
AMD's wide-EP folds the DeepSeek shared expert into FusedMoE as a synthetic routed expert (top-k slot via
`grouped_topk`) so a **single** fused dispatch covers shared+routed experts. It is **designed to preserve the
numerics** of the unfused shared-MLP + residual path — but still accuracy-gate after enabling it.

## Verify
Numeric parity vs a torch reference MoE (greedy/temp=0) after switching all-to-all backend or quant; a small
eval (e.g. gsm8k) — AITER MLA caused a gsm8k loss in one DP2TP4 case, so EP/quant changes get the same
treatment. Reduction-order deltas are expected; a *large* delta or eval regression is a bug.

## Sources
- prob-mult-in-combine, FP8 dispatch / BF16 combine: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- shared-expert fusion preserves numerics: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- FNUZ/OCP 2× hazard: [[hardware/shared/dtype_numerics.md]], [[backends/sglang_kernels/overview.md]].
