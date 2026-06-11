---
title: moe_dispatch_combine — numerics
kind: technique
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# moe_dispatch_combine — numerics & parity

## What changes the numbers
Dispatch/combine **moves** data and does **one reduction** (combine gathers k expert outputs per token and
multiplies the routing weight). Two numeric levers: the **on-the-wire quant** of dispatch, and the
**reduction order/dtype** of combine.

## fp8 dispatch / bf16 combine
- **Dispatch fp8** (E4M3FNUZ on gfx942, OCP on gfx950) quantizes tokens on the wire to halve bytes and
  returns per-token/per-block scales — this introduces **quant error before the expert GEMM**. It is the
  standard recipe and generally accuracy-safe for inference, but it **is** an accuracy gate (re-run a small
  eval when enabling).
- **Combine bf16** even when dispatch is fp8: the gather+weight-multiply runs in bf16/fp32 to avoid
  compounding quant error on the reduction. Don't combine in fp8.
- fnuz vs OCP: gfx942 is **fnuz** (exponent bias off-by-one vs OCP) — a token quantized in the wrong fp8
  dialect is off by exactly 2×. Match the dialect to the arch.

## Combine reduction order
Combine sums the k expert contributions (and any shared-expert contribution) per token. The **order**
differs from a dense reference and across backends (MoRI vs DeepEP vs a torch reference), so:
- expect small bf16 differences; gate with **greedy/temp=0 parity over ≥10 prompts**, not byte match.
- the routing **weight** multiplied in combine must be the **unbiased** routing weight (DeepSeek uses bias
  only for *selection*) — a common bug is carrying the biased score into combine.

## Where the multiply lands (and why it matters)
The routed-weight multiply can live in: the router (don't, under EP), **stage-1 of the grouped GEMM**
(`doweight_stage1`), **stage-2 epilogue** (`MulRoutedWeight1`), or **combine** (MoRI-EP's prob-mult). Each
choice changes the rounding point. Keep it consistent with the reference and in bf16/fp32, not fp8.

## Static-shape padding (a correctness trap, not just perf)
HIP-graph capture forces **static** tensor sizes, but EP token counts are **dynamic**. Padding to a fixed
`max_num_inp_token_per_rank` must use a value (0 / a sentinel) that **does not** contribute to the combine
reduction — a non-zero pad token leaks into an expert's output. Verify the pad is masked.

## Verification recipe
1. Isolated: dispatch→(identity expert)→combine round-trip must reconstruct the input within fp8 tolerance
   (catches a broken inverse map / wrong `src_info`).
2. Full MoE layer vs a torch reference (greedy) after switching all2all backend or enabling fp8 dispatch.
3. eval (e.g. gsm8k) when enabling fp8 dispatch — it's a quant gate.

## Sources
- fp8 dispatch / bf16 combine, layouts, src_info inverse map: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- shared-expert fusion preserves numerics, prob-mult in combine: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- fnuz vs OCP fp8 off-by-2×: [[scaled_quant_gemm]] numerics; CDNA3/4 ISA.
