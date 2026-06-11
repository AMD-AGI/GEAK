---
title: gather_scatter — numerics
kind: technique
operator: gather_scatter
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
---

# gather_scatter — numerics

## gather: byte-exact
Pure relocation → bit-identical for every dtype. Oracle `torch.equal(out, in[idx])`. A numeric delta means
an indexing bug (OOB index, wrong stride), not precision.

## scatter-reduce: order-dependent fp
When multiple inputs accumulate into one output via **atomics**, the reduction order is **non-deterministic**
(lane/wave/block scheduling). For fp16/bf16/fp32 this yields a small, run-to-run-varying delta vs a
sequential reference. Consequences:
- **Don't gate on byte parity** for scatter-reduce — gate **task accuracy** (e.g. greedy/temp=0 eval) or use
  a relaxed `allclose` (bf16 atol/rtol ~1e-2).
- For the **MoE unpermute**, `topk` rows reduce into each token *and* the router weight is multiplied in;
  the reduction order differs from a dense reference path → re-check end-to-end accuracy after a backend swap
  ([[operators/moe_dispatch_combine/overview.md]] makes the same point for combine).
- Integer scatter-add is exact (no rounding) regardless of order.

## Determinism knob
If reproducibility matters more than speed, replace atomics with a **sorted segmented reduction** (sort by
output index, then contiguous per-segment reduce) — deterministic order, no atomic contention, at the cost of
the sort. This is the same align-sort that improves coalescing ([[operators/gather_scatter/tuning.md]] §4).

## fp8 gather/scatter
Moving fp8 bytes is exact and **dialect-agnostic** (no value interpreted). The FNUZ (gfx942) vs OCP (gfx950)
hazard only appears if you **reduce** fp8 (a scatter-add in fp8 dequantizes/requantizes) — do the reduction
in fp32 and requantize once. See [[hardware/shared/dtype_numerics.md]].

## Sources
- HW atomic reduction-order non-determinism: https://llvm.org/docs/AMDGPUUsage.html
- MoE unpermute reduction + router weight (accuracy-gate, not parity): https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
- fp8 dialect: [[hardware/shared/dtype_numerics.md]], [[languages/triton_amd/pitfalls.md]].
