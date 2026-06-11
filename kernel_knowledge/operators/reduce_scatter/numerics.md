---
title: reduce_scatter — numerics
kind: technique
operator: reduce_scatter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
---

# reduce_scatter — numerics & parity

## It IS a reduction (unlike all-gather)
Reduce-scatter sums across ranks → it has an **accumulation dtype and order**. RCCL accumulates in fp32;
ring vs tree and channel counts change the **order** → benign bf16 rounding deltas, not regressions. Gate
with greedy/temp=0 parity; don't require byte parity across algo changes.

## SP-rewrite equivalence
Replacing `all_reduce` with `reduce_scatter + local RMSNorm + all_gather` is **mathematically equivalent**
to AR-then-norm only if the norm is the **same RMSNorm on the same data** — but it now runs on the sharded
tensor before the gather. Verify the SP path matches the non-SP path (greedy parity); a mismatch usually
means the norm's reduction axis or epsilon differs, or the shard boundaries are wrong.

## fp8 in the fused SP kernel
The fused `reduce_scatter_rmsnorm_quant_all_gather` kernel can **fp8-quantize** between the norm and the
all-gather (`_quantize_fp8_stage`) — that quant is an **accuracy gate** (fnuz on gfx942, wrong dialect =
2× off). Validate end-to-end when the quant stage is enabled.

## Verification recipe
1. Numeric: RS output (reduced shard) vs an fp32 reference — within bf16 tolerance.
2. SP equivalence: SP path (RS+norm+AG) vs non-SP (AR+norm) — greedy parity.
3. Quant gate: a small eval when the fused kernel's fp8 stage is on.

## Sources
- RCCL reduction semantics: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- fused SP kernel fp8 stage: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
