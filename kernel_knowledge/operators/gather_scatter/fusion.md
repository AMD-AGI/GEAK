---
title: gather_scatter — fusion (into grouped-GEMM prologue/epilogue; embedding)
kind: technique
operator: gather_scatter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
  - https://gau-nernst.github.io/amd-a2a/
---

# gather_scatter — fusion

A gather/scatter is memory-bound and adds an HBM round-trip; the win is to **fold it into the kernel on
either side** so the permuted tensor is never materialized.

## Fusion targets
| pattern | fuse into | effect | link |
|---|---|---|---|
| **MoE permute (gather)** | grouped-GEMM **prologue** — the stage-1 GEMM reads tokens through the sort index directly | no materialized permuted activation | [[operators/fused_moe_grouped_gemm/overview.md]] |
| **MoE unpermute (scatter-reduce)** | grouped-GEMM **epilogue** — fold `+= router_weight·out` into the down-proj write-back | removes a separate scatter kernel + the router-weight multiply | [[operators/fused_moe_grouped_gemm/overview.md]], [[operators/moe_dispatch_combine/overview.md]] |
| **embedding gather** | the following norm/RoPE | one fused pass token-load→norm | [[operators/rmsnorm/overview.md]], [[operators/rope/overview.md]] |
| **gather + dequant** | dequantize fp8/int8 rows during the gather | one HBM pass does move+dequant | [[operators/quant_dequant_fp8/overview.md]] |
| **all-to-all dispatch/combine** | EP token routing **is** a distributed gather/scatter; MoRI-EP folds the **router-weight multiply into combine** (scatter side) | prob-mult fused into the collective | [[operators/all_to_all_dispatch_combine/overview.md]], [[backends/mori_rccl/mori_ep.md]] |

## How aiter does it (on-box)
`aiter.fused_moe` calls `moe_sorting` to produce `sorted_token_ids`/`sorted_expert_ids` + a padded block
layout, then the two-stage grouped GEMM **consumes those indices directly** — the gather is the GEMM
prologue, the combine (`MulRoutedWeight1`) is the stage-2 epilogue. No standalone permute tensor.
(`ROCm/aiter@a6bb49937:aiter/fused_moe.py`, `aiter/ops/moe_sorting.py`.) See [[backends/aiter/fmoe.md]].

## The Triton fusion limit (honest gap)
A *fully* fused down-proj + scatter is blocked in Triton: it can't do **2-D scalar indexing into an
accumulator** (`acc[m,:]`), so the scatter stays a separate (still gather-tiled, atomic) kernel. HIP/CK can
fuse the write-back fully; Triton can fuse the *prologue* gather but not the epilogue scatter. The
single-kernel a2a study (gau-nernst) reaches the fused ideal in raw HIP by writing combine directly.

## When NOT to fuse
A standalone, well-coalesced gather is fine for **embedding** at prefill (it feeds a different kernel
shape) and for debugging. The anti-pattern is a materialized MoE permute tensor between two GEMMs.

## Cross-links
[[operators/gather_scatter/tuning.md]] · [[operators/fused_moe_grouped_gemm/overview.md]] ·
[[operators/moe_dispatch_combine/overview.md]] · [[backends/aiter/fmoe.md]].

## Sources
- aiter moe_sorting + grouped-GEMM consume-index, MulRoutedWeight epilogue: ROCm/aiter@a6bb49937:aiter/fused_moe.py, aiter/ops/moe_sorting.py.
- Triton 2-D scalar-index limit (separate scatter kernel): https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
- Single-kernel fused combine reference: https://gau-nernst.github.io/amd-a2a/
