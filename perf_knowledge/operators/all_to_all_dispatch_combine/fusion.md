---
title: all_to_all_dispatch_combine — fusion (prob-mult, shared-expert, combine→GEMM epilogue)
kind: technique
operator: all_to_all_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://gau-nernst.github.io/amd-a2a/
  - https://arxiv.org/abs/2506.04667
---

# all_to_all_dispatch_combine — fusion

The collective brackets the MoE grouped GEMM; every fusion removes a kernel/round-trip from the EP hot path.

## Fusion targets
| pattern | how | status on AMD | link |
|---|---|---|---|
| **router-weight × combine** | combine multiplies the per-expert routing weight during the gather (`combine(..., weights=...)`) instead of a separate kernel | ✅ done in MoRI-EP | [[backends/mori_rccl/mori_ep.md]] |
| **shared-expert → single dispatch** | treat the DeepSeek shared expert as a synthetic routed expert (top-k slot via `grouped_topk`) → one fused dispatch for shared+routed | ✅ AMD wide-EP (flag-gated, MoRI-side; `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is *incompatible* with MoRI) | [[backends/aiter/fmoe.md]], [[operators/shared_expert_fusion/overview.md]] |
| **dispatch → grouped-GEMM prologue** | grouped GEMM reads the dispatched (3D-packed) tokens directly via `convert_dispatch_output`/`packed_recv_*` | ✅ via DeepEP-compatible adapter (`ENABLE_STANDARD_MOE_ADAPT=ON`) | [[operators/fused_moe_grouped_gemm/overview.md]] |
| **combine → down-proj GEMM epilogue** | fold the combine reduction into the down-proj epilogue so tokens are never re-materialized (FlashDMoE single-kernel north-star) | ⚠ **partial** — MoRI's zero-copy registered buffers move toward it; not yet a single fused kernel on AMD (honest gap) | arXiv 2506.04667 |

## The AITER seam
`aiter/moe_op/mori_all2all.py` (`MoriAll2AllManager`) wraps dispatch/combine for AITER FusedMoE; vLLM/SGLang
register MoRI as an all-to-all backend (`--all2all-backend mori_low_latency`). The 3D DeepEP-compatible layout
(`packed_recv_x`, `packed_recv_count`, `packed_recv_src_info`, `packed_recv_layout_range`) is what the
grouped GEMM consumes — built only with `ENABLE_STANDARD_MOE_ADAPT=ON` (CMake default OFF), else those
methods `RuntimeError`.

## Overlap (a fusion of time, not kernels)
Split phases (`dispatch_send/recv`, `combine_send/recv`) let dispatch overlap stage-1 prep and combine overlap
stage-2 — the practical substitute for a true single kernel today. See [[operators/all_to_all_dispatch_combine/tuning.md]] §5.

## Cross-links
[[operators/moe_dispatch_combine/overview.md]] (this op as a generic MoE primitive) ·
[[operators/fused_moe_grouped_gemm/overview.md]] · [[operators/gather_scatter/fusion.md]] (the local analog) ·
[[backends/mori_rccl/mori_ep.md]].

## Sources
- prob-mult-in-combine, dispatch_standard_moe / packed_recv_* layout, ENABLE_STANDARD_MOE_ADAPT: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- shared-expert fusion (synthetic routed expert), MoRI incompat flag: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- combine→GEMM epilogue north-star (FlashDMoE): https://arxiv.org/abs/2506.04667 · https://gau-nernst.github.io/amd-a2a/
