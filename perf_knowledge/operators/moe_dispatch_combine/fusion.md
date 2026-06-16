---
title: moe_dispatch_combine — fusion
kind: technique
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://arxiv.org/abs/2506.04667
---

# moe_dispatch_combine — fusion

The MoE EP pipeline is `route → dispatch → grouped GEMM (gate/up, down) → combine`. Fusion here means
collapsing those boundaries and **overlapping comm with compute**.

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **prob-mult → combine** | routing-weight multiply done during the combine gather | MoRI-EP `combine(..., weights=)` | removes a separate weighted-combine kernel |
| **shared-expert → dispatch** | shared experts injected as synthetic routed experts (top-k slots via `grouped_topk`) → one dispatch for shared+routed | Wide-EP / [[shared_expert_fusion]] | one fused dispatch, no separate shared Linear+add |
| **fp8 quantize → dispatch** | token fp8 cast + scale computed inside the dispatch send | MoRI-EP fp8 dispatch | halves wire bytes, no separate quant pass |
| **split-phase comm ↔ grouped GEMM overlap** | `dispatch_send/recv`, `combine_send/recv` interleaved with the grouped GEMM mainloop | MoRI-EP split API + HIP graph | hides comm under compute |
| **dispatch + grouped-GEMM (partial)** | the a2a reference fused grouped-GEMM into the same launch sweep | gau-nernst study (345→292 µs) | one kernel boundary removed |

## The AITER FusedMoE seam (DeepEP-compatible 3D layout)
MoRI-EP's native layout is **2D** `[num_tokens, hidden]`; AITER's grouped GEMM wants the **3D**
`packed_recv_x / packed_recv_count / packed_recv_src_info / packed_recv_layout_range`. The bridge is
`dispatch_standard_moe()` / `combine_standard_moe()` / `convert_dispatch_output()` — **requires building
mori with `ENABLE_STANDARD_MOE_ADAPT=ON`** (CMake default OFF), else those methods `RuntimeError`. The wire
point is `aiter/moe_op/mori_all2all.py` (`MoriAll2AllManager`). This is how dispatch hands off to
[[fused_moe_grouped_gemm]] without re-materializing tokens.

## The north-star (not yet shipped on AMD)
**Fold the combine reduction into the down-proj GEMM epilogue** so expert outputs are never
re-materialized to global memory before the gather (cf. FlashDMoE/FlashMoE single-kernel design, NVIDIA
CUTLASS+NVSHMEM, arXiv 2506.04667). MoRI-EP's **zero-copy registered buffers**
(`get_registered_combine_input_buffer`) move toward this, but a single fused combine+GEMM kernel does not
exist on AMD yet — flag it as a gap.

## Decode-path discipline
Capture dispatch→GEMM→combine into **one HIP graph** (pad token counts static), hoist all `torch.empty`
out, use the **low-latency** kernel mode (`InterNodeV1LL`/`AsyncLL`). The decode a2a is the latency tail of
the MoE layer.

## Cross-links
[[moe_routing_topk]] (feeds dispatch) · [[fused_moe_grouped_gemm]] (consumes dispatch) ·
[[shared_expert_fusion]] · [`backends/mori_rccl/mori_ep.md`](../../backends/mori_rccl/mori_ep.md) ·
[`backends/aiter/fmoe.md`](../../backends/aiter/fmoe.md).

## Sources
- prob-mult-in-combine, 3D adapter, ENABLE_STANDARD_MOE_ADAPT, zero-copy buffers: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- shared-expert fusion (Wide-EP): https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- single-kernel combine+GEMM north-star: https://arxiv.org/abs/2506.04667
