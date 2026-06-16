---
title: all_to_all_dispatch_combine — overview
kind: operator_overview
operator: all_to_all_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://gau-nernst.github.io/amd-a2a/
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
---

# all_to_all_dispatch_combine  (EP token routing)

## TL;DR
The two **sparse** all-to-all collectives of expert-parallel MoE: **dispatch** routes each token to the GPUs
holding its ≤top-k experts, **combine** gathers the expert outputs back and (fused) multiplies the router
weight. It is a **distributed gather/scatter over xGMI (intra-node) + RDMA (inter-node)** — pure
communication, bandwidth- and latency-bound, and the bottleneck of wide-EP DeepSeek serving. This op **is**
[[operators/moe_dispatch_combine/overview.md]] viewed as a collective; the SOTA backend on Instinct is
**MoRI-EP**, with DeepEP-on-ROCm / UCCL-EP as portable alternatives.

## Math contract
- **dispatch**: `input[num_tokens, hidden]` + `indices[num_tokens, topk]` (int32) → per-rank packed token
  buffers `(output, weights, scales, indices, recv_count)`. Sparse: a token is sent only to its ≤topk
  expert-ranks, **not** a dense N×N exchange — sparsity is where scalability comes from.
- **combine**: gather expert outputs back to origin tokens; the **per-expert routing weight is multiplied
  during combine** (prob-mult fused), reducing `topk` partials → one token. Returns `(output, weights)`.
- Quant: typically **FP8 dispatch / BF16 combine** (send compressed tokens, accumulate in higher precision).
- Split phases for overlap: `dispatch_send/recv`, `combine_send/recv`.

## Shape regimes
- **prefill / high-concurrency decode**: large batches → **throughput** kernels (MoRI `InterNodeV1`,
  DeepEP "normal"); bulk dispatch overlapping grouped-GEMM.
- **low-concurrency decode**: small batches, latency-critical → **low-latency** kernels (MoRI `InterNodeV1LL`
  /`AsyncLL`, DeepEP "low-latency"). MoRI **auto-switches** by concurrency.
- DeepSeek-V3 dims: hidden 7168, topk 8, up to 256 experts, ~4096 tok/rank.

## Where it matters (Amdahl)
On a **dense TP** model, comm is single-digit % — not here. On **wide-EP MoE** the dispatch/combine pair
brackets every MoE layer across the EP group and becomes a primary cost once compute is sharded thin. AMD's
wide-EP DeepSeek (32× MI300X, 2P2D, MoRI-EP + AITER) reaches **32.3k in / 12.4k out tok/s per node**, and
EP16 gives **1.3×** output throughput over EP8 under SLO — gains that exist **only** because dispatch/combine
are fast and overlapped. Spend here only when EP is the regime ([[backends/mori_rccl/overview.md]] decision rule).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| mori | 🟢 sota (first-party AMD EP all-to-all; HIP-graph-capturable) | [backends/mori.md](backends/mori.md) |
| aiter | 🟢 sota (integration seam: `MoriAll2AllManager` wraps dispatch/combine for FusedMoE) | [backends/aiter.md](backends/aiter.md) |
| hip | 🟢 (the authored single-kernel a2a; raw xGMI symmetric-memory) | [backends/hip.md](backends/hip.md) |
| (DeepEP-ROCm / UCCL-EP) | 🟡 competitive (portable) | [[backends/mori_rccl/deepep.md]] |

## Fusion neighbors
**prob-mult → combine** (done by MoRI-EP); **shared-expert fusion** (DeepSeek shared expert as a synthetic
routed expert → single fused dispatch); the north-star **combine → grouped-GEMM down-proj epilogue**
(fold the combine reduction into the GEMM so tokens are never re-materialized — partial on AMD today). See
[fusion.md](fusion.md), [[operators/fused_moe_grouped_gemm/overview.md]].

## Numerics
combine reduction order + FP8 dispatch quant differ from a dense reference → accuracy-gate (greedy/temp=0),
not byte parity. See [numerics.md](numerics.md).

## How to bench
Fabric first: `rccl-tests all_reduce_perf` → expect ~316–330 GB/s busbw on a healthy 8× MI300X node (lower =
slow xGMI link, fix HW first). Then MoRI-EP / DeepEP `test_internode.py`/`test_low_latency.py` for
dispatch/combine GB/s + latency; rocprofv3 to confirm overlap with grouped-GEMM. Cross-link the bandwidth
table in [[backends/mori_rccl/mori_ep.md]].

## Sources
- MoRI-EP API, sparse dispatch/combine, prob-mult-in-combine, kernel modes: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- Wide-EP DeepSeek 32-GPU numbers, EP16 vs EP8, shared-expert fusion: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- Authored single-kernel a2a (grid_size=304, 292 µs): https://gau-nernst.github.io/amd-a2a/
- xGMI bandwidth / fabric baseline: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
