---
title: MoRI-EP — MoE all-to-all dispatch/combine on Instinct
kind: backend
backend: mori
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://github.com/ROCm/mori
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://gau-nernst.github.io/amd-a2a/
---

# MoRI-EP — expert-parallel all-to-all (dispatch / combine)

## TL;DR
MoRI-EP is AMD's **native** MoE dispatch/combine library — the all-to-all that routes tokens to the GPU
holding their top-k experts and gathers the outputs back. It is the **SOTA EP backend on Instinct**
(co-designed with AITER FusedMoE, HIP-graph-capturable, GPU-initiated comm) and the one AMD uses in its
own Wide-EP DeepSeek deployments. Choose it (over DeepEP, [deepep.md](deepep.md)) for production MoE EP on
MI300X/MI355X; DeepEP/UCCL-EP are the portable alternatives. `operator: moe_dispatch_combine`,
`all_to_all_dispatch_combine`.

## Concepts — the two sparse collectives
- **dispatch**: each token goes only to the ≤top-k expert-ranks it selected (sparse, not dense all-to-all)
  → scalability comes from this sparsity. Native API: `dispatch(input[num_tokens,hidden], indices
  [num_tokens,topk] int32)` → returns (output, weights, scales, indices, recv_count).
- **combine**: gather expert outputs back to origin; the **per-expert routing weight is multiplied during
  combine** (`combine(..., weights=...)`) rather than as a separate kernel — i.e. **prob-mult fused into
  combine**. Returns (output, reconstructed weights).
- **Split phases for overlap**: `dispatch_send()`/`dispatch_recv()`, `combine_send()`/`combine_recv()`.

## Tensor layouts (the AITER FusedMoE seam)
- **Native (2D)**: `[num_tokens, hidden_dim]` + int32 indices `[num_tokens, topk]` — the fast path.
- **DeepEP-compatible (3D)**: `dispatch_standard_moe()` / `combine_standard_moe()` /
  `convert_dispatch_output()` produce the standard 3D MoE layout (`packed_recv_x`, `packed_recv_count`,
  `packed_recv_src_info`, `packed_recv_layout_range`) that AITER's grouped-GEMM consumes. **Requires
  building with `ENABLE_STANDARD_MOE_ADAPT=ON`** (CMake default OFF) — else those methods `RuntimeError`.
- AITER integration point: `aiter/moe_op/mori_all2all.py` (`MoriAll2AllManager` wraps dispatch/combine
  for FusedMoE). vLLM/SGLang register MoRI as an all2all backend (`--all2all-backend mori_low_latency`).

## Kernel modes (`EpDispatchCombineKernelType`)
| mode | value | when |
|---|---|---|
| `IntraNode` | 0 | single node, xGMI only |
| `InterNode` | 1 | baseline multi-node |
| `InterNodeV1` | 2 | **throughput** priority (large batches, prefill / high-concurrency decode) |
| `InterNodeV1LL` | 3 | **low latency** (small batches, low-concurrency decode) |
| `AsyncLL` | 4 | async low-latency, pipelined |

MoRI **auto-switches** high-throughput vs low-latency kernels by concurrency; pre-tuned launch configs via
`MORI_EP_LAUNCH_CONFIG_MODE=AUTO` (e.g. InterNodeV1: block 96 / rdma 64 / warp 8).

## Config knobs (`EpDispatchCombineConfig`)
- Grid/block: `block_num` (default 80), `warp_num_per_block` (8), `rdma_block_num` (0), `num_qp_per_pe` (1).
- Model dims: `hidden_dim` (7168 for DeepSeek-V3), `num_experts_per_token` (top-K=8), `num_experts_per_rank`,
  `max_num_inp_token_per_rank` (4096).
- `use_external_inp_buf` (True) → zero-copy vs external buffer; per-call overrides for block/rdma/warp.
- Init: `mori.shmem.shmem_torch_process_group_init("default")`; `reset()` between iterations (or
  `call_reset=True` in combine).

## Fusion neighbors (where the wins are)
- **prob-mult → combine** (done by MoRI-EP, above).
- **shared-expert fusion**: AMD's Wide-EP fuses DeepSeek shared experts into AITER FusedMoE by treating
  them as synthetic routed experts that get top-k slots via `grouped_topk` → a **single** fused dispatch
  for shared + routed experts (flag-gated; preserves numerics). Note `VLLM_ROCM_USE_AITER_FUSION_SHARED_
  EXPERTS` is incompatible with MoRI — fusion is done MoRI-side in the Wide-EP path.
- **combine → grouped-GEMM epilogue**: the design north-star (cf. FlashDMoE single-kernel, below) is to
  fold the combine reduction into the down-proj GEMM epilogue so tokens are never re-materialized; MoRI-EP's
  zero-copy registered buffers (`get_registered_combine_input_buffer`) move toward this but it is not yet a
  single fused kernel on AMD.

## Measured perf (version-tagged)
- MoRI-EP bandwidth (4096 tok, hidden 7168, top-8, **FP8 dispatch + BF16 combine**, EP8), mori v1.2.0 2026-06:
  - MI300X + CX7: **307 GB/s dispatch / 330 GB/s combine** (XGMI).
  - MI355X + AINIC: **345 GB/s dispatch / 420 GB/s combine** (XGMI); latency @128 tok **31 µs dispatch /
    36 µs combine**.
- Kernel-level optimizations "reduced latency by up to **82%**", driving HBM/XGMI/RDMA close to roofline
  (AMD-reported, mori repo). Scales to **64 GPUs** with SOTA perf (2025/09 milestone).
- Wide-EP end-to-end (32× MI300X, MoRI-EP + AITER): **32.3k in / 12.4k out tok/s per node**, ROCm 6.3.1,
  2025-11 (AMD-reported).
- **Reference single-kernel a2a study** (gau-nernst, GPU MODE AMD Distributed Challenge, MI300X, num_experts
  256 / topk 8 / hidden 7168 / max 256 tok / world 8): hand-written dispatch+combine reached **292 µs**
  (from 93,540 µs reference): P2P symmetric memory 517 µs → fuse grouped-GEMM 345 µs → 303 µs → 292 µs.
  Key finding: **`grid_size=304`** (exactly the MI300X CU count, 256 didn't work) gave ~3× combine-send
  speedup; biggest non-obvious win was hoisting `torch.empty`/memset out of the kernel (caching-allocator
  malloc showed up in traces).

## Pitfalls
- **JIT first-iteration cost** (`~/.mori/jit/`); precompile (`MORI_PRECOMPILE=1`) and warm before timing.
- Forgetting `ENABLE_STANDARD_MOE_ADAPT=ON` → no DeepEP-compatible 3D API.
- Static-shape requirement for HIP graphs: EP input sizes are **dynamic** (per-routing) — Moreh had to
  pad/static-ize tensor sizes to capture MoRI-EP dispatch/combine into a HIP graph. Without graph capture
  you eat CPU launch overhead on the decode hot path.
- **Expert load imbalance**: naive contiguous expert sharding showed up to **2× imbalance** across GPUs
  (Moreh); use an EPLB-style frequency-balanced grouping (256 experts → 8 sets of 32 by activation freq).

## Verify
- rocprofv3 trace: dispatch/combine kernels present, overlapping grouped-GEMM, no `hipMalloc` in the steady
  state. Confirm kernel mode (LL vs throughput) matches concurrency.
- Numeric parity vs a torch reference MoE (greedy/temp=0) — combine reduction order differs from a dense path.

## Alternatives / cross-links
[deepep.md](deepep.md) (portable DeepEP / UCCL-EP) · [rccl_tuning.md](rccl_tuning.md) (the RDMA/xGMI knobs
under MoRI) · [overview.md](overview.md) · operators: `moe_dispatch_combine`, `fused_moe_grouped_gemm`.

## Sources
- MoRI-EP guide (API, layouts, `ENABLE_STANDARD_MOE_ADAPT`, kernel types, config): https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- MoRI repo (bandwidth/latency table, 82% latency cut, 64-GPU scaling): https://github.com/ROCm/mori @ v1.2.0
- Wide-EP shared-expert fusion + 32-GPU numbers: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html (ROCm 6.3.1, 2025-11)
- gau-nernst MI300X a2a (grid_size=304, 345/292 µs, malloc/memset fix): https://gau-nernst.github.io/amd-a2a/
- FlashDMoE/FlashMoE single-kernel design reference (NVIDIA, CUTLASS+NVSHMEM): arXiv 2506.04667 — https://arxiv.org/abs/2506.04667
