---
title: moe_dispatch_combine — tuning
kind: technique
operator: moe_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-09
sources:
  - https://gau-nernst.github.io/amd-a2a/
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://github.com/ROCm/mori
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# moe_dispatch_combine — tuning

## What you actually tune
A bandwidth-bound, latency-sensitive sparse all-to-all. Levers: kernel mode (throughput vs low-latency),
grid size, on-the-wire dtype (fp8 dispatch), buffer reuse (zero-copy / no per-call malloc), and overlap
with the grouped GEMM.

## The MI300X reference findings (gau-nernst, GPU MODE AMD challenge)
The hand-written dispatch+combine that hit **292 µs** (from a 93,540 µs reference) at E=256/topk=8/hidden
7168/world=8 surfaced the load-bearing knobs:
- **`grid_size = 304`** — *exactly* the MI300X CU count. 256 "didn't work"; 304 gave **~3× combine-send
  speedup**. Size the grid to the full CU count, not a round power of two.
- **Hoist `torch.empty` / memset out of the kernel** — the caching-allocator malloc showed up in traces
  as a real cost. Pre-allocate registered buffers once and reuse (the biggest non-obvious win).
- **P2P symmetric memory** (each rank maps peers' buffers) → direct `load/store`/`put` instead of staged
  copies. The optimization ladder was: P2P symmetric 517 µs → fuse grouped-GEMM 345 µs → 303 µs → 292 µs.

## MoRI-EP knobs (production path)
- **Kernel type** (`EpDispatchCombineKernelType`): `IntraNode` (xGMI only), `InterNodeV1` (throughput),
  `InterNodeV1LL` (low latency), `AsyncLL`. MoRI **auto-switches** by concurrency
  (`MORI_EP_LAUNCH_CONFIG_MODE=AUTO`). For single-node intra-node, `IntraNode` is the path.
- **Grid/block config** (`EpDispatchCombineConfig`): `block_num` (default 80), `warp_num_per_block` (8),
  `rdma_block_num` (0 for intra-node), `num_qp_per_pe` (1). Pre-tuned launch configs per kernel type
  (e.g. InterNodeV1: block 96 / rdma 64 / warp 8).
- **`use_external_inp_buf=True`** → zero-copy from an external buffer (vs an internal copy);
  `get_registered_combine_input_buffer` for the registered zero-copy path.
- **`max_num_inp_token_per_rank`** (4096), `hidden_dim` (7168), `num_experts_per_token` (8) — model dims.
- **Arch pin**: `MORI_GPU_ARCHS=gfx942`/`gfx950` to stop wrong-arch JIT; `MORI_PRECOMPILE=1` to avoid the
  first-iteration JIT cost.

## On-the-wire dtype
- **fp8 dispatch + bf16 combine** is the standard recipe: halve dispatch bytes (fp8 quantize tokens, send
  scales), keep combine in bf16 for accuracy. Measured (mori v1.2.0, EP8, 4096 tok, hidden 7168, top-8):
  MI300X+CX7 **307 GB/s dispatch / 330 GB/s combine**; MI355X+AINIC **345 / 420 GB/s**.
- **Quantized all-to-all (FP4 dispatch + FP8 combine)**: a **2.56× round-trip bandwidth reduction**
  (28672 → 11200 B/token, LMSYS MoRI 2026-05-28). MoRI-EP combine (EP8, BF16, 4096 tok, hidden 7168)
  fp8_blockwise **~736 µs** vs BF16 ref **~907 µs**; the adaptive **InterNodeV1LL** path gives
  **1.52× dispatch / 1.82× combine** at ≤256 tok/rank.
- fp4 dispatch is possible on CDNA4 (MI350/355) where FP4 HW exists; **not** on gfx942.

## Overlap with compute (the other half of the win)
- **HIP-graph-capture** the dispatch/combine so decode doesn't eat CPU launch overhead. EP input sizes are
  **dynamic** (per-routing) → you must **pad/static-ize** tensor sizes to capture (Moreh did this).
- Split phases (`dispatch_send`/`dispatch_recv`, `combine_send`/`combine_recv`) to **interleave** comm with
  the grouped GEMM rather than serialize.
- Confirm in a rocprofv3 trace that dispatch/combine kernels **overlap** the grouped GEMM and that there is
  **no `hipMalloc`** in steady state.

## Expert load balance (a tuning concern, not just correctness)
Naive contiguous expert sharding showed up to **2× imbalance** across GPUs (Moreh) — one rank's experts get
all the tokens, stalling the collective. Use an **EPLB-style frequency-balanced** grouping (256 experts → 8
sets of 32 by activation frequency) so dispatch traffic is even.

## How to verify a tuning win
- Bandwidth harness at your (tokens, hidden, topk, dtype, EP) → compare GB/s vs the mori table.
- rocprofv3: kernel mode matches concurrency (LL vs throughput), grid = 304 (single-kernel) / configured
  block_num (MoRI), overlap with GEMM, no malloc in steady state.
- Per-GPU token-count histogram to catch expert imbalance.

## Sources
- 292 µs / grid_size=304 / malloc+memset hoist / P2P ladder: https://gau-nernst.github.io/amd-a2a/
- MoRI-EP kernel types, config, AUTO launch, fp8/bf16 bandwidth: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md ; https://github.com/ROCm/mori
- Quantized A2A 2.56× BW (28672→11200 B/tok), fp8_blockwise combine ~736 vs BF16 ~907 µs, InterNodeV1LL 1.52×/1.82× ≤256 tok/rank: https://www.lmsys.org/blog/2026-05-28-mori/ (2026-05-28)
- 2× imbalance + EPLB grouping (Moreh): https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/
