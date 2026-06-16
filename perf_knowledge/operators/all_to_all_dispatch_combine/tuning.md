---
title: all_to_all_dispatch_combine — tuning (xGMI/RDMA, kernel mode, grid=304, overlap)
kind: technique
operator: all_to_all_dispatch_combine
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
updated: 2026-06-09
sources:
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://gau-nernst.github.io/amd-a2a/
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://github.com/ROCm/mori
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# all_to_all_dispatch_combine — tuning

Comm-bound: tune the **fabric**, the **kernel mode**, the **grid**, and the **overlap** — and quantize the
payload.

## 1. The fabric (xGMI / RDMA) — baseline before software
An 8× MI300X node is **fully-connected xGMI** (direct link to each of the other 7), ~64 GB/s/link theoretical,
**~45–48 GB/s realized** → ~316–330 GB/s all-reduce busbw on a healthy node. No central switch (unlike
NVSwitch), so collective algos behave differently than on H100. **Always baseline first**:
`rccl-tests all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1` — below ~316 GB/s means a degraded link; fix HW
before touching software. Stay **within one xGMI island (≤8 GPU)** for the intra-node path; cross-node goes
over RDMA (CX-7 / Broadcom Thor2 / AMD Pensando), which is the inter-node bottleneck.

## 2. Kernel mode (throughput vs low-latency)
MoRI-EP `EpDispatchCombineKernelType`: `IntraNode`(0), `InterNode`(1), **`InterNodeV1`(2, throughput)**,
**`InterNodeV1LL`(3, low-latency)**, `AsyncLL`(4). DeepEP mirrors this as "normal" vs "low-latency". MoRI
**auto-switches** by concurrency; pre-tune launch configs with `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`
(e.g. InterNodeV1: block 96 / rdma 64 / warp 8). Match mode to regime: bulk prefill → throughput; small-batch
decode → low-latency.

## 3. Grid sizing — `grid_size = 304` (the non-obvious win)
The authored single-kernel a2a study found that launching with **`grid_size = 304`** (exactly the MI300X CU
count) gave a **~3× combine-send speedup** vs 256 (256 "didn't work"). MoRI defaults: `block_num=80`,
`warp_num_per_block=8`, `rdma_block_num=0`, `num_qp_per_pe=1` — tunable per call. Lesson: size the grid to the
**physical CU count**, not a round number.

## 4. Hoist allocation out of the kernel (the biggest single fix)
The same study's **largest non-obvious win** was hoisting `torch.empty`/memset out of the hot path — the
caching-allocator `hipMalloc` showed up in traces and dominated. MoRI's `use_external_inp_buf=True` /
`get_registered_combine_input_buffer` give **zero-copy registered buffers** for exactly this. Never allocate
or memset inside the steady-state dispatch/combine.

## 5. Overlap with compute (split phases)
Use `dispatch_send`/`dispatch_recv` and `combine_send`/`combine_recv` to overlap the collective with the
grouped-GEMM (dispatch overlaps stage-1 prep; combine overlaps stage-2). MoRI is **HIP-graph-capturable** —
but EP input sizes are **dynamic** (per-routing), so you must **pad/static-ize** tensor sizes to capture
into a graph (Moreh's approach), else you eat CPU launch overhead on the decode hot path.

## 6. Quantize the payload
**FP8 dispatch / BF16 combine**: send tokens compressed (halves dispatch bytes), accumulate combine in bf16.
This is the standard MoRI-EP / DeepEP config and the basis of the published bandwidth table
([[backends/mori_rccl/mori_ep.md]]). Going further, the **quantized all-to-all (FP4 dispatch + FP8 combine)**
gives a **2.56× round-trip bandwidth reduction** (28672 → 11200 B/token, LMSYS MoRI 2026-05-28). Measured
combine (EP8, BF16, 4096 tok, hidden 7168): fp8_blockwise **~736 µs** vs BF16 ref **~907 µs**; the adaptive
**InterNodeV1LL** path gives **1.52× dispatch / 1.82× combine** at ≤256 tok/rank.

## 7. Expert load balance
Naive contiguous expert sharding showed up to **2× imbalance** across GPUs (Moreh) → a few ranks bottleneck
the whole collective. Use an **EPLB-style frequency-balanced** grouping (e.g. 256 experts → 8 sets of 32 by
activation frequency). Balance is a *comm* lever, not just a compute one.

## Measured anchors (version-tagged)
- MoRI-EP (4096 tok, hidden 7168, top-8, FP8 dispatch/BF16 combine, EP8), mori v1.2.0 2026-06:
  MI300X+CX7 **307/330 GB/s** dispatch/combine; MI355X+AINIC **345/420 GB/s**, latency @128 tok **31/36 µs**.
- Authored HIP a2a: **93,540 µs → 292 µs** (P2P sym-mem 517 → fuse grouped-GEMM 345 → 303 → 292), MI300X.
- Kernel opts "reduced latency up to **82%**"; scales to **64 GPUs** (mori, AMD-reported).
- Quantized A2A (FP4 dispatch + FP8 combine): **2.56×** round-trip BW (28672 → 11200 B/token); MI355X+MoRI
  SGLang **1.25× tok/s/GPU vs B200** at iso-latency (LMSYS MoRI, 2026-05-28).

## Verify
`rccl-tests` fabric baseline; rocprofv3 → dispatch/combine present, **overlapping** grouped-GEMM, **no
`hipMalloc`** in steady state; confirm kernel mode (LL vs throughput) matches concurrency; banner
(`mori_low_latency`) in server log.

## Sources
- MoRI kernel modes, config knobs, registered buffers, FP8/BF16: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md · https://github.com/ROCm/mori
- grid_size=304, malloc/memset hoist, 292 µs ladder: https://gau-nernst.github.io/amd-a2a/
- xGMI ~45–48 GB/s/link, 316–330 GB/s busbw, fabric baseline: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- 2× expert imbalance / EPLB: [[backends/mori_rccl/mori_ep.md]] (Moreh).
- Quantized A2A 2.56× BW, fp8_blockwise combine ~736 vs BF16 ~907 µs, InterNodeV1LL 1.52×/1.82× ≤256 tok/rank, 1.25× tok/s/GPU vs B200: https://www.lmsys.org/blog/2026-05-28-mori/ (2026-05-28)
