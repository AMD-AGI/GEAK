---
title: allgather — tuning
kind: technique
operator: allgather
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/all_gather.py
---

# allgather — tuning

## What you actually tune
Pure data movement → tune **algorithm by message size**, **overlap with the adjacent GEMM**, and **copy
engine** (SDMA vs CU). Full RCCL knob table:
[`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).

## Levers
- **SDMA offload**: `HSA_ENABLE_SDMA=1` (default) — AG/RS can run on the SDMA copy engines, freeing CUs for
  the overlapping GEMM. MoRI-IO/AllGather explicitly offloads to SDMA.
- **Channels (sub-island)**: `NCCL_MIN_NCHANNELS=112` for TP=2/4 (bypasses tuning model — A/B).
- **MSCCL++ small-msg**: `RCCL_MSCCLPP_THRESHOLD` raises the cutoff for the fast small-message kernels.
- **Graph capture**: `-G 1` (~3–5% small-msg launch latency).
- **GPU-initiated (Iris)**: aiter Triton `all_gather` uses `iris.put` push-based, persistent-grid PID
  mapping, grid sized to CU count — for fusing AG into a Triton kernel (e.g. the SP fused kernel).

## Overlap with compute (the main win)
All-gather standalone is small; the win is **AsyncTP** overlapping it with the producing/consuming GEMM.
`TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, `SGLANG_ROCM_USE_MULTI_STREAM=1`. Verify in a trace
that AG runs concurrently with the GEMM (and ideally on SDMA, not CUs).

## xGMI facts
Same fully-connected mesh as all-reduce: ~45–48 GB/s/link, slowest link caps it, use all 8 GPUs, keep
within one island. Ring for large gathers, tree/1-shot for small.

## Pitfalls
- AG on CUs (not SDMA) steals cycles from the overlapping GEMM — prefer SDMA offload.
- MIN_NCHANNELS disables the tuning model — A/B.
- Off-island (multi-node) AG falls onto the NIC — tune RDMA (`NCCL_NET_GDR_LEVEL=2`).

## How to verify
`rccl-tests all_gather_perf -b 8 -e 16G -f 2 -g 1 -G 1`; rocprof for AG↔GEMM overlap + SDMA usage; e2e SP
tok/s.

## Sources
- RCCL env / SDMA / MSCCL++: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- aiter Triton Iris all_gather: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/all_gather.py`.
- full knob table: [`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).
