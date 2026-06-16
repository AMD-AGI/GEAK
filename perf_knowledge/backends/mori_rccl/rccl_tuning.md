---
title: RCCL tuning — env knobs, xGMI topology, custom all-reduce on Instinct
kind: backend
backend: rccl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html
---

# RCCL tuning — the TP/collective backend on Instinct

## TL;DR
RCCL is the NCCL-API-compatible collective library on ROCm — the default for **TP** all-reduce/all-gather/
reduce-scatter. On a single 8× MI300X node it usually sustains ~316–330 GB/s busbw out of the box; tuning
gains are single-digit % **except** for sub-island TP (TP=2/4), EP all-to-all, and multi-node, where the
right knob is worth real time. First **baseline the fabric**, then turn one knob at a time and A/B it
(several knobs *disable* RCCL's MI300X tuning model). For EP MoE all-to-all prefer MoRI-EP/DeepEP over raw
RCCL ([mori_ep.md](mori_ep.md), [deepep.md](deepep.md)).

## Concepts — xGMI topology (8× MI300X)
| property | value |
|---|---|
| interconnect | xGMI / Infinity Fabric, **fully connected** mesh (dedicated link to each of 7 peers) |
| per-link BW | 64 GB/s theoretical, **45–48 GB/s realized** |
| per-GPU aggregate (unidir) | 7×64 = 448 theoretical, **315–336 GB/s realized** |
| 8-GPU all-reduce busbw (16G msg) | **~316–330 GB/s** (graph mode `-G 1` slightly higher) |
| bottleneck | the **slowest link** caps the collective |

No NVSwitch analog → ring/tree algos and channel counts behave differently than H100. **Use all 8 GPUs**
for full bandwidth; a 2/4-GPU collective uses only a fraction of links (RCCL compensates with more channels:
32 @TP2, 24 @TP4). Keep TP **within one xGMI island (≤8)**; use PP across nodes.

## The levers — RCCL/NCCL env table
| env var | default/range | effect | MI300X guidance |
|---|---|---|---|
| `NCCL_MIN_NCHANNELS` | per-topo (32@TP2, 24@TP4) | min channels (collective parallelism) | **`112` for single-node e2e** (AMD rec); biggest sub-8-GPU TP knob. **Bypasses the tuning model** — A/B it |
| `NCCL_MAX_NCHANNELS` | topo | cap channels | rarely needed; also bypasses tuning model |
| `NCCL_THREAD_THRESHOLDS` | tuned | LL/LL128 thresholds | setting it (or MIN/MAX_NCHANNELS) **disables** the channel tuning model |
| `RCCL_MSCCLPP_THRESHOLD` | 1 MB | msg-size cutoff for MSCCL++ fast kernels | raise to push more small-msg TP/EP collectives through MSCCL++ |
| `NCCL_PROTO` | auto | `LL`/`LL128`/`Simple` | leave auto |
| `NCCL_ALGO` | auto | `Ring`/`Tree`/`CollnetDirect` | leave auto on single-node mesh |
| `NCCL_IGNORE_CPU_AFFINITY` | 0 | use GPU affinity | `=1` helps multi-node |
| `NCCL_P2P_LEVEL` | auto | P2P transport threshold | keep P2P on (xGMI is the point) |
| `NCCL_NET_GDR_LEVEL` / `RCCL_NET_GDR_LEVEL` | — | GPUDirect RDMA level | `=2` for **multi-node** |
| `NCCL_IB_GID_INDEX` | — | RoCE GID index | set (e.g. `3`) for RoCE multi-node |
| `NCCL_IB_HCA` | — | IB/RoCE HCA list | bind to the right NICs (e.g. `ionic_0..7` on Pollara) |
| `RCCL_ENABLE_CONTEXT_TRACKING` | 0 | per-GPU context tracking | `=1` can help in certain scenarios |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 0 | P2P over PCIe (large BAR) | `=1` for PCIe-connected GPUs |
| `HSA_ENABLE_SDMA` | 1 | SDMA copy engines | keep on; MoRI-IO/AllGather offload to SDMA |
| `NCCL_DEBUG` + `RCCL_DEBUG_SUBSYS=INIT,GRAPH` | — | see chosen algo/channels | diagnosis |

> Deprecation note: upstream NCCL deprecated `NCCL_MIN_NCHANNELS` for `NCCL_MIN_CTAS`, but **RCCL still
> documents and honors `NCCL_MIN_NCHANNELS`** — keep using it on ROCm.

MSCCL++ kernels are on by default on MI300X (efficient small-message AR/AG); the path that matters is the
MSCCL++ kernel, controlled by `RCCL_MSCCLPP_THRESHOLD` (some builds turned legacy MSCCL API symbols into
no-ops for link compat).

## Framework custom all-reduce (often beats RCCL for small/decode messages)
| framework | mechanism | engage |
|---|---|---|
| sglang | AITER custom AR/AG | `SGLANG_USE_AITER_AR=1`, `SGLANG_USE_AITER_AG=1` (needs `SGLANG_USE_AITER=1`) |
| vLLM | Quick Reduce (quantized custom AR) | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION={FP,INT8,INT6,INT4}`, `_CAST_BF16_TO_FP16=1`, `_MAX_SIZE_BYTES_MB` |
| both | xGMI P2P custom AR | on by default within an island |

Quick Reduce **quantizes the reduction** to cut wire bytes → an **accuracy gate** (any non-`NONE` changes
reduced values). AITER custom AR has had stability bugs on MI300X (aiter #1542 segfault) — fall back to
`SGLANG_USE_AITER_AR=0` if the AR path crashes.

## Overlap comm with compute
`TORCH_NCCL_HIGH_PRIORITY=1` (force high-priority RCCL streams), `GPU_MAX_HW_QUEUES=2` (separate HW queues
for comm+compute), sglang `SGLANG_ROCM_USE_MULTI_STREAM=1`, RCCL `-G 1` (HIP-graph-captured collectives,
~3–5% on small-msg launch latency), `--bind-to numa`. Verify in a rocprofv3 trace that the AR kernel
overlaps the next layer's GEMM rather than serializing.

## When is comm the bottleneck? (decision guide)
| signal | comm-bound? | action |
|---|---|---|
| TP=2/4, small model | likely | `NCCL_MIN_NCHANNELS=112`, custom AR |
| TP=8 single island, dense decode | usually not | leave tuning model on; small gains |
| EP MoE, high concurrency | **yes** (all-to-all) | MoRI-EP / DeepEP, `RCCL_MSCCLPP_THRESHOLD` |
| multi-node / cross-island | **yes** (off xGMI) | PP across nodes; `NCCL_NET_GDR_LEVEL=2`, IB/RoCE tune |
| AR not overlapping GEMM in trace | yes | high-priority streams, `GPU_MAX_HW_QUEUES=2` |
| prefill (large M) | rarely | compute dominates; AR amortized |

## Verify
```bash
mpirun -np 8 --bind-to numa -env NCCL_DEBUG=VERSION \
  rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1
# expect ~316-330 GB/s busbw on a healthy 8x MI300X node; lower => a slow xGMI link
./TransferBench a2a 64M 8   # per-link a2a sanity (catches a degraded link)
```
If busbw is well below ~310 GB/s you have a hardware/link problem — fix that **before** any software tuning.

## Pitfalls
- Setting MIN/MAX_NCHANNELS or THREAD_THRESHOLDS **disables** the MI300X channel tuning model → can regress
  shapes the model handled well. Always A/B.
- Quantized custom AR is an accuracy gate by design; never ship it without a small eval.
- Don't stretch TP across nodes — you fall off xGMI onto NIC fabric.

## Alternatives / cross-links
[overview.md](overview.md) · [mori_ep.md](mori_ep.md) / [deepep.md](deepep.md) (EP all-to-all) ·
operators `allreduce`, `allgather`, `reduce_scatter`, `fused_allreduce_rmsnorm`.

## Sources
- RCCL usage tips (env vars, tuning model, channel defaults): https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- Understanding RCCL bandwidth & xGMI on MI300X: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- MI300X workload optimization (NCCL_MIN_NCHANNELS, affinity): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- vLLM serving on AMD MI300X best practices (NCCL tuning): https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html
- AITER all-reduce segfault (issue #1542): https://github.com/ROCm/aiter/issues/1542
- vLLM V1 ROCm optimization (Quick Reduce): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
