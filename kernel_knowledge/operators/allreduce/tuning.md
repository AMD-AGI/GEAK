---
title: allreduce — tuning
kind: technique
operator: allreduce
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, int6, int4]
regimes: [prefill, decode, both]
updated: 2026-06-09
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html
  - https://github.com/vllm-project/vllm/pull/19744
  - https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html
---

# allreduce — tuning

Full knob table lives in [`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md);
this is the operator-level decision guide.

## Step 0 — baseline the fabric (do this first)
`rccl-tests all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1` → expect **~316–330 GB/s** busbw on a healthy 8×
MI300X node (graph mode `-G 1` slightly higher). Below ~310 GB/s = a slow xGMI link → **fix hardware before
software**. `TransferBench a2a 64M 8` catches a degraded link.

## The 3-way adaptive dispatch (the SOTA selection)
vLLM/SGLang auto-pick the AR implementation by message size / TP degree — match it, don't fight it:
1. **Custom AllReduce (CR)** — smallest messages (**<~512KB–2MB**), lowest latency (decode).
2. **QuickReduce (QR)** — two-shot with **INT4/INT6/INT8/FP4 inline compression**, wins mid/large messages
   **up to 3× vs RCCL** (up to 2.25× on 2×/4× MI300X). **MI300-only**; FP4 added on **MI355** (≈ INT4
   perf/accuracy). Crossover vs CR ≈ **1MB @ TP2, 4MB @ TP8**. ⚠ QR lifts **TTFT, not TPOT** — decode AR
   volume is tiny, so QR helps **prefill** comm only; don't expect a decode win.
3. **RCCL** — largest messages.

## The levers (one at a time, A/B each)
| signal | action |
|---|---|
| TP=2/4 (sub-island, fewer links) | `NCCL_MIN_NCHANNELS=112` (biggest sub-8 knob; **bypasses the tuning model** — A/B); custom AR |
| TP=8 single island, dense decode | leave the tuning model on; small gains; try custom AR for small msgs |
| decode (tiny msgs, latency-bound) | 1-shot custom AR / Quick Reduce; tree algo |
| AR not overlapping the next GEMM in trace | `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, sglang `SGLANG_ROCM_USE_MULTI_STREAM=1` |
| prefill (large M) | compute dominates; AR amortized — don't over-tune |

## xGMI topology facts (drive the algo choice)
- Fully-connected mesh, 7×64 GB/s/link theoretical, **45–48 GB/s realized/link**, ~315–336 GB/s aggregate/GPU.
- **Slowest link caps the collective.** No NVSwitch → ring/tree channel counts differ from H100.
- Use **all 8 GPUs** for full bandwidth; keep TP **within one island (≤8)**; PP across nodes.
- RCCL compensates for sub-island TP with more channels (32 @TP2, 24 @TP4).

## Custom AR (the decode win)
- sglang: `SGLANG_USE_AITER_AR=1` (needs `SGLANG_USE_AITER=1`); AITER 1-shot/2-shot xGMI P2P.
- vLLM: **QuickReduce** quantized custom AR — `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION={FP,INT8,INT6,INT4}`
  (FP4 on MI355), `_CAST_BF16_TO_FP16=1`, `_MAX_SIZE_BYTES_MB`. **MI300-only**, two-shot + inline
  compression → **up to 3× vs RCCL** on mid/large messages (crossover ~1MB @TP2, ~4MB @TP8); lifts **TTFT
  not TPOT**. Quantizing cuts wire bytes but needs an **accuracy gate** ([numerics.md](numerics.md))
  (vLLM PR #19744).
- `RCCL_MSCCLPP_THRESHOLD` raises the msg-size cutoff for MSCCL++ fast small-message kernels.
- **ROCm 7.2** RCCL: native 4-NIC topology / rail-aligned patterns + NCCL 2.28 backports; a **rocSHMEM GDA**
  backend removes the CPU from the critical path. QuickReduce / QuickReduce-FP4 were the MLPerf v6.0 TP comm.

## Overlap with compute
`-G 1` HIP-graph-captured collectives (~3–5% small-msg launch latency); high-priority streams; separate HW
queue. Verify in a rocprofv3 trace that the AR kernel overlaps the next layer's GEMM, not serializes.

## Pitfalls
- `NCCL_MIN_NCHANNELS` / `NCCL_THREAD_THRESHOLDS` **disable** the MI300X channel tuning model → can regress
  shapes it handled well. Always A/B.
- AITER custom AR has had segfaults on MI300X (aiter #1542) → `SGLANG_USE_AITER_AR=0` fallback.
- Quantized custom AR without an eval = shipping an accuracy regression.

## Sources
- xGMI busbw / mesh / algos: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- env / tuning model / MSCCL++: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- QuickReduce up to 3× vs RCCL, INT4/INT6/INT8 inline compression, MI300-only, crossover, TTFT-only: https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html ; FP4 on MI355: https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html ; vLLM PR #19744: https://github.com/vllm-project/vllm/pull/19744
- ROCm 7.2 RCCL 4-NIC / NCCL 2.28 / rocSHMEM GDA: https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html
- full knob table + custom AR flags: [`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).
