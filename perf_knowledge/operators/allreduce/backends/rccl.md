---
title: allreduce on RCCL/MSCCL++ — SOTA card
kind: sota_card
operator: allreduce
backend: rccl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html
  - https://github.com/vllm-project/vllm/pull/19744
  - https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
---

# allreduce × RCCL / MSCCL++

## TL;DR
> RCCL is the **default TP all-reduce** on Instinct (NCCL-API-compatible) and the **largest-message tier of a
> 3-way adaptive dispatch** that vLLM/SGLang now auto-select: Custom AllReduce (small, <~512KB–2MB) →
> **QuickReduce** (mid/large, up to **3× vs RCCL**, two-shot + INT4/INT6/INT8/FP4 inline compression,
> MI300-only; crossover ~1MB @ TP2, ~4MB @ TP8; lifts **TTFT not TPOT**) → RCCL (largest). MSCCL++ fast
> small-message kernels are on by default on MI300X; RCCL sustains ~316–330 GB/s busbw on a healthy 8-GPU
> node out of the box. **ROCm 7.2** adds native 4-NIC topology / NCCL 2.28 backports and a **rocSHMEM GDA**
> backend that removes the CPU from the critical path. QuickReduce is the key new lever and could become its
> own backend card later; for now see [hip.md](hip.md) / [vllm_kernels.md](vllm_kernels.md). Key constraint:
> xGMI is **point-to-point, not switched** — full bandwidth needs all 8 GPUs and all 7 links active
> simultaneously; TP=2/4 see only a fraction.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| RCCL ring/tree all-reduce | `ROCm/rccl` | gfx942/950; bf16/fp16 | **~316–330 GB/s busbw** @ 8× MI300X, 16G msg, `-G 1` (AMD blog); ~310–330 GB/s practical per-GPU aggregate | **largest** messages in the 3-way dispatch |
| QuickReduce (two-shot, inline INT4/INT6/INT8/FP4) | vLLM PR #19744, [vllm_kernels.md](vllm_kernels.md) | **MI300-only**; FP4 on MI355 | **up to 3× vs RCCL** (up to 2.25× on 2×/4× MI300X); crossover ~1MB @TP2, ~4MB @TP8; lifts **TTFT not TPOT** | mid/large messages (the new lever) |
| Custom AllReduce (1-/2-shot xGMI P2P) | aiter `custom_all_reduce.cu`, [hip.md](hip.md) | gfx942/950 | lowest latency <~512KB–2MB | small/decode messages |
| MSCCL++ fast kernels | RCCL (default on MI300X) | gfx942/950 | faster small-msg AR/AG | small messages (gated by `RCCL_MSCCLPP_THRESHOLD`) |
| aiter Triton GPU-initiated collective (fused) | `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` (on Iris) | gfx942/950 | fuses RS+RMSNorm+fp8-quant+AG into one kernel | TP norm-after-AR where you want a single GPU-initiated kernel |

### 3-way adaptive dispatch (the SOTA selection)
vLLM/SGLang auto-select the AR implementation by message size and TP degree:
1. **Custom AllReduce (CR)** — lowest latency for small messages **<~512KB–2MB** (decode comm volume is tiny).
2. **QuickReduce (QR)** — two-shot with **INT4/INT6/INT8/FP4 inline compression**, wins mid/large messages
   **up to 3× vs RCCL** (up to 2.25× on 2×/4× MI300X). **MI300-only** (`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`);
   FP4 added on MI355 (≈ INT4 perf/accuracy). Crossover vs CR ≈ **1MB @ TP2, 4MB @ TP8**. QR lifts **TTFT
   not TPOT** — it helps prefill comm, not the tiny decode AR.
3. **RCCL** — largest messages, falls back to ring/tree busbw.

QuickReduce is the **key new lever**; it could warrant its own backend card later (kept under
[vllm_kernels.md](vllm_kernels.md) / [hip.md](hip.md) for now).

### ROCm 7.2 RCCL improvements
Native **4-NIC topology** + rail-aligned patterns, **NCCL 2.28** backports, and a **rocSHMEM GDA** backend
that removes the CPU from the critical path. QuickReduce / QuickReduce-FP4 are used as the MLPerf v6.0 TP comm.

### xGMI topology (why TP degree matters)
MI300X = 8 GPUs in a **fully-connected mesh**: each GPU has **7 xGMI links** to its peers. Per link: ~64 GB/s
paper (Infinity Fabric), ~48 GB/s usable after CRC/protocol overhead. Aggregate per GPU = 7×64 = **448 GB/s
paper**, ~336 GB/s usable, 310–330 GB/s practical. Because it's **point-to-point** (no switch / no in-network
reduce), a pair-only collective is capped at one link:
- **TP=2** → ~64 GB/s (one link)
- **TP=4** → ~189 GB/s
- **TP=8** → full mesh (~316–330 GB/s busbw)
This is the core reason small-TP all-reduce is comparatively slow on MI300X and why decode prefers a custom AR.

### Measured perf (busbw by TP degree, AMD blog)
| config | metric | value @ hw / date | source |
|---|---|---|---|
| all_reduce, 8× MI300X, 16G msg, `-G 1` | busbw | **~316–330 GB/s** @ MI300X gfx942 (AMD blog) | mi300x-rccl-xgmi blog |
| per-GPU xGMI aggregate (7 links) | usable bw | ~336 GB/s (448 paper) | mi300x-rccl-xgmi blog |
| TP=2 (one link) | AR ceiling | **~64 GB/s** | mi300x-rccl-xgmi blog |
| TP=4 | AR ceiling | **~189 GB/s** | mi300x-rccl-xgmi blog |
| optimized CPX mode, single OAM | AR peak | ~340 GB/s | mi300x-rccl-xgmi blog |

These are vendor/AMD-blog numbers; re-measure with rccl-tests on your node (a single slow link drags the whole
collective). The TP=2/4 caps are the practical reason to keep TP=8 within the island or use a custom AR.

### GPU-initiated alternative (aiter Triton comms)
aiter ships **GPU-initiated** collectives in Triton over **Iris** (`aiter/ops/triton/comms/`): `reduce_scatter.py`
(pull-based `iris.load`), `all_gather.py` (push-based `iris.put`), and the fully-fused
`fused/reduce_scatter_rmsnorm_quant_all_gather.py` (RS → RMSNorm → optional per-token fp8 quant → AG in one
kernel). These bypass RCCL's host-launched channels entirely for the TP+norm pattern; see [[fused_allreduce_rmsnorm]]
and [[reduce_scatter]].

## Config space / knobs
- `NCCL_MIN_NCHANNELS=112` (sub-8 TP; **bypasses the MI300X tuning model** — A/B it). RCCL already pre-sets
  32 channels for TP=2, 24 for TP=4.
- `RCCL_MSCCLPP_THRESHOLD` — message size below which MSCCL++ fast kernels engage.
- `NCCL_ALGO` / `NCCL_PROTO` — leave **auto** single-node.
- Multi-node: `NCCL_NET_GDR_LEVEL=2`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`.
- Overlap: `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, `rccl-tests -G 1`.
- Full table: [`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md).

## Numerics / parity
fp32 accumulate; algo/channel changes reorder the reduction → benign bf16 deltas. Parity-safe. fp8 collectives
(MSCCL++/custom) add a quant gate. See [numerics.md](../numerics.md).

## Integration (rebind seam)
torch distributed / framework AR dispatches to RCCL automatically; tune via env. `NCCL_DEBUG=VERSION` +
`RCCL_DEBUG_SUBSYS=INIT,GRAPH` to see the chosen algo/channels. To replace it for the TP+norm pattern, wire the
aiter Triton fused collective instead.

## Pitfalls & anti-patterns
- MIN/MAX_NCHANNELS / THREAD_THRESHOLDS **disable** the MI300X tuning model → regressions; always A/B.
- Stretching TP across nodes (off xGMI onto the NIC) — use PP across nodes; keep TP within the 8-GPU island.
- A single slow xGMI link caps the whole collective — baseline with rccl-tests / TransferBench first.
- Expecting TP=2/4 to scale like TP=8 — the point-to-point topology caps small TP at one/few links.
- Leaving MSCCL++ off where small-message AR dominates (decode) — confirm it's engaged or use a custom AR.

## How to verify
`rccl-tests all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1` → ~316–330 GB/s on 8 GPUs; `TransferBench` AllToAll for
the xGMI ceiling; per-TP-degree busbw to confirm the point-to-point cap; e2e TP tok/s; trace overlap.

## Worked example (decide RCCL vs custom AR, Llama-70B)
TP=8 prefill, bf16, AR message ~4 MB (large) → **RCCL ring** at ~316–330 GB/s busbw is right; confirm all 8
GPUs / all links active (`rccl-tests`). Same model **decode**, M=1, AR message a few KB → RCCL ring overhead
dominates; switch to aiter custom 1-shot AR ([hip.md](hip.md)) or ensure MSCCL++ is below `RCCL_MSCCLPP_THRESHOLD`.
TP=4 anywhere → expect ~189 GB/s cap; if AR-bound, prefer TP=8 or a custom AR.

## Alternatives / cross-links
[[allreduce]] · [hip.md](hip.md) (custom 1-shot/2-shot) · [vllm_kernels.md](vllm_kernels.md) (Quick Reduce) ·
[[fused_allreduce_rmsnorm]] · [[reduce_scatter]] · [[allgather]] ·
[`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md) · [overview.md](../overview.md) ·
[numerics.md](../numerics.md).

## Sources
- xGMI busbw / topology / TP=2,4,8 caps: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- RCCL env / MSCCL++ / tuning model / pre-set channels: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- QuickReduce up to 3× vs RCCL, two-shot + INT4/INT6/INT8 inline compression, MI300-only: https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html ; FP4 on MI355: https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html ; vLLM PR #19744: https://github.com/vllm-project/vllm/pull/19744
- ROCm 7.2 RCCL 4-NIC / NCCL 2.28 / rocSHMEM GDA: https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html
- aiter Triton GPU-initiated fused collective (Iris): `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`, `aiter/ops/triton/comms/{reduce_scatter,all_gather}.py`.
