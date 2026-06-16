---
title: moe_dispatch_combine on Triton — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: experimental
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/all_gather.py
  - https://github.com/ROCm/iris
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
---

# moe_dispatch_combine × Triton

## TL;DR
> Triton handles the **local** permute/scatter (the gather/scatter inside a single-GPU fused-MoE Triton
> kernel) well, and there is an **experimental** GPU-initiated comms path: aiter ships Triton
> reduce-scatter / all-gather built on **Iris** (a SHMEM-like Triton library), and ByteDance's
> Triton-distributed builds EP all-to-all on MoRI-SHMEM. For production EP dispatch/combine, use MoRI-EP
> (HIP) — Triton is the local-permute path and an emerging distributed-comm authoring surface.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton fused-MoE local permute/scatter | vLLM/sglang `fused_moe.py` Triton | gfx942/950 | — | single-GPU MoE permute (no EP) |
| aiter Triton comms (Iris-based) | `aiter/ops/triton/comms/{all_gather,reduce_scatter}.py` | gfx942/950 | — (experimental) | authoring GPU-initiated comms in Triton |
| Triton-distributed EP a2a (MoRI-SHMEM) | ByteDance Triton-distributed on MoRI-SHMEM | gfx942 | — | research EP all-to-all in Triton |

Recommend: Triton for local permute; **MoRI-EP (HIP)** for distributed dispatch/combine.

## Config space / knobs
- Local permute: `BLOCK` over tokens; wave64 `num_warps=2–4`; `knobs.amd.use_buffer_ops` for masked
  scatter; fp32 index math.
- Iris comms: GPU-initiated `iris.put`/`load`/`store` over symmetric heap; persistent-grid PID mapping;
  `grid` sized to CU count. Heap size via `calculate_heap_size`.

## Numerics / parity
Same as the operator: fp8 dispatch quant gate, bf16 combine, masked pad. Triton local permute is a pure
gather (lossless) — the numeric risk is downstream. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM/SGLang: `VLLM_ROCM_USE_AITER=0` (or Triton fused-MoE) uses the Triton permute.
- aiter Iris comms: `aiter/ops/triton/comms/` (requires the `iris` package).

## Pitfalls & anti-patterns
- Triton GPU-initiated comms are **experimental** — don't ship as the production EP path.
- `num_warps=8` from NVIDIA → spill on the permute kernel; use 2–4.
- Iris requires the symmetric heap + the `iris` library installed.

## How to verify
rocprof: confirm the Triton permute/comms kernel ran; round-trip identity + greedy parity; for Iris,
sanity-check bandwidth vs the MoRI-EP table (expect lower — it's experimental).

## Alternatives / cross-links
[mori_rccl.md](mori.md) (production EP) · [hip.md](hip.md) · [aiter.md](aiter.md) ·
[`languages/triton_amd/`](../../../languages/triton_amd/overview.md) · [overview.md](../overview.md).

## Sources
- aiter Triton Iris comms: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/{all_gather,reduce_scatter,iris}.py`.
- Iris (Triton SHMEM): https://github.com/ROCm/iris
- Triton fused-MoE reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- Triton-distributed on MoRI-SHMEM: https://github.com/ROCm/mori (MoRI-SHMEM backing ByteDance Triton-distributed).
