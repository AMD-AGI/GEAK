---
title: allgather on HIP / Triton-Iris — SOTA card
kind: sota_card
operator: allgather
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/all_gather.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
  - https://github.com/ROCm/iris
---

# allgather × HIP / Triton-Iris (GPU-initiated)

## TL;DR
> The custom all-gather path on AMD is **GPU-initiated comm**: aiter's Triton `all_gather` (push-based
> `iris.put` over a symmetric heap) and the fused SP kernel. Reach for it to **fuse** the AG into a Triton
> kernel (the RS+RMSNorm+quant+AG SP pipeline) or to author an xGMI P2P gather with custom overlap. RCCL is
> the default for a standalone AG; this is the fusion/authoring seam.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter Triton `all_gather` (Iris) | `aiter/ops/triton/comms/all_gather.py` | gfx942/950; bf16/fp16/fp8 | — (GPU-initiated, fusible) | fusing AG into a Triton kernel |
| fused RS+RMSNorm+quant+AG | `.../comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` | gfx942/950 | — | the SP rewrite of all-reduce (one kernel) |
| HIP xGMI P2P gather | `languages/hip_cpp/` + IPC buffers | gfx942/950 | — | bespoke gather/overlap |

## Config space / knobs
- Iris: symmetric heap (`calculate_heap_size`), `iris.put` push, persistent-grid PID mapping, grid = CU
  count. Requires the `iris` package + a symmetric memory init.
- HIP P2P: IPC handle exchange, `__builtin_amdgcn_*` / `global_load` over xGMI; SDMA offload for the copy.
- Overlap: separate stream / `GPU_MAX_HW_QUEUES=2`.

## Numerics / parity
Pure copy → parity-safe; the fused SP kernel's RMSNorm/quant stages carry the numeric concern (see
[[fused_allreduce_rmsnorm]]), not the gather. See [numerics.md](../numerics.md).

## Integration (rebind seam)
aiter Triton comms under `aiter/ops/triton/comms/`; the fused SP kernel is the drop-in for AR+norm in an SP
layout. Edit the Triton/HIP source for a custom gather.

## Pitfalls & anti-patterns
- Iris requires the symmetric heap + `iris` installed; experimental — benchmark vs RCCL.
- AG on CUs vs SDMA — prefer SDMA for the standalone copy; the fused kernel intentionally uses CUs (it's
  doing norm/quant too).

## How to verify
rocprof confirms the Triton/Iris kernel ran; structural shard-layout test; bandwidth vs RCCL (expect lower
standalone — the value is fusion).

## Alternatives / cross-links
[mori_rccl.md](rccl.md) (default) · [[reduce_scatter]] · [[fused_allreduce_rmsnorm]] ·
[`languages/triton_amd/`](../../../languages/triton_amd/overview.md) · [overview.md](../overview.md).

## Sources
- aiter Triton Iris all_gather + fused SP kernel: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/all_gather.py`, `.../fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- Iris: https://github.com/ROCm/iris
