---
title: context_parallel_attention on aiter — SOTA card
kind: sota_card
operator: context_parallel_attention
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# context_parallel_attention × aiter

## TL;DR
aiter supplies both halves of CP on AMD: the **tuned local FA tile** (MHA/MLA prefill) and the
**collectives** — including **Iris GPU-initiated Triton comm primitives** (`all_gather`, `reduce_scatter`,
and a fused `reduce_scatter_rmsnorm_quant_all_gather`) plus `custom_all_reduce`/`quick_all_reduce` RCCL
bypass. There is no single aiter "ring_attention" op; CP is assembled from these. Use aiter when you want
the tuned local kernel + in-kernel comm overlap; the ring orchestration still comes from the framework.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter MHA/MLA prefill (local tile) | `ROCm/aiter@a6bb49937:aiter/ops/mha.py`, `aiter/mla` | gfx942/950; bf16/fp16/fp8 | MHA prefill up to **14×** vs naive; long-context up to **2×** (AMD vendor, MI300X, 2025) | the local FA tile at `seq/cp` |
| aiter Iris comm primitives | `aiter/ops/triton/comms/{iris,all_gather,reduce_scatter}.py`, `comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` | gfx942/950 | GPU-initiated, Triton-based; no public CP-scaling number | in-kernel comm/compute overlap |
| custom/quick all-reduce | `aiter/ops/custom_all_reduce.py` | gfx942/950 | RCCL bypass | small-message collectives |

> Vendor 14×/2× are **local-attention** speedups vs naive, not CP-scaling numbers. No published end-to-end
> AMD ring-attention scaling figure as of 2026-06.

## Config space / knobs
- `SGLANG_USE_AITER=1` / `VLLM_ROCM_USE_AITER=1` (engage aiter); `SGLANG_USE_AITER_AR/AG` for custom AR/AG.
- `VLLM_ALL2ALL_BACKEND="allgather_reducescatter"`, `--disable-nccl-for-dp-synchronization` on ROCm.
- Stay within the 8-GPU XGMI island (TP) before CP; `NCCL_MIN_NCHANNELS=112`, `HSA_NO_SCRATCH_RECLAIM=1`.

## Numerics / parity
fp32 LSE-merge; fp8 KV around the ring must keep the fnuz dialect consistent across ranks. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
aiter is the live path — engage via framework flags; the Iris primitives are importable Triton ops you can
fuse into a custom ring loop. Verify: `AITER_LOG_MORE=1` (local tile on tuned vs fallback), profile XGMI
overlap.

## Pitfalls & anti-patterns
- aiter custom AR has had segfaults (#1542) → `SGLANG_USE_AITER_AR=0` to fall back.
- gfx942 coverage gaps: newest paths may be gfx950-only → Triton fallback (slower local tile).
- No turnkey aiter ring op — you still orchestrate CP; don't expect a one-call kernel.

## How to verify
TTFT scaling vs `cp`; CP parity vs single-GPU; `AITER_LOG_MORE=1`; `rocprofv3` XGMI utilization.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [sglang_kernels.md](sglang_kernels.md) ·
backend: [[aiter]] · [[mori_rccl]] (EP all-to-all) · collectives: [[allgather]] · [[reduce_scatter]] ·
[[fused_allreduce_rmsnorm]].

## Sources
- aiter comm primitives (Iris, all_gather, reduce_scatter, fused): `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/` (on-box).
- aiter local-attention speedups (14× MHA, 2× long-context) + Iris: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- ROCm topology / allgather_reducescatter: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
