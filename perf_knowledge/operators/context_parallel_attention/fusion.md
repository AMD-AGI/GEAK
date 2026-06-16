---
title: context_parallel_attention — fusion
kind: technique
operator: context_parallel_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# context_parallel_attention — fusion

## The fusion that defines CP: comm ⊗ compute
CP's performance is about **overlapping** the collective with the local attention, and **fusing** comm
into the kernel where possible:

| fused/overlapped piece | where | benefit |
|---|---|---|
| **KV-block P2P send/recv ⊗ local FA tile** | ring loop | hide XGMI transfer under compute (the whole point) |
| **GPU-initiated comm in-kernel** | aiter Iris (`comms/iris.py`) | Triton kernel issues all-gather/reduce-scatter inline → tighter overlap than a separate RCCL call |
| **fused reduce-scatter + RMSNorm + quant + all-gather** | aiter `comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` | collapses a comm+norm+quant chain (TP/SP epilogue) into one pass |
| **incremental LSE-merge** | per received KV block | merge as blocks arrive; don't buffer all then merge |
| **split-KV transfer for PD-disagg** | prefill→decode KV handoff | overlap KV transfer with the next ring step (SGLang proposal) |

## What does NOT fuse
- The local FA tile and the LSE-merge stay as the standard FA kernels ([[attention_prefill_fmha]]); CP
  wraps them, it doesn't replace the matmul.
- All-to-all (Ulysses) does a single local full-seq FA with no cross-rank merge — fewer fusion seams but
  bigger messages; pick ring vs all-to-all by message-size vs overlap tradeoff.

## Cross-links
- Collectives: [[all_to_all_dispatch_combine]] · [[allgather]] · [[reduce_scatter]] ·
  [[fused_allreduce_rmsnorm]] (related fused comm).
- Local attention: [[attention_prefill_fmha]] · [[chunked_prefill]] (extend/prefix path).
- KV: [[paged_kv_copy]] · [[kv_cache_quant]].
- Languages/backends: [[triton_amd]] · [[aiter]] · [[sglang_kernels]] · [[mori_rccl]] (EP all-to-all).

## Sources
- aiter fused comm + Iris GPU-initiated primitives: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/` (on-box) ; https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- SGLang split-KV transfer + zigzag ring: https://github.com/sgl-project/sglang/issues/22223
