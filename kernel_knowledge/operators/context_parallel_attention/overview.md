---
title: context_parallel_attention — overview
kind: operator_overview
operator: context_parallel_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# context_parallel_attention  (ring / all-to-all context parallel)

## TL;DR
Context (sequence) parallelism splits a **single long sequence across GPUs** so that the KV cache and the
O(N²) attention compute are sharded — essential when one device's HBM can't hold a 256K+ context.
**Ring attention** keeps Q local and **rotates K/V blocks around a GPU ring** (overlapping the
point-to-point KV send with the local FA tile), accumulating with online-softmax LSE-merge.
**All-to-all (Ulysses)** instead shards the sequence and does an all-to-all to switch to head-parallel for
the attention, then all-to-all back. On AMD this is **two layers**: the local attention tile is the same
FA kernel ([[attention_prefill_fmha]] via Triton/aiter), and the **partial-result merge across ranks**
(LSE rescale) + the **collective** (P2P ring / all-to-all). The single most important fact on MI300X: the
collective rides **Infinity Fabric / XGMI (~448 GB/s/dir)**, slower than NVLink, so **stay within one
8-GPU XGMI island (TP) before reaching for CP**, and use `allgather_reducescatter`, not NCCL all-to-all.

## Math contract
Standard attention `O = softmax(QKᵀ·scale + mask)V`, computed with the sequence sharded across `cp` ranks.
- **Ring**: rank `r` holds `Q_r, K_r, V_r`. Over `cp` steps each rank computes a local FA tile against the
  currently-held K/V block, **merges** into a running `(O, m, l)` via online-softmax LSE, then **sends**
  its K/V block to the next rank and receives the previous. Causal load-balancing uses a **zigzag**
  assignment so every rank does equal work.
- **All-to-all (Ulysses)**: all-to-all on the head dimension → each rank gets all tokens for a subset of
  heads → local full-seq FA → all-to-all back. One collective per direction, larger messages.
- The cross-rank merge is the FA **LSE-rescale** (`merge_state` / `merge_attn_states`): combine partial
  outputs with their log-sum-exp, fp32.

## Shape regimes
- **Prefill only** (TTFT for very long context, 128K–1M). Decode generally uses TP/paged KV, not CP.
- The local tile is a normal FA prefill at `seq/cp`; the operator's new cost is the **ring P2P / all-to-all
  + LSE merge** overlapped with compute. CP shines when `seq/cp` is still large (compute hides comm).

## Where it matters (Amdahl)
Long-context prefill is O(N²); CP is what makes 256K+ TTFT feasible when KV exhausts HBM. On MI300X the
limiter is interconnect: a poorly-overlapped ring exposes XGMI latency and tanks scaling. The win comes
from (a) overlapping the KV rotation with the local FA tile, (b) zigzag causal balancing, (c) keeping CP
inside the 8-GPU island. ROCm long-seq FA already lags CUDA ~20–25% at 32K+, so the local kernel must be
tuned too ([[attention_prefill_fmha]]).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 (local FA tile + LSE merge; the portable CP kernel layer) | [backends/triton.md](backends/triton.md) |
| aiter | 🟢 (local FA + `merge_attn_states` + Iris GPU-initiated comm primitives) | [backends/aiter.md](backends/aiter.md) |
| sglang_kernels | 🟡 (zigzag ring-attention + split-KV transfer; PD-disagg, maturing on ROCm) | [backends/sglang_kernels.md](backends/sglang_kernels.md) |

## Fusion neighbors
LSE-merge of partial outputs ([[reduction]]), the collective ([[all_to_all_dispatch_combine]] / allgather /
reduce-scatter), KV transfer for PD-disaggregation, fp8 KV. See [fusion.md](fusion.md).

## Numerics
LSE-merge must be associative & fp32; ring vs all-to-all must match a single-GPU reference. See
[numerics.md](numerics.md).

## How to bench
Long-context prefill (e.g. 128K, cp=8 on one node); measure TTFT scaling vs cp and comm/compute overlap;
oracle = single-GPU full attention at the same seq (if it fits) or a CP-disabled reference at shorter seq.
See [tuning.md](tuning.md).

## Sources
- SGLang prefill CP / zigzag ring attention + split-KV transfer (256K+, PD-disagg): https://github.com/sgl-project/sglang/issues/22223
- ROCm: stay within 8-GPU XGMI island (TP), `allgather_reducescatter` not NCCL all-to-all: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; vLLM ROCm guidance
- aiter Iris GPU-initiated comm (reduce-scatter/all-gather), long-context up to 2×: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- vLLM ROCm extend/chunked-context path, long-seq gap: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
