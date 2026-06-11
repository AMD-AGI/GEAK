---
title: sliding_window_attention on aiter — SOTA card
kind: sota_card
operator: sliding_window_attention
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/aiter
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# sliding_window_attention × aiter

## TL;DR
aiter is the **dispatcher** that serves SWA in production on sglang/vLLM: its MHA / paged-attention /
unified-attention entrypoints carry a `sliding_window` argument and route to the fastest available impl
(CK / Triton). There is **no separate "SWA kernel"** in aiter — SWA is a parameter on the FMHA/paged
kernels. Use aiter via the framework (`SGLANG_USE_AITER=1` / `VLLM_ROCM_USE_AITER=1`); for the SWA
*mask* itself the underlying engine is CK or the aiter Triton FA (above).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter MHA prefill (`sliding_window`) | `ROCm/aiter@a6bb49937:aiter/ops/mha.py`, `aiter/ops/attention.py` | gfx942/950; bf16/fp16/fp8 | aiter MHA prefill up to **14× vs naive** (AMD vendor, MI300X, 2025-03) — full-attn figure, not SWA-specific | supported MHA SWA models (Llama/Qwen/Mistral) |
| aiter paged decode (windowed KV scan) | `aiter/paged_attn.py` | gfx942/950 | scales gains with context (AMD: up to 2×) | SWA decode, paged KV |
| aiter Triton unified attention | `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` | gfx942/950 | launch-bound small batch | chunked-prefill + decode in one kernel |

> The vendor 14×/2× figures are **full-attention** speedups vs a naive baseline; no SWA-specific measured
> number is published. Treat them as upper context, not an SWA claim.

## Config space / knobs
- Pass `sliding_window` / `window_size` through the framework attention layer; aiter selects CK vs Triton.
- `SGLANG_USE_AITER=1` (master gate), `--attention-backend aiter`; `SGLANG_USE_AITER_UNIFIED_ATTN=1` for
  the unified Triton path on small batches.
- vLLM: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MHA=1`; SWA models still often need the CK mask
  path (`VLLM_USE_TRITON_FLASH_ATTN=0`).

## Numerics / parity
fp32 accumulate; fp8 KV uses fnuz on gfx942 (accuracy-gate). Window/sink semantics inherited from the
underlying CK/Triton kernel. See [numerics.md](../numerics.md).

## Integration (rebind seam)
aiter is the live serving path — to change SWA behavior you change the framework flag / the underlying
CK or Triton kernel, not aiter's dispatch. Verify engagement: `AITER_LOG_MORE=1` (hot shape → asm/CK vs
Triton fallback); backend banner.

## Pitfalls & anti-patterns
- gfx942 coverage gaps: some newest SWA/sparse paths exist only on gfx950 → gfx942 falls back to Triton
  (several × slower). Verify a tuned path exists for your shape.
- Don't assume aiter has a bespoke SWA kernel — it's a flag on FMHA; the mask correctness lives in CK/Triton.
- AITER attention has had accuracy regressions on some models — gate after enabling.

## How to verify
`AITER_LOG_MORE=1` + `grep` the dispatch; isolated bench vs CK/Triton SWA at the served shape; greedy
temp=0 parity vs dense band-mask reference.

## Alternatives / cross-links
[overview.md](../overview.md) · [ck.md](ck.md) · [triton.md](triton.md) ·
[flash_attention_rocm.md](fa_rocm.md) · backend: [[aiter]] (overview) · core: [[attention_decode_paged]].

## Sources
- aiter MHA/paged with sliding_window: `ROCm/aiter@a6bb49937:aiter/ops/mha.py`, `aiter/paged_attn.py`, `aiter/ops/attention.py` (on-box).
- aiter vendor speedups (14× MHA prefill, 2× long-context): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- ROCm SWA / CK fallback / 7 backends: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
