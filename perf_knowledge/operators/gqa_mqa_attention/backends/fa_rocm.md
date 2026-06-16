---
title: gqa_mqa_attention on flash_attention_rocm — SOTA card
kind: sota_card
operator: gqa_mqa_attention
backend: fa_rocm
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/ROCm/aiter
---

# gqa_mqa_attention × flash_attention_rocm (`fa_rocm`)

## TL;DR
MQA/GQA is a first-class feature of **both** FA-ROCm backends: CK (default, fp16/bf16, head_dim ≤256) and
Triton (aiter kernels, + fp8 / arbitrary head dim). `flash_attn_func` just takes a smaller
`num_kv_heads` and broadcasts. It is the portable GQA API; for serving the dedicated AITER FA path usually
wins (1.2–4.4× TPS). Pick CK for stable half-precision GQA, Triton for fp8 KV / head_dim>256.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FA-ROCm CK backend (GQA) | `Dao-AILab/flash-attention` + composable_kernel | gfx90a/942/950; fp16/bf16; head_dim ≤256 | mature default | stable half-precision GQA, training |
| FA-ROCm Triton backend (GQA + fp8) | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`; aiter kernels | + fp8, arbitrary head dim | feature-rich | fp8 KV / head_dim>256 GQA |

## Config space / knobs
`num_kv_heads < num_q_heads` (the GQA setting); backend select via
`FLASH_ATTENTION_TRITON_AMD_ENABLE`; Triton autotune / config-json. vLLM `VLLM_USE_TRITON_FLASH_ATTN`.
See [triton.md](triton.md) and [ck.md](ck.md).

## Numerics / parity
GQA bit-identical to MHA-with-shared-KV; fp32 accumulate. CK no fp8; fp8 KV GQA is Triton + accuracy gate.
See [../numerics.md](../numerics.md).

## Integration (rebind seam)
`flash_attn_func(q, k, v, ...)` with `k`/`v` at `num_kv_heads`. In vLLM this is `ROCM_AITER_FA` (the AITER
FA path, recommended default).

## Pitfalls & anti-patterns
- CK head_dim ≤256 / no fp8; for fp8 KV or head_dim>256 GQA use Triton.
- Triton SWA is WIP — for SWA GQA models use CK.
- Dedicated AITER FA beats generic FA-ROCm on serving — prefer it.
- Don't pre-`repeat_kv`.

## How to verify
CK: `pytest tests/test_flash_attn_ck.py`; Triton: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pytest
tests/test_flash_attn_triton_amd.py`. Confirm active backend from logs; bench at model ratio vs AITER FA;
greedy temp=0 parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [ck.md](ck.md) ·
`backends/flash_attention_rocm/overview.md` · [[../overview.md]].

## Sources
- MQA/GQA both FA-ROCm backends: https://github.com/Dao-AILab/flash-attention
- AITER FA beats generic FA (1.2–4.4× TPS): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- Triton FA kernels = aiter: https://github.com/ROCm/aiter
