---
title: mla_attention on flash_attention_rocm — SOTA card
kind: sota_card
operator: mla_attention
backend: fa_rocm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: na
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
---

# mla_attention × flash_attention_rocm (`fa_rocm`)

## TL;DR (status: na — use aiter MLA instead)
`fa_rocm` (Dao-AILab FlashAttention ROCm, CK + Triton backends) is a **standard MHA/GQA FlashAttention**
package — it has **no dedicated MLA path**. MLA's compressed-latent + decoupled-RoPE attention (effective
head_dim 576, MQA-on-latent with matrix absorption) is not the FA-2 contract `fa_rocm` implements (CK FA
caps head_dim at 256; the Triton backend has no absorbed-MLA kernel). DeepSeek MLA on AMD goes through
**aiter** (`mla_decode_fwd` / `mla_prefill_fwd`, surfaced in vLLM as `ROCM_AITER_MLA` /
`ROCM_AITER_TRITON_MLA`). **Use [aiter.md](aiter.md); there is no `fa_rocm` MLA seam to wire.**

## Why na (not just "slower")
- **CK FA backend**: head_dim ≤256 hard limit → cannot represent the 576-wide latent+rope attention.
- **Triton FA backend** (aiter kernels): implements MHA/GQA FA, not the absorbed-MLA decode (`Wuk`/`Wuv`
  folding + MQA on the latent). The MLA kernels live separately in `aiter/mla.py` and
  `aiter/ops/triton/attention/mla_decode.py`, not in the FA `flash_attn_func` surface.
- vLLM exposes MLA as its own backends (`ROCM_AITER_MLA`, `TRITON_MLA`), distinct from the FA backends
  (`ROCM_AITER_FA`) — confirming MLA is not served through `fa_rocm`.

## What to do instead
- Serving: `--attention-backend ROCM_AITER_MLA` (vLLM) / `--attention-backend aiter` (sglang) → aiter
  `mla_decode_fwd`. See [aiter.md](aiter.md).
- Reference/editable: Triton MLA (`mla_decode.py`) — [triton.md](triton.md).

## Sources
- FA-ROCm is MHA/GQA FA, CK head_dim ≤256, no MLA kernel: https://github.com/Dao-AILab/flash-attention
- MLA served via dedicated AITER MLA backends (not FA): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- MLA kernels live in aiter, not the FA surface: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py`.
