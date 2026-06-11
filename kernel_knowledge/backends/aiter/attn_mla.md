---
title: aiter attention & MLA — flash_attn, paged decode, mla_decode_fwd
kind: backend
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
---

# aiter attention & MLA

## TL;DR
aiter owns the attention kernels on AMD serving: **`flash_attn_func`** (MHA prefill), **paged/decode
attention**, and **`mla_decode_fwd`** (DeepSeek Multi-head Latent Attention decode). The MLA decode kernel
is the headline: AMD reports up to **17× vs naive decode** on MI300X, achieved via matrix-absorption (run
MQA instead of MHA) + a hand-tuned asm kernel that the Triton MLA path is several × slower than. These are
the *default* attention kernels; `--attention-backend` only overrides which one runs, not whether aiter is
on.

## Concepts

### MLA decode (`aiter/mla.py:mla_decode_fwd`)
```python
mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
               max_seqlen_q, sm_scale=None, logit_cap=0.0, num_kv_splits=None, ...)
```
- `q`: `[B*q_seqlen, num_heads, kv_lora_rank + qk_rope_head_dim]` (e.g. 512 + 64).
- `kv_buffer`: `[num_pages, page_size, num_heads_kv(=1), qk_head_dim]`; in decode `num_heads_kv==1` and
  `page_size==1` use the original (unpaged) representation.
- `o`: `[B*q_seqlen, num_heads, kv_lora_rank]`.
- `sm_scale` defaults to `1/sqrt(qk_head_dim)`.

**Matrix absorption**: the `kv_proj_up` weight is split — `Wuk` absorbed into `q_nope`, `Wuv` into the
attention output — so the layer runs as MQA. This collapses bandwidth and lets the asm kernel saturate
the MFMA pipe.

**`num_kv_splits` (split-KV)**: `get_meta_param` auto-picks the KV-split count and builds
`num_kv_splits_indptr` from batch/total-KV/heads heuristics; a Triton stage-2 (`_fwd_kernel_stage2_asm`)
combines the per-split partials. This is the decode analog of split-K — it fills CUs when batch is small.
Leave `num_kv_splits=None` (auto) unless you are an expert.

### MHA prefill / paged decode
`aiter.flash_attn_func(q, k, v, causal=..., softmax_scale=...)` is the flash MHA prefill entry (asm/CK/
Triton under the hood). Paged/decode attention is exposed via `aiter.paged_attn` / `attention.py`. FP8
KV-cache and fp8 fmha are supported on gfx942 (FNUZ); newer sparse-MLA / paged-MQA-logits paths may be
gfx950-only.

## The levers
- Pick the right entry: `mla_decode_fwd` for DeepSeek decode, `flash_attn_func` for MHA prefill.
- vLLM: `VLLM_MLA_DISABLE=0`, `VLLM_USE_AITER_MLA`, `VLLM_USE_TRITON_FLASH_ATTN=0` to keep the asm/CK MLA
  path; `--attention-backend` selects the kernel but `VLLM_ROCM_USE_AITER=1` is still required.
- KV-cache dtype (bf16 vs fp8 FNUZ) trades accuracy for bandwidth.

## Numerics / parity
MLA matrix absorption is algebraically equivalent to standard MLA (parity-safe in bf16). fp8 KV-cache /
fp8 fmha introduce quant error — validate model accuracy. The Triton MLA reference exists for
correctness cross-checks but is much slower.

## Pitfalls
- gfx942 may fall back to Triton for the newest MLA variants (sparse prefill/decode) → several × slower;
  confirm an asm path exists for your model+shape on gfx942.
- `num_heads_kv`/`page_size` must match the decode contract (`==1`) to hit the fast unpaged path.
- Don't hand-set `num_kv_splits` unless you have measured it; the auto heuristic is shape-aware.

## How to verify
`AITER_LOG_MORE=1` to confirm the asm MLA kernel (not Triton) fires; benchmark decode tok/s. The AMD
MLA blog + AI-Developer-Hub notebook give a runnable `mla_decode_fwd` example to confirm the path on-box.

## Alternatives / cross-links
operators: `mla_attention`, `attention_prefill_fmha`, `attention_decode_paged` ·
[`backends/flash_attention_rocm/`](../flash_attention_rocm/) · [integration.md](integration.md).

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`: `aiter/mla.py` (`mla_decode_fwd`,
  `get_meta_param`, `_fwd_kernel_stage2_asm`).
- 17× MLA decode + matrix absorption (AMD-reported, MI300X, tested 2025-03):
  https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- `mla_decode_fwd` signature/example: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
