---
title: speculative_decode_verify on vLLM kernels — SOTA card
kind: sota_card
operator: speculative_decode_verify
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
  - https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
  - https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus.html
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# speculative_decode_verify × vLLM kernels

## TL;DR
vLLM V1 has **native, first-class spec-decode**: EAGLE / EAGLE3, Medusa, n-gram, and draft-model methods,
with a **rejection sampler** that preserves the target distribution. On MI300X it delivers **2.31×**
(vendor, vLLM) and **3.6×** combined with FP8 (Llama-3.1-405B). The verify attention rides vLLM's ROCm
attention backends (`TRITON_ATTN` / `ROCM_AITER_FA` / `ROCM_ATTN`); the spec logic is framework-level. Use
the V1 spec-decode config; **always** keep `VLLM_ROCM_USE_AITER=1` even when forcing `--attention-backend`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM V1 spec-decode (EAGLE/Medusa/ngram) + rejection sampler | `vllm-project/vllm:vllm/v1/spec_decode/` | gfx942/950; bf16/fp16/fp8 | 2.31× (vLLM, MI300X); 3.6×+FP8 on Llama-3.1-405B (AMD vendor) | production spec-decode on vLLM |
| verify attention | vLLM ROCm backends (`TRITON_ATTN`/`ROCM_AITER_FA`/`ROCM_ATTN`) | gfx942/950 | per the ROCm attention backend card | the verify forward |

## Config space / knobs
- `--speculative-config '{"method":"eagle","model":...,"num_speculative_tokens":N}'` (V1 JSON config);
  also `ngram`, `medusa`, draft-model.
- `VLLM_ROCM_USE_AITER=1` (master, even with `--attention-backend`), `VLLM_ROCM_USE_AITER_MHA=1`.
- `--attention-backend {TRITON_ATTN, ROCM_AITER_FA, ROCM_ATTN}` for the verify forward.
- Combine with FP8 (fnuz on gfx942) for the multiplicative win. See [tuning.md](../tuning.md).

## Numerics / parity
Rejection sampler preserves the target distribution; greedy spec must be token-exact vs non-spec. bf16
MFMA + GQA custom-mask correctness on CDNA3. fnuz fp8. See [numerics.md](../numerics.md).

## Integration (rebind seam)
Spec-decode is configured via `--speculative-config`; the verify attention is the `--attention-backend`
enum. Custom verify kernels wire in via the attention backend, not the spec scheduler.

## Pitfalls & anti-patterns
- Forgetting `VLLM_ROCM_USE_AITER=1` when forcing a backend → AITER GEMM/RMSNorm/MoE stay off (only the
  attention kernel is overridden).
- `ROCM_ATTN` decode fallback cliff (2.7–4.4× slower) when KV head size is unsupported by HIP paged attn →
  the verify pass falls to Triton decode. Verify the head size is supported.
- FP4 batched matmul crashes gfx942 (`VLLM_ROCM_USE_AITER_FP4BMM=0`) — unrelated but a common MI300X trap.
- CDNA3 bf16/GQA custom-mask verify bugs (see SSD) — gate acceptance rate.

## How to verify
Greedy token-exactness vs non-spec; accepted tokens/step; rocprofv3 Top-N to see the verify kernel
(`*ck_*`/`fmha_*` = AITER/CK, Python name = Triton, `paged_attention_ll4mi_*` = vLLM custom HIP); confirm
the spec method in the log.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [aiter.md](aiter.md) ·
[sglang_kernels.md](sglang_kernels.md) · backend: [[vllm_kernels]] · core: [[attention_decode_paged]].

## Sources
- vLLM spec-decode (EAGLE/Medusa/ngram, rejection sampler, V1 config): https://docs.vllm.ai/en/latest/features/speculative_decoding/
- AMD spec-decode MI300X (2.31× / 3.6×+FP8): https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html ; https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus.html
- vLLM ROCm attention backends / `VLLM_ROCM_USE_AITER` requirement: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
