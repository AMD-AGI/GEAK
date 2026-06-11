---
title: chunked_prefill on vllm_kernels — SOTA card
kind: sota_card
operator: chunked_prefill
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# chunked_prefill × vllm_kernels

## TL;DR
This card is the **vLLM (V1) dispatch** for chunked prefill. vLLM V1 does chunked prefill by default
(scheduler mixes prefill chunks + decode tokens per step) and routes the attention through `TRITON_ATTN`
(the **unified attention** kernel — one launch for prefill+decode), `ROCM_AITER_FA`, or `ROCM_ATTN`. Use
the auto-selected default (`ROCM_AITER_FA` / `TRITON_ATTN` unified); the chunk size is the
`--max-num-batched-tokens` lever.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `TRITON_ATTN` unified attention | `vllm-project/vllm:vllm/v1/attention/backends/triton_attn.py` (aiter `unified_attention`) | gfx942/950; bf16/fp16/fp8 | one-launch prefill+decode; part of AITER FA 1.2–4.4× TPS (vendor) | launch-bound mixed batches |
| `ROCM_AITER_FA` (prefill+decode) | `rocm_aiter_fa.py` | as above | AITER FA path | default MHA mixed serving |

## Config space / knobs
`--max-num-batched-tokens N` / `--enable-chunked-prefill` (scheduler chunk; V1 default on).
`--attention-backend {TRITON_ATTN, ROCM_AITER_FA, ROCM_ATTN}`. Master `VLLM_ROCM_USE_AITER=1` +
`VLLM_ROCM_USE_AITER_MHA=1`. `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (AITER FA, concurrency ≥32).
`--kv-cache-dtype fp8_e4m3` (fnuz). See [../tuning.md](../tuning.md) and
`backends/vllm_kernels/rocm_kernels.md`.

## Numerics / parity
fp32 online-softmax over segmented KV; chunk-invariance gate; reduction order differs across backends →
re-check greedy temp=0 parity. fp8 KV accuracy gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Pure dispatch: `--attention-backend` enum + `VLLM_ROCM_USE_AITER*` hierarchy. Use upstream
`vllm/vllm-openai-rocm` (V1 only). Dispatch order on gfx942: ROCM_ATTN → ROCM_AITER_UNIFIED_ATTN →
TRITON_ATTN (`vllm/platforms/rocm.py`).

## Pitfalls & anti-patterns
- V0-era flags ignored on V1.
- Unified kernel causal only.
- AITER CK crash under HIP-graph capture for novel shapes → force Triton.
- Confirm active backend from logs, not the flag.

## How to verify
rocprofv3 Top-N (one unified kernel vs two); chunk-invariance test; greedy temp=0 parity after a swap;
throughput + TPOT + TTFT at the chunk size.

## Alternatives / cross-links
[triton.md](triton.md) · [aiter.md](aiter.md) · [sglang_kernels.md](sglang_kernels.md) ·
`backends/vllm_kernels/overview.md`, `rocm_kernels.md` ·
[[../../attention_decode_paged/backends/vllm_kernels.md]] · [[../overview.md]].

## Sources
- vLLM `TRITON_ATTN` unified attention: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
- ROCm dispatch order: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- AITER FA mixed-serving TPS (vendor): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
