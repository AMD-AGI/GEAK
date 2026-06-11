---
title: attention_decode_paged on flash_attention_rocm — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: fa_rocm
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/ROCm/aiter
---

# attention_decode_paged × flash_attention_rocm (`fa_rocm`)

## TL;DR
The FA-ROCm package supports **paged attention** through its **Triton backend** (aiter kernels) — the CK
FA backend is prefill/training-oriented and does not cover the paged/decode feature set. So `fa_rocm`
decode = the Triton-backend paged path, which the dedicated **AITER FA decode backend usually beats**
(1.2–4.4× TPS). Use `fa_rocm` as the portable `flash_attn_*` decode API; for serving, route to AITER FA
([aiter.md](aiter.md)) or vLLM-HIP ([hip.md](hip.md)).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FA-ROCm Triton backend, paged | `Dao-AILab/flash-attention` (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) + `third_party/aiter` | gfx90a/942/950; fp16/bf16/fp8; arbitrary head dim; paged | below dedicated AITER FA decode | portable paged `flash_attn` API |
| (CK backend) | composable_kernel submodule | fp16/bf16, head_dim ≤256 | **no paged/decode feature set** — prefill/training | not the decode path |

## Config space / knobs
Enable Triton backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (install + runtime). Autotune
`FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` or pin `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`. Decode
levers are the Triton paged-decode ones (`NUM_KV_SPLITS`, `BLOCK_N`, `waves_per_eu`) — see
[triton.md](triton.md).

## Numerics / parity
fp32 accumulate; splitKV reduce; fp8 KV (fnuz on gfx942) task-accuracy gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
`flash_attn_with_kvcache(...)` / paged `flash_attn_*` API with the Triton backend enabled. In vLLM this
surfaces as `ROCM_AITER_FA` (the AITER FA path), which is the recommended decode default.

## Pitfalls & anti-patterns
- The **CK backend has no paged/decode feature set** — for decode you must be on the Triton backend (or
  AITER FA).
- Triton backend must be enabled at **install** too, or kernels aren't built.
- Dedicated AITER FA decode beats this generic path → prefer it for serving.
- vLLM V1 may report a different active backend than the legacy flag.

## How to verify
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pytest tests/test_flash_attn_triton_amd.py`; confirm active
backend from logs; isolated decode bench vs AITER FA / vLLM-HIP at the served batch; greedy temp=0 parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) · [vllm_kernels.md](vllm_kernels.md) ·
`backends/flash_attention_rocm/triton_backend.md` · [[../overview.md]].

## Sources
- FA-ROCm Triton backend paged / feature split (CK no paged): https://github.com/Dao-AILab/flash-attention
- AITER FA decode beats generic FA (1.2–4.4× TPS): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- Triton FA kernels = aiter: https://github.com/ROCm/aiter
