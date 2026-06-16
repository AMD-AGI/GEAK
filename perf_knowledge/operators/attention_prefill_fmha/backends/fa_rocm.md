---
title: attention_prefill_fmha on flash_attention_rocm — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: fa_rocm
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32, fp8_e4m3]
regimes: [prefill]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/ROCm/flash-attention
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_prefill_fmha × flash_attention_rocm (`fa_rocm`)

## TL;DR
`fa_rocm` is the ROCm build of Dao-AILab FlashAttention — the `flash_attn_func` API most code imports.
It is **not one kernel**: it ships **two backends**, **CK (default)** and **Triton (aiter kernels)**,
selectable by env. Choose CK for stable fp16/bf16 within head_dim ≤256 (and SWA); choose Triton for fp8 /
arbitrary head dim / ALiBi. For serving, the newer AITER FA backends usually beat the generic `fa_rocm`
paths — so `fa_rocm` is the **portable drop-in**, not the serving SOTA.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK backend (default) | `Dao-AILab/flash-attention` + `ROCm/composable_kernel` submodule | gfx90a/942/950; fp16/bf16; head_dim ≤256 fwd+bwd | mature default; ~CK-Tile FMHA perf | stable half-precision, SWA, training bwd |
| Triton backend (aiter kernels) | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`; `third_party/aiter` | + fp32 + fp8, arbitrary head dim, ALiBi/rotary/paged | feature-rich | fp8 / head_dim>256 / ALiBi |

## Config space / knobs
- Backend select: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (install **and** runtime) → Triton; unset → CK.
- Build arch: `FX_GFX_ARCHS=gfx90a;gfx942` (CK); `BUILD_TRITON=1`; `FA_BRANCH` pins the CK branch;
  `BUILD_FA=0` drops FA → SDPA.
- Triton autotune: `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` or pin
  `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`.
- vLLM: `VLLM_USE_TRITON_FLASH_ATTN` (=0 → CK; needed for SWA).

## Numerics / parity
Both backends: fp32 online-softmax accumulate. CK has **no fp32** (falls to Triton/SDPA). fp8 is Triton
(FA-v3 interface), FNUZ on gfx942. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
`flash_attn_func(q,k,v,causal=...,softmax_scale=...)` — standard FA-2 API. Backend is an env/build choice,
not a separate import. vLLM wraps it as `ROCM_AITER_FA` (the AITER FA path) on V1.

## Pitfalls & anti-patterns
- **head_dim > 256 → CK can't run** (hard limit); switch to Triton.
- **CK has no SWA in the core FA path / Triton SWA is WIP** → for SWA, generally `VLLM_USE_TRITON_FLASH_ATTN=0` (CK SWA-supporting path) — confirm per model/version.
- Triton backend must be **enabled at install** too (not just runtime), or kernels aren't built.
- vLLM V1 may report a different active backend than the legacy flag requests — verify from logs.

## How to verify
CK: `pytest tests/test_flash_attn_ck.py`; Triton:
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pytest tests/test_flash_attn_triton_amd.py`. Confirm the active
backend from the log (don't trust the env flag alone on V1). Micro-bench CK vs Triton vs AITER FA.

## Alternatives / cross-links
[ck_tile.md](ck.md) · [triton.md](triton.md) · [aiter.md](aiter.md) ·
`backends/flash_attention_rocm/overview.md`, `ck_backend.md`, `triton_backend.md` · [[../overview.md]].

## Sources
- FA-ROCm two backends / env flags / head_dim ≤256 / feature table: https://github.com/Dao-AILab/flash-attention ; https://github.com/ROCm/flash-attention
- AITER FA beats generic FA (1.2–4.4× TPS): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
