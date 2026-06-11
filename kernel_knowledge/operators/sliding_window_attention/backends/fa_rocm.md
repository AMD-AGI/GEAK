---
title: sliding_window_attention on FlashAttention-ROCm — SOTA card
kind: sota_card
operator: sliding_window_attention
backend: fa_rocm
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/ROCm/flash-attention
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# sliding_window_attention × FlashAttention-ROCm (`fa_rocm`)

## TL;DR
FlashAttention-ROCm exposes SWA via `window_size=(left,right)` over **two backends**: **CK** (default,
the reliable SWA path) and **Triton** (kernels from the aiter submodule; SWA historically WIP). For SWA
models on ROCm the documented recipe is the **CK** backend (`VLLM_USE_TRITON_FLASH_ATTN=0`); pick the
Triton backend only when you need fp8 / arbitrary head dim / ALiBi and your version's Triton SWA is ready.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FA-ROCm **CK** backend + `window_size` | `Dao-AILab/flash-attention` (ROCm), `ROCm/flash-attention` | gfx90a/942/950; fp16/bf16; head_dim ≤256 | default; the SWA path vLLM selects with `VLLM_USE_TRITON_FLASH_ATTN=0` | general SWA (Mistral/Gemma/Qwen2/3) |
| FA-ROCm **Triton** backend (aiter kernels) | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | gfx942/950; +fp8, arbitrary head dim, ALiBi | SWA WIP upstream — verify per version | fp8 / head_dim>256 / ALiBi SWA |

## Config space / knobs
- Backend select: `FLASH_ATTENTION_TRITON_AMD_ENABLE` (FA level), `VLLM_USE_TRITON_FLASH_ATTN` (vLLM).
- `window_size=(left,right)`; `(-1,-1)` = full; causal SWA = `(W-1,0)`.
- Triton backend autotune: `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` or pin
  `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`.
- Build: CK arch list `FX_GFX_ARCHS=gfx90a;gfx942`, `BUILD_TRITON=1` (Triton backend must be enabled at
  **install** too, not just runtime).

## Numerics / parity
fp32 online-softmax; CK FA is fp16/bf16 only (no fp32 → use Triton/SDPA). Window edge / sink semantics
per [numerics.md](../numerics.md). fp8 (Triton only) accuracy-gate.

## Integration (rebind seam)
- vLLM: `VLLM_USE_TRITON_FLASH_ATTN=0` → CK (SWA); on V1 verify the banner (legacy flag can be ignored).
- Direct: `flash_attn_func(..., window_size=(left,right))`.

## Pitfalls & anti-patterns
- **CK has SWA; Triton SWA is WIP** → default to CK for SWA models. Confirm per model/version.
- CK head_dim ≤256 and fp16/bf16 only.
- Triton backend must be built at install (`BUILD_TRITON=1`) or its kernels aren't present.
- vLLM V1 may report Triton even with the legacy flag set.

## How to verify
Tests: CK `pytest tests/test_flash_attn_ck.py`; Triton
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pytest tests/test_flash_attn_triton_amd.py`. Micro-bench CK vs
Triton at `(B,H,S,D,window,causal,dtype)`; greedy temp=0 parity vs dense band mask.

## Alternatives / cross-links
[overview.md](../overview.md) · [ck.md](ck.md) · [triton.md](triton.md) · [aiter.md](aiter.md) ·
backend: [[flash_attention_rocm]] (overview) · core: [[attention_prefill_fmha]].

## Sources
- FA window_size + two-backend + build flags: https://github.com/Dao-AILab/flash-attention ; https://github.com/ROCm/flash-attention
- ROCm SWA → CK, V1 banner caveat, 7 backends: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
