---
title: sliding_window_attention on Triton — SOTA card
kind: sota_card
operator: sliding_window_attention
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/ROCm/aiter
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# sliding_window_attention × Triton

## TL;DR
Triton is the **editable** SWA path: the band mask + KV-block early-out is a few lines on top of the FA
body. Upstream `Dao-AILab/flash-attention`'s **Triton** backend historically marked SWA WIP, but the
aiter-supplied Triton attention kernels carry a `sliding_window` argument (used by sglang's
`unified_attention` / paged-decode paths). Use Triton SWA when you need a feature the CK backend lacks
(fp8, arbitrary head dim, ALiBi) or you're customizing the mask; otherwise CK is the safer default on
ROCm. Honest limit: generic Triton attention is several × slower than a tuned kernel (vLLM/DeepSeek-V4
bring-up).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter Triton FA (`sliding_window` arg) | `ROCm/aiter:aiter/ops/triton/_triton_kernels/attention/` (mha.py, unified_attention.py, flash_attn_triton_amd/) | gfx942/950; bf16/fp16/fp8 | no public per-shape SWA number — measure on-box | fp8 SWA, arbitrary head dim, paged decode SWA |
| FA-ROCm Triton backend | `Dao-AILab/flash-attention` `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | gfx942/950 | SWA WIP upstream — verify per version | when CK feature gaps block you |
| portable NSA/FLA-style band kernel | author in Triton (band loop bound) | gfx942/950 | runs on MI300X via [[triton_amd]] | research / custom mask |

## Config space / knobs
- The SWA-defining lever is the **KV loop bound** (skip blocks fully outside `[i-left, i+right]`), not a
  post-multiply mask — see [tuning.md](../tuning.md).
- AMD Triton knobs: `matrix_instr_nonkdim=16`, `num_warps=4` (avoid 8 → spill), `num_stages=1` (fused
  FA), `waves_per_eu=2–3`, `schedule_hint=attention`, `knobs.amd.use_buffer_ops=ON` for edge loads.
- fp8: use **fnuz** (`fp8e4b8`/`fp8e5b16`) on gfx942, not OCP `e4m3fn`.

## Numerics / parity
fp32 online-softmax; window off-by-one and sink semantics are the real risk. fp8 SWA must be
accuracy-gated (fnuz dialect). See [numerics.md](../numerics.md).

## Integration (rebind seam)
- sglang: `--attention-backend triton` (and the aiter unified path with `SGLANG_USE_AITER_UNIFIED_ATTN=1`).
- vLLM: `VLLM_USE_TRITON_FLASH_ATTN=1` (Triton FA); `TRITON_ATTN` backend on V1.
- The `@triton.jit` SWA kernel is a clean Python rebind seam (overlay a tuned config JSON via
  `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`).

## Pitfalls & anti-patterns
- **Masking instead of skipping** → full-attention cost, zero SWA benefit (the #1 SWA bug).
- `num_warps=8` carried from NVIDIA → VGPR spill (3–5× slower).
- OCP fp8 into `tl.dot` on gfx942 → unsupported; use fnuz.
- Upstream Triton-FA SWA may be WIP on your version → fall back to CK.

## How to verify
Wall-clock scales with `window` not `seq`; greedy temp=0 parity vs dense band-mask reference (prompts
longer than `W`); `TRITON_PRINT_AUTOTUNING=1` to confirm the chosen config; backend banner in log.

## Alternatives / cross-links
[overview.md](../overview.md) · [ck.md](ck.md) · [aiter.md](aiter.md) ·
[flash_attention_rocm.md](fa_rocm.md) · language: [[triton_amd]] · core: [[attention_prefill_fmha]].

## Sources
- aiter Triton FA `sliding_window`: `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/attention/` (on-box).
- FA Triton backend SWA WIP: https://github.com/Dao-AILab/flash-attention
- AMD Triton knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
