---
title: speculative_decode_verify on aiter — SOTA card
kind: sota_card
operator: speculative_decode_verify
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
  - https://github.com/sgl-project/sglang/issues/16027
  - https://github.com/ROCm/ATOM
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# speculative_decode_verify × aiter

## TL;DR
aiter provides the **tuned verify attention** (unified attention with a verify mode) used by sglang's
`SGLANG_AITER_UNIFIED_VERIFY`, and AMD's **ATOM** (AiTer Optimized Model) ships MTP/EAGLE spec-decode with
AITER kernels. aiter is the fastest verify path **when it engages cleanly** — but EAGLE + the AITER
attention backend have hit real integration bugs on ROCm (HIP-graph draft-extend crash). Use aiter via the
framework with `VLLM_ROCM_USE_AITER=1` / `SGLANG_USE_AITER=1`; keep a Triton verify fallback ready.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter unified verify attention | `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/attention/unified_attention.py` (verify mode) | gfx942/950; bf16/fp16/fp8 | part of AMD's spec stack (2.31× / 3.6×+FP8 vendor) | supported EAGLE/MTP verify |
| ATOM (MTP + EAGLE proposer, AITER kernels) | `ROCm/ATOM` | gfx942/950; +FP8/MXFP4/INT8 | AMD-curated spec-decode model stack | turnkey AITER spec-decode |

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` (master — required even with `--attention-backend`), `VLLM_ROCM_USE_AITER_MHA=1`.
- sglang: `SGLANG_USE_AITER=1`, `--attention-backend aiter`, `SGLANG_AITER_UNIFIED_VERIFY` for the unified
  verify kernel.
- Combine with FP8 (fnuz on gfx942) for the multiplicative speedup. See [tuning.md](../tuning.md).

## Numerics / parity
fp32 accumulate; greedy token-exact vs non-spec; AITER attention has shown eval regressions on some
models — accuracy-gate. fnuz fp8. See [numerics.md](../numerics.md).

## Integration (rebind seam)
aiter is the live verify path; engage via flags. Verify engagement: `AITER_LOG_MORE=1`, backend banner. If
EAGLE draft-extend crashes under HIP-graph capture, fall back to Triton verify or disable graph capture
for the draft phase.

## Pitfalls & anti-patterns
- **EAGLE + AiterAttnBackend HIP-graph crash** (sglang #16027: missing `max_split_per_batch` during draft-
  extend graph capture) — the main-model graph captures fine; the draft-extend one fails. Known, maturing.
- AITER MLA/attn accuracy regressions on some models → gate.
- gfx942 coverage gaps → Triton fallback (slower verify).
- Don't assume aiter has a bespoke "tree attention" op — verify is a mode on unified attention + custom mask.

## How to verify
Greedy token-exactness vs non-spec; accepted tokens/step; `AITER_LOG_MORE=1` for the verify path; test
EAGLE draft-extend under HIP-graph capture before trusting it.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [sglang_kernels.md](sglang_kernels.md) ·
[vllm_kernels.md](vllm_kernels.md) · backend: [[aiter]] · core: [[attention_decode_paged]].

## Sources
- aiter unified attention (verify mode): `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/attention/unified_attention.py` (on-box).
- AMD spec-decode (2.31× / 3.6×+FP8): https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
- ATOM (MTP/EAGLE + AITER): https://github.com/ROCm/ATOM
- EAGLE+AITER HIP-graph crash: https://github.com/sgl-project/sglang/issues/16027
