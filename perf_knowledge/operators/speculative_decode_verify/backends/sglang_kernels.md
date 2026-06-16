---
title: speculative_decode_verify on SGLang kernels — SOTA card
kind: sota_card
operator: speculative_decode_verify
backend: sglang_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/16027
  - https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
  - https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
---

# speculative_decode_verify × SGLang kernels

## TL;DR
SGLang owns the **spec-decode orchestration** on AMD: EAGLE / EAGLE3 / Medusa / n-gram / MTP draft
workers, the efficient **tree-mask builder** (`build_tree_kernel_efficient`, `TreeMaskMode`), the
**fused KV materialize** Triton op, and the verify dispatch over AITER/Triton attention. This is the most
complete spec-decode path on MI300X; the verify *attention* underneath is AITER or Triton. Integration is
**maturing** (EAGLE+AITER HIP-graph bug), so verify the chosen draft algorithm engages on your model.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| sglang spec workers (EAGLE/EAGLE3/ngram/MTP) | `sgl-project/sglang:python/sglang/srt/speculative/` (`eagle_worker*.py`, `ngram_worker.py`, `multi_layer_eagle_*`) | gfx942/950; bf16/fp16/fp8 | AMD spec stack (2.31× / 3.6×+FP8 vendor) | production spec-decode serving |
| tree-mask builder + fused KV materialize | `speculative/eagle_utils.py` (`build_tree_kernel_efficient`, `TreeMaskMode`), `triton_ops/fused_kv_materialize.py` | gfx942/950 | collapses the small-kernel storm | building/verifying the draft tree |
| unified verify | `SGLANG_AITER_UNIFIED_VERIFY` | gfx942/950 | launch-bound small batch | one-kernel verify |

## Config space / knobs
- `--speculative-algorithm {EAGLE,EAGLE3,NEXTN,STANDALONE,NGRAM}`, `--speculative-num-steps`,
  `--speculative-eagle-topk`, `--speculative-num-draft-tokens` (tree shape).
- `--attention-backend {triton,aiter}` (verify kernel), `SGLANG_AITER_UNIFIED_VERIFY`,
  `TreeMaskMode.FULL_MASK`/`QLEN_ONLY`. ROCm: `SGLANG_USE_AITER=1`, `HSA_NO_SCRATCH_RECLAIM=1`.

## Numerics / parity
Greedy token-exact vs non-spec; tree-mask + GQA + bf16 MFMA correctness; reduction order differs across
backends. See [numerics.md](../numerics.md).

## Integration (rebind seam)
Configured at launch; the spec workers are the orchestration surface (`eagle_worker.py` etc.), the verify
attention is the `--attention-backend` choice. The tree-mask builder / fused-KV-materialize are the
custom-kernel rewrite seams.

## Pitfalls & anti-patterns
- **EAGLE + AiterAttnBackend HIP-graph crash** during draft-extend capture (#16027) → fall back to Triton
  verify or disable draft-phase graph capture.
- Backend swap changes reduction order → re-gate greedy exactness.
- Large trees → big mask buffers; use partial-packed mode and verify the small-kernel storm is fused away.

## How to verify
Greedy token-exactness; accepted tokens/step; confirm the spec algorithm + verify backend in the server
log; `rocprofv3` for kernel-launch overhead; test EAGLE draft-extend under graph capture.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [aiter.md](aiter.md) ·
[vllm_kernels.md](vllm_kernels.md) · backend: [[sglang_kernels]] · core: [[attention_decode_paged]].

## Sources
- sglang spec workers + tree builder + fused KV materialize: `sgl-project/sglang:python/sglang/srt/speculative/` (on-box).
- EAGLE+AITER HIP-graph crash: https://github.com/sgl-project/sglang/issues/16027
- AMD SSD (tree decode, custom masks, CDNA3 bugs): https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
- SGLang AMD docs: https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
