---
title: speculative_decode_verify on Triton — SOTA card
kind: sota_card
operator: speculative_decode_verify
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# speculative_decode_verify × Triton

## TL;DR
Triton is the **editable** verify path: the tree-mask FA verify is a small-M custom-mask FlashAttention,
and the tree-mask + fused-KV-materialize helpers are Triton (sglang `triton_ops/fused_kv_materialize.py`,
`build_tree_kernel_efficient`). Use Triton when you need to customize the tree mask / verify layout or when
the AITER verify path has integration bugs (it has — see [aiter.md](aiter.md)). The custom-mask attention
must be correct on **GQA** (the AMD bug-prone case) and use the right **bf16 MFMA intrinsic**.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton custom-mask verify FA | aiter `unified_attention` / FA Triton (`_triton_kernels/attention/`) | gfx942/950; bf16/fp16/fp8 | enables tree verify; part of AMD's 2.31× / 3.6×+FP8 stack (vendor) | editable verify, GQA tree mask |
| sgl tree-mask + fused KV materialize | `sgl-project/sglang:speculative/{eagle_utils.py,triton_ops/fused_kv_materialize.py}` | gfx942/950 | collapses the small-kernel storm | building the tree mask / writing tree KV |

> No hand-tuned CK/asm tree-verify kernel on AMD as of 2026-06; the verify path is custom-mask Triton FA.

## Config space / knobs
- `BLOCK_M ≈ T` (tree size), `matrix_instr_nonkdim=16`, `num_warps=2–4`, `num_stages=1`,
  `schedule_hint=memory-bound-attention`, `knobs.amd.use_buffer_ops=ON`.
- `TreeMaskMode.QLEN_ONLY` (chain) vs `FULL_MASK` (branching); partial-packed mask for large trees.
- Prefix/suffix split to avoid a full-sequence mask. See [tuning.md](../tuning.md).

## Numerics / parity
Greedy must be token-exact vs non-spec; tree-mask + GQA row-mapping + bf16 MFMA dtype are the risks. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
- sglang: `--speculative-algorithm EAGLE/EAGLE3/NEXTN/...`, draft model/head; verify rides the
  `--attention-backend` choice (`triton`). `SGLANG_AITER_UNIFIED_VERIFY` toggles the unified verify path.
- The `@triton.jit` verify FA + tree-mask builder are Python rebind seams.

## Pitfalls & anti-patterns
- **bf16 MFMA type-confusion** in a rowsum/verify kernel → acceptance collapse. Match intrinsic to dtype.
- **GQA custom-mask row-mapping** bug on CDNA3 → corrupted outputs only on the tree path. Test GQA trees.
- Many tiny kernels (un-fused tree decode) → launch-bound; fuse the KV materialize + mask build.

## How to verify
Greedy token-exactness vs non-spec (≥10 prompts); accepted tokens/step stable; `rocprofv3` shows no
small-kernel storm; `TRITON_PRINT_AUTOTUNING=1` for the verify FA config.

## Alternatives / cross-links
[overview.md](../overview.md) · [aiter.md](aiter.md) · [sglang_kernels.md](sglang_kernels.md) ·
[vllm_kernels.md](vllm_kernels.md) · languages: [[triton_amd]] · core: [[attention_decode_paged]] ·
[[gqa_mqa_attention]].

## Sources
- aiter Triton verify / unified attention: `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/attention/` (on-box).
- sgl tree mask + fused KV materialize: `sgl-project/sglang:python/sglang/srt/speculative/` (on-box).
- AMD SSD (bf16 rowsum, GQA custom-mask bugs): https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
- AMD Triton knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
