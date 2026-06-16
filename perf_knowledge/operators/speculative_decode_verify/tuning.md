---
title: speculative_decode_verify — tuning
kind: technique
operator: speculative_decode_verify
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
---

# speculative_decode_verify — tuning

## The lever is acceptance rate × cheap verify, not raw kernel TFLOPs
Spec-decode speedup ≈ `accepted_tokens_per_step / target_passes`. Two things to maximize: the
**acceptance rate** (algorithmic — draft quality, tree shape) and the **verify pass cost** (kernel). The
verify attention is small-M and cheap; over-optimizing it while acceptance is low wastes effort.

## Tree shape (the biggest knob)
- **Tree size `T`** and **width/depth**: bigger tree → more candidates accepted per pass, but a bigger
  verify pass and more draft cost. Sweep `T ∈ {8,16,32,64}` and pick by tok/s, not acceptance alone.
- **Linear vs branching**: chain draft → `TreeMaskMode.QLEN_ONLY` (simpler, cheaper mask); branching tree
  → `FULL_MASK`. n-gram/MTP drafts are often chains; EAGLE/Medusa branch.
- **Partial-packed tree mask** (sglang) shrinks the mask buffer for large trees.

## Verify-attention kernel knobs (custom-mask decode FA — [[triton_amd]])
- It's a small-M FA over paged KV with a **custom tree mask**. `BLOCK_M` ≈ `T` (one tree per program),
  `matrix_instr_nonkdim=16`, `num_warps=2–4` (small-M, avoid spill), `num_stages=1`,
  `schedule_hint=memory-bound-attention`, `knobs.amd.use_buffer_ops=ON`.
- **Prefix/suffix split** (xFormers-style): attend the cached prefix densely, the tree suffix with the
  custom mask — avoids materializing a full mask over the whole sequence.
- **Fused KV materialize** (sglang `triton_ops/fused_kv_materialize.py`): write the tree's draft KV into
  the paged cache in one fused kernel rather than many tiny copies (the SSD blog warns naive multi-round
  sampling launches "numerous small kernels").

## HIP-graph capture (the AMD integration gotcha)
EAGLE's **draft extend** phase has crashed under HIP-graph capture on ROCm (sglang #16027:
`AiterAttnBackend` missing `max_split_per_batch`). The draft and target graphs are captured separately;
the draft-extend capture is the fragile one. Make spec metadata static/capture-safe, or disable graph
capture for the draft phase if it crashes.

## Combine with FP8
FP8 target + spec-decode multiply: FP8 cuts per-pass bandwidth, spec cuts the number of passes — AMD
reports **3.6×** on Llama-3.1-405B (FP8 + spec) vs **2.31×** spec alone. fnuz on gfx942; accuracy-gate.

## Autotune sketch
Sweep `(T, tree_width, draft_len)` by end-to-end tok/s on a representative trace; the verify-FA config is
keyed on `(T, head_dim)`. Re-measure acceptance per workload — it's data-dependent.

## Verify the speedup is real
Track **accepted tokens/step** and tok/s together. If tok/s drops while acceptance looks fine, the verify
pass or the small-kernel overhead dominates — profile `rocprofv3` for many tiny launches. If acceptance
collapses, suspect a tree-mask or bf16 MFMA bug (see [numerics.md](numerics.md)).

## Sources
- AMD spec-decode (2.31× / 3.6× +FP8, tree attention): https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
- AMD SSD (small-kernel overhead, tree decode, custom masks): https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
- sglang tree mask + fused KV materialize: `sgl-project/sglang:python/sglang/srt/speculative/{eagle_utils.py,triton_ops/fused_kv_materialize.py}` (on-box).
- EAGLE draft-extend HIP-graph crash: https://github.com/sgl-project/sglang/issues/16027
