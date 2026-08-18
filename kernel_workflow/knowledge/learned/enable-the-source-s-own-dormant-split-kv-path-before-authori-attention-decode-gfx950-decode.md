---
key: a Triton paged decode attention op on gfx950 whose source already contains a split-KV/reduce path the public wrapper never enables, over a mix of sliding-window and 1-KV-head full-attention cases
type: lever
confidence: ★★
effect: 1.73x isolated geomean in one round against the frozen baseline, the largest step of a 2.11x geomean / 1.52x time-weighted final stack; per case ~2.4-2.8x on the 1-KV-head full-attention decode cases and only ~1.15x on the sliding-window cases at that point; naive un-windowed activation was -10% on the highest-weight sliding case
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode
description: A Triton paged decode op may already ship an unreached split-KV + reduce path; enabling it from the wrapper with window-aware segmentation won 1.73x
keywords: ['attention-decode', 'split-kv', 'paged-attention', 'decode', 'triton', 'flash-decoding', 'host-wrapper', 'long-context', 'gfx950']
kernels: ['kernel_unified_attention_3d', 'reduce_segments']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-13
roofline: the low-parallelism geometry sat at ~22% of achievable HBM with only 64 real workgroups against 256 CUs (a parallelism wall, not a bandwidth wall); after the split the dominant geometry runs at ~90% and later ~98% of the measured streaming roof
levers: ['algo.split-kv', 'host.launch-shape']
---
# Enable the source's own dormant split-KV path before authoring a new one
- lever: Before authoring new KV parallelism, read the op's source for a partition/reduce path the public wrapper never reaches because it allocates no scratch and passes no segment count; enabling it needs no signature change and can be the largest single step of the campaign.
- apply: Allocate the partial/scratch tensors and pick the segment count inside the wrapper; compute the segmentation window bit-for-bit identically in both the main and the reduce kernel, since both recompute tiles-per-segment independently.
- verify: Confirm the split path is what actually ran (segment count > 1 reaching both kernels) and re-time per case: the gain concentrates on the parallelism-starved geometry, so check the heaviest-weighted case did not pay for it.
- pitfall: Even segmentation over the whole sequence launches segments that fall outside a sliding window; the existing early-return does not cull them, so they iterate an empty tile loop and still write full zero partials the reduce reads back -> -10% on the highest-weight case -> derive segment bounds from the window.
The reduce hardcoded a tile-size constant while the main kernel launches the branch-selected one -> they coincide only on the caller path shipped today -> pass the actual tile size through.
A constexpr kwarg added to the kernel signature but not to the launch call silently defaults -> most work items never launch and the output is plausibly shaped garbage -> the tell was identical error counts across four mappings that should each have covered a different item set.
- caution: Also verify the segment count against scratch round-trip bytes and not only against a workgroups-per-CU line: two geometries here whose optimal working workgroup counts differed 8x shared one optimal scratch footprint, and each extra segment costs tokens*heads*head_size*4 bytes written plus the same read back.
Also re-price any launch-param win found before the split: it attacked the same parallelism deficit, and stacked unguarded it cost occupancy (geomean 1.47 vs 1.73); a candidate gated inert under the likely winner merges for free.
- source: run mi355x_vllm_triton_unified_attention_gemma4-bmk7-12h, round 1, 2026-08-17; director-validated weighted 1.5147 / geomean 2.1071, correctness pass
