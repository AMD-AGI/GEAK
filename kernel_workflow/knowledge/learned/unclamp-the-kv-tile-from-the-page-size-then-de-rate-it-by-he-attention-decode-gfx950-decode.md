---
key: paged attention in Triton on gfx950 whose host wrapper clamps the KV tile to the cache page size and floors the split count, launching thousands of near-empty workgroups
type: lever
confidence: ★★
effect: ~1.37x weighted from the launch-config direction alone and +11% more from an occupancy-correct split count; per-case it is carried by the head<=256 cases, while forcing the same wide tile onto the head-512 case is 1.5x-3x slower from fp32 accumulator pressure, so the de-rate is what makes the rule safe
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-17
name: unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode
description: Paged attention wrappers that clamp the KV tile to the page size and floor the split count over-subscribe the CUs; unclamping with a head de-rate is ~1.37x.
keywords: ['launch-config', 'tile-size', 'paged-attention', 'decode', 'split-k', 'register-pressure', 'grid-occupancy', 'triton', 'empty-workgroups']
kernels: ['kernel_unified_attention_3d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-17
---
# Unclamp the KV tile from the page size, then de-rate it by head width
- lever: Before touching the body, print the grid the wrapper actually launches: a tile clamped to the paged block size plus a split-count floor can put thousands of near-empty workgroups against a few hundred CUs.
- apply: Raise the tile for head widths up to 256, cut the split floor to a small constant, and size the split count against the real (seqs x kv_heads x splits) grid for a couple of waves per SIMD; de-rate the tile by head width (a factor per 128 of head) rather than by a flat element budget.
- stack: total ~1.70x weighted isolated vs frozen baseline (director-verified) = several directions compounded
  - 1. launch-config unclamp + head de-rate - ~1.37x standalone (round 1, verified) - the bulk of the win
  - 2. all-decode body specialisation +19% and host-side reduce-dispatch collapse +8% on top of (1) (round 1, verified)
  - 3. occupancy-correct split count on the real 3-D grid +11% (round 2, verified); cache policy, deferred value load and reduce-side wave count added the remainder a percent or two at a time
  - note: attribution is incremental in landing order, not independent
- verify: Re-time every candidate against the frozen baseline after the config moves, and dump the launched grid, split count and tile per case as a debug fingerprint so you can prove the config flipped rather than infer it.
- pitfall: a memory direction verified 1.06x standalone became a net regression once the config landed -> it was parameterized against the old grid and its gates were neutralized -> re-measure every shelved direction stacked on the config winner, not standalone
num_warps / num_stages / waves_per_eu sweeps returned nothing here -> occupancy is supplied by the grid, not by registers, once the tile and split count are right -> price the grid before spending a slot on occupancy knobs
- caution: Also verify the widest head separately: the accumulator is block rows times head width in fp32, so the tile rule that wins everywhere else can invert there.
- source: GEAK 12h per-kernel time-budget campaign, run mi355x_vllm_triton_unified_attention-bmk7-12h, 2026-08-17, rounds 1/2/4/13, director-validated accepted, correctness PASS
