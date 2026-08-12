---
key: latency/dep-stall-bound fp8 block-scale grouped MoE GEMM on gfx950 with a small-token case, where high dep_wait is misread as grid underfill
type: anti-pattern
confidence: ★★
effect: 0.851x overall vs the incumbent (net regression); on the small 2-token case the split variants got monotonically slower: KBatch=2 ~5.1x, =3 ~7.4x, =4 ~9.6x the fused non-split time, while staying numerically exact (err_ratio 0, cosine diff ~8e-8)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-11
name: a-56-dep-wait-on-a-grouped-moe-gemm-is-not-evidence-of-an-un-moe-grouped-gemm-gfx950-small-batch
description: Anti-pattern: split-K/KBatch on a grouped MoE GEMM whose block grid already exceeds CU count regresses monotonically; dep-stall is not grid underfill
keywords: ['split-k', 'moe', 'grouped-gemm', 'occupancy', 'dep-stall', 'grid-fill', 'anti-pattern', 'gfx950']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: small-batch
layer: learned
lifecycle: archived
cost: L3
verified_on: 2026-08-11
---
# A ~56% dep_wait on a grouped MoE GEMM is not evidence of an underfilled grid
- lever: Before spending a round on split-K/KBatch, count the actual blocks: (sorted token-ids / block_m) x N-tiles, and compare with CU count. Here that was ~512 M-blocks x 12 N-tiles over 304 CUs, i.e. ~20 block-waves — already well filled, so the stall was per-block dependency latency at occupancy 2, an axis split-K cannot touch.
- apply: The cheap check is arithmetic on the sort output plus the tile dims; the expensive check is plumbing the KBatch path. Do the arithmetic first, and only treat dep_wait as underfill when the block count is genuinely below CU count.
- verify: If split-K is still plumbed, price its fixed cost explicitly: a K-wide padded fp32 partial buffer that is memset-zeroed every call, a cross-partial reduction that grows with KBatch, and a deferred activation pass — that overhead dwarfed the fused GEMM on the small case.
- pitfall: Numerically exact result read as 'mechanism works, just needs tuning' -> the mechanism did work, the premise did not -> increasing KBatch made it monotonically worse, which is the signature of pure added overhead rather than a mistuned knob.
- caution: Also verify the roofline class before generalising this: on a genuinely memory- or grid-starved grouped GEMM split-K can still pay; the disconfirmation here is conditioned on a grid already many block-waves deep and a latency/dep-stall bound.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11
