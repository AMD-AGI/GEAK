---
key: multi-XCD CDNA4 (gfx950) grouped GEMM whose expert-weight operand is re-read across workgroups — the grid-to-L2-slice mapping, not the inner loop
type: lever
confidence: ★★
effect: largest single item of the campaign: first-stage HBM read traffic -47% at 2048 tokens and -67% at 65536, carrying the cumulative from 1.29x to 1.35x; a later per-grid-size chunk adds +0.9% more, with the mid case monotone in chunk size (-0% / -1.9% / -2.4% for 24 / 384 / 816)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed
description: Chunk-interleaving the linear workgroup id inverts the hardware XCD round-robin and restores weight reuse inside each XCD's L2 slice.
keywords: ['xcd-swizzle', 'l2-reuse', 'workgroup-mapping', 'bucket-routing', 'moe', 'grouped-gemm', 'fp8-blockscale']
kernels: ['fmoe_fp8_blockscale_g1u1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-09
---
# Invert the XCD round-robin with a chunk-interleaved workgroup remap
- lever: the hardware distributes workgroups across XCDs on the LINEAR id, so tiles sharing an operand scatter; remap the id in chunks so a run of tiles sharing weights lands on one XCD's L2 slice.
- apply: index remap inside the gridwise header, gated on exact divisibility (total % (NumXcd*Chunk) == 0) with a fallback to the stock mapping; the logical XCD count that won was half the physical one, and the chunk value itself is the tuned quantity.
- verify: enumerate the remap host-side and prove it is a bijection before it is allowed to be timed; then confirm the HBM read counter drops, and swap prebuilt modules in place so an arm is never confounded with a build.
- pitfall: a remap looked ~12% faster -> the id formula was not a bijection and silently dropped 2 of 12 tiles at one N extent -> full-speed wrong answer that a speed-only A/B accepts; bijection proof first.
one global chunk value scored neutral for ten rounds -> the optimum tracks grid size, and the best mid-case value costs +8.2% at the small case -> express the chunk as a function of grid size.
- caution: also verify the reuse you are targeting is contiguous in the remapped id — a second-stage operand strided by n-block is structurally unreachable by the same swizzle, and that stage's grid order was pinned by its epilogue's row addressing.
- source: run kernel_20_geak_0808_16h, 2026-08-08..09, gfx950/MI355X; 10-cycle interleaved paired A/B, correctness PASS per arm
