---
key: bf16 paged decode attention with a ragged kv_indptr layout, page granularity 1 and GQA 32/4, shipped as jinja-templated HIP through a ctypes wrapper on gfx950/MI355
type: lever
confidence: ★★
effect: 1.35x weighted / 1.34x geomean isolated vs the frozen baseline, independently re-measured by the director; per-case 1.29-1.41x on the eight decode-shape cases and 1.19-1.21x on the long-context streaming case, run-to-run spread under 1% over 3 repeats, oracle parity PASS
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode
description: HIP paged decode attention on gfx950: per-tensor NT policy, then global-to-LDS DMA bought for prefetch depth, then a transposing LDS read: 1.35x weighted
keywords: ['gfx950', 'attention-decode', 'paged-attention', 'paged-kv', 'decode', 'lds-staging', 'lds-tiling', 'prefetch', 'non-temporal-loads', 'cache-modifier', 'bank-conflict', 'launch-bounds', 'occupancy', 'raw-hip', 'memory-bound', 'isa-diff']
kernels: ['aiter_paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
roofline: memory-bound throughout; the main dispatch ends at ~99.8% of a re-measured blended DRAM roof, and the occupancy limiter moves VGPR -> LDS once the DMA staging lands
levers: ['mem.lds-staging', 'mem.load-cache-policy', 'compute.launch-bounds']
origin_kernels: ['paged_attention_ragged']
---
# Buy prefetch depth with a global-to-LDS DMA on bandwidth-bound decode attention
- lever: On a bandwidth-bound HIP decode attention kernel, stage the V tile through the arch's global-to-LDS DMA and hoist the whole first-group prefetch above the QK phase; the DMA on its own is worth ~0, the prefetch depth it frees registers for is the value. Around it, choose the non-temporal hint per tensor and replace the V transpose scalar gather with a transposing LDS read.
- apply: Four seams in the kernel template: a cache-policy bit chosen per tensor instead of kernel-wide; the LDS-DMA intrinsic with an XOR swizzle of its global source so LDS rows dodge the 32-bank row stride; a transposing LDS read with per-lane addressing in place of the gather; and an explicit launch-bounds attribute pinning the wave count the allocator would otherwise give away.
- stack: total 1.35x weighted isolated (director-verified) = four directions compounded over 7 rounds
  1. per-tensor non-temporal loads - 1.25x standalone (round 1, verified) - the bulk of the win; V-only 1.26x, K-only 1.13x, kernel-wide 1.04x
  2. global-to-LDS DMA V staging + prefetch hoisted above QK - +1.9% on top of (1) (round 3, verified)
  3. transposing LDS read + XOR swizzle - +1.04% on top of (1,2), uniform across all 9 cases (round 4, verified)
  4. two tiles per workgroup in one online-softmax loop + launch-bounds - +0.44% weighted on top of (1-3) (round 5, paired, 4/4 positive)
  note: attribution is incremental in landing order, not independent; (2) alone regresses the streaming case to ~0.99x and only pays once a companion head-rotation lands. Roughly +0.6% more came from host-derived template constants and reverting an earlier rotation.
- verify: Diff the emitted ISA for both arms before believing any cache-policy change - a metadata-only difference never survives a runtime branch, while an intrinsic-immediate one does - and read occupancy from the compiler's kernel-resource-usage 'Occupancy [waves/SIMD]' line rather than an ELF VGPR count or a profiler VGPR column, both of which report something else. Then re-time every case against the frozen baseline, paired.
- pitfall: kernel-wide non-temporal loads gave ~1.04x while V-only gave ~1.26x -> the two operands have opposite reuse (K is re-read across heads, V is streamed) -> pick the hint per tensor, not per kernel.
every later vmcnt wait in the region degraded to vmcnt(0) -> one outstanding LDS-DMA poisons the compiler's waitcnt bracket -> keep the DMA window tight; a vestigial barrier also drains in-flight LDS-DMA and cost ~0.7% here.
the DMA with a default aux operand cost ~20% -> the non-temporal bit on the DMA source is load-bearing -> set it and confirm it in the ISA, not in the C++.
dropping the launch-bounds attribute landed the allocator at 136 unified registers / 3 waves and lost ~3.3% -> the tile-per-workgroup change alone does not hold the wave count -> pin it explicitly and re-read occupancy.
- caution: Also verify which budget is binding after the DMA lands - the limiter flipped from registers to LDS capacity here, and the next direction's headroom is on whichever budget is now full. Also verify a hand-port of a winning idea into the golden path against the mechanical merge of its diff: porting the idea beat stacking the patch by ~11.5% in this run.
- source: run paged_attention_ragged-own16h, 2026-08-12 (16h per-kernel campaign, 7 rounds, 20 direction-credits, director-validated accepted)
