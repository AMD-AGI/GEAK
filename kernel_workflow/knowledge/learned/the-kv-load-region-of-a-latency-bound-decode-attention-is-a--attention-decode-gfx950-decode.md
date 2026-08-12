---
key: KV-streaming load region of a Triton paged decode attention on gfx950 that is latency-bound rather than bandwidth-starved
type: anti-pattern
confidence: ★★
effect: 0 of 4 memory-side directions beat the frozen baseline: K/V cache_modifier '.cg' cost +20% (K) and +12% (V) on the long-context case (all-.cg 1.16x slower than incumbent), '.cs' is a hard compile error on this arch, a loop-carried double-buffer prefetch of the page index was +3% worse on that same case, and loop-split was 4-5% worse; the short case (c2) and mid case (c32) stayed within noise throughout.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: the-kv-load-region-of-a-latency-bound-decode-attention-is-a--attention-decode-gfx950-decode
description: Anti-pattern: on latency-bound paged decode attention already issuing 128-bit KV loads, cache hints, prefetch and loop-split all measured null-to-negative.
keywords: ['cache-modifier', 'software-prefetch', 'vectorization', 'latency-bound', 'attention-decode', 'paged-kv', 'anti-pattern', 'memory-bound']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-08-11
roofline: already ~55-65% of achievable HBM bandwidth with a wait-to-busy ratio ~3.2, i.e. latency-bound not bandwidth-bound
levers: ['mem.cache-modifier', 'mem.software-prefetch']
---
# The KV load region of a latency-bound decode attention is a closed axis
- lever: Spend the round elsewhere when the streaming loads already emit 128-bit widths and the stall profile is latency- rather than bandwidth-shaped: cache hints, manual prefetch and loop-splitting each measured negative here.
- apply: Check the generated ISA for the load widths first (ten 128-bit buffer loads here means vector width has no headroom), then read the wait-to-busy ratio; that pair predicted every negative below.
- verify: Disassemble the candidate and count load widths, and A/B each hint against the frozen baseline per case rather than on the geomean, since the short launch-floored case masks a regression in the long one.
- pitfall: A cache hint that reads as a demotion added an L2 round trip instead of removing one -> the default full-L1 path plus two-stage pipelining was already hiding the latency -> reverting to the default hint recovered the loss.
- caution: Also verify this on a kernel with more than ~1 workgroup per CU before assuming it carries: the closure was measured where occupancy is one workgroup per CU, and a bandwidth-starved variant may still pay.
- source: 16h per-kernel time-budget campaign (chuschen16h wave, 61 passes), 2026-08-11
