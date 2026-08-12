---
key: diagnosing a repeatable two-group split in per-case timings on a large-KV-buffer decode benchmark whose cases are listed in consecutive shape order, gfx950
type: anti-pattern
confidence: ★★
effect: a repeatable ~2.6% split between two case groups tracked the KV buffer's allocation placement rather than the context length it was confounded with - the fast/slow assignment reverses when the allocator phase flips - and the affected cases carried ~51% of the weighted metric; the kernel-side direction built to remove it moved that axis ~1.00x and shipped an unrelated +0.38%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode
description: A period-2 per-case timing split can be the caching allocator moving a large buffer, not a shape effect: the timing follows the pointer, not the shape
keywords: ['measurement', 'harness-artifact', 'ab-methodology', 'measurement-discipline', 'frozen-baseline', 'kv-cache', 'attention-decode', 'decode', 'anti-pattern', 'gfx950', 'noise-floor']
kernels: ['aiter_paged_attention_ragged']
platforms: ['gfx950']
kernel_class: method
regime: decode
layer: learned
lifecycle: archived
cost: L0
verified_on: 2026-08-12
levers: ['method.measurement']
---
# Test the allocator before designing a kernel fix for a periodic per-case split
- lever: When per-case timings fall into two groups with period 2 along a parameter the fixture happens to list in order, test the allocator hypothesis first: re-time with the large buffer re-allocated and see whether the timing follows the pointer or the parameter. Only the second one is a kernel property worth a direction.
- apply: Log the big buffer's base address per case; flip the allocation phase (allocate and free a throwaway buffer before the fixture builds the cache) and re-time. If fast and slow swap groups while every shape is unchanged, it is a caching-allocator placement ping-pong, and no kernel change reaches it.
- verify: Keep a max/min spread check across repeats of the same case - it caught every occurrence of a single-case spike artifact here - and quote paired deltas only, since the box itself drifted on the order of 1-2.6% per hour and cross-session per-case comparison was meaningless under this fixture.
- pitfall: a context-parity penalty read as an address-phase / channel-camping defect and a direction was spent building the transform -> the fixture listed contexts consecutively, so parity was perfectly confounded with allocation order -> the sign reversed with the allocator phase, closing the axis by construction (the swept transform topped out at +0.19%).
- caution: Also verify how much of the weighted metric sits on the artifact before reading a stop decision off the per-case table: about half the weight sat on cases whose split no kernel change can reach here, which flatters or penalises every later A/B on that axis.
- source: run paged_attention_ragged-own16h, 2026-08-12 (16h per-kernel campaign, round 6 memory direction plus the round-1 to round-6 per-case tables; director-validated accepted)
