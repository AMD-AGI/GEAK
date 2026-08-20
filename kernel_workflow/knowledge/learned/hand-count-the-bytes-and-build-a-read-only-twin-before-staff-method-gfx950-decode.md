---
key: deciding whether a memory-bound decode kernel still has headroom when a profiler traffic counter is the only evidence for the target
type: method
confidence: ★★
effect: zero-patch round: the twin's wall time came within 1.02x of the real kernel at batch=64 (compute fully hidden), and the hand count showed the profiler byte figure was 2x low, so the round's target implied above-nameplate bandwidth; the round verified 1.00x and promoted nothing
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode
description: Hand-count traffic and time a math-stripped read-only twin before staffing a memory round: it closed the largest open lane here at zero patch.
keywords: ['roofline', 'profiler-error', 'read-only-twin', 'memory-bound', 'anti-pattern', 'attention-decode', 'method']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: method
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-08
origin_kernels: ['_fwd_grouped_kernel_stage1']
---
# Hand-count the bytes and build a read-only twin before staffing a memory round
- lever: Derive the traffic by hand from the problem (batch x tokens x (key_dim + value_dim) x dtype width, times how often each byte is actually read) and reconcile it with the profiler before believing any headroom estimate built on the counter.
- apply: Build a read-only twin: identical grid and identical gather addressing, all attention math stripped, and a deliberately trivial consumer. Time it against the real kernel; separately time a plain contiguous read on the same box to get the achievable-rate denominator.
- verify: A twin within a few percent of the real kernel means compute is already hidden behind the loads, so occupancy, ILP and scheduling levers are dead by construction; express the result as a fraction of the measured achievable rate, not of nameplate.
- pitfall: The first twin came out ~30% slower than the real kernel, i.e. a floor above the thing it was bounding -> its full-tile fp32 reduction consumer was itself the bottleneck -> make the twin's consumer trivial.
Grouping the profiler's dispatch table by grid size alone merged two shapes that launch the same WG count -> group by (grid size, register count) instead (92 vs 108 separated them here).
- caution: A round that promotes a zero-byte patch can still be its most valuable one, since it prevents a mis-specified successor - also verify a suspicious profiler figure a second independent way (hand arithmetic plus a stripped probe) before declaring the axis closed.
- source: run _fwd_grouped_kernel_stage1-own16h, 2026-08-08, round-3 deep_explore, verified 1.6243x at parity
