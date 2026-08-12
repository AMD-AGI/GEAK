---
key: Triton attention decode with a serial online-softmax recurrence and an already-lean host launcher, gfx950
type: anti-pattern
confidence: ★★
effect: Disconfirming, no patch shipped: manual QK/PV pipelining that carries the raw score tile across the loop back-edge regresses 40-56% on every case; reordering the PV dot ahead of the accumulator rescale is a 0% no-op; wrapper-level graph capture regresses per-case c2 -49.6%, c32 -40.2%, c64 -29.0%.
confirms_cited: 2
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: prove-the-floor-before-restructuring-ilp-or-wrapping-the-lau-attention-decode-gfx950-decode
description: At an attention decode's ISA and launcher floor, manual SW-pipelining and graph capture both regress; probe occupancy and host share first
keywords: ['attention', 'decode', 'software-pipelining', 'num-stages', 'launch-overhead', 'graph-capture', 'occupancy', 'dead-end']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# Prove the floor before restructuring ILP or wrapping the launch
- lever: Spend one cheap probe on the floor before an ILP or host-side round: dump the ISA for register count, spills and occupancy, and measure the host share of the wall against the compute share.
- apply: Read three things — the MFMA variant actually emitted (a 16-wide one is already optimal for a 16-row block, so non-K-dim tuning is a no-op), register count vs spills, and waves per CU. Those decide whether ILP or the grid is the limiter.
- verify: Re-time any restructure interleaved against the frozen baseline on every case; a compiler-visible reorder that shows exactly 0% is evidence the scheduler already did it, not evidence of a measurement bug.
- pitfall: Hand software-pipelining regressed instead of helping -> carrying the score tile plus prefetched keys across the back-edge multiplies live state by the stage count and fights Triton's own num_stages pipeliner -> reverted and left the pipelining to the compiler knob.
- caution: Also verify where the wall actually sits before wrapping the launch: with a lean launcher the host floor already sat under the graph-launch cost and back-to-back async enqueues overlapped with prior compute, so capture only added cost — and a case at one wave per CU is grid-quantization-limited, which no ILP edit reaches.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11; ledger directions r1_d0_ilp_restructure and r1_d0_hipgraph_16h, both dead_end
