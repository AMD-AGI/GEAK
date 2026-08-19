---
key: small-batch attention decode already sitting on its serial online-softmax dependency chain behind an async, launcher-lean host wrapper on gfx950
type: anti-pattern
confidence: ★★
effect: closed axis: wrapper graph capture regressed per case -49.6% / -40.2% / -29.0%; manual software pipelining -40 to -56% on every case; PV-dot reorder exactly 1.00x; recovering occupancy on the mid-context case regressed monotonically, with a single deep-prefetched wave the optimum
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode
description: With a lean launcher and an MFMA-optimal tile, decode graph capture, manual SW pipelining and occupancy recovery all measured at or below 1.00x.
keywords: ['attention', 'decode', 'launch-overhead', 'graph-capture', 'software-pipelining', 'occupancy', 'wave-quantization', 'anti-pattern']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
verified_on: 2026-07-29
origin_kernels: ['kernel_unified_attention_2d']
---
# Four host and compute directions that a latency-floored decode kernel gives back
- lever: Read the ISA before restructuring: register count with zero spills says whether occupancy is even the constraint, and the MFMA variant says whether tuning its shape can pay at all.
- apply: Where the residual limiter is wave quantization (about 1.5 waves of workgroups over the CU count), the payoff lives in the grid, not in the inner recurrence.
- verify: Check the host cost is above the graph-launch cost before capturing; an async back-to-back launcher already overlaps enqueue with the previous call's compute, so capture only adds a fixed per-call charge.
- pitfall: Manual QK/PV software pipelining carrying the raw score tile across the loop back-edge regressed 40-56% -> live state times the stage count inflates and fights the compiler's own pipeliner -> leave staging to num_stages, and expect reordering the PV dot ahead of the rescale to be a no-op the scheduler already did.
- caution: Also verify how thin wins are timed: an average-of-many-reps harness washed a ~1% win and the driver reported no improvement while the verified patch was real, so re-measure with interleaved min/median A/B over at least 8 reps and confirm the patch marker landed.
- source: run kernel_unified_attention_2d-ch16h, 2026-07-29, directions r1_d0_hipgraph_16h, r1_d0_ilp_restructure, r2_d0
