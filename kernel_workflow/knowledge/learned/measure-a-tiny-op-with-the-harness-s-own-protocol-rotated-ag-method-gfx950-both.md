---
key: measuring single-dispatch tiny ops on gfx950 where the timing harness itself, not the kernel, owns most of the reported number
type: method
confidence: ★★
effect: a graph-replay probe overstated reported impact ~4x and inverted small signs (replay spread across cases 3x the harness spread); an unrotated interleaved comparator read the first slot slow on an identical body and manufactured a ~14% fake win; inside the plateau band, added device work is discounted ~4x (transfer function 2.10 / 2.10 / 2.08 / 2.07 across 4x growing device work, breakout only at 16x)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both
description: For tiny ops, measure with the harness protocol plus position rotation and an A/A null: replay probes overstate ~4x and slot bias fakes ~14%
keywords: ['method', 'measurement', 'ab-methodology', 'graph-replay', 'dispatch-bound', 'noise-floor', 'measurement-floor', 'negative-control', 'gfx950']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: method
regime: both
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
levers: ['method.ab-protocol']
---
# Measure a tiny op with the harness's own protocol, rotated, against an A/A null
- lever: build the probe around one dispatch sandwiched between the harness's own barrier packets, rotate variant position every round, and carry an identical-body A/A pair plus an exact-equality parity gate in every generation.
- apply: replace throughput-style replay loops with the single-dispatch protocol; derive a transfer function first by sweeping a synthetic in-window kernel from empty dispatch upward, so you know whether your op sits in the discount plateau or past its breakout.
- verify: the A/A null should read within a few tenths of a percent; a first-slot bias on an identical body means the rig is not rotated yet, and any candidate delta smaller than that null is not a result.
- pitfall: replay-based probe disagreed with the harness in magnitude and sometimes in sign -> replay measures dispatch throughput and hides exposed latency -> re-time with the single-dispatch protocol.
- pitfall: two of three probe generations manufactured a win before the third was right -> unrotated slot bias -> rotate positions and publish the A/A null alongside every number.
- pitfall: a byte-identical artifact re-measured 2.53x in one session and 2.60x in another -> box drift across sessions -> bound what any sub-5% claim can mean, or re-measure both arms in one session.
- caution: also verify where the op sits on that transfer function before calling a device direction dead: inside the plateau a real device win is discounted several-fold before it reaches the reported number, and outside it the same win reads clean.
- source: run kernel_20_geak_0808_16h, memory-movement scatter lane, round 4 deep_explore, 2026-08-12
