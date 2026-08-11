---
key: paged KV attention on gfx950 whose 4-way block scatter is read-once with near-zero L2 reuse — which axes are already closed there
type: anti-pattern
confidence: ★★
effect: Disconfirming, per-case: native fp8 PV dot 1.006x full-mix / 1.003x heavy subset, with the two time-dominant long-context shapes flat at 1.000x-1.004x and only the compute-lightest shape at +1.05%; fp8 QK dot fails parity outright (worst abs diff 0.1825 vs a 5e-2 gate, identical whether the whole path or only QK is narrowed); group-8 mxfp4 KV saves ~37% traffic rather than 50% and still misses the worst-element gate; partition size 512/1024 gives 0.72x (live K/V register arrays double, 21-register spill) and 128 fails correctness on frozen scratch; occupancy re-tune 0.97x; graph replay of the 2-dispatch shape ~0.86x on every shape.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-11
name: on-paged-attention-at-the-scatter-roofline-precision-and-occ-attention-decode-gfx950-decode
description: Anti-pattern: on paged attention pinned at the block-scatter roofline, fp8 dots, fp4 KV, partition resize, occupancy and graph replay all return noise
keywords: ['paged-attention', 'hbm-bound', 'roofline', 'fp8', 'mxfp4', 'occupancy', 'launch-overhead', 'dead-end', 'gfx950', 'decode']
kernels: []
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: archived
verified_on: 2026-08-11
roofline: memory-bound at ~0.54 of achievable roofline; the residual gap is the scatter itself, not occupancy
---
# On paged attention at the scatter roofline, precision and occupancy axes return noise
- lever: Read the bound first: when the KV stream is a read-once 4-way block scatter with near-zero L2 reuse, the only axis that pays is bytes moved — so spend rounds on traffic (storage width, cache policy) before compute precision, occupancy or launch shape.
- apply: Cheap triage before committing a round: take the register count from the code object's .vgpr_count, check whether a spill was actually produced, and check whether the time-dominant heavy shapes move at all under a candidate; a lever that only moves the compute-lightest shape is a tail, not the bottleneck.
- verify: Re-measure any apparent precision win on the time-dominant shapes alone; if those are flat the geomean gain is coming from the light tail and will not survive a real workload.
- pitfall: launch_bounds and waves_per_eu changes appeared to be applied but nothing moved → with .vgpr_count at a hard floor and no spill produced, they are no-ops → the only remaining register lever is a genuine source-level footprint cut, which on a fully-preloaded kernel trades away the latency-hiding parallelism.
Moving the V load after the QK dot to shorten liveness → LLVM re-hoisted it → ~2-5% worse than leaving the scheduler alone.
fp8 QK looked like an oversight in dead code → it is unreachable for a numeric reason: the error localizes entirely to the QK dot while P and V tolerate fp8.
- caution: Also verify the tolerance shape before writing off a narrow KV format: these verdicts are under a worst-element allclose, and a group-scaled format that clears an RMS-style gate can still miss that one.
- source: 16h single-kernel time-budget campaign, run chuschen16h, 2026-08-11
