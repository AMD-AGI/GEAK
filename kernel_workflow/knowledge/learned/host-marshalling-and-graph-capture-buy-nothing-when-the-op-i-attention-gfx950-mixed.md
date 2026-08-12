---
key: paged/ragged attention on gfx950 whose GPU work per call dwarfs host marshalling and its two dispatches — closing the launch-overhead axis
type: anti-pattern
confidence: ★★
effect: Host-marshalling memoization 1.009x (real CPU saving, invisible to the per-call event window); wrapper-level graph capture of both launches 0.986x geomean — net regression, helping only 1 of 9 cases (+3%) and costing ~2% on the other short-context cases; graph-off reproduced the baseline exactly
confirms_cited: 2
confirms_blind: 0
losses: 1
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: host-marshalling-and-graph-capture-buy-nothing-when-the-op-i-attention-gfx950-mixed
description: GPU-bound op: real host-side savings and a correct wrapper HIP-graph capture measured 1.009x and 0.986x; a below-noise delta cannot be gated into a win.
keywords: ['launch-overhead', 'hip-graph', 'graph-capture', 'host-runtime', 'gpu-bound', 'harness-noise', 'per-shape-gate', 'attention', 'gfx950', 'anti-pattern']
kernels: ['paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention
regime: mixed
layer: learned
lifecycle: active
---
# host marshalling and graph capture buy nothing when the op is GPU-bound
- lever: Before spending a round on the launch floor, compare per-call GPU work against host + dispatch cost: where GPU work is an order of magnitude larger, the async queue hides the collapsed host cost and the per-call event window cannot see it at all.
- apply: If the axis still has to be closed for confidence, close it at the WRAPPER level (capture both launches into one graph) rather than at the launcher level, and check that graph-off reproduces the baseline to prove the capture was real and correct.
- verify: Compare replay vs eager per case against the harness noise floor first; a true delta of ~1-2% under a ~1.2% noise floor is not a measurable win however the geomean is sliced.
- pitfall: capture is correct and passes parity yet the geomean drops -> the two dispatches per call were never a serialized floor -> retire the axis instead of tuning the capture.
- caution: A per-shape never-regress gate can rescue a lever only when its true delta clears the harness noise floor — also verify that before assuming case selection can convert a mixed-sign result into a positive geomean.
- source: 16h single-kernel time-budget campaign, run id chuschen16h, round 1 direction d1 + round 2 host-runtime direction d0, 2026-08-11
