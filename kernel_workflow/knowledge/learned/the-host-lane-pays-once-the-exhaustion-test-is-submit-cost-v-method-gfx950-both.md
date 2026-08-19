---
key: deciding when the host/launch lane on a short elementwise GPU op is spent, on a harness whose event bracket costs more than the smallest case
type: anti-pattern
confidence: ★★
effect: one prebind win (1.68x paired on the smallest case) then ~1.00x across four directions: a 48% per-launch host cut moved the wall 0%, a further -12.3% per launch bought +0.55% at the wall, the event-bracketed NOP wall was flat from grid 1 to 32768 and across num_warps 1/2/4, and 1-node graph replay cost more than the binder it replaced
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both
description: After one prebind win, four more host/launch directions returned ~1.00x on a short elementwise op; exhaustion test: submit cost vs smallest case GPU time
keywords: ['launch-overhead', 'host-runtime', 'dispatch-cache', 'hip-graph', 'anti-pattern', 'measurement-floor', 'quantize-cast', 'gfx950']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: method
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['host.launch-overhead']
origin_kernels: ['_per_token_group_quant_fp8']
---
# The host lane pays once; the exhaustion test is submit cost versus the smallest case's kernel time
- lever: prebind the dispatch once — cache the compiled handle and the argument binding inside the timed region, refining the JIT's own specialization key so a wrong binary can never be selected — then treat submit-cost-below-smallest-case-GPU-time as the signal that the lane has paid out and later rounds are better spent in the kernel
- apply: measure an event-bracketed NOP launch to get the harness floor, and compare each case's GPU time against it before planning any host direction; the smallest case here sat under that bracket, so its wall was the harness, not the op
- verify: before funding a wall-minus-GPU residual, check the estimator: here the residual grew with case size and was an artifact of comparing a min against a mean over pipelined launches, not overhead — three rounds spent budget against that phantom
- pitfall: reproducible -2.7% to -6.8% cuts in the tiny case's GPU time moved the wall only +0.2-0.4% -> that case is pinned to a single timer quantum (~1.4%) under the event bracket -> compute the geomean ceiling implied by the floor first; here it capped the run at 4.36x and made a 4.7x target arithmetically unreachable
- caution: also verify that a graph-capture replay actually removes the cost you are paying for: a 1-node graph still sits between the same two event records, and measured costlier than the direct binder
- source: run _per_token_group_quant_fp8-own16h, 2026-08-12, rounds 1 and 3-6 and 10, director-validated
