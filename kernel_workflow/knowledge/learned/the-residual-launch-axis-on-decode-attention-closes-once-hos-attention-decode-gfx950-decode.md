---
key: the residual host/launch axis on Triton decode attention at gfx950 after dispatch is already collapsed — graph capture, monomorphic launch, and launch-knob re-sweeps
type: anti-pattern
confidence: ★★
effect: closed axis, four directions: wrapper-level graph replay 0.652x / 0.696x / 0.755x of eager at batch B=2 / 32 / 64 (a regression at every case); a monomorphic launcher that met its own host target measured exactly 1.000x; a full launch-knob re-sweep kept every bucket's winner within 0.2%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode
description: Once host time sits far under GPU time, more launch-path work on decode attention returns 1.00x or worse — graph replay measured 0.65-0.76x of eager
keywords: ['launch-overhead', 'host-dispatch', 'cuda-graph', 'dispatch-floor', 'decode', 'anti-pattern', 'paged-attention']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
levers: ['host.launch-overhead']
---
# The residual launch axis on decode attention closes once host time is far under GPU time
- lever: Before funding another launch-path round, compare host time per call against pure-GPU time per call; when host is not within a small multiple of binding, a round spent in the kernel body pays and one spent on the launch path does not.
- apply: Measure a five-line empty kernel at each of your grid sizes and subtract that per-dispatch floor from any 'the preamble costs X' claim before it sizes a lever; here the floor was flat across a 32x span in program count and sat inside every tile-count fit intercept.
- verify: Judge on the isolated A/B against the frozen baseline, not on the sub-metric the direction targeted: a direction can cut its own host number by a third and still land at exactly 1.000x end to end.
- pitfall: An apparently large fixed preamble → about 80% of it was the grid-independent dispatch floor → the real preamble was under 4% of the smallest case, so the lever sized against it could not have paid.
Graph capture engaged correctly yet regressed → one graph launch cost roughly 2x a direct C launcher call → no per-case gate turns it on at these sizes.
- caution: Also verify the harness print resolution before chasing small deltas: here one printed quantum was 0.4-0.8% of the signal, so effects under about 1.5% at the larger batches were unresolvable at any repeat count.
- source: run kernel_20_geak_0808_16h, rounds 3-5 dead-end evidence + director validation, 2026-08-12
