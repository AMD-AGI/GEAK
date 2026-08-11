---
name: separate-the-scored-per-call-bracket-from-device-time-before-quantize-cast-gfx950-memory-bound
description: Attribute a small case's flat per-call floor to the harness bracket first: graph capture bought ~12% real throughput there and scored exactly zero
keywords: [graph-capture, dispatch-floor, measurement-method, launch-overhead, quant, memory-bound, harness-artifact]
kernels: [_per_token_group_quant_fp8]
platforms: [gfx950]
kernel_class: quantize_cast
regime: memory-bound
key: the smallest case of a per-token-group fp8 quant/cast op (Triton, gfx950) sitting on the harness's own per-call event bracket rather than on the Triton dispatch path
lifecycle: active
type: anti-pattern
confidence: ★★
effect: The smallest case sat at a constant floor across every config and ended at 1.1944x while the two large cases reached 3.0994x and 3.2328x - it holds a third of the geomean weight and caps it below ~2.5x. Wrapping the launch in a HIP graph capture/replay was bit-exact and bought ~12% pure throughput on that case, but did not move the scored number at all (the harness event.record() bracket survives replay), and applying it unconditionally regressed the two large cases 4-7%; size-gating it to the small case left net +0.3%, i.e. noise. A persistent grid-strided rewrite was also bit-exact and never beat the plain one-program-per-row launch on that case under a same-warmup 3-repetition A/B.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.70h / 56 passes, 2026-08-11
last_seen: 2026-08-11
---
# Separate the scored per-call bracket from device time before optimizing dispatch on the smallest case
- lever: When a small case's per-call time is a flat constant across configs, first attribute that constant: if it is the measurement harness's own per-call host work (event record plus Python dispatch) rather than the Triton dispatch path, then graph capture, persistent grids and any other device- or dispatch-side rewrite can show a real throughput win and still score zero.
- apply: Measure the same candidate both ways — a raw throughput loop and the scored path — and let the scored delta decide; size-gate a dispatch rewrite to the case it targets rather than applying it across the sweep.
- verify: A/B any such change on the large cases too before shipping, since the same rewrite that helps a launch-floored case can cost several percent where the kernel is bandwidth-bound; discard a warmup-boosted number on sight in this regime.
- pitfall: a bit-exact graph capture bought real throughput on the small case and moved the scored number not at all -> the scored bracket is the harness's own event record, which survives replay -> attribute the flat constant before funding any dispatch-side rewrite against it.
- caution: Also verify the net across the sweep before shipping a size-gated fix: unconditional application regressed the two bandwidth-bound cases 4-7% here, and the gated version landed inside the noise band.
- source: chuschen 16h time-budget campaign run, 15.70h / 56 passes, 2026-08-11
