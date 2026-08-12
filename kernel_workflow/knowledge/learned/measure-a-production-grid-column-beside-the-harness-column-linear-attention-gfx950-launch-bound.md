---
key: harness-constructed vs upstream-caller launch grid for a Triton chunked linear-attention forward kernel on gfx950, packed/varlen layouts
type: method
confidence: ★★
effect: verified geomean 22.68x on the harness grid vs 2.09x re-measured on the production grid; the gap tracked the over-launched dimension exactly — ~1.0x artifact on the smallest batch case (3.26x vs 3.33x) but ~23x and ~57x on the two larger batch cases (38.2x/93.7x vs 1.63x/1.67x); the second column reclassified 4 of 13 directions as harness-only
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 13
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
name: measure-a-production-grid-column-beside-the-harness-column-linear-attention-gfx950-launch-bound
description: Report every gain in a production-grid column beside the harness column: it cut a 22.68x harness geomean to 2.09x and reclassified 4 of 13 directions
keywords: ['measurement-method', 'control-experiment', 'harness-artifact', 'production-grid', 'launch-bound', 'launch-overhead', 'kernel-cache']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: launch-bound
layer: learned
lifecycle: active
---
# Measure a production-grid column beside the harness column
- lever: Stand up a second bench that launches the op the way the upstream caller does, and report every gain in BOTH columns from round 1 — a benchmark that constructs the launch grid itself can issue work the real caller never issues, and deleting that work is a legal patch that then dominates the headline.
- apply: Read the upstream call site for how the grid and tensor packing are derived — under packed/varlen layouts a batch dimension routinely collapses to 1 while the bench still launches batch*heads, leaving programs that recompute byte-identical tiles — then keep a small side bench pinned to the caller's grid and label any gain that moves only the harness column.
- verify: A gain is real when both columns move together; if only the harness column moves, the patch removed work production never issued.
- caution: Also verify the two columns' BASELINES and not just their optimized wall-clock — the columns can agree on the optimized side while the ratio keeps its artifact; and also verify any memoized/cached launch fast path against freshly allocated tensors, since a bench steady state that reuses the same tensor objects pays a caller that allocates per step nothing.
- source: run kernel_20_geak_0808_4h 2026-08-08
