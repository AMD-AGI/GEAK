---
key: locating work in a large template GEMM kernel on gfx950 - deciding which code region deserves the next round before any GPU time is spent
type: method
confidence: ★★
effect: the epilogue returned +6.8% and the prologue +2.75% (uniform across all three per-case shapes) inside a 1.84x stack, after six rounds of hot-loop work had returned ~1.00x; the prologue alone was 36% of static instructions, larger than the epilogue
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a
description: Census the ISA per region (prologue / main loop / epilogue) on CPU before tuning the hot loop: the two unexamined regions carried the wins the loop refused
keywords: ['isa-census', 'profiling-method', 'prologue', 'epilogue', 'hot-loop', 'cpu-locate-gpu-price', 'serialisation', 'composable-kernel']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
---
# Per-region ISA census before hot-loop tuning; locate on CPU, price on GPU
- lever: Build a per-region static-ISA census - instruction and full-wait counts for prologue, main loop and epilogue - and fund the region by its share of instructions and serialised waits rather than by intuition about where a GEMM spends time.
- apply: Emit device-only assembly on the CPU with the compiler flags scraped from the existing build file, then segment the listing by region with a short script. No GPU lock, minutes rather than a round.
- verify: A census verdict is a locator, so price every region-local fix on the frozen-baseline A/B with interleaved controls; several structural closures in this run cost zero GPU time but none of them counted as a win until measured.
- pitfall: a round concluded the kernel was closed from a disassembly that had only ever covered the main loop -> the region scope of the closure was implicit -> state which region a closure covers; the scope error cost two rounds
  - pitfall: deleting a quarter of a region's instructions measured null with the sign flipping across arms, while deleting that same region's serialised memory round trips paid +2.75% -> count full waits, not instructions
- caution: Also verify where a wait you plan to delete resolves and whether co-resident waves already cover it - two removals with clean ISA proof and strictly better register use both priced null because the row they loaded was cache-resident.
- source: run moe_stage1-own16h, 2026-08-12 - 13-round campaign, director-validated
