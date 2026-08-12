---
key: isolated A/B methodology for long multi-round campaigns on JIT/ninja-built GPU kernels where candidates are compared across sessions and later stacked
type: method
confidence: ★★
effect: a stale resident shared object measured 1.3120 while a forced rebuild of identical sources measured 1.3371 on the same per-case set - a ~2% phantom that drove five rounds of false refutation; and two file-disjoint winners each verified ~1.385x stacked to 1.132x, a 22% regression
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a
description: Force a rebuild and re-measure the head in-session: a stale resident binary and file-disjoint stacking each produced multi-round phantom results
keywords: ['measurement-rig', 'ab-methodology', 'stale-binary', 'stacking', 'noise-floor', 'counter-guided', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# Force the rebuild, pair the blocks, dump the registers before stacking
- lever: Adjudicate candidates against a measured run-to-run floor using paired, rotated A/B blocks over freshly rebuilt objects, and re-measure the current head in the same session rather than comparing to a stored cross-session number.
- apply: Build each arm as its own shared object, rotate the swap order across >=10 paired blocks, obtain the floor from an identical-vs-identical control in the same session, and force the compile step explicitly since the build graph may not track header dependencies.
- verify: Ship when the per-case delta is disjoint across blocks AND clears the measured floor; for a stack, dump descriptors, registers and scratch of the merged build and compare against each part before believing the sum.
- pitfall: Five rounds refuted their own real wins -> they were compared against a number produced by a binary that was never rebuilt -> a forced rebuild of identical sources moved the same measurement by ~2%.
Two winners touching disjoint files merged into a large regression -> both spent the same scalar resource, so the merged build spilled inside the K loop -> file-disjointness is not resource independence.
- caution: Instruction count is a description rather than a gate - one winner improved time while adding instructions, and a later change moved the ISA measurably for zero time; also verify every count-derived claim against time on the current head.
- source: run moe_stage2-own16h, rounds 13-27, 2026-08-12; floor and stack measured on the validated rig
