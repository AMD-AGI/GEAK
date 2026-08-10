---
key: moe grouped gemm · gfx950 · mixed
type: lever
confidence: ★★
effect: director-verified 1.49x geomean end state; this lever ALONE measured 1.29x, per-case 1.21x at the smallest token count and 1.34x / 1.34x at the two large token counts (16x and 32x more tokens) — it doubled resident workgroups per CU (2 -> 4) with an unchanged instruction stream and bit-exact output, and it outweighed every tile and instance knob tried in the same round.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm7.2 / torch2.11.0 / hip (CK C++ templates, JIT-compiled)
last_seen: 2026-08-08
---
# Size shared memory from the instantiated variant, not the generic max
- lever: In a templated GEMM / grouped-GEMM family, check that every __shared__ declaration is sized for the variant actually instantiated: the shared-size helper returns a generic max over all code paths (operand staging vs the epilogue's shuffle buffer), and a second ping-pong buffer often reuses that same max while only one term ever addresses it — so a template flag that zeroes a term (an operand preshuffled straight into registers, never staged) leaves a large dead allocation that halves resident workgroups per CU.
- apply: Evaluate each term of the shared-size helper for the instantiated template arguments, then size each declared buffer to the span its own consumer addresses; the arithmetic is compile-time, so the launch bound changes and nothing in the loop body does.
- verify: The dispatched code object's group-segment size falls and workgroups/CU rises while the instruction histogram and the output bytes stay unchanged — a shared-memory-only edit that moved the instruction count changed something else too. Read the occupancy inputs (vgpr count, group segment) from the code-object metadata rather than a profiler's register column: the profiler matched a cold non-main-loop variant of the same symbol here, and its numbers implied twice the true occupancy for two rounds before anyone noticed.
- caution: Also verify what binds after the reclaim before funding a second round on the same budget: a deeper shrink of the same allocation was worth 0% here once registers had become the binder, and the reclaimed capacity had no consumer at all (that pipeline never stages the preshuffled operand through shared memory, so 'spend the slack on prefetch depth' would have had to build the staging path first). Also re-measure anything that was tuned against the pre-fix occupancy — a coarser-tile patch worth +14% against the starved build regressed every case once stacked on the fix.
- source: run kernel_20_geak_0808_4h 2026-08-08
