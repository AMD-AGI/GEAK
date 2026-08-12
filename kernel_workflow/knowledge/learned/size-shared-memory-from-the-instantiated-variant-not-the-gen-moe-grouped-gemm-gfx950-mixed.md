---
key: dead shared-memory allocation from a generic shared-size helper in a CK/HIP templated fp8 block-scaled MoE grouped GEMM on gfx950 — either fused pipeline stage, ping/pong 2-LDS variant
type: lever
confidence: ★★
effect: reproduced on a second, independent stage of the same template family: 1.195x director-verified geomean (per-case 1.13x at the smallest token count, 1.22x and 1.24x at 16x and 32x more tokens), bit-exact, two full benchmarks 0.07% apart with a same-session stock control within 0.6% of the frozen denominator. First campaign, other stage: 1.29x for this lever alone (1.21x / 1.34x / 1.34x) inside a 1.49x end state. Both times the mechanism is the same — resident workgroups per CU rise with an unchanged instruction stream.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 7
toolchain: rocm7.2 / torch2.11.0 / hip (CK C++ templates, JIT-compiled)
last_seen: 2026-08-12
name: size-shared-memory-from-the-instantiated-variant-not-the-gen-moe-grouped-gemm-gfx950-mixed
description: Size each shared-memory buffer from the instantiated template variant, not the generic max: 1.19-1.29x on MoE grouped GEMM by raising resident workgroups per CU
keywords: ['lds', 'occupancy', 'vgpr', 'moe', 'template-instantiation', 'isa-check', 'code-object', 'dead-allocation', 'lds-padding', 'lds-bank-conflict', 'composable-kernel', 'spill', 'harness-artifact']
kernels: ['moe_stage1', 'moe_stage2', 'kernel_moe_gemm_2lds']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
---
# Size shared memory from the instantiated variant, not the generic max
- lever: In a templated GEMM / grouped-GEMM family, check that every shared-memory declaration is sized for the variant actually instantiated: the shared-size helper returns a generic max over all code paths (operand staging vs the epilogue's shuffle buffer), and a second ping-pong buffer often reuses that same max while only one term ever addresses it — so a template flag that zeroes a term (an operand preshuffled straight into registers, never staged) leaves a large dead allocation that pins resident workgroups per CU.
- apply: Evaluate each term of the shared-size helper for the instantiated template arguments and size each declared buffer to the span its own consumer addresses (only the ping buffer is handed to the epilogue). Then spend a few bytes of the reclaim on the operand tile's row stride: a derived-class override that replaces the stock XOR-modulo permuted A layout with a plain M-major layout padded by one row (`(MPerBlock+1)*AK1`) de-conflicts the banks for ~0.3% of the tile. All arithmetic is compile-time — the launch bound changes and the loop body does not.
- stack: total 1.195x director-verified geomean = two co-landed levers in one patch, never isolated from each other in this run.
  - 1. pong-buffer right-sizing — the occupancy step (2 -> 3 blocks/CU); measured alone at 1.29x on the other stage in an earlier run.
  - 2. padded naive A-LDS descriptor — the bank-conflict half; its coarse cousin (the stock `ExtraM`-style pad knob) measured 1.018x standalone (round 1, verified) and is dead code once the override replaces the base descriptor, so the fine form strictly subsumes it.
  - note: attribution is across runs, not an isolated split inside this one.
- verify: The dispatched code object's group-segment size falls and workgroups/CU rises while the instruction histogram, spill count and output bytes stay unchanged — a shared-memory-only edit that moved the instruction count changed something else too. Read occupancy inputs from the code object, and confirm scratch/spills stayed at zero.
- pitfall: occupancy read as twice its true value for two rounds → the profiler's register column had matched a cold non-main-loop variant of the same symbol → read the occupancy inputs (VGPR count, group segment) from the code-object metadata instead.
- caution: Also verify what binds after the reclaim before funding a second round on the same budget: a deeper shrink was worth 0% once registers had become the binder on the other stage, and here shrinking the remaining epilogue shuffle span (repeats 4x2 -> 1x1) regressed to 0.918x while widening the epilogue's store vector regressed to 0.44x. Also re-measure anything tuned against the pre-fix occupancy — a coarser-tile patch worth +14% against the starved build regressed every case once stacked on the fix.
- source: run kernel_20_geak_0808_4h 2026-08-08; run kernel_20_geak_0811_2h_kb_new 2026-08-12 (director-validated)

- caution: promoted to ★★★ by the run that wrote it, with no independent confirmation; reset to ★★ per the self-confirmation cap — also verify it engages on your shapes.
