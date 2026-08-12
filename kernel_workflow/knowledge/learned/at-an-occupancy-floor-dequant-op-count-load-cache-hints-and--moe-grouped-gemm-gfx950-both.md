---
key: micro-optimizing the dequant/load path of an occupancy-2 int4 weight-only GEMM on gfx950 that already carries a manual register pipeline
type: anti-pattern
confidence: ★★
effect: 3 independent directions, none positive. Magic-number bit-cast dequant: bit-exact, ISA-confirmed conversion-op removal (17 -> 1 per loop body), yet 0.988x overall and 0.978x on the batch-2 case. Load cache modifier '.cg': 0.83x on batch-32 and 0.89x on batch-64. Compiler auto num_stages>=2: -15% to -36% at BLOCK_M 256/512, and num_stages=3 exceeds the LDS capacity outright.
confirms_cited: 1
confirms_blind: 0
losses: 1
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: at-an-occupancy-floor-dequant-op-count-load-cache-hints-and--moe-grouped-gemm-gfx950-both
description: On a register-tight int4 GEMM at occupancy 2, cheaper dequant math, load cache modifiers and compiler auto-pipelining all measured neutral to clearly negative.
keywords: ['dequant', 'cache-modifier', 'num-stages', 'software-pipeline', 'lds', 'int4', 'grouped-gemm', 'noise-floor', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# At an occupancy floor, dequant op-count, load cache hints and auto-pipelining are all off the critical path
- lever: Treat the dequant ALU chain as already overlapped once the loop is at its occupancy floor: the scalar work hides under the matrix pipeline, so removing conversion instructions buys nothing and the payoff sits in tile shape and reuse instead.
- apply: If the loop already carries a hand-written register double-buffer, leave the compiler's auto-staging at its default: layering an explicit stage count on top either does nothing (default), only disables the automatic staging (one stage, noise-level), or multi-buffers operands through LDS and regresses. Default L1 caching of the read-once streamed quantized weight is the fast setting.
- verify: Confirm the edit actually reached the machine code before believing a null: the cache-modifier lane was only trustworthy because the '.cg' variant regressed measurably, which proves the hint landed; a flat result with no ISA delta is an un-applied patch, not a negative.
- pitfall: A ~0.6% gain on one bucket looked like a win -> the per-bucket spread across repeat launches is wider than that -> re-run the A/B and treat anything inside a ~1-2% band as noise rather than banking it.
- caution: Also verify this on your own occupancy: the finding is conditioned on being register-pinned with the accumulator near the ceiling, and on a loop that is not accumulator-limited the same dequant and staging levers can still pay.
- source: 16h single-kernel time-budget campaign, run id chuschen16h, 32 passes, 2026-08-11
