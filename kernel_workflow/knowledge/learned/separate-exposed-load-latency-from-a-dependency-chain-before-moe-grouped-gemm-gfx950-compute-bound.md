---
name: separate-exposed-load-latency-from-a-dependency-chain-before-moe-grouped-gemm-gfx950-compute-bound
description: A regressing double-buffer diagnoses a dependency chain, not exposed load latency: one -14% arm closed the whole prefetch/pipeline axis on a MoE GEMM
keywords: [prefetch, pipeline-stages, num-stages, occupancy, moe, dequant, control-experiment]
kernels: [fused_moe_kernel_gptq_awq]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: software prefetch / pipeline-depth / occupancy axis on a packed-weight fused-MoE grouped GEMM (Triton) on gfx950, batch 32-64
lifecycle: active
type: anti-pattern
confidence: ★★
effect: TRUE-NEG closing the whole pipeline-mechanism axis at a 42.2393x incumbent, source restored byte-identical, no patch. A genuine double-buffer of the operand actually on the L2/HBM critical path - the one the traffic analysis pointed at, numerically valid, cos gate PASSED - REGRESSED -14% at batch 32 and -12% at batch 64. num_stages>1 was net-negative for this body even at depth 2 (a smallest-case-only num_stages 1->2 was worth 41.5747 -> 41.9525 cumulative, i.e. under 1%). A second wave (NSUB=2) cost -55% and occupancy 2 overall was net-negative ~76%, despite being reachable and numerically valid. The existing prefetch env knob turned out not to be a depth knob at all: it is a single 1-deep double-buffer, and values outside its two live settings merely DISABLE it (-43%).
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.57h / 35 passes, 2026-08-11
last_seen: 2026-08-11
---
# Separate exposed load latency from a dependency chain before funding a software prefetch or deeper pipeline
- lever: A regressing double-buffer is a diagnosis, not just a loss: if adding a buffer for the operand on the critical path makes things slower, the residual is the unpack->scale->dot dependency chain plus operand reuse, not exposed load latency, and no pipeline-depth or occupancy knob reaches it - the extra buffer's register pressure only tightens the chain. Run that one-shot buffer experiment early and let it retire the whole axis. Occupancy is worth raising when a single wave is latency-STALLED; when the second wave doubles per-CU streamed traffic against a small L2 it buys thrash instead.
- apply: Build one arm that double-buffers the suspected critical operand and one that raises occupancy, and read the sign before planning a restructure.
- verify: Sign of the buffered arm on the large cases, and whether the knob you swept changed the generated code at all.
- pitfall: a planned 'deeper prefetch' env sweep read as a large measured negative at every new value -> the knob was a boolean in numeric clothing (a single 1-deep double-buffer) whose off-values merely disabled it, -43% -> read what a knob controls in the source before sweeping it, and price a code change instead.
- caution: Also verify a load already carrying the loop is not freed in the process; removing the existing 1-deep buffer here cost -40 to -43%.
- source: chuschen 16h time-budget campaign run, 15.57h / 35 passes, 2026-08-11
