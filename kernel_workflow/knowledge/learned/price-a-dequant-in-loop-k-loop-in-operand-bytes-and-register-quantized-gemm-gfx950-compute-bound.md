---
name: price-a-dequant-in-loop-k-loop-in-operand-bytes-and-register-quantized-gemm-gfx950-compute-bound
description: Size a dequant-in-loop K loop in operand bytes and register/LDS budget, not instructions: instruction count correlated NEGATIVELY with time on four ladders
keywords: [dequant, quantized-gemm, vgpr, lds, occupancy, counters, measurement-method, control-experiment, compute-bound]
kernels: [_gemm_a8w8_blockscale_kernel]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: sizing candidate arms of a narrow-operand dequant-in-loop k loop on a Triton fp8 blockscale GEMM, gfx950, small- and large-M cases
lifecycle: active
type: method
confidence: ★★
effect: no speedup of its own -- a screen. On the run's director-verified end state (12.90x geomean; 10.38x at the smallest M, 14.24x and 14.52x at the two large-M cases) the instruction-count-to-time correlation was NEGATIVE over four ordered ladders: removing the in-loop dequant entirely (bit-exact, verified equal on every arm) was slower on all of them, -0.5% to -106%; one earlier variant cut loop instructions 20.9% and cost +83% time; and adding a single instruction cost 4.7%.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950 CDNA4
source: run kb_on_0810 2026-08-11
last_seen: 2026-08-11
---
# Price a dequant-in-loop k loop in operand bytes and register/LDS budget, not in instruction count
- lever: in a k loop that unpacks a narrow quantized operand to a wider dot input, the loop is usually bound by operand BYTES moved through LDS/VGPR rather than by the unpack VALU, so size every candidate in bytes-per-matrix-op and in register/LDS budget before you size it in instructions; keeping the operand narrow until the last moment can be worth more than deleting the unpack.
- apply: before timing anything, compile each arm and read n_regs, spill count and shared bytes, and drop arms that cross the occupancy-relevant register step or grow LDS (here a 2x-wider in-loop operand took VGPR 416 -> 476-493 and LDS 65536 -> 98304 and produced the body's first spills); when double-buffering, buffer exactly one tile and make it the narrower one.
- verify: confirm with a deletion probe that HOLDS OPERAND WIDTH FIXED -- hoist the work out of the loop into a pre-pass and time both, and separately time the pre-pass, so the width change and the work removal are not read off the same variant; grade arms interleaved in one window against a byte-identical control with an exactness gate.
- pitfall: an arm that cut loop instruction count 20.9% ran +83% slower -> the wider in-loop operand crossed a register/LDS step and spilled -> read spills and shared bytes out of the compiled arm before it is timed.
- caution: also verify what an instruction census is being used for: a per-line ISA/DWARF attribution is a census of instructions, not of time, and a lane whose payoff argument is 'this is N% of the loop's instructions' should be re-argued in bytes or occupancy before it is funded.
- source: run kb_on_0810 2026-08-11
