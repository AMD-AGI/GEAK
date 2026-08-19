---
key: VALU-bound per-token-group quantization on gfx950 Triton, where dividing each element by its group scale dominates the instruction mix
type: lever
confidence: ★★
effect: 1.16x standalone on top of the cast fix (2.66x -> 3.09x cumulative geomean), additive with the disjoint host lever to 4.12x; improved every per-case size, largest on the two large cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both
description: On a VALU-bound quant cast, a bit-exact reciprocal + FMA replacing per-element division cut VALU/wave 1216->768 for 1.16x, with format constants folded in
keywords: ['quantize-cast', 'valu-bound', 'reciprocal', 'division', 'bit-exact', 'fp8', 'gated-lever', 'gfx950']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
levers: ['compute.exact-reciprocal']
origin_kernels: ['_per_token_group_quant_fp8']
---
# Divide by the group scale is a correctly-rounded reciprocal, not a division
- lever: replace the per-element divide with a correctly-rounded reciprocal plus two FMAs, and fold any constant format factor (here a x2 from the fp8 flavour) into the divisor so it is free; where surrounding code already excludes overflow/underflow, a 3-op rcp + one Newton step is enough
- apply: gate the short reciprocal on a geometry constexpr the launcher already computes so both arms live in one kernel, and prove bit-identity exhaustively over the reachable divisor patterns rather than sampling
- verify: bit-identical to the correctly-rounded reference over all reachable divisors, then a paired interleaved A/B on min and median against the incumbent in the same lock window
- pitfall: a rule of engagement that fenced off 'the divide region' of the file hid the two largest compute levers for several rounds -> the restriction named a code region instead of the property that mattered -> state such rules about the property (approximate vs correctly-rounded), so a rewrite that preserves it stays in play
two levers keying on the same physical resource anti-composed four separate times (the naive union lost) -> resolve by making one lever yield to the other's existing gate instead of choosing between them
- caution: also verify packed math at depth greater than 1 on your shapes: freeing ~20 VGPRs met the precondition three separate ways here and every arm still lost (one at -7.1% on the mid-size case), because each inline-asm block is a scheduling barrier loads cannot move across
- source: run _per_token_group_quant_fp8-own16h, 2026-08-12, rounds 3 and 8, director-validated
