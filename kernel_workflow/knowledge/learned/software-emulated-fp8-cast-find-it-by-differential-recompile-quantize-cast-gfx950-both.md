---
key: per-token-group fp8 quantization in Triton on gfx950, where the requested fp8 flavour is not the one the hardware convert implements
type: lever
confidence: ★★
effect: 1.66x standalone on the coarsened kernel (cumulative 1.60x -> 2.66x geomean); held on every per-case size, and the run total was 2.93x/4.47x/3.95x per case (3.73x geomean, interleaved)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both
description: Non-OCP fp8 output makes the compiler emulate the cast in software; native packed convert + bitcast cuts VALU/wave 852->338 on a quant cast
keywords: ['fp8', 'quantize-cast', 'dtype-emulation', 'valu-bound', 'native-convert', 'bitcast', 'bit-exact', 'gfx950']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
roofline: VALU-issue bound (>100% of device VALU issue on the two large cases) -> HBM bound at 94-99% of the measured no-math bandwidth roof
levers: ['compute.native-convert']
---
# Software-emulated fp8 cast: find it by differential recompile, kill it with the native convert
- lever: when an elementwise cast shows a ~200:1 VALU:memory instruction ratio, suspect the dtype is emulated; emit the hardware's own packed convert and bitcast into the target flavour (biases differing by a constant are bit-exact over the clamped range), then fix -0 in the fp32 domain
- apply: differential recompile: change ONLY the output dtype to the hardware-native twin and diff the ISA VALU count (852 -> 290 here); that delta is the emulation, and it is the size of the available win
- stack: total 3.73x geomean isolated, drift-free interleaved (per case 2.93x / 4.47x / 3.95x) = four directions compounded
  - 1. coarsening: G groups per program, 2-D tile, dwordx4 loads, num_warps=1 — 1.60x standalone (round 1, verified); fewer warps won monotonically
  - 2. native convert + bitcast (this card) — 1.60x -> 2.66x cumulative (round 2, verified) — the largest single step
  - 3. exact reciprocal + prebound host dispatch, textually disjoint — 2.66x -> 4.12x cumulative (round 3, both verified)
  - 4. geometry-gated packed multiply and a short reciprocal at small tiles — ~+2% (rounds 7-8, verified)
  - note: 1-4 are cumulative against the FROZEN baseline table, which carried ~1.13x of box drift; the 3.73x total is the drift-free re-measure, so the parts do not multiply to it
- verify: paired interleaved A/B inside one GPU lock — candidate, incumbent HEAD and a byte-identical null as separate processes from their own directories, randomized order, min AND median, 2-4 windows
- pitfall: two plain-Triton reformulations passed the harness correctness cases yet were wrong on 8 of 94 NaN/eps configs (including all-zeros with eps=0) -> a clamp lowers to a med3 instruction that maps NaN to a number while the hardware convert maps it to the max code -> keep the med3 in the chain and sweep NaN / +-0 / eps separately
frozen-denominator scoring handed out ~1.13x of free geomean -> the frozen table was 17-21% slower than a same-day remeasure on two of three cases -> score every candidate against an incumbent measured in the same window
- caution: also verify which -0 fixup form is cheapest on your target: abs as a free source modifier measured below the maximum, byte-mask and xor forms here, and the ordering may differ elsewhere
- source: run _per_token_group_quant_fp8-own16h, 2026-08-12, director-validated (compile PASS, correctness PASS, 4 interleaved A/B rounds)
