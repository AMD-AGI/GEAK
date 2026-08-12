---
key: fnuz-vs-OCP-e4m3 fp8 operand dialect at the dot site in a Triton fused-MoE grouped GEMM on CDNA4/gfx950, small- and large-batch cases
type: lever
confidence: ★★
effect: 15.4-16.7x geomean from the one dialect edit alone, reproduced on two independent runs and holding on every case; in the second run it hand-merged with in-body M-coarsening (2.84x alone) to a director-verified 40.8x geomean - 21.3x small-batch, 56.6x and 57.0x on the two large-batch cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: check-whether-the-fp8-operand-dialect-is-the-one-this-part-e-moe-grouped-gemm-gfx950-mixed
description: Bitcast fp8 dot operands to the native OCP e4m3 dialect to delete a software cast, then re-open M-coarsening: 40.8x stacked on a fused-MoE grouped GEMM
keywords: ['fp8', 'dtype-dialect', 'mfma', 'valu-emulation', 'isa-check', 'moe', 'dequant', 'grouped-gemm', 'm-coarsening', 'operand-reuse', 'nameplate-peak', 'measurement-drift']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
verified_on: 2026-08-12
roofline: VALU-emulation-bound (matrix core under 1% busy, fp16 emulation path at ~0.01 of fp8 nameplate) -> operand-supply/memory-bound at ~0.57 of the ACHIEVABLE MFMA roof, where that roof is itself only ~0.40 of the fp8 nameplate; HBM stays around 0.12 of spec, so the limiter is the cache/vector operand path, not DRAM
---
# Check whether the fp8 operand dialect is the one this part executes natively
- lever: On CDNA4 the native fp8 dialect is OCP e4m3; operands typed in the older fnuz dialect are lowered by Triton to a per-element software conversion before the dot (a NaN-check and select storm measured at ~410 VALU instructions per MFMA, VALU 86% busy, MFMA under 1%), so the matrix core is idle while the vector unit emulates the cast. Once it is removed, immediately re-open the arithmetic-intensity axis: the emulation was hiding the memory system, so a tile-shape/reuse direction that looked worthless before now pays.
- apply: Bitcast both dot operands to the native OCP e4m3 type at the dot site and fold the resulting 2^-2 exponent-bias factor into a scale multiply the kernel already performs (per-block, per-channel or per-tensor), so the inner loop gains no new math; masked loads with other=0.0 stay dialect-safe because +0 has the same encoding in both. Then coarsen in the kernel body over same-group row tiles (BLOCK_M 16 -> 256 here) so one program reuses the weight tile, with the coarsen factor chosen adaptively for the small-grid vs large-grid case.
- stack: total 40.8x isolated geomean (director-verified) = two directions, authored separately and then hand-merged
  - 1. fp8 dialect bitcast + folded power-of-two rescale - 16.7x standalone (round 1, verified) - the enabling lever
  - 2. in-body M-coarsening over same-expert row tiles, adaptive two-level coarsen factor - 2.84x standalone (round 1, verified); a single global coarsen factor caps at 2.18x, so the two-level split is load-bearing
  - note: they compose far above additive but below the product of the standalones (~0.86 of it), because the occupancy collapse coarsening pays for (waves/SIMD 7 -> 2) only becomes affordable once the dot is native. A later codegen knob (matrix instruction nonkdim 32) added +0.4% and is below this harness's own session drift, so treat it as inside the noise band.
- verify: grep the emitted .amdgcn for the native fp8 MFMA opcode (should appear) and for the fnuz NaN-compare instructions (should fall to zero), then re-run correctness before banking the win - a faster clock alone does not prove the intended instruction is issued. Cross-check the MFMA instruction count times its FLOP/instr against the known workload FLOPs.
- pitfall: the two directions could not be applied together by patch tooling -> both edit the prologue and the dot site -> budget a hand-merge for same-site directions instead of expecting an automatic apply; the merged build is the thing to re-time, not either half.
- caution: the bitcast is exact only where the operand bytes carry no fnuz-NaN code (0x80) and no exponent-field-15 codes, which is a byte-histogram property of the real tensors and cheap to check offline; also verify it there, and where it is violated a packed-dword clamp costs about 1 op per 4 elements against the ~25 ops per element removed, so the direction can survive the check failing.
- source: run kernel_20_geak_0808_4h 2026-08-08

- caution: promoted to ★★★ by the run that wrote it, with no independent confirmation; reset to ★★ per the self-confirmation cap — also verify it engages on your shapes.
