---
name: hand-write-the-fp8-upcast-as-packed-dword-swar-quantized-gemm-gfx950-compute-bound
description: Hand-write the fp8 upcast as packed-dword SWAR inline asm when the operand dtype has no native matrix-core path: 5.92x, then 1.35x more from dword packing
keywords: [fp8, dtype-dialect, valu-emulation, isa-check, inline-asm, swar, dequant, quantized-gemm, config-sweep]
kernels: [_w8a8_triton_block_scaled_mm]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: software fp8 (fnuz) upcast in the k loop of a Triton W8A8 block-scaled GEMM on gfx950, where the operand dtype has no native matrix-core path
lifecycle: archived
type: lever
confidence: ★★
effect: the hand-written bit-trick upcast alone measured 5.92x over the stock kernel, and moving the same bit arithmetic to packed dwords a further 1.35x; the cost is per loaded element so both held on every case (the run's final verified geomean, with tiling and a hand prefetch stacked on top, was 13.43x — 11.36x on the smallest-M case, 14.45x and 14.76x on the two large-M cases)
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# Hand-write the fp8 upcast as packed-dword SWAR
- lever: When the quantized operand dtype has no native matrix-core path on the target gfx (fnuz fp8 here, which the compiler notes it upcasts to fp16), the emitted per-element software upcast — not the matrix core — is the bottleneck: hand-write that conversion, and do its bit arithmetic on packed 32-bit registers rather than on the sub-32-bit tensor, since gfx9 has no packed 16-bit AND and one mask line on a uint16 tile lowers to 4 VALU per 2 elements.
- apply: Census the k-loop ISA per source line to find the widened lines, then express the conversion as tl.inline_asm_elementwise over dwords: read the byte operand through an int16 pointer view with pack=2 so the asm gets one dword = 4 fp8 values, size outputs as ceil(n*sizeof(dtype)/4) registers, and restore k order with reshape(join(lo,hi)); fold any constant scale as an exact power of two.
- pitfall: odd elements came out silently zeroed while the kernel compiled and ran -> a plain '=v' output constraint let LLVM alias the asm's scratch register onto its input -> declare scratch outputs early-clobber '=&v'.
- verify: Re-census the loop — instructions and VALU per iteration should fall several-fold with the MFMA count unchanged and lane-ops per converted element approaching 2 — then re-sweep the launch config on the new body (the previous table goes stale: re-sweeping after the body edit was worth +8% and +17.3% over stacking the old one) and A/B on the official runner, alternating binaries and discarding each binary's first pass.
- caution: Also verify whether the native low-precision matrix-core path is really the faster target before assuming it is the goal — here it compiled and ran yet landed below the hand-written fp16 route once the emulation storm was gone, and it could not reproduce a bit-exact output gate; and check the packed form preserves the k order the reduction sees, because the natural two-output dword split separates even-k from odd-k elements.
- source: run kernel_20_geak_0808_4h 2026-08-08
