---
name: check-whether-the-fp8-operand-dialect-is-the-one-this-part-e-moe-grouped-gemm-gfx950-mixed
description: Bitcast fp8 dot operands to the part's native OCP e4m3 dialect to delete a per-element software cast: 15.4x geomean on a fused-MoE grouped GEMM
keywords: [fp8, dtype-dialect, mfma, valu-emulation, isa-check, moe, dequant]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: mixed
key: fnuz-vs-OCP-e4m3 fp8 operand dialect at the dot site in a Triton fused-MoE grouped GEMM on CDNA4/gfx950, small- and large-batch cases
lifecycle: active
type: lever
confidence: ★★
effect: 15.4x geomean from one edit when the operands were in the non-native fp8 dialect; it held on every case, and the round's integrated build ran 14.5x on the small-batch case and 17.8-18.2x on the two large-batch cases
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
source: run kernel_20_geak_0808_4h 2026-08-08
last_seen: 2026-08-08
---
# Check whether the fp8 operand dialect is the one this part executes natively
- lever: On CDNA4 the native fp8 dialect is OCP e4m3; operands typed in the older fnuz dialect are lowered by Triton to a per-element software conversion before the dot (a NaN-check and select storm measured at 410 VALU instructions per MFMA, VALU 86% busy, MFMA under 1%), so the matrix core is idle while the vector unit emulates the cast.
- apply: Bitcast both dot operands to the native OCP e4m3 type at the dot site and fold the resulting 2^-2 exponent-bias factor into a scale multiply the kernel already performs (per-block, per-channel or per-tensor), so the inner loop gains no new math; masked loads with other=0.0 stay dialect-safe because +0 has the same encoding in both.
- verify: grep the emitted .amdgcn for the native fp8 MFMA opcode (should appear) and for the fnuz NaN-compare instructions (should fall to zero), then re-run correctness before banking the win - a faster clock alone does not prove the intended instruction is issued.
- caution: the bitcast is exact only where the operand bytes carry no fnuz-NaN code (0x80) and no exponent-field-15 codes, which is a byte-histogram property of the real tensors and cheap to check offline; also verify it there, and where it is violated a packed-dword clamp costs about 1 op per 4 elements against the ~25 ops per element removed, so the direction can survive the check failing.
- source: run kernel_20_geak_0808_4h 2026-08-08
