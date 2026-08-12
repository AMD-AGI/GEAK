---
key: porting an fp4/mxfp4 weight-storage win onto a latency-bound gfx950 grouped MoE GEMM that already runs on f8f6f4 MFMA
type: anti-pattern
confidence: ★★
effect: 0.9993x of the incumbent (1.4644x vs 1.4655x cumulative) — no movement on any of the three cases (2/32/64 tokens) against an expected 1.95x; the axis closed with three independent reasons rather than one bad measurement
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-11
name: fp4-weight-storage-buys-nothing-on-an-f8f6f4-mfma-that-is-la-moe-grouped-gemm-gfx950-mixed
description: Anti-pattern: downgrading B to fp4/mxfp4 on a gfx950 f8f6f4-MFMA GEMM cuts zero op-count and cannot help a latency-bound kernel
keywords: ['mxfp4', 'fp4', 'low-precision', 'mfma', 'moe', 'grouped-gemm', 'anti-pattern', 'gfx950']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: archived
cost: L3
verified_on: 2026-08-11
---
# fp4 weight storage buys nothing on an f8f6f4 MFMA that is latency-bound
- lever: Treat a low-precision weight-storage port as three checks, cheapest first: does the toolchain define the native fp4 type at all; does the target MFMA change its K-per-instruction with operand precision; and is the kernel actually memory-bound.
- apply: On gfx950 the f8f6f4 MFMA consumes the same K regardless of operand format — fp4 is a format-select field on the same instruction, so halving weight width cuts no instruction count. The only remaining upside is bytes read, which pays only on a bandwidth-bound op.
- verify: Confirm the type actually compiles before attributing a null result to tuning: the native packed-fp4 HIP type was undefined on this toolchain, so the low-precision dispatch was compiled out entirely and the run silently measured the incumbent path.
- pitfall: Expected a large win from a sibling kernel's fp4 result -> that sibling was memory-bound and gated on SNR, this one is latency/dep-stall-bound and gated on cosine -> the port measured flat; compare bound class and correctness-gate type before porting.
- caution: Also verify the bound class first on any low-precision-storage port; where the op is genuinely HBM-bound the same change can pay, and this disconfirmation does not extend there.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11
