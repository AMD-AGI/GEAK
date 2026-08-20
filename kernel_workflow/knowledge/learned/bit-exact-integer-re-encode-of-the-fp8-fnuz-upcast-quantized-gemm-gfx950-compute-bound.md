---
key: W8A8 block-scaled GEMM with fp8-e4m3FNUZ operands and bf16 out, Triton on gfx950/MI355 — the software FNUZ->fp16 decode path in the K loop
type: lever
confidence: ★★
effect: 16.50x isolated geomean vs the frozen baseline (director-verified, -0.11% vs the lane's claim); per-case 14.02x at M=2048, 17.81x at M=32768, 17.99x at M=65536 — the decode rewrite alone measured 3.82x in the first round
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound
description: Block-scaled fp8-FNUZ GEMM on gfx950: replacing the emitted per-element FNUZ upcast with a bit-exact packed integer re-encode dominates the win.
keywords: ['fp8', 'fnuz', 'dequant', 'block-scaled', 'bit-exact', 'packed-valu', 'quantized-gemm', 'gfx950', 'triton']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-08
origin_kernels: ['_w8a8_triton_block_scaled_mm']
---
# Bit-exact integer re-encode of the fp8-FNUZ upcast
- lever: When the compiler emits a software per-element FNUZ fp8 -> fp16 upcast inside the K loop (Triton warns about it), hand-write the same upcast as an integer bit shuffle on packed dwords; the dequant VALU chain is then the thing you delete, not the thing you tune.
- apply: Kernel-source level: shift/mask the packed byte pairs with 16-bit packed ops (a packed arithmetic-shift plus a sign-hole mask got it to ~3 VALU per output dword), keep the loads as uint32 so tile geometry and vector-load widths are unchanged, and pair it with a two-tier launcher config table (num_warps, waves_per_eu as an allocator directive, mask elision for even-K) plus a once-per-call B decode pre-pass.
- stack: total 16.50x isolated geomean (director-verified) = six directions compounded, incremental in landing order
  - 1. bit-exact integer re-encode of the FNUZ upcast — 3.82x standalone (round 1, verified) — the bulk of the win
  - 2. two-tier launcher/config table with a plan cache — ~+30% cumulative on top (rounds 1-5, verified)
  - 3. dword-granular decode then packed 16-bit ops — +22.8% then +4.4% on top (rounds 2 and 10, verified)
  - 4. per-token scale re-layout in the launcher — +3.2% (round 8, verified); the strided gather was ~36% of L2 demand at 16x read amplification
  - 5. decode B once per call in a pre-pass — +6.2% (round 9, verified)
  - 6. joint re-tile 256x64 -> 128x128 anchored with the decode plan — +2.1% (round 12, verified)
  - note: attribution is incremental in landing order, not independent.
- verify: Compare the isolated A/B ratio against the frozen baseline on every M case, and compare the candidate's output to the predecessor with an integer equality test on the raw bit view — every promotion here was bit-identical, and an independent re-check at five seeds plus three off-tile M values stayed bit-identical.
- pitfall: the arch's hardware fp8 converter looked like the obvious replacement -> its scaled-convert form destroys the sign of codes 0x7F/0xFF and the legacy forms decode OCP rather than FNUZ, so it is both wrong and 4 ops/dword against the software 3 -> keep the software decode and enumerate all 256 input codes to prove exactness
- caution: also verify exactness with an integer compare rather than the harness tolerance: when the output's smallest magnitude is ~1e-9 against an rms near 28, a max-relative-error gate of 1e-2 IS bit-exactness in disguise, so a scale re-association that reads 'within tolerance' can still be a numeric change.
- source: run _w8a8_triton_block_scaled_mm-own16h, 2026-08-08 campaign, director validation accepted 2026-08-12
