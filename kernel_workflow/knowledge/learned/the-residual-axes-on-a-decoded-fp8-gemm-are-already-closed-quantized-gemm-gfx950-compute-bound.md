---
key: where NOT to spend rounds on a compute-bound Triton block-scaled fp8 GEMM on gfx950 once the decode chain has already been removed
type: anti-pattern
confidence: ★★
effect: ~1.00x across four axes: graph-capture replay measured -10.5% at M=2048 and -5.1% at M=32768 against direct dispatch; pre-pass fusion has a ceiling under 0.2% on every case; nine LDS round-trip / barrier-count directions all landed under the 1.02x gate; the third workgroup priced at +9.1% against the 25% tile cost of reaching it
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound
description: Once the dequant chain is gone from a compute-bound fp8 GEMM on gfx950, LDS, barrier, occupancy and host-launch directions all return ~1.00x.
keywords: ['closed-axis', 'lds-tiling', 'occupancy', 'launch-overhead', 'hip-graph', 'quantized-gemm', 'gfx950', 'triton', 'measurement-discipline']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-08
origin_kernels: ['_w8a8_triton_block_scaled_mm']
---
# The residual axes on a decoded fp8 GEMM are already closed
- lever: Before committing rounds to LDS round-trip removal, barrier reduction, graph capture, pre-pass fusion or a third workgroup on such a kernel, price each ceiling arithmetically first — here all four closed at roughly parity, and the remaining >10% idea (a preshuffled MFMA-fragment operand buffer) is not expressible through tile-language loads at all.
- apply: Ceiling arithmetic before code: express the candidate's best case as a fraction of the measured kernel ratio (host-side prep versus kernel time, removable fill versus intrinsic per-MFMA work, occupancy gain versus the tile shrink that buys it) and drop anything whose ceiling sits under the promotion gate.
- verify: Same-session interleaved medians with identity-config control rows before and after each block in both orders; an identical tree re-read across five sessions spread about 1%, so a candidate under ~1.02x is inside that spread.
- pitfall: a software-pipelining direction hit both of its stated success criteria (prologue share and stall ratio each improved substantially) and ran 4.6% slower -> the counters were proxies, not causes, and the predictor scoreboard finished 0-for-4 on issue count and 0-for-2 on prologue share -> state every target as a ratio on the launched binary before writing code
- caution: also verify the warm-up launch count is held fixed whenever a candidate changes the early dispatch count: 66 extra warm-up launches alone measured +3.4% at M=2048 and +1.0% at M=32768 with no kernel change, which is larger than most candidates here.
- source: run _w8a8_triton_block_scaled_mm-own16h, 2026-08-08 campaign, director validation accepted 2026-08-12
