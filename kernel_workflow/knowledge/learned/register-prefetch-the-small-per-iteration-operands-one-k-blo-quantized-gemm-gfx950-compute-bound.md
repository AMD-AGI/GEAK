---
key: staging small scale/metadata global loads into registers one k-block ahead inside the register-saturated MFMA k-loop of a block-scaled fp8 GEMM (Triton, gfx950)
type: lever
confidence: ★★
effect: paired on the two large-M cases of a 24.99x director-verified end state (per-case 24.39x / 24.87x / 25.73x at M=2k / 32k / 64k): the 16-dword/lane B operand at depth 1 gave +6.7% / +4.5% (M=32k / 64k), the two 2-dword scale loads at depth 1 a further +3.9% / +7.0%, and the same scales at depth 2 a further +3.4% / +4.0%; ~0% on the small-M case, whose short path the constexpr gate leaves untouched. Re-seen on a second, independently frozen lane of the same op class: the depth-1 scale prefetch alone was +9.0% over that round's canonical control (bit-identical output) and rode into a 19.12x director-verified end state, per-case 17.28x / 20.16x / 20.06x at M=2k / 32k / 64k.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 10
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950 CDNA4
last_seen: 2026-08-12
name: register-prefetch-the-small-per-iteration-operands-one-k-blo-quantized-gemm-gfx950-compute-bound
description: Register-prefetch the SMALL per-iteration operands one k-block ahead as the body's last VMEM: +3.9-9.0% on the large-M cases of a block-scaled GEMM
keywords: ['prefetch', 'vgpr', 'isa-check', 'operand-reuse', 'compute-bound', 'quant', 'dequant', 'mfma', 'convert-layout', 'cross-lane', 'lds']
kernels: ['_gemm_a8w8_blockscale_kernel', '_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
---
# register-prefetch the SMALL per-iteration operands one k-block ahead, as the last VMEM of the body; size the candidate list by dwords/lane
- lever: on a register-saturated MFMA k-loop, try staging a loop operand's global load one (or two) k-blocks ahead into registers; affordability is set by BYTES PER LANE, not by the operand's importance — 2-dword scale/metadata loads stayed spill-free to depth 6, a 16-dword tile operand paid only at depth 1, and a 128-dword operand never paid. On the second lane the payload turned out to be not the arithmetic but the LAYOUT CONVERSION the scale broadcast lowers to: a per-row scale broadcast across the N tile becomes load + multiply + LDS write + barrier + strided LDS reads, and in the stock body that chain sat at the loop bottom with a barrier of its own, directly ahead of its consumers — priced by ablation at ~10% of the loop against ~6.5% for the FMAs themselves. Prefetching it one k-block ahead lets the scheduler fold the conversion onto the already-existing operand-staging barrier.
- apply: issue the prefetched load as the LAST vector-memory statement of the body; prefetch a group of related small loads together rather than one of them (either scale alone was 0.91-0.96x, the pair +2.1% geomean); constexpr-gate it to the loop variant you tuned, since the same patch on a num_stages=2 variant of the same kernel was 0.93x. Statement order within the body is part of the patch, not cosmetic: tile loads first and the scale prefetch after won, the reverse order lost ~1.5%.
- stack: total 19.12x director-verified isolated geomean on the second lane (per-case 17.28x / 20.16x / 20.06x at M=2k / 32k / 64k) = four directions compounded, incremental in landing order: 1. native unscaled fp8 MFMA in place of the software upcast -> 10.91x, the big lever; 2. host launch-path collapse -> ~1.00x alone but it fixes the small-M floor and compounds later; (1+2 integrated 12.38x); 3. scalar-scale fold into the row scale plus a from-scratch tile/config re-derivation on the fp8 body -> 17.40x; 4. THIS card's lever, depth-1 scale register prefetch, plus a cached direct module-launch closure -> 19.12x (+2.2% over the best single direction; the prefetch was +9.0% standalone against a same-session control, the launcher +2.7%). Attribution is incremental, not independent.
- verify: read the first s_waitcnt of the loop body before and after — the acceptance tell that predicted the sign three rounds running is that the prefetched load has LEFT the wait chain (its vmcnt term disappears), not that any counter improved; then confirm 0 spilled dwords, shared-memory size (it can FALL, as the conversion scratch stops being live across the barrier), and bit-identical output. Instruction and barrier counts are a poor proxy on a stall-limited body: the winning loop grew ~40% in instructions and from 2 to 6 barriers and still ran 9% faster.
- pitfall: the identical prefetch measured 0.99x with fresh spills -> the load had been placed at the top of the body, extending every live range across it (mid-body was 2.7-3.5% worse) -> move it to the last vector-memory statement of the body.
- caution: also verify the deeper depth on YOUR body before shipping it — depth 2 was monotone-decreasing but still positive on the first lane (1.034 / 1.032 / 1.031 / 1.027 at depth 2/3/4/6) and a 12% LOSS on the second, reproduced independently there by a depth-2 operand rotation losing to its depth-1 form — and quote the expectation on the graded geomean rather than the paired delta: a gate that excludes the small-M case divided a +3.4-4.0% paired win down to +0.58% geomean.
- source: run kb_on_0810 2026-08-10
