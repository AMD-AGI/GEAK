---
name: register-prefetch-the-small-per-iteration-operands-one-k-blo-quantized-gemm-gfx950-compute-bound
description: Register-prefetch the SMALL per-iteration operands one k-block ahead as the body's last VMEM: +3.9-7.0% on the large-M cases of a block-scaled GEMM
keywords: [prefetch, vgpr, isa-check, operand-reuse, compute-bound, quant, dequant, mfma]
kernels: [_gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: staging small scale/metadata global loads into registers one k-block ahead inside the register-saturated MFMA k-loop of a block-scaled fp8 GEMM (Triton, gfx950)
lifecycle: active
type: lever
confidence: ★★
effect: paired on the two large-M cases of a 24.99x director-verified end state (per-case 24.39x / 24.87x / 25.73x at M=2k / 32k / 64k): the 16-dword/lane B operand at depth 1 gave +6.7% / +4.5% (M=32k / 64k), the two 2-dword scale loads at depth 1 a further +3.9% / +7.0%, and the same scales at depth 2 a further +3.4% / +4.0%; ~0% on the small-M case, whose short path the constexpr gate leaves untouched.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 9
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950 CDNA4
source: run kb_on_0810 2026-08-10
last_seen: 2026-08-10
---
# register-prefetch the SMALL per-iteration operands one k-block ahead, as the last VMEM of the body; size the candidate list by dwords/lane
- lever: on a register-saturated MFMA k-loop, try staging a loop operand's global load one (or two) k-blocks ahead into registers; affordability is set by BYTES PER LANE, not by the operand's importance — 2-dword scale/metadata loads stayed spill-free to depth 6, a 16-dword tile operand paid only at depth 1, and a 128-dword operand never paid.
- apply: issue the prefetched load as the LAST vector-memory statement of the body; prefetch a group of related small loads together rather than one of them (either scale alone was 0.91-0.96x, the pair +2.1% geomean); constexpr-gate it to the loop variant you tuned, since the same patch on a num_stages=2 variant of the same kernel was 0.93x.
- verify: read the first s_waitcnt of the loop body before and after — the acceptance tell that predicted the sign three rounds running is that the prefetched load has LEFT the wait chain (its vmcnt term disappears), not that any counter improved; then confirm 0 spilled dwords, unchanged shared-memory size, and bit-identical output.
- pitfall: the identical prefetch measured 0.99x with fresh spills -> the load had been placed at the top of the body, extending every live range across it (mid-body was 2.7-3.5% worse) -> move it to the last vector-memory statement of the body.
- caution: also verify the deeper depth before shipping it — gain was monotone-decreasing past depth 2 here (1.034 / 1.032 / 1.031 / 1.027 at depth 2/3/4/6) — and quote the expectation on the graded geomean rather than the paired delta: a gate that excludes the small-M case divided a +3.4-4.0% paired win down to +0.58% geomean.
- source: run kb_on_0810 2026-08-10
