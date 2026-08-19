---
key: an already MFMA-bound fp16 dense GEMM on gfx950 CDNA4, deciding whether host, swizzle, epilogue or split-K rounds are still worth buying
type: anti-pattern
confidence: ★★
effect: ~1.00x each across six directions on all three cases (graph capture/replay, XCD + GROUP_M swizzle, cache modifiers, epilogue bias-fold, split-K and M-adaptive dual-tile, LLIR/AMDGCN scheduling flags); the negatives are unambiguous — split-K measures 0.53x on the small-M case and dropping the LDS round-trip measures 2.5x slower
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-12
name: the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound
description: Once an fp16 dense GEMM sits at ~2/3 of its MFMA roof, six host/layout/algorithm directions each returned ~1.00x; the residual gap is a backend lowering gap
keywords: ['dense-gemm', 'roofline', 'mfma', 'split-k', 'launch-overhead', 'hip-graph', 'xcd-swizzle', 'convert-layout', 'gfx950', 'compute-bound', 'anti-pattern']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
layer: learned
lifecycle: active
origin_kernels: ['_gemm_a16_w16_kernel']
---
# The in-source ceiling of an MFMA-bound dense GEMM
- lever: Roofline the incumbent before planning more rounds: at roughly 65% of the MFMA roof (ceiling ~99% of peak) with one dispatch per call and the host launch fully hidden, the residual gap is the async global->LDS software pipeline, and host, swizzle, epilogue and split-K rounds all return ~1.00x.
- apply: Read the fraction-of-roof and the dispatch count first; a compute-bound op at 2/3 of its roof with a hidden launch tail is telling you the remaining lever is in codegen, not in the wrapper or the grid ordering.
- verify: Confirm each scheduling flag actually exists in the deployed compiler by diffing the generated assembly with and without it — here the codegen was byte-identical, so the flags were inert rather than ineffective.
- pitfall: the convert_layout round-trip through LDS read as a missing optimization -> loading directly in dot layout removes the coalescing the round-trip buys and measures 2.5x slower -> keep the round-trip, the backend already pipelines it; separately, the async global->LDS primitive feeding a DotOperand fails to lower at LLVM translation (unrealized_conversion_cast), so that path needs a compiler-side fix rather than another source round.
- caution: also verify the grid-ordering knobs on your own shape before treating them as closed here: GROUP_M was razor-sharp at 4 and the XCD count was already optimal on this shape, which is a property of this tile/CU ratio.
- source: run _gemm_a16_w16_kernel-ch16h (16h single-kernel budget, 44 passes), 2026-08-12
