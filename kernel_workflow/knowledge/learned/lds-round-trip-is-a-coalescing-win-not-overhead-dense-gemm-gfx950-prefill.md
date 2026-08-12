---
key: removing the shared-memory layout round-trip / hand-pipelining global-to-LDS in a Triton or Gluon fp16 GEMM on gfx950
type: anti-pattern
confidence: ★★
effect: Loading directly into the dot layout to skip the round-trip measured 2.5x slower on the same shapes; a faithful reimplementation of the async-pipelined reference reproduced 2.667x vs the incumbent 2.664x weighted, i.e. no gain on any of the three cases, leaving the op at ~65% of its MFMA-efficiency ceiling.
confirms_cited: 3
confirms_blind: 0
losses: 1
attempts: 9
toolchain: unknown
last_seen: 2026-08-12
name: lds-round-trip-is-a-coalescing-win-not-overhead-dense-gemm-gfx950-prefill
description: The shared-memory layout round-trip in a Triton GEMM is a coalescing win: removing it is 2.5x slower, and the async-copy replacement does not lower.
keywords: ['dense-gemm', 'convert-layout', 'lds-tiling', 'async-copy', 'software-pipeline', 'coalescing', 'roofline', 'anti-pattern', 'gfx950']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: prefill
layer: learned
lifecycle: active
---
# lds-round-trip-is-a-coalescing-win-not-overhead
- lever: Before attacking a layout round-trip as overhead, price the alternative: the round-trip buys coalesced global loads and the backend already software-pipelines it, so the visible traffic is the cheap half of the trade.
- apply: Price it by A/B-ing one variant that loads straight into the dot layout; if that is slower, the remaining roofline gap is a backend-lowering question (async global-to-shared feeding a dot operand fails to lower in this toolchain) rather than a source-level one.
- verify: Compute the fraction of the MFMA-efficiency ceiling attained before and after; if a candidate reproduces the incumbent ratio to within noise it is the same schedule under a new spelling, not a new lever.
- pitfall: Scheduling flags copied from a reference implementation silently produced byte-identical code -> the flags do not exist in the deployed compiler version -> diff the generated assembly before scoring a flag direction at all.
- caution: Also verify the shape regime a reference kernel was tuned for: the ping-pong / async-pipeline reference here reached its ceiling on large square shapes and scopes itself out of skinny, low-K ones.
- source: 16h per-kernel time-budget campaign chuschen16h, 44 passes, 2026-08-11
