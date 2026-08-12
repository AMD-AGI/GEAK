---
key: bf16 dense GEMM + bias with a frozen launch config, large-M compute-bound, gfx950/MI355 — the editable surface is one Python file that owns dispatch
type: lever
confidence: ★★
effect: 4.05x geomean director-verified, paired in-session, non-overlapping (two invocations 4.0448/4.0461); per-case 3.53x at M=2048, 4.36x at M=32768, 4.30x at M=65536
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound
description: Own the launch/dispatch layer of a frozen bf16 dense GEMM, then race Triton vs hand HIP vs tuned vendor library per shape: 4.05x geomean.
keywords: ['dense-gemm', 'bf16', 'gfx950', 'dispatch-shim', 'backend-routing', 'vendor-library', 'hipblaslt', 'launch-config', 'argmin-dispatch', 'codegen']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
---
# Own the dispatch layer, then race backends behind it
- lever: When the editable file owns the launcher, treat the frozen launch config as the defect first, then reuse that shim as a per-shape backend selector (in-tree kernel / hand-authored HIP / tuned vendor library) chosen by measured argmin, not by reasoning.
- apply: Replace the fixed grid with a real tile (256x256x64, 16 warps, 3 stages, tuned GROUP_SIZE_M); JIT extra backends from the same file via cpp_extension load_inline; freeze per-shape vendor algo indices from an exhaustive solution sweep; cache plans keyed by shape+algo+pointer; fuse bias into the library epilogue for a single copy-free dispatch; keep named fallback tiers.
- stack: total 4.05x director-verified geomean = three directions compounded, each landing at the shim
  - 1. tile/launch-config escape - 2.78x standalone (round 1, verified) - the whole cumulative for four rounds and the enabler of the rest
  - 2. hand-authored HIP kernel, same tile/waves/MFMA - +21% on top of (1) (round 5, verified) - isolates codegen, not algorithm, as the wall
  - 3. tuned vendor-library backend - +17% on top of (1,2) (round 7, verified) - largest single win
  - note: attribution is incremental in landing order; (2) and (3) are mutually exclusive backends, so (3) alone over (1) is roughly the product of the two.
- verify: Paired golden-vs-candidate A/B inside one process, repeated invocations; confirm the intended tier actually engaged (one-shot named warning on every fallback transition, strict env flag to make them fatal) - otherwise a build failure reads as 'the fast path was chosen'.
- pitfall: A shared build cache served a stale artifact so the old binary was timed -> validate any new instrument against a known-good reference dispatch before trusting one number from it.
An in-process sweep reported a zero duration for every algo while parity showed cos 1.000000 -> the timer, not the kernels, was broken.
- caution: Frozen algo indices are card- and library-version specific; also verify they still resolve on your build - an unresolved index degrades to the framework path silently, correct but roughly 5% slower, and shapes outside the frozen table lose ~1.11x at the smallest M.
- source: run _gemm_a16_w16_kernel-own16h, 2026-08-12, kernel_workflow 16h campaign, director-validated
