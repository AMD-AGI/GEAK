---
key: small MoE router top-k + softmax + bitmatrix pack, bf16, tiny grid on gfx950 - the in-tree Triton Python launcher, not the device code, is the bottleneck
type: lever
confidence: ★★
effect: 1.96x geomean director-verified (three runs 1.94/2.03/1.96, non-overlapping); per case 2.27x at 2048 rows, 2.12x at 32768, 1.54x at 65536 - the gain shrinks as device time grows
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both
description: Tiny dispatch-bound Triton op: memoize compile, bake launch opts, call the C launch entry directly - ~1.96x geomean, largest on the smallest case
keywords: ['launch-overhead', 'host-runtime', 'dispatch-bound', 'triton', 'moe-router', 'topk', 'memoization', 'gfx950']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: moe_router_topk
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['host.launch-overhead']
origin_kernels: ['_topk_forward']
---
# Bypass the JIT launcher for a dispatch-bound Triton op
- lever: memoize compile+bind once, bake the launch options into the cached handle, then call the compiled kernel's C launch entry with pre-resolved device pointers and a 1-slot monomorphic inline cache in closure cells.
- apply: edit the Python launcher that ships next to the kernel: cache the CompiledKernel on first call, skip JITFunction.run and the launcher's scratch wrapper, and compile num_warps/num_stages in so no kwargs cross the binder per call.
- stack: total 1.96x director-verified geomean = three host-side steps plus one device step
  - 1. memoized compile + C launcher (round 1) - 1.48x - host issue per launch down ~2.3x, uniform across all three cases
  - 2. launch options baked into the handle, num_warps=2 / num_stages=1 (round 2) - 1.79x cumulative; LDS per block falls to zero, the reduction stays intra-wave
  - 3. raw launch entry + int pointers + inline cache (round 3) - 1.88x cumulative; host issue now ~1.1x the empty-kernel direct-C floor
  - 4. softmax as exp2 + one reciprocal (round 3, device-only) - pays only at the largest case; integrated 1.96-2.00x
  - note: attribution is incremental in landing order, not independent.
- verify: paired A/B against the frozen baseline plus a per-launch host-issue counter; re-prove the fast path is live on the merged file, and re-check parity because the exp2 step costs up to ~1 bf16 ULP.
- pitfall: two real device wins measured null in the first round -> the host floor hid them -> re-issue device candidates after the launcher patch lands.
- pitfall: the best num_warps lost its gain to per-call kwarg binding -> the constants were passed per launch -> bake them into the memoized handle instead.
- caution: also verify graph capture/replay before assuming it is the cheapest dispatch path - here replay measured about 2x worse than the direct memoized launch.
- source: run kernel_20_geak_0808_16h lane _topk_forward, 2026-08-12, director_validation.json + tech_lead_report.md
