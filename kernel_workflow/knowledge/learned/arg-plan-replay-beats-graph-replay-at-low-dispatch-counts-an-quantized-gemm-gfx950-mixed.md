---
key: host/launch path of a block-scaled fp8 linear under Triton on gfx950, a handful of large dispatches per call
type: lever
confidence: ★★
effect: host lane worth 13.5% of the final geomean on all three cases (M=2048/32768/65536), plus 1.044x cumulative from the scale restage; a per-case decision between replay and eager measured 0.000%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed
description: Arg-plan replay beats device-graph capture at low dispatch counts (13.5% of geomean); its free extra dispatch funds a host restage of a scale operand
keywords: ['launch-overhead', 'host-runtime', 'graph-replay', 'quantized-gemm', 'scale-operand', 'block-scale', 'cache-line']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
---
# Arg-plan replay beats graph replay at low dispatch counts, and pays a second time
- lever: Memoize and replay the Triton launcher's argument plan (keyed on the callable identity) instead of capturing a device graph: at these dispatch counts the plan replay's CPU cost was well under half the graph's. The plan captures N launches, so an additional small dispatch inside the captured region is nearly free -- which funds a host-side restage the kernel could not do for itself.
- apply: Wrapper-level: build the arg plan once, replay it per call. Then pre-transpose the operand whose in-loop access stride is smaller than a cache line into a pooled buffer as a second captured dispatch, so the tile read becomes contiguous; the kernel needs no edit when it already takes strides as runtime args (four in-kernel attempts at the same gather failed on register pressure).
- verify: Alternate replay and eager arms with >=4 reps and paired in-process medians (the eager arm's cross-run spread is wide on the mid case); confirm the restage by cache lines touched per scale tile rather than by clock alone; run the parity gate through the replay path, not only the eager path.
- pitfall: a pooled staging buffer keyed by shape handed two distinct tensors the same buffer -> plan replay retargets the stream and destroys the program order that shape-keying silently assumes -> re-key the pool and exercise replay inside the correctness run, which surfaced corruption on 19 of 20 iterations.
- caution: also verify the in-loop stride really is below the cache line before restaging: generalizing the same transform to the two main operands returned exactly 1.000x because both were already line-aligned.
- source: run gemm_a8w8_blockscale-own16h, 2026-08-12, rounds 2-4 host lane and round 11 direction d1, director-validated
