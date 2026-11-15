# Match the native kernels' reduction order and rounding points

The stage's oracle judges a stage against a float64 reference with a tolerance
([stage_authoring.md](stage_authoring.md#oracle-unchanged-never-edited)); this page is about the drift
that passes a per-layer tolerance but shows up downstream. The source of truth is the native kernel
itself: read the files in the stage's native-source list (in its task statement) for its reduction order,
exponent base and rounding points. Backend-specific facts that cost the most time to rediscover are in
[native_reference_notes.md](native_reference_notes.md).

## When

Per-layer errors are small but the FP8 quantization of the next input flips codes, sequence checks drift,
or greedy outputs fork early; or a fused stage's error exceeds the sum of its parts' errors.

## Recipe

1. **RMSNorm sum of squares in the native reduction tree** (per-thread chunk summed sequentially, then
   the native kernel's lane / group tree, then the same rsqrt). A different order flips the FP8
   quantization of the next input now and then even when the norm itself is within tolerance.
2. **Fused all-reduce + residual + RMSNorm semantics:** check where native rounds. Typically the reduced
   sum plus residual enters RMSNorm in FP32 unrounded and is rounded once, when the residual is written
   back as BF16.
3. **FP8 activation quant from the FP32 normed value**, with the native group size and the native rule for
   amax 0.
4. **Activations and softmax as the native backend computes them**: SwiGLU form, and the exponent base of
   the online softmax (some backends use base 2 with a folded log2e, others the natural exponent).
5. **Attention sinks as native applies them**: some backends start the running max at the sink, so P is
   rounded against it; applying the sink after the online softmax is closer to exact math but further
   from native.
6. **Causal limit and window from `seq_lens` / query start locations** (query row j of a step sees up to
   seq_len − query_len + j), while RoPE uses `positions`. Mixing the two is wrong for multi-token steps.
7. **FP8 KV:** read and write in the native format and scale; the current token's K/V also go through the
   same quantize–dequantize before they enter attention.
8. **Per-expert results rounded where native rounds them** before the sum.
9. **TP sums in rank order** on every rank
   ([all-reduce fusion](handoffs_and_sync.md#fuse-the-tp-all-reduce-into-its-neighbours)).

## What is left after all this (so do not chase it)

GEMM / MoE accumulation orders (MFMA vs the native library) differ by ~1 ulp now and then. Greedy outputs
then fork only at near-ties; native itself forks there between batch sizes or eager vs graph mode. Compare
the kernel's agreement with native against native's agreement with itself, and use an accuracy benchmark
as the end-to-end criterion, not bit-identity.

## Check

The stage oracle as usual; for norm / quant stages also count FP8 code flips of the output
against the native op on the captured inputs (should be 0 for norm + quant).
