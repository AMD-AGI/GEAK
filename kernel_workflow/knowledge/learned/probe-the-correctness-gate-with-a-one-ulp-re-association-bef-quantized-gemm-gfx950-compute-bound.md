---
key: whether a tight max_rel correctness gate admits a reordered reduction on a block-scaled quantized GEMM (Triton) on gfx950, largest shape
type: method
confidence: ★★
effect: No speedup -- it retires lanes. A one-ulp scalar re-association of a rescale term measured max_rel 0.545 against a <1e-2 bar on the largest shape; four rounds of briefs were nevertheless aimed at a native low-precision matrix-core path whose nominal prize was ~1.6x the achievable roof, and when finally probed it produced 100% NaN from quantizer saturation, then cos=0.999555 / max_rel=1.5e5 with saturation patched, then a reduction 412x less accurate than the wider-dot path against an fp64 reference with saturation removed by construction.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950 CDNA4
last_seen: 2026-08-11
name: probe-the-correctness-gate-with-a-one-ulp-re-association-bef-quantized-gemm-gfx950-compute-bound
description: Probe the correctness gate with a one-ulp re-association before funding any reduction-reordering lane: it retired four rounds aimed at a ~1.6x roof
keywords: ['correctness-gate', 'measurement-method', 'dtype-dialect', 'mfma', 'quantized-gemm', 'roofline', 'control-experiment']
kernels: ['_gemm_a8w8_blockscale_kernel', '_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
lifecycle: active
---
# Probe the correctness gate with a one-ulp re-association before funding any lane that reorders the reduction
- lever: When the harness scores correctness with a tight max_rel or bitwise comparison, treat 'does this gate admit a different summation order' as a measurable precondition, not an assumption: any lane that changes the reduction (a narrower matrix-core dtype, split-K or stream-K, atomics, algebraic re-association of a scale) is only worth funding if the gate tolerates reorder at all.
- apply: Spend a few minutes first on the cheapest possible reorder -- re-associate one scalar expression so the result differs by about one ulp, and run an atomics-free control that merely regroups the k reduction into two fp32 partials -- and read the gate's max_rel on the largest shape; if a one-ulp change already fails, the whole family is closed and the slots can go elsewhere.
- verify: Report max_rel and the count of violating elements, not cosine similarity, which reads 1.000000 over millions of violating elements; when the probe is a dtype change, also check the operand map separately from the accumulation.
- pitfall: a narrow matrix-core probe returned 100% NaN and looked like an accumulation-order verdict -> saturating codes in one fp8 dialect reinterpret as NaN in another, so it was a value-level operand-map bug -> patch saturation first and re-read the gate before attributing the failure to reduction order.
- caution: Also verify the roof the lane is being sized against: if the reorder-dependent path is inaccessible, the vendor number for that path is not a reachable target and any gap measured against it will keep re-funding closed lanes -- re-baseline on the widest dtype the gate actually admits.
- source: run kb_on_0810 2026-08-11
