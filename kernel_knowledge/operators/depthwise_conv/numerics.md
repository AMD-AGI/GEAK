---
title: depthwise_conv — numerics
kind: operator_overview
operator: depthwise_conv
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# depthwise_conv — numerics

Medium-depth (vision-territory). For the LLM-relevant 1D variant's numerics see
[[causal_conv1d]]/numerics.

## The contract
A per-channel spatial dot with **fp32 accumulation** for fp16/bf16 inputs (MIOpen convs accumulate in
fp32, cast to the output dtype on store). The reduction is over the spatial window only (`R×S`, typically
9 for 3×3) — short, so accuracy is dominated by output rounding, not accumulation drift. Parity vs a
reference depthwise conv (`F.conv2d(..., groups=C)`) is **same-math** and tight:
- bf16: `atol≈1e-2/rtol≈1e-2`; fp16 tighter; fp32 near-exact.
A different solver (direct vs Winograd vs CK-inline) gives a *different* rounding but stays within the
same band — solver swaps are parity-safe, not bit-identical.

## Winograd caveat
The Winograd solver (the fast 3×3 stride-1 path) transforms input/filter into the Winograd domain,
multiplies, and transforms back. That transform introduces **slightly larger numerical error** than
direct convolution — usually still within an `atol≈1e-2` bf16 band, but if you see a parity test fail
*only* on 3×3 stride-1 shapes, suspect Winograd and re-check the tolerance (or force a direct solver to
isolate). This is a known, accepted CNN tradeoff, not a bug.

## int8 / quant
int8 depthwise needs a **task-accuracy gate**, not byte parity — per-channel scales, int32 accumulate,
requantize on store. Depthwise is more sensitive to per-channel quant than a dense conv because there's
no channel averaging to smooth scale error; validate end-task accuracy, not just per-tensor MSE.

## Layout does not change math
NHWC vs NCHW is a **layout** change, not a math change — same fp32-accumulate result within rounding.
NHWC is purely a performance choice (fast solvers + fused path); it must not change correctness. If NHWC
and NCHW disagree beyond the rounding band, it's a descriptor/stride bug, not numerics (see MIOpen NHWC
descriptor friction, issue #2001).

## Determinism
MIOpen conv solvers are generally deterministic for a fixed solver + shape, but a FindDb miss can pick a
*different* solver run-to-run (different rounding) — pin the solver (seed FindDb with one NORMAL find) if
you need run-to-run reproducibility.

## Verify
```python
y = conv_depthwise(x_nhwc)                          # MIOpen via channels_last
y_ref = F.conv2d(x, W, bias, stride, pad, groups=C) # reference
torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)   # bf16 band; loosen for Winograd 3x3s1
```

## Sources
- fp32 accumulate for fp16/bf16 conv; grouped/depthwise via group count: https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
- Fusion (conv+bias+act) NHWC path: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
- NHWC = layout not math; pin solver for determinism: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
