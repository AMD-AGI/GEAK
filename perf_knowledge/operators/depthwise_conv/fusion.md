---
title: depthwise_conv — fusion
kind: operator_overview
operator: depthwise_conv
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# depthwise_conv — fusion

Medium-depth (vision-territory). For the LLM-relevant 1D variant's fusion (conv+SiLU+QKV-split in GDN)
see [[causal_conv1d]]/fusion.

## The fusion that matters: conv + bias + activation
Because depthwise conv is memory-bound and cheap, the win is **not re-reading its output** for the bias
and activation — fuse them into one kernel.
- MIOpen **fusion plan**: build a plan, add ops in **data-flow order** (conv → bias → activation),
  **compile once** (costly), then reuse with different argument buffers. Execute with
  `miopenExecuteFusionPlan_v2` — which is **NHWC/NDHWC only**. Recent MIOpen also added
  **grouped-conv + activation** and **grouped-conv + bias + activation** fusions plus NHWC batchnorm,
  which is exactly the depthwise case in modern vision backbones.
- Operator *order* is part of the contract: `conv→bias→act` ≠ `act→conv`; build the plan to match the
  graph.

## Depthwise → pointwise (the separable-conv chain)
A depthwise-separable conv is **depthwise conv → 1×1 pointwise conv**. The 1×1 is a plain GEMM
(hipBLASLt/MIOpen GEMM solver), **not** a depthwise op. They do **not** fuse into one kernel (one is
bandwidth-bound spatial, the other is matrix-core-bound channel-mixing) — keep them as two kernels, each
on its fast path, with the activation fused into whichever produces the consumed tensor.

## Batchnorm fold
At inference, fold batchnorm into the conv weights/bias offline (standard CNN transform) so BN is free;
at training, MIOpen offers NHWC batchnorm as a separate (now default-NHWC) op around the conv. Folding is
a graph rewrite, not a kernel fusion — do it in the model export, not at runtime.

## Fusion support matrix
| fusion | MIOpen fusion plan | Triton (authored) | HIP (authored) |
|---|---|---|---|
| + bias | yes (NHWC) | yes (epilogue) | yes |
| + activation (relu/silu/…) | yes (NHWC, op-order) | yes | yes |
| + bias + act (grouped/depthwise) | **yes** (recent MIOpen) | yes | yes |
| + batchnorm (training) | separate NHWC BN op | manual | manual |
| depthwise + pointwise (1×1) | no (two kernels) | no | no |

## What does NOT fuse (and why)
- depthwise + the following 1×1 (different boundness, different tiling) — see above.
- NCHW input → **no fused path at all** (the fused executor is NHWC-only); convert to channels-last first
  or you lose the fusion entirely.

## Where fusion moves wall time
In a CNN/vision tower the depthwise conv is a memory-bound tail; fusing bias+act removes a full extra
read+write of its output tensor (often the bigger cost than the conv math). On NCHW inputs the layout
transform itself can cost more than the fusion saves — channels-last end-to-end is the real lever.

## Sources
- MIOpen fusion API (`_v2` NHWC-only, op order = data-flow, compile-once): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
- Grouped-conv + (bias +) activation fusions, NHWC batchnorm default: https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- Channels-last end-to-end / fuse epilogue guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
