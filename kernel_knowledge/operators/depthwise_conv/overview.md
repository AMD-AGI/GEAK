---
title: depthwise_conv — overview
kind: operator_overview
operator: depthwise_conv
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://github.com/Dao-AILab/causal-conv1d
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# depthwise_conv  (`groups == in_channels`, one filter per channel)

## TL;DR
Depthwise convolution = a grouped conv with `groups == in_channels`, so each channel is convolved with
its own filter (no cross-channel mixing). It is **memory-bound** (very low arithmetic intensity — one
small filter per channel, no channel reduction), which makes it a poor fit for matrix cores and a classic
"falls off the implicit-GEMM fast path" op. On AMD this is **MIOpen territory** (it has no dedicated
depthwise API — you set `miopenSetConvolutionGroupCount(groups=channels)` on the general conv path), and
it is **vision/CNN, not the LLM hot path**. The one LLM-relevant depthwise op — the *short causal 1D*
variant in Mamba/GDN — has its own dedicated kernels: see [[causal_conv1d]] (do not route that through
MIOpen). This card covers the 2D/3D spatial depthwise conv used in vision backbones (MobileNet-style,
ViT conv stems, diffusion).

## Math contract
Grouped cross-correlation with `groups = C`:
`Y[n,c,p,q] = Σ_{r,s} X[n,c, p·s_h+r·d_h, q·s_w+s·d_w] · W[c,1,r,s] (+ bias[c])`, stride `s`, dilation
`d`, padding. **Depthwise constraint in MIOpen**: dilation must be **1** for group/depthwise convs.
dtype: fp16/bf16/int8 in, **fp32 accumulate**. A depthwise-separable conv is this op followed by a 1×1
("pointwise") conv — the 1×1 is a plain GEMM (hipBLASLt/MIOpen), *not* this op.

## Shape regimes
- Vision backbones: `[N, C, H, W]` with C in the hundreds–thousands, 3×3 (sometimes 5×5/7×7) filters,
  stride 1 or 2. Channels-last (NHWC) is the perf layout on ROCm.
- This op rarely dominates a model's FLOPs (it's deliberately cheap) but can dominate **wall time** when
  it's memory-bound and the rest is matrix-core-bound — i.e. it's an Amdahl tail in CNN inference.
- 1D depthwise on long sequences exists (audio, Conv-former) but the LLM-relevant 1D case is the short
  causal one — [[causal_conv1d]].

## Where it matters (Amdahl)
In a transformer LLM: **absent**. In a VLM's vision tower or a diffusion U-Net: present but small —
the win is keeping it on a fast NHWC MIOpen/CK solver instead of a generic GEMM fallback, and **fusing**
conv+bias+activation so it's one kernel. Optimize the GEMMs first; this is a tail.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| miopen | 🟢 sota (the production conv library; grouped/depthwise via group count, NHWC, CK-inline solvers) | [backends/miopen.md](backends/miopen.md) |
| triton | 🟡 competitive (authorable; PyTorch-Inductor can lower conv to Triton under max-autotune when it beats MIOpen) | [backends/triton.md](backends/triton.md) |
| hip | 🟡 competitive (hand-written for a fixed shape; only when MIOpen lacks a fast solver) | [backends/hip.md](backends/hip.md) |
| ck | 🟢 (the grouped-conv implicit-GEMM solvers MIOpen calls inline) | see [[conv2d]] / [`../../backends/composable_kernel_lib/`](../../backends/composable_kernel_lib/) |

## Fusion neighbors
conv + bias + activation (MIOpen fusion plan), depthwise→pointwise (separable) chaining, batchnorm fold
→ see [fusion.md](fusion.md).

## Numerics
fp32-accumulate over a small spatial window; same-math parity vs a reference depthwise conv; int8 needs a
quant gate → see [numerics.md](numerics.md).

## How to bench
`MIOpenDriver conv --group-count <C> ... --layout NHWC -t 1` (replay the exact shape); PyTorch:
`model.to(memory_format=torch.channels_last)` + `MIOPEN_ENABLE_LOGGING_CMD=1` to see the chosen solver.

## Sources
- MIOpen grouped/depthwise = set group count to #channels; dilation must be 1: https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
- MIOpen find/immediate + NHWC fast path: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- LLM-relevant depthwise is the short causal 1D variant (separate kernels): https://github.com/Dao-AILab/causal-conv1d
- NHWC / ≥1024-grid / memory-bound tail guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
