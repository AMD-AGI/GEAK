---
title: MIOpen convolution — solvers, NHWC, find/immediate, conv+bias+act fusion
kind: backend
backend: miopen
operator: conv2d
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
  - https://github.com/ROCm/MIOpen/issues/2001
---

# MIOpen convolution (conv2d / depthwise / grouped)

## TL;DR
MIOpen owns `conv2d` (forward + bwd-data + bwd-weights), grouped/depthwise conv, and the conv+bias+act
fusion that vision encoders and CNNs rely on. The two performance decisions: (1) **layout** — use
**NHWC/NDHWC** (channels-last) to hit the fast solvers and the only fused path; (2) **solver selection** —
seed **FindDb** with a NORMAL find once, then run with the default HYBRID mode. In an LLM stack this is the
**VLM vision tower** lever, not the language-model lever. Operators: `conv2d`, `depthwise_conv`.

## Math contract
Standard cross-correlation conv: `Y[n,k,p,q] = sum_{c,r,s} X[n,c, p·s_h+r·d_h, q·s_w+s·d_w] · W[k,c,r,s]
(+ bias)`, with stride `s`, dilation `d`, padding, groups. fp32 accumulate for fp16/bf16 inputs.

## Solver families
| solver | best for |
|---|---|
| implicit GEMM / GEMM | general dense conv, large channels |
| **Winograd** | 3×3 stride-1 conv (the ViT/CNN sweet spot) |
| direct | small/odd shapes |
| FFT | large kernels (rare in transformers) |
| **Composable Kernel (CK) inline** | MFMA-backed conv solvers, built inline in MIOpen (CHANGELOG) |

Which solver runs is chosen by FindDb / immediate-mode heuristic — you select the **mode**, not the solver
directly (though `MIOpenDriver`/logging reveals it).

## Layout — NHWC is the fast path
- ROCm/MIOpen perf and the **fused** execution path (`miopenExecuteFusionPlan_v2`) are **NHWC/NDHWC only**.
- Set tensor descriptors with `miopenSetNdTensorDescriptorWithLayout` (developers have hit friction wiring
  NHWC descriptors — issue #2001). PyTorch: use `model.to(memory_format=torch.channels_last)` and
  channels-last inputs so the ROCm conv backend selects NHWC solvers.

## Find / immediate mode (selecting + caching a solver)
- **Find** (`miopenFindConvolution{Forward,BackwardData,BackwardWeights}Algorithm`): benchmarks applicable
  solvers — expensive; **call once**, store the returned algo + workspace, reuse for the process lifetime.
- **Immediate mode**: `miopenConvolution*GetSolutionCount` → `*GetSolution` → `*Immediate`; optional
  `*CompileSolution` to pre-populate the kernel cache (first `*Immediate` JIT-compiles otherwise).
- **`MIOPEN_FIND_MODE`**: `HYBRID`/3 (default — FindDb hit, else full find), `FAST`/2 (FindDb hit, else
  immediate fallback — fast start, possible perf loss), `NORMAL`/1 (benchmark all — slow start, best perf),
  `DYNAMIC_HYBRID`/5. On a FindDb miss, immediate mode uses the AI heuristic
  (`MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=ON`, ~90% solver-prediction accuracy) or a throughput index.

## Conv + bias + activation fusion
- Build a fusion plan from the input descriptor; **operator creation order = data-flow order** (conv→bias→
  act differs from act→conv). Compile the plan **once** (costly), then reuse with different argument buffers.
- Execute with `miopenExecuteFusionPlan_v2` (NHWC/NDHWC; takes a workspace for layout transforms). Recent
  MIOpen added **grouped-conv + activation** and **grouped-conv + bias + activation** fusions, plus NHWC
  batchnorm — relevant to modern vision backbones.

## Pitfalls
- NCHW input → no fused path and slower solvers; convert to channels-last first.
- Cold find per step = throughput collapse; reserve+reuse algo and workspace.
- First immediate/fused call JIT-compiles → precompile (`*CompileSolution`) for steady-state latency.
- Recompile the fusion plan only when descriptors/op-params change, not per call.

## How to bench
```bash
MIOPEN_ENABLE_LOGGING_CMD=1 python infer.py   # prints the MIOpenDriver command for each conv
# replay / sweep a single conv:
MIOpenDriver conv -n <N> -c <C> -H <H> -W <W> -k <K> -y <R> -x <S> -p <pad> -u <stride> --layout NHWC -V 1
```
Use `--layout NHWC`, `-t 1` for timing; compare against NCHW to confirm the channels-last win on your shapes.

## Alternatives / cross-links
[overview.md](overview.md) · Inductor conv lowering under max-autotune:
[../pytorch_inductor/max_autotune.md](../pytorch_inductor/max_autotune.md) · operators `conv2d`,
`depthwise_conv`.

## Sources
- Find APIs & immediate mode (FIND_MODE values, AI fallback, FindDb): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- Fusion API (`_v2` NHWC-only, op order, compile-once): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
- CHANGELOG (grouped-conv fusion, NHWC batchnorm, CK inline, HYBRID default): https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- NHWC descriptor wiring friction: https://github.com/ROCm/MIOpen/issues/2001
