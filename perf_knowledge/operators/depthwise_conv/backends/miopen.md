---
title: depthwise_conv on MIOpen — SOTA card
kind: sota_card
operator: depthwise_conv
backend: miopen
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
---

# depthwise_conv × MIOpen

## TL;DR
MIOpen is the production depthwise-conv backend on AMD. There is **no dedicated depthwise API** — you call
the general convolution path and set `miopenSetConvolutionGroupCount(C)` with `C == in_channels`. The two
performance decisions are the same as any MIOpen conv: **NHWC layout** (the fast solvers + the only fused
path) and **solver selection** (seed FindDb with one NORMAL find, then run default HYBRID). Depthwise is
memory-bound, so the winning solvers are **direct / Winograd / CK-inline direct**, not matrix-core
implicit-GEMM. This is the VLM vision-tower / CNN lever, not the language-model lever — for the LLM 1D
case use [[causal_conv1d]].

## SOTA implementation(s)
| impl | source | gens/dtypes/shapes | measured perf | when best |
|---|---|---|---|---|
| MIOpen grouped conv, group_count=C, NHWC | `miopenSetConvolutionGroupCount` + Find/immediate; solvers direct/Winograd/CK-inline | gfx908–950; fp32/fp16/bf16/int8; dilation=1 | shape-dependent; Winograd is the 3×3 stride-1 sweet spot (no on-box measurement here — vision tail) | production CNN/VLM depthwise conv on ROCm |
| MIOpen fusion plan (conv+bias+act) | `miopenExecuteFusionPlan_v2` (NHWC) | same | fewer launches/HBM passes | inference, channels-last |

Honest gap: no on-box depthwise benchmark was taken for this card (no vision workload on the box). Numbers
are shape-specific; replay with `MIOpenDriver` (below) on your shapes before quoting.

## Config space / knobs
- `miopenSetConvolutionGroupCount(C)` — **before** any Find/workspace call; `C==channels` = depthwise.
- Layout: NHWC/NDHWC (`miopenSetNdTensorDescriptorWithLayout`; PyTorch `channels_last`).
- `MIOPEN_FIND_MODE` = NORMAL/1 (benchmark all, seed FindDb) | FAST/2 | **HYBRID/3 default** | DYNAMIC_HYBRID/5.
- `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` (default ON), `MIOPEN_USER_DB_PATH`, `MIOPEN_ENABLE_LOGGING_CMD=1`.
- `miopenConvolution*CompileSolution` to precompile (avoid first-call JIT stall).
- **dilation = 1 only** for depthwise/grouped.

## Numerics / parity
fp32 accumulate; solver swaps parity-safe within `atol≈1e-2` (bf16). Winograd (3×3 s1) has slightly
larger error — loosen tolerance there. int8 → task-accuracy gate. See [../numerics.md](../numerics.md).

## Integration (how it gets used)
PyTorch routes `nn.Conv2d(groups=C)` → MIOpen automatically. To hit the fast path:
`model.to(memory_format=torch.channels_last)` + channels-last inputs. There is **no Python "rebind seam"**
to inject a custom kernel into MIOpen's dispatch — to override, you bypass MIOpen at the PyTorch level
(custom op / Inductor lowering), not inside MIOpen.

## Pitfalls & anti-patterns
- ⚠ Forgetting `setConvolutionGroupCount` before Find → wrong/slow solver or error.
- ⚠ NCHW input → no fused path + slower solvers; convert to channels-last first.
- ⚠ Cold `miopenFind*` per step → throughput collapse; reserve algo+workspace once.
- ⚠ Dilated depthwise → unsupported group solver (dilation must be 1).
- ⚠ First immediate/fused call JIT-compiles → precompile for steady-state latency.
- Repo moved: MIOpen ≤ROCm 6.4.3 is `ROCm/MIOpen`; current source is `ROCm/rocm-libraries`.

## How to verify
```bash
MIOPEN_ENABLE_LOGGING_CMD=1 python infer.py    # shows chosen solver + replay line
MIOpenDriver conv --group-count C -n N -c C -H H -W W -y R -x S -p pad -u stride --layout NHWC -t 1 -V 1
```
Compare NHWC vs NCHW and HYBRID vs NORMAL-find; seed FindDb with NORMAL for production shapes.

## Alternatives / cross-links
[triton.md](triton.md) (authored / Inductor max-autotune) · [hip.md](hip.md) (hand-written fixed shape) ·
[../overview.md](../overview.md) · backend deep-dive: [`../../../backends/miopen/`](../../../backends/miopen/)
(conv.md, overview.md) · related op: [[conv2d]] · LLM 1D variant: [[causal_conv1d]].

## Sources
- Grouped/depthwise via group count; dilation=1; group-count before Find: https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
- Find/immediate, FIND_MODE, FindDb, AI fallback: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- Fusion (conv+bias+act, NHWC-only, op-order): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
- Grouped-conv fusions, NHWC default, CK-inline solvers, repo move: https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
