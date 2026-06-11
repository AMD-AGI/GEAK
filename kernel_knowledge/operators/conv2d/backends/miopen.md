---
title: conv2d on MIOpen — SOTA card
kind: sota_card
operator: conv2d
backend: miopen
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
  - https://github.com/ROCm/MIOpen/issues/2001
---

# conv2d × MIOpen

## TL;DR
MIOpen is **the** production conv2d backend on AMD: forward + backward-data + backward-weight, with a
benchmark-driven solver DB. For the matrix-core path it calls **Composable Kernel XDL implicit-GEMM
solvers inline** (CHANGELOG) — so "MIOpen conv2d" on the MFMA path is CK under the hood. The two perf
decisions: **NHWC layout** (fast solvers + the only fused path) and **solver selection** (seed FindDb with
one NORMAL find, then default HYBRID). In an LLM stack this is **idle**; it's the VLM vision-tower /
diffusion / CNN lever. For the LLM 1D conv use [[causal_conv1d]].

## SOTA implementation(s)
| impl | source | gens/dtypes/shapes | measured perf | when best |
|---|---|---|---|---|
| MIOpen conv (Find/immediate, NHWC) | `miopenFindConvolution*` / immediate-mode + FindDb; solvers implicit-GEMM(XDL)/Winograd/direct/FFT/CK-inline | gfx908–950; fp32/fp16/bf16/int8 | shape-dependent; implicit-GEMM ~GEMM-class on large-channel convs (~45–55% of peak); Winograd best for 3×3 s1. No on-box measurement (vision tail). | production conv2d on ROCm |
| MIOpen fusion plan (conv+bias+act) | `miopenExecuteFusionPlan_v2` (NHWC) | same | fewer launches/HBM passes | inference, channels-last |

Honest gap: no on-box conv2d benchmark (no vision workload on the box); quote a `MIOpenDriver` replay on
your shapes.

## Config space / knobs
- Layout: NHWC/NDHWC (`miopenSetNdTensorDescriptorWithLayout`; PyTorch `channels_last`).
- `MIOPEN_FIND_MODE` = NORMAL/1 (benchmark all, seed FindDb) | FAST/2 | **HYBRID/3 default** | DYNAMIC_HYBRID/5.
- `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` (default ON, ~90% pick), `MIOPEN_USER_DB_PATH`,
  `MIOPEN_ENABLE_LOGGING_CMD=1`, `miopenConvolution*CompileSolution` (precompile).
- Reserve + reuse algo + workspace (a too-small workspace forces a slow algorithm).

## Numerics / parity
fp32 accumulate; solver swaps parity-safe within `atol≈1e-2` (bf16); Winograd 3×3 s1 slightly looser;
int8/fp8 → task-accuracy gate (fp8 dialect must match arch). See [../numerics.md](../numerics.md).

## Integration (how it gets used)
PyTorch `nn.Conv2d` → MIOpen automatically; hit the fast path with `model.to(memory_format=
torch.channels_last)` + channels-last inputs. **No Python rebind seam** to inject a custom kernel into
MIOpen's dispatch — override at the PyTorch level (custom op / Inductor) or call CK directly
([composable_kernel.md](ck.md)).

## Pitfalls & anti-patterns
- ⚠ NCHW input → no fused path + slower solvers; convert to channels-last first.
- ⚠ Cold `miopenFind*` per step → throughput collapse; reserve algo+workspace once.
- ⚠ FindDb miss + AI fallback OFF → generic GEMM solver = slow; run NORMAL find once to seed.
- ⚠ NHWC descriptor wiring friction (issue #2001) — set the layout descriptor correctly.
- ⚠ First immediate/fused call JIT-compiles → precompile.
- Repo moved: ≤ROCm 6.4.3 `ROCm/MIOpen`; current `ROCm/rocm-libraries`.

## How to verify
```bash
MIOPEN_ENABLE_LOGGING_CMD=1 python infer.py    # chosen solver + replay line
MIOpenDriver conv -n N -c C -H H -W W -k K -y R -x S -p pad -u stride --layout NHWC -t 1 -V 1
```
Compare NHWC vs NCHW and HYBRID vs NORMAL-find; seed FindDb with NORMAL for production shapes.

## Alternatives / cross-links
[composable_kernel.md](ck.md) (the XDL solvers MIOpen calls; author directly) ·
[hip.md](hip.md) · [../overview.md](../overview.md) · backend deep-dive:
[`../../../backends/miopen/`](../../../backends/miopen/) (conv.md) · related ops: [[depthwise_conv]],
[[causal_conv1d]].

## Sources
- Find/immediate, FIND_MODE, FindDb, AI fallback, precompile: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- Fusion (conv+bias+act, NHWC-only, op-order): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
- CK-inline implicit-GEMM solvers, NHWC default, HYBRID default, repo move: https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- NHWC descriptor friction: https://github.com/ROCm/MIOpen/issues/2001
