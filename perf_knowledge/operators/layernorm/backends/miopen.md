---
title: layernorm on miopen — SOTA card
kind: sota_card
operator: layernorm
backend: miopen
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both, training]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
---

# layernorm × miopen

## TL;DR
MIOpen (the cuDNN analog) **has a LayerNorm primitive** (`miopenLayerNormForward`), but on the LLM/serving
path it is **effectively idle** — norms come from AITER/Triton fused kernels, not MIOpen. MIOpen LayerNorm
matters where PyTorch routes through MIOpen: **vision encoders / CNN+norm front-ends, diffusion, and
training graphs** that don't go through the aiter path. For a transformer decoder, use [aiter.md](aiter.md)
or [triton.md](triton.md); MIOpen is the fallback when the framework's norm dispatch lands in MIOpen.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `miopenLayerNormForward` (solver-selected) | MIOpen (ROCm/rocm-libraries) | gfx908..950, fp32/bf16/fp16 | data-driven solver pick (FindDb); not benchmarked here vs aiter | PyTorch `layer_norm` lands in MIOpen (non-LLM) |
| AITER/Triton (recommended for LLM) | see [aiter.md](aiter.md)/[triton.md](triton.md) | gfx942/950 | bandwidth-bound | the actual serving path |

## Config space / knobs
- Solver selection: `MIOPEN_FIND_MODE` (HYBRID default), AI immediate-mode fallback
  (`MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=ON`). FindDb caches the chosen solver per shape.
- Precompile chosen solutions to avoid first-call JIT stall.
- See [[backends/miopen/overview]] for the find-vs-immediate machinery.

## Numerics / parity
fp32 μ/σ², biased variance; MIOpen's reduction differs from aiter/Triton → re-gate parity if you swap a
norm from aiter to MIOpen (rare). Training: MIOpen provides the backward.

## Integration (rebind seam)
- Implicit: PyTorch `F.layer_norm` may route to MIOpen on ROCm for some shapes/dtypes; there is **no
  serving-time rebind seam** to force it on the LLM path (and you wouldn't — aiter is faster).
- For vision towers in VLMs, MIOpen handles conv+norm; the LayerNorm there is MIOpen's.

## Pitfalls & anti-patterns
- ⚠ Don't try to wire MIOpen LayerNorm into an LLM decoder — there's no benefit and no clean seam; aiter/
  Triton own it.
- A cold `miopenFind*` per step destroys throughput — reserve once (general MIOpen rule).
- Repo moved: current source is `ROCm/rocm-libraries` (was `ROCm/MIOpen` ≤ 6.4.3).

## How to verify
`MIOPEN_ENABLE_LOGGING_CMD=1` to confirm a MIOpen LayerNorm solver ran (only relevant on the vision/CNN
path); for LLM, confirm norms are aiter/Triton in rocprofv3 (they should be).

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [hip.md](hip.md) · [[backends/miopen/overview]] ·
[[backends/miopen/conv]].

## Sources
- MIOpen find/immediate + LayerNorm primitive: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html.
- MIOpen CHANGELOG (norm solvers, repo move): https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md.
- MIOpen idle at LLM inference: perf_knowledge [[backends/miopen/overview]].
