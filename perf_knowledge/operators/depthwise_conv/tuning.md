---
title: depthwise_conv — tuning
kind: operator_overview
operator: depthwise_conv
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
  - https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# depthwise_conv — tuning

Medium-depth: this is a vision/MIOpen-territory op, light on the LLM path. The LLM-relevant 1D variant
is [[causal_conv1d]] — tune that, not this, for language models.

## The decision first
Depthwise conv is **memory-bound** (no channel reduction → low arithmetic intensity → matrix cores
mostly idle). You do **not** hand-write MFMA tiles for it. The levers are MIOpen's: (1) **NHWC layout**,
(2) **solver selection** (`MIOPEN_FIND_MODE` + FindDb), (3) **conv+bias+act fusion**. Only drop to an
authored Triton/HIP kernel if MIOpen has no fast solver for your exact shape (verify with logging first).

## MIOpen levers (the production path)
| lever | knob | note |
|---|---|---|
| **group count** | `miopenSetConvolutionGroupCount(C)` | required *before* any Find/workspace call; `C == channels` = depthwise |
| **layout** | NHWC/NDHWC (channels-last) | the fast solvers + the fused path are channels-last only; PyTorch `to(memory_format=torch.channels_last)` |
| **solver selection mode** | `MIOPEN_FIND_MODE` = NORMAL/1, FAST/2, **HYBRID/3 (default)**, DYNAMIC_HYBRID/5 | NORMAL benchmarks all (slow start, best perf) — run **once** to seed FindDb for production shapes; HYBRID reuses FindDb then full-finds on miss |
| **immediate fallback** | `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` (default ON) | AI heuristic (~90% solver-pick accuracy) on FindDb miss; OFF → throughput index (can pick a generic GEMM = slow) |
| **workspace** | reserve + reuse the algo + workspace | a too-small workspace forces a slow algorithm; a cold Find per step collapses throughput |
| **precompile** | `miopenConvolution*CompileSolution` | avoid first-call JIT stall |
| **dilation** | must be **1** for depthwise/grouped | a dilated depthwise conv has no MIOpen group solver |

## Why there's no MFMA tuning here
Each output element is a small spatial dot over a single channel — there is no K-reduction to feed a
matrix-core K-loop. The fast solvers are **direct** / Winograd (3×3 stride-1) / CK-inline direct, all
bandwidth-bound. So `matrix_instr_nonkdim`, `num_stages`, split-K are **not** the levers — coalesced
NHWC loads and solver choice are. Winograd is the sweet spot for the common 3×3 stride-1 depthwise.

## If you author (Triton/HIP)
The tunables are tile/occupancy only: channel-tile `BLOCK_C` (coalesced, e.g. 64–256), spatial tile
`BLOCK_H×BLOCK_W`, `num_warps` (wave64 → 2–4), grid ≥1024 workgroups to fill 304 CUs. Hold the small
filter in registers; stage the input halo in LDS if the spatial tile reuses it. Bake per shape; a wide
autotune is wasted on a memory-bound op. See [backends/triton.md](backends/triton.md),
[backends/hip.md](backends/hip.md).

## How to verify a tune helped
```bash
MIOPEN_ENABLE_LOGGING_CMD=1 python infer.py     # prints the chosen solver + a MIOpenDriver replay line
MIOpenDriver conv --group-count <C> -n N -c C -H H -W W -y R -x S -p pad -u stride --layout NHWC -t 1 -V 1
```
Compare NHWC vs NCHW (confirm the channels-last win) and HYBRID vs a NORMAL-find run on your shapes; if
they diverge, seed FindDb with one NORMAL find for production. Memory-bound → judge against achieved
HBM BW, not FLOPs.

## Sources
- FIND_MODE values, FindDb, AI fallback, immediate mode: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- Grouped/depthwise = set group count; dilation=1 constraint; group-count before Find: https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
- NHWC fast path, CK-inline solvers, channels-last on ROCm 7: https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- ≥1024 grid / memory-bound tail: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
