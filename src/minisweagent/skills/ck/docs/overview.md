# CK / ck_tile Overview

## What CK is

Composable-Kernel (CK) is AMD's **C++ template + code-generation** library for
high-performance GPU operators on CDNA (MI300X / gfx942) and RDNA. There are two
generations in the upstream repo:

- **legacy CK** (`include/ck/`, `tensor_operation/`) — device-op templates.
- **ck_tile** (`include/ck_tile/`, `example/ck_tile/`) — the current
  tile-programming model used for new kernels (GEMM, FMHA, MoE, norm, quant).

The defining property: you do not hand-write a kernel, you **instantiate a
template** with a tile configuration. Performance comes from picking the right
instance, not from clever per-line C++. Because instantiation is expensive at
compile time, CK uses **codegen scripts** (`generate.py` / `gen_instances.py`)
to emit one `.cu`/`.cpp` per instance so they build in parallel.

## Why this matters for a rewrite agent

On an AMD box the practical CK surface is **aiter**, which ships pre-codegen'd,
pre-tuned CK instances behind Python ops and selects the best instance per shape
from a tuned CSV at call time. So the win path is:

0. wire the shipped aiter CK op (see `shipped_aiter_ck_ops.md`),
1. tune the CK instance for your shape via the codegen tuners
   (`instance_tuning.md`),
2. only then author a ck_tile program from an example (`ck_tile_authoring.md`).

You almost never write raw CK template metaprogramming from scratch.

## The ck_tile programming model

A ck_tile kernel is composed from a small set of orthogonal pieces. Authoring or
adapting an example means filling these in (see `example/ck_tile/03_gemm/`,
`01_fmha/`, `15_fused_moe/`):

- **Problem** — the math + types: A/B/C dtypes, accumulator dtype, layouts
  (row/col-major), element-wise ops. (e.g. `GemmProblem`, `FmhaPipelineProblem`).
- **Traits / Config** — the tile shape and warp layout: BlockTile dims
  (`MPerBlock`, `NPerBlock`, `KPerBlock`), warp tile dims, warps layout
  (`MWaves`, `NWaves`), block size. This is the main perf knob surface.
- **Pipeline** — the memory→compute schedule. ck_tile ships swappable pipelines
  (basic / memory-bound / compute `COMPUTE_V3` / weight-preshuffle). Swapping the
  pipeline type is how you trade off for a shape regime without rewriting logic.
- **Block GEMM / Warp GEMM** — how warps iterate and map to MFMA instructions.
- **Epilogue** — moves the accumulator from registers to global memory; the place
  for post-fusion (activation, type cast, quant, topk-weight multiply).
- **Tile distribution** — the encoding that maps threads to tile elements
  (`include/ck_tile/tile_program/tile_distribution/`, demoed in example
  `51_tile_distr_enc_reg_map`).

A ck_tile program then has two layers:
- a **host launcher** (the `.cpp` / `.cu`): parses args, picks the instance,
  computes grid/block, launches.
- the **device template** (the `.hpp`): the kernel body parameterized by the
  pieces above. Keeping these separated is what enables parallel codegen builds.

## When CK wins

CK is the right tool for **compute-heavy, regular-shaped** operators where an MFMA
pipeline dominates:

- **dense GEMM** (bf16/fp16/fp8/int8/int4-B), batched and grouped GEMM.
- **MoE / fused-MoE** — group-GEMM over experts with sorting + activation +
  topk-weight fused (the 2-stage path is the big aiter win).
- **block-scale / preshuffled fp8 GEMM** — quantized inference GEMM.
- **FMHA** (attention) and **norm/quant** fused ops (rmsnorm, smoothquant).

CK is **not** the lever for memory-bound elementwise/reduction kernels where the
win is fusion or layout — those rarely beat a tuned ck_tile only because of CK.
For those, prefer fusing into an existing op or a lighter DSL.
