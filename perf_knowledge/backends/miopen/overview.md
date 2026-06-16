---
title: MIOpen — AMD's deep-learning primitive library (overview)
kind: backend
backend: miopen
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
---

# MIOpen — the cuDNN analog on ROCm

## TL;DR
MIOpen is AMD's deep-learning primitive library (the **cuDNN analog**): convolutions, pooling,
batchnorm/layernorm, RNN, activation, softmax, with a benchmark-driven solver-selection database. For
**transformer LLM inference it is largely idle** — no convolutions, and norms/softmax come from
AITER/Triton fused kernels, not MIOpen. It matters for **vision encoders** (ViT/CLIP towers in VLMs),
Whisper/conv front-ends, diffusion, and CNN training. PyTorch routes `conv2d`/`batch_norm` here. The two
things to know: how MIOpen **picks a solver** (find vs immediate mode + FindDb) and how to **fuse**
conv+bias+activation. Conv specifics in [conv.md](conv.md).

## Concepts
- **Solvers**: for each conv stage MIOpen has many algorithms (GEMM/implicit-GEMM, direct, Winograd, FFT,
  and inline **Composable Kernel** solvers). Selection is data-driven, recorded in **FindDb**.
- **Find APIs** (`miopenFindConvolution*`): benchmark applicable solvers at runtime — **expensive** in time
  and workspace; call **once per process lifetime** and cache the chosen algo+workspace.
- **Immediate mode** (v2.0+): query supported solutions without a find run; on a FindDb miss it falls back
  to an **AI heuristic** (`MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=ON`, default, ~90% accuracy predicting the
  optimal solver) or a throughput-index estimate (older builds fell back to a generic GEMM).
- **FindDb**: persistent DB of find results; system DB ships with the install, user DB grows at runtime
  (`MIOPEN_USER_DB_PATH`).
- **Fusion API**: merge conv + bias + activation (and batchnorm) into one kernel; modern path is
  `miopenExecuteFusionPlan_v2`, **NHWC/NDHWC only**.

## The levers
| lever | knob | note |
|---|---|---|
| solver-selection speed/quality | `MIOPEN_FIND_MODE` = `NORMAL`/1, `FAST`/2, `HYBRID`/3 (**default**), `DYNAMIC_HYBRID`/5 | NORMAL benchmarks everything (slow start, best perf); FAST trusts FindDb+immediate fallback (fast start, may lose perf); HYBRID = FindDb hit else full find |
| immediate-mode fallback | `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` (default ON) | AI heuristic on DB miss; OFF → throughput-index |
| user DB path | `MIOPEN_USER_DB_PATH` | where runtime-tuned results land |
| logging | `MIOPEN_ENABLE_LOGGING=1`, `MIOPEN_ENABLE_LOGGING_CMD=1` | see which solver ran; CMD form is reproducible via `MIOpenDriver` |
| layout | NHWC/NDHWC for the fast fused path | channels-last is the perf layout on ROCm 7 |
| precompile | `miopenConvolution*CompileSolution` | populate kernel cache to avoid first-call JIT stall |

## Where it sits in the stack
PyTorch `conv2d`/`batch_norm`/`pooling`/RNN → MIOpen. Dense `matmul`/`Linear` does **not** go here (that's
hipBLASLt/rocBLAS). Triton is **not** used if MIOpen/rocBLAS is faster — PyTorch picks the library backend.
PyTorch Inductor *can* lower conv to Triton under `max-autotune`, but only when it beats MIOpen
([pytorch_inductor/max_autotune.md](../pytorch_inductor/max_autotune.md)).

## Pitfalls
- A cold `miopenFind*` per inference step destroys throughput — reserve algo+workspace once and reuse.
- First `*Immediate` call JIT-compiles the kernel (latency spike) → precompile chosen solutions.
- `miopenExecuteFusionPlan_v2` is **NHWC-only**; an NCHW input means no fused path (or a layout transform).
- A FindDb miss with the AI fallback disabled can pick a generic GEMM solver = slow; run NORMAL find once to
  seed FindDb for production shapes.
- Repo moved: MIOpen for ROCm 6.4.3 and earlier is `ROCm/MIOpen`; current public source is in
  **`ROCm/rocm-libraries`** — pin accordingly.

## Verify
- `MIOPEN_ENABLE_LOGGING_CMD=1` → confirms the chosen solver and lets you replay with `MIOpenDriver`.
- Compare immediate-mode vs a NORMAL-find run on your shapes; if they diverge, seed FindDb with NORMAL find.

## Sources
- Find APIs & immediate mode (FindDb, AI fallback, FIND_MODE): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- Fusion API (`_v2`, NHWC-only, operator order): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
- MIOpen CHANGELOG (HYBRID default, CK-inline, grouped-conv fusion, NHWC): https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- ROCm ecosystem mapping (MIOpen mostly idle at LLM inference): perf_knowledge ../../index/README.md
