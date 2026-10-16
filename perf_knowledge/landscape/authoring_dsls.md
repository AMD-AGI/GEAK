---
title: Kernel authoring options
kind: landscape
updated: 2026-10-16
scope: AMD kernel authoring options and toolchain references
status: survey
---

# Kernel authoring options

Choose a language that can express the required dataflow and integrate through the
task's public callable. There is no required escalation ladder: a library, Python
DSL, HIP or assembly can be appropriate. Toolchain availability and evaluator
support must be checked on the actual target.

| Option | Useful capability | Local detail |
|---|---|---|
| Triton | Tile operations, reductions, matmul and compiler-managed layouts | [AMD backend](../languages/triton_amd/overview.md); inspect the installed `HIPOptions` |
| Gluon | Explicit thread layouts and lower-level scheduling in the Triton toolchain | [Programming model](../languages/gluon/programming_model.md); APIs are experimental |
| FlyDSL | Explicit layouts, control flow and AMD intrinsics | [Authoring](../languages/flydsl/authoring_tile_programming.md); distinguish authored kernels from aiter wrappers |
| HIP/C++ | Runtime integration, custom device code and compiler intrinsics | [Toolchain](../languages/hip_cpp/overview.md) |
| CK / CK-Tile | C++ template kernels, tile distributions and fused pipelines | [CK-Tile](../languages/composable_kernel/ck_tile.md); check instance support before invoking |
| TileLang | Tile primitives with generated backend code and autotuning | [AMD guide](../languages/tilelang/overview.md); target/version support varies |
| HipKittens | Tile primitives and explicit AMD schedules | [Guide](../languages/hipkittens/overview.md); pin the source and target branch |
| rocWMMA | Matrix fragments inside HIP kernels | [Guide](../languages/rocwmma/overview.md); fragment ownership and supported types matter |
| Mojo | A separate compiler/runtime with GPU programming support | [AMD status](../languages/mojo/status_amd.md); qualify dependencies and integration |

CUTLASS/CuTe examples can inform layouts but their CUDA instructions are not a
portable AMD implementation; see [porting limits](../languages/cutlass_port/status_amd.md).
For compiler-level integration, [IREE's HIP target](https://iree.dev/guides/deployment-configurations/gpu-rocm/)
and [rocMLIR](https://github.com/ROCm/rocMLIR) are separate options.

Published ping-pong/interleaved schedules are useful experiments, not universal
rules against producer/consumer specialization. Verify the generated work,
resources and full-operation latency for the chosen source and shape.

## Sources

- [Triton AMD compiler](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)
- [Gluon](https://triton-lang.org/main/gluon/index.html)
- [TileLang](https://github.com/tile-ai/tilelang)
- [CK-Tile](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/index.html)
- [HipKittens](https://github.com/HazyResearch/HipKittens)
- [Mojo GPU fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals/)
