---
title: Comparing TileLang and Triton
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16]
regimes: [both]
status: competitive
updated: 2026-10-16
---

# Comparing TileLang and Triton

TileLang and Triton expose different controls over tile operations, layouts and
lowering. Compare the implementations available for the required shape, dtype,
device and numerical contract. Assembly or a library is another candidate, not a
universal ceiling imposed by language choice.

The AMD TileLang attention example reports 0.36 ms against 0.55 ms for Triton and
0.97 ms for PyTorch at `B=1,H=8,S=4096,D=128`, with ROCm 7.0.1 / TileLang 0.1.7.
These are source-reported results for that benchmark; they do not establish current
performance or explain which compiler mechanism caused the difference.

Use matching full-operation correctness and unprofiled timing, including helper
kernels, workspace and required state updates. Inspect generated layouts or
pipelines only when needed to test an explanation. Preserve useful slower variants
when a different schedule or decomposition remains worth developing.

## Sources

- [AMD TileLang attention example](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html)
- [TileLang](https://github.com/tile-ai/tilelang)
- [AMD Triton compiler](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)
