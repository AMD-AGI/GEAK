---
title: TileLang AMD pitfalls
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16]
regimes: [both]
status: competitive
updated: 2026-10-16
---

# TileLang on AMD: checks that matter

- Check the installed compiler/backend and target support. Historical CDNA3
  measurements neither qualify nor rule out a current gfx950 implementation.
- Match emitted tile shape, thread count and stages to the requested configuration.
  Budget registers and shared memory for the actual target; do not assume tuning
  automatically rejects every invalid configuration.
- Keep tuning records tied to shape, dtype and source/toolchain. Revalidate after
  relevant changes rather than treating a published winning configuration as portable.
- Inspect generated code when the performance hypothesis depends on an intrinsic,
  layout or pipeline. API presence does not establish support on the selected path.
- Run the task's numerical/state checks and unprofiled operation timing. Greedy
  token agreement alone is not kernel-level correctness.

## Sources

- [TileLang repository](https://github.com/tile-ai/tilelang)
- [AMD attention example](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html)
- [Autotuning](autotune.md), [primitives](primitives.md)
