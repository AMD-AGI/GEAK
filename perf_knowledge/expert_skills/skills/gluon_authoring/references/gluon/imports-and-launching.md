# Gluon Imports And Launching

Use this file when a task needs Gluon imports, `@gluon.jit`, launcher wiring, or
host-created layouts. For layout authoring continue with `layout-reference.md`;
for matrix lowering continue with `matrix-reference.md`. Target/dtype/API support
lives in `../hardware/capability-matrix.md`.

## Supported Imports

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

Do not probe availability with `from triton import gluon`.

## Launcher Contract

Gluon keeps Triton's host launcher model:

- `@gluon.jit`;
- `kernel[grid](...)`;
- `gl.program_id`;
- `gl.constexpr`;
- launch attributes such as `num_warps` and `num_ctas`.

Build host-dependent layouts outside generated `@gluon.jit` code and pass them as
`gl.constexpr`. This includes `BlockedLayout`, `SliceLayout`, `DotOperandLayout`,
and `AMDMFMALayout` when they depend on shape, target, or launch configuration.

For `@gluon.jit` kernels, compiler hints such as `waves_per_eu` and `num_stages`
can be passed as launch keyword arguments just like Triton JIT kernels:

```python
kernel[grid](args..., waves_per_eu=2, num_stages=1)
```

Do this before enabling full autotune when the source has commented-out or
disabled hint configs.

## JIT Entry And Host Feeding

Gluon evidence requires a launched helper whose output feeds correctness and
timing.

Rules:

- keep grid, launch attributes, target guards, fallback selection, and layout
  factories outside generated `@gluon.jit` code;
- build derived layouts on the host and pass them as `gl.constexpr`;
- do not guess helper names or import paths;
- preserve public wrapper ABI and output feeding path;
- compile success is not integration evidence until the helper is launched by the
  measured wrapper.

Source-proven exception: trusted production code may contain
`layout: gl.constexpr = gl.BlockedLayout(...)` inside the JIT body. Preserve that
pattern when it already exists; generated patches should prefer host factories.

The `@gluon.jit` function must be defined in a Python source file. Avoid
interactive definitions, `python -c`, stdin, or `exec` for probes; Triton source
inspection can fail before the real Gluon path is exercised.

For sub-ms operators, host layout construction can dominate the measured
boundary. Precompute fixed layouts at module scope, cache layouts by shape key,
and measure wrapper-only overhead when layouts are built per call (the sub-ms
table in `../phases/harness.md` is the acceptance gate).

## `tl.*` Boundary Inside `@gluon.jit`

| Usually keep when source-proven | Replace in generated Gluon dataflow |
| --- | --- |
| `tl.constexpr`, scalar launch math, scalar/control flow, source-proven elementwise numerics | `tl.arange`, tensor `tl.load/store`, `tl.zeros/full`, `tl.dot/dot_scaled` |

Audit remaining `tl.*` by dataflow role. The issue is not spelling; it is whether
tensor layout, memory, matrix, or reduction state is still outside the Gluon plan.

`tl.dot` is not a safe partial-migration bridge. Its result does not carry a
Gluon distributed layout, so later Gluon broadcasts, elementwise ops, or stores
can fail with distributed-type errors. Keep the kernel plain Triton, or lower the
dot fully through a target-specific MFMA plan (`matrix-reference.md`).

## Core API Surface

Common language and shape APIs:

- `program_id`, `num_programs`, `num_warps`, `num_ctas`, `constexpr`;
- `arange`, `zeros`, `zeros_like`, `full`, `full_like`, `cast`, `to_tensor`;
- `broadcast`, `expand_dims`, `reshape`, `permute`, `split`, `join`, `ravel`,
  `map_elementwise`;
- `load`, `store`, `gather`, `where`;
- scalar/math APIs such as `cdiv`, `minimum`, `maximum`, `exp`, `exp2`, `floor`,
  `ceil`, `sqrt`, `rsqrt`, and `abs`;
- reductions such as `sum`, `max`, `min`, `reduce`, `reduce_or`, `xor_sum`;
- specialized APIs such as `associative_scan`, `histogram`, and generic atomics.

Common layout and memory objects (CDNA / gfx950-gfx942):

- `BlockedLayout`, `SliceLayout`, `DotOperandLayout`;
- `DistributedLinearLayout`;
- `SwizzledSharedLayout`, `PaddedSharedLayout`;
- `AMDMFMALayout` (`version=4` gfx950 / `version=3` gfx942);
- `allocate_shared_memory`, `barrier`, `to_linear_layout`, `set_auto_layout`.

Treat scans, histograms, atomics, auto-layout, and async paths as source-first
features. Do not invent names from memory; on a missing symbol use
`../missing-doc-protocol.md`.

## Rewrite Table

| Plain Triton pattern | First Gluon rewrite | Notes |
| --- | --- | --- |
| `tl.arange(0, X)` | `gl.arange(0, X, layout=layout)` | Generated performance paths should pass an explicit layout. |
| `tl.load` / `tl.store` | `gl.load` / `gl.store` | Move to AMD buffer ops only with evidence (`memory-reference.md`). |
| `tl.zeros` / `tl.full` | `gl.zeros(..., layout=layout)` / `gl.full(..., layout=layout)` | Accumulators and fallbacks need explicit layout. |
| Device scalar math | `gl.cdiv`, `gl.minimum`, `gl.maximum`, `gl.exp`, `gl.where` | Host launch math can stay in Python/Triton. |
| `tl.max` / `tl.sum` | `gl.max` / `gl.sum` with matching layout assumptions | Audit reduction identity, axis, and dtype. |
| Shape APIs | Gluon shape APIs on layout-aware tensors | Preserve transformation semantics. |
| `tl.dot` / `tl.dot_scaled` | target-specific MFMA lowering or keep plain Triton | No generic `gl.dot`; check `../hardware/capability-matrix.md`. |
