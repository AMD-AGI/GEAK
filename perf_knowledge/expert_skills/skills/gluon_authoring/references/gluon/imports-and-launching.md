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

For `@gluon.jit` kernels, compiler hints can be passed as launch keyword arguments
just like Triton JIT kernels:

```python
kernel[grid](args..., waves_per_eu=2)
```

Do this before enabling full autotune when the source has commented-out or
disabled hint configs.

### `waves_per_eu` is a per-launch decision, and `0` is not a value

Two things about that one line are worth spelling out, because the common reading of both is wrong.

**`0` does not request zero waves, and it is not "the compiler's choice" in the sense a sweep
means.** The AMD backend guards the attribute emission on it: `if options.waves_per_eu != 0` before
`add_fn_attr("amdgpu-waves-per-eu", "N,N")` (checked on `v3.8.0`; the default in the options class
is `0`). So `0` means **the attribute is never attached** and LLVM's own heuristic runs unbounded,
while any non-zero `N` attaches a *pair* `N,N` — a lower bound equal to the upper bound, i.e. a
pin, not a hint. Switching a launch from `2` to `0` is therefore not a small change in the same
dial; it removes the constraint entirely.

**It is a per-launch value, not a per-kernel one.** The same `@gluon.jit` function, in one file,
is routinely launched with different values chosen by a host-side expression over the shape — the
clean form is a small host helper (`_compiler_options(m)` or equivalent) returning the launch
kwargs, rather than the value being inlined at each call site. The switch is not always on `M`:
production examples also key it on `N`, on the batch structure (ragged versus paired), and on the
role the launch is playing inside one host function, with **non-monotone** value sequences across
the bins. Do not infer a monotone relation and then sweep as if it were one.

Two boundaries:

- **This is not the same statement as an autotune sweep that includes `0`.** A sweep over
  `waves_per_eu` including `0` (`../workloads/attention.md`, its per-dispatch sweep section)
  asks which single value to freeze. This is about a value that stays *different per bin* in the
  shipped source. Neither is evidence for the other.
- **A per-shape branch can be dead.** One surveyed dispatcher reads `4 if rows == 1 else 0` while
  its registered signatures never include `rows == 1`, so the branch is source that no call
  reaches. Intersect the branch's constant set with the registered signature set before quoting it
  as an instance of anything — `../tile-programming/pipeline.md ### The four layers, ordered by what production reaches for`.

**Do not copy a champion's `waves_per_eu` across with the rest of its config.** It is not a hint:
it reaches LLVM as `amdgpu-waves-per-eu` and **caps** occupancy outright. Measured, a transcribed
anchor left without it ran at 3 waves/SIMD and was the faster arm; adding the champion's
`waves_per_eu=2` left the VGPR count unchanged, dropped it to 2 waves/SIMD, and gave a slower clock.
The throttle was tuned for plain's register-heavy *pipelined* body, which is not what a hand-built
Gluon anchor is. Carry the tile shape over; leave this one off until it earns its way back in.

`num_stages` is accepted here too, and on the Gluon path it does nothing — dead in 3.8.0 and
on every older minor — it is recorded into the IR and never read
(`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`). Passing it
costs nothing but buys nothing; it must not appear in an autotune config space,
where it silently multiplies the sweep by a dead axis. Keep it only as a budget parameter and a
champion-record field.

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
table in `../method/benchmark-hygiene.md` is the acceptance gate).

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
`../method/triage.md`.

## Pointers as data (gl.pointer_type)

Kernel arguments arrive as pointers from the launcher. When the *addresses themselves* live in
device memory — a table of peer base pointers, a paged index chain, a per-expert weight table —
the body has to turn loaded integers into pointers, and `gl.pointer_type` is the dtype that lets
it. Re-exported into `gl.__all__` on all four versions (checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0).

```python
addrs = gl.load(table_ptr + idx)                     # int64 tensor: raw addresses
ptrs  = addrs.to(gl.pointer_type(gl.uint32))         # now dereferenceable
vals  = gl.load(ptrs + off)                          # ordinary load through them
```

`gl.pointer_type(element_ty, address_space=1, const=False)` builds a **dtype**, not a value;
`address_space=1` is global and is what you want unless you have a specific reason otherwise.
`const=True` on a read-only table is worth setting where it is true.

Three rules follow from how the cast is defined, and the first two are easy to get backwards:

- **The element type and the address width are unrelated.** `gl.pointer_type(gl.uint32)` describes
  a 64-bit address pointing at `uint32` data. The table you load from must hold 64-bit integers
  regardless of the element type you attach.
- **Integer -> pointer accepts any integer width**, so loading the table as `int32` truncates every
  address with no error at all. The reverse direction is stricter: pointer -> integer is defined
  only at 64 bits (plus a 1-bit form that is a null test, not an address).
- **The cast is a bit-level reinterpretation**, not a lookup. Nothing checks that the address is
  mapped in this process, so a stale, wrong-device, or wrongly-strided table entry faults at the
  *dereference*, several lines away from the cast that caused it.

The pointer tensor inherits the layout of the integer tensor it came from, so coalescing on the
subsequent `gl.load` is decided by the layout of the table load, not by anything you can express at
the cast (`layout-reference.md`).

Where the addresses belong to *another device*, the cast is only the first half of the problem —
the ordering discipline that makes a peer's data safe to read is in
`../workloads/collective.md ## Peer pointers arrive as integers`.

## Rewrite Table

| Plain Triton pattern | First Gluon rewrite | Notes |
| --- | --- | --- |
| `tl.arange(0, X)` | `gl.arange(0, X, layout=layout)` | Generated performance paths should pass an explicit layout. |
| `tl.load` / `tl.store` | `gl.load` / `gl.store` | Move to AMD buffer ops only with evidence (`memory-reference.md`). |
| `tl.zeros` / `tl.full` | `gl.zeros(..., layout=layout)` / `gl.full(..., layout=layout)` | Accumulators and fallbacks need explicit layout. |
| Device scalar math | `gl.cdiv`, `gl.minimum`, `gl.maximum`, `gl.exp`, `gl.where` | Host launch math can stay in Python/Triton. |
| `tl.max` / `tl.sum` | `gl.max` / `gl.sum` with matching layout assumptions | Audit reduction identity, axis, and dtype. |
| Shape APIs | Gluon shape APIs on layout-aware tensors | Preserve transformation semantics. |
| `tl.dot` / `tl.dot_scaled` | target-specific MFMA lowering or keep plain Triton | No generic `gl.dot`; check `../hardware/capability-matrix.md`. For tiles that cannot fill a matrix instruction, `matrix-reference.md ## When the matrix core is the wrong instrument (gl.dot_fma)`. |
