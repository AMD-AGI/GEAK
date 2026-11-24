# Gluon Imports And Launching

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

`from triton import gluon` raises `ImportError` on 3.8.0 even where Gluon is installed, so it is not an
availability probe. Layouts: `layout-reference.md`; matrix
lowering: `matrix-reference.md`; target/dtype support: `../hardware/capability-matrix.md`.

## Launcher Contract

Gluon keeps Triton's launcher model: `@gluon.jit`, `kernel[grid](...)`, `gl.program_id`,
`gl.constexpr`, launch attributes `num_warps` / `num_ctas`, and compiler hints as launch kwargs
(`kernel[grid](..., waves_per_eu=2)`).

- Build shape/target/`num_warps`-dependent layouts (`BlockedLayout`, `SliceLayout`,
  `DotOperandLayout`, `AMDMFMALayout`) on the host and pass them as `gl.constexpr`. A
  `layout: gl.constexpr = gl.BlockedLayout(...)` inside the JIT body is fine where source already
  does it.
- Define `@gluon.jit` in a `.py` file — `python -c`, stdin and `exec` break source inspection.
- For sub-ms operators, cache host-built layouts by shape key (`../method/benchmark-hygiene.md`).
- **`num_stages` is accepted and inert on the Gluon path** (3.6.0–3.8.0; no pass reads it, 3.8.0
  source-checked) — sweeping it in an autotune space times the same kernel repeatedly
  (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

### `waves_per_eu` is a per-launch decision, and `0` is not a value

**On v3.8.0 the AMD backend always attaches `"amdgpu-waves-per-eu"="N, N"` (unconditional in `make_llir`): a non-zero `N` is a
pin (min = max = N), and the default `0` emits `"0, 0"`, which LLVM reads as no limit.** So going
`2 -> 0` removes the constraint entirely rather than nudging it. GPU-measured on gfx950 / 3.8.0: the pin also
**caps** a light kernel (a 40-VGPR kernel went from 8 to 2 resident workgroups per CU at `waves_per_eu=2`, LLVM
padding `.amdhsa_next_free_vgpr` to the 2-wave rung), and an `llvm_fn_attrs` `amdgpu-waves-per-eu` pair overrides
it.

- It is per launch: production picks it with a host helper over shape (M, N, batch structure, role),
  often non-monotone across bins, so a monotone relation is not implied; a per-shape branch quoted from
  production is evidence only if the registered signatures reach it
  (`../tile-programming/pipeline.md ### The four layers, and how often surveyed kernels use them`).
- **A champion's `waves_per_eu` is tuned to its own body** — copied with the rest of its config onto a
  different body it can cap occupancy (one measured case lost a wave/SIMD and ran slower;
  `../../playbook.md ## Knobs & pitfalls`); re-sweep it per body.

## `tl.*` Boundary Inside `@gluon.jit`

Keep `tl.constexpr`, scalar launch math and control flow; replace tensor dataflow (`tl.arange`,
`tl.load/store`, `tl.zeros/full`, `tl.dot/dot_scaled`) with `gl.*` + explicit layouts.

**`tl.dot` is not a partial-migration bridge: its result carries no Gluon distributed layout**, so
later Gluon broadcasts / elementwise ops / stores fail with distributed-type errors (3.8.0
compile-checked: "expected broadcast right input to be a distributed_type"). A
`gl.load -> tl.dot -> gl.store` candidate does not compile; the working forms are plain Triton or the dot
lowered fully through MFMA (`matrix-reference.md`). For tiles that cannot fill an MFMA:
`matrix-reference.md ## When the matrix core is the wrong instrument (gl.dot_fma)`.

`gl` names recalled from other APIs are often absent; non-existent spellings are listed in
`appendix-api.md ## Spellings that do not exist on the CDNA Gluon path`.

## Pointers as data (gl.pointer_type)

When addresses live in device memory (peer base pointers, paged index chains, per-expert weight
tables), cast loaded integers to a pointer dtype. In `gl.__all__` on 3.6.0-3.8.0.

```python
addrs = gl.load(table_ptr + idx)                     # int64 tensor: raw addresses
ptrs  = addrs.to(gl.pointer_type(gl.uint32))         # now dereferenceable
vals  = gl.load(ptrs + off)
```

`gl.pointer_type(element_ty, address_space=1, const=False)` builds a **dtype** (1 = global).

- **Element type and address width are unrelated**: `pointer_type(gl.uint32)` is a 64-bit address
  to `uint32` data; the table must hold 64-bit integers.
- **Integer -> pointer requires a 64-bit integer** (Triton 3.8.0, compile-verified): an `int32` / `int16` /
  `int8` tensor fails at compile time (`'tt.int_to_ptr' op operand #0 must be 64-bit signless integer`). The
  silent failure is loading a 64-bit table as `int32` and widening with `.to(gl.int64)` to make the cast
  compile, which sign-extends truncated addresses; load the table as `int64` / `uint64`. Pointer -> integer
  works only to `int64` / `uint64` (plus `int1` for a null test).
- The cast is a bit reinterpretation: a bad entry faults at the dereference, lines away.
- The pointer tensor inherits the integer tensor's layout, so coalescing is decided by the table
  load's layout.

Peer-device ordering: `../workloads/collective.md ## Peer pointers arrive as integers`.
