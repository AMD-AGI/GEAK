---
title: "FlyDSL — legacy API migration (raw MLIR dialects → fx.*)"
kind: language
gens: [gfx942, gfx950, gfx1250]
updated: 2026-09-21
source_commit: ROCm/aiter@b0ced008
sources:
  - ROCm/aiter@b0ced008:.claude/skills/flydsl-kernel-code-cleanup/SKILL.md
  - ROCm/aiter@b0ced008:aiter/ops/flydsl/kernels/
  - ROCm/aiter@b0ced008:requirements.txt
  - ROCm/aiter@04c7b808:.claude/skills/review-pr/rules.md
---

> **Reference (how-to), not a verdict.** Ingested from the aiter FlyDSL cleanup skill. It maps *legacy*
> kernel constructs onto the `fx.*` surface of a **pinned** FlyDSL version — it is a refactoring
> reference, not an optimization lever. Nothing here is expected to make a kernel faster; several
> entries can make one slower if applied to a scheduled hot loop (see §3c, §7b). Write new kernels
> from [`authoring_tile_programming.md`](authoring_tile_programming.md); optimize via
> [`authoring_optimization.md`](authoring_optimization.md); triage the result with
> [`debugging.md`](debugging.md).

# FlyDSL legacy API migration

## Overview

FlyDSL kernels written against earlier versions of the DSL reach into raw upstream MLIR dialects
(`arith`, `scf`, `vector`, `llvm`, `memref`, `math`, `rocdl`) and into now-superseded FlyDSL helpers
(`ArithValue`, `buffer_ops`, `SmemAllocator`/`SmemPtr`, `*_atom_call`). The current surface expresses
almost all of it through typed `fx.*` values and Python operators.

**The rule of thumb:** inside a `@flyc.kernel` / `@flyc.jit` body, prefer `fx.*` and Python operators.
Drop to a raw dialect only at a hard boundary where no wrapper exists, and keep that boundary local.

**Scope:** behavior-preserving refactoring of existing kernels. This is *not* a performance workflow
and not a correctness workflow. A migration that changes numerics, ISA, or schedule has failed, even
when it compiles and passes.

---

## Version boundary (read before applying anything)

Every mapping below is stated against **`flydsl==0.3.2`**, the version aiter pins in
`requirements.txt` at the cited commit. This matters concretely:

- The `scf` loop contract in §3 (which `range` form converts bounds, which accepts `init=`, which
  integer widths are rejected) is **version-specific behavior**, not a stable API guarantee.
- A sibling FlyDSL checkout on the machine can expose APIs newer than the pinned version supports.
  Resolve the imported version and module path before assuming a replacement exists.
- Migration never requires bumping the dependency or editing FlyDSL itself. If a mapping needs a
  newer FlyDSL, the kernel stays on the legacy construct.
- Honor the task's explicit version, architecture and single-/multi-GPU scope; a cleanup does not
  widen any of them.

Other docs in this base pin older FlyDSL versions for their own claims. Check the pin on any doc
before combining its guidance with this one.

---

## Where the code lives (aiter tree)

| Location | Role |
|---|---|
| `aiter/ops/flydsl/kernels/` | `@flyc.kernel` device kernels and shared helpers (`tensor_shim.py`, `kernels_common.py`, `buffer_ops.py`, `act.py`, …) |
| `aiter/ops/flydsl/*.py` | Launch wrappers, compile helpers, public op entry points |
| `op_tests/test_flydsl_*.py` | Top-level FlyDSL correctness / perf tests |
| `op_tests/flydsl_tests/` | Additional FlyDSL kernel tests |

Shared helpers already have owners; reuse them rather than adding a parallel copy:

- `kernels/tensor_shim.py` — compilation, cached dispatch, failure recovery, pointer/base/dtype
  extraction (`_run_compiled`, `ptr_arg`).
- `kernels/kernels_common.py` — `LOG2E`, host/wrapping-integer `ceildiv`.
- `kernels/act.py` — activations.
- Family `*_common.py` modules — reductions and specialized memory operations.

`moe_kernels._run_compiled(exe, args)` is a **tuple-argument** adapter for existing MoE/AOT callers,
distinct from `tensor_shim._run_compiled(exe, *args)`. Consolidating launch code has to preserve that
contract.

Prefer typed `fx.min` / `fx.max` / `fx.ceildiv` where their signedness, NaN and overflow semantics
match the code being replaced.

---

## Cautions

- **Surgical and behavior-preserving.** Minimal diffs, matching local style. A migration is a refactor.
- **Pervasively legacy kernels are not migration targets.** Some kernels use `_scf.IfOp` / `_raw`
  throughout (the fmha gfx950 family, tuned MLA decode kernels). Converting one construct inside them
  in passing is fine; a mass rewrite is a separate, measured piece of work.
- **Verify, do not assume.** Offset, type and SSA changes can shift results. Compare before/after
  numerics *and* generated code with `FLYDSL_RUNTIME_ENABLE_CACHE=0`, using a fresh dump directory per
  specialization so one shape cannot overwrite another.
- **Raw boundaries carry semantics.** Exact scope/ordering, volatile and alias metadata, raw SSA
  contracts, and integer widths unsupported by the pinned API are all reasons a raw dialect call stays.
  Record the specific reason. Hiding a dialect behind a new facade is not a migration.

---

## 1. `ArithValue` and index helpers (deprecated in `expr/arith.py`)

| Deprecated | Replacement |
|---|---|
| `ArithValue(x)` (wrap for operators) | `fx.Int32/Int64/Float32/Vector` — already overload `+ - * / % << >> == < >` |
| `arith.unwrap(v)` / `arith._to_raw(v)` | `v.ir_value()`, only where a raw `ir.Value` is needed |
| index-typed arithmetic counters | `fx.Int64(...)` or `fx.Int32(...)` when the consumer permits a fixed-width integer |
| `arith.index_cast(T.index, v)` at an index-typed boundary | `fx.Index(v)` |

`fx.Index` maps to MLIR `index`. Prefer explicit-width `fx.Int64` / `fx.Int32` for arithmetic,
choosing width and signedness deliberately. Keep `fx.Index` where a launch, layout, loop or other API
requires the index type — replacing it merely to remove the name changes the IR contract. Widening an
`i32` counter or narrowing an index requires checking the consumer and the supported bounds.

```python
# Before
acc  = ArithValue(val) + peer
lane = ArithValue(tid) % fx.Index(64)
cond = arith.unwrap(idx >= limit)
off  = arith.index_cast(T.index, x)
# After
acc  = val + peer                    # val already fx.Float32 / fx.Vector
lane = tid % fx.Int64(64)
cond = (idx >= limit).ir_value()     # only if a raw scf.IfOp needs it
off  = fx.Index(x)                   # preserve this consumer's index contract
```

When an operand is a raw `ir.Value`, wrap it once at the source (`fx.Float32(v)`) rather than with
`ArithValue` per use. An explicit `arith.*FOp` is still warranted for non-default fastmath.

One raw boundary runs the other way and survives this migration: **`arith.bitcast` requires its
operand already unwrapped**. Passing a DSL value straight into `arith.bitcast(val, ty)` — the result
of an arithmetic op, a load, or a `const_expr` — is a JIT-time type error, and Python raises nothing
statically, so it surfaces only when that dtype branch is first compiled (ROCm/aiter#3944, a bf16/f16
output path). Keep `arith.unwrap(val)` on that call rather than deleting it as part of a sweep.

### 1b. Redundant `fx.*` wraps

Wrapping *introduces* a type (from a Python literal or a raw `ir.Value`) or *changes* one. Re-wrapping
an already-typed value is noise; double-wrapping is dead.

```python
# Before
for i in range_constexpr(fx.Int32(N)):
    off = fx.Int64(fx.Int64(base) + fx.Int64(4))
tile = fx.make_layout(fx.Int32(BLOCK), fx.Int32(1))
idx  = fx.Int32(tx)                  # tx already fx.Int32
# After
for i in range_constexpr(N):
    off = base + fx.Int64(4)
tile = fx.make_layout(BLOCK, 1)      # builders take Python ints
idx  = tx
```

- Compile-time shapes, strides and bounds (`make_layout`, `make_shape`, `range_constexpr`,
  `Constexpr`) take plain Python ints.
- Wrap a runtime value once, at first typed use.
- A real cast (`fx.Int64(i32)` widen, `fx.Int32(index)` narrow) is not redundant — it is what replaces
  `arith.index_cast`.

---

## 2. `buffer_ops` → `make_buffer_tensor` + copy atoms

`create_buffer_resource` plus manual offsets is the legacy form. Building a buffer-resource view with
`fx.rocdl.make_buffer_tensor()` and then using layout ops + `fx.copy` (§7b) constructs the
OOB-checked V# descriptor for you.

```python
# Before (manual offsets)
rsrc = buffer_ops.create_buffer_resource(A, max_size=True)
data = buffer_ops.buffer_load(rsrc, row * K + k, vec_width=4, dtype=fx.Float32)
buffer_ops.buffer_store(data, rsrc, row * N + col)
# After
bufA = fx.rocdl.make_buffer_tensor(A)
tA   = fx.make_view(fx.get_iter(bufA), fx.make_layout((M, K), (K, 1)))
copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
fx.copy(copy, fx.slice(tA, (None, tid)), rA)   # after partitioning tA (§7b)
```

- `make_buffer_tensor(tensor, max_size=True)` mirrors `create_buffer_resource`. Pass
  `num_records_bytes=` for a const byte count, or `max_size=False` to derive from the layout.
- **`max_size=True` is not a safe default on a runtime-ragged dimension.** It declares the buffer as
  large as its *allocation*, so when the bound dimension is a runtime extent — M, a token count,
  `num_valid`, a per-expert count — the hardware `num_records` field permits reads past the live
  data: silent garbage, no fault. Full-size tensors (weights, `sin_cache`/`cos_cache`, `rms_weight`,
  `block_table`) are correct with `max_size=True`; the suspicious sites are those bound to M, token
  or expert counts. Same defect shape in `make_tensor_descriptor_2d` when `oob_outer_bound` is set
  and `oob_inner_bound` is not, and in a 32-bit `voffset` that overflows on a >4 GB weight. Measured
  instance: `preshuffle_gemm.py` bound `arg_scale_a` with `max_size=True` while `scale_a` is per-row
  in a ragged M (ROCm/aiter#4151, fixed in #4546) — the same commit's comment claimed B and the
  scales were exact-multiple in N, which was true for `arg_scale_b` and false for `arg_scale_a`.
  This is distinct from int32 index overflow: nothing overflows in Python, the read leaves the
  allocation on the GPU.
- gfx1250 TDM uses a different atom — `fx.rocdl.make_tdm_atom` (raw VA, not a buffer resource).
- A scalar-base + per-thread-offset load with no layout form legitimately stays on `buffer_ops`.
- `buffer_load` / `buffer_store` `offset` is in **elements** (multiplied by `sizeof(dtype)`
  internally). Reading it as bytes is a recurring source of `//4`-style offset bugs.
- aiter still ships `aiter/ops/flydsl/kernels/buffer_ops.py` for legacy kernels; it is not removed.

---

## 3. Raw upstream dialects → `fx.*` and Python

### `arith`
| Raw | Preferred |
|---|---|
| `arith.constant(42, index=True)` | `fx.Index(42)` when the consumer requires MLIR `index`; otherwise an explicit-width `fx.Int32/Int64(42)` |
| `arith.mulf/addf(a,b)` | `a * b` / `a + b` |
| `arith.trunc_f(ty, v)` / `ext_f` | `v.to(<target fx type>)` |
| `arith.index_cast(T.i32, v)` | `fx.Int32(v)` |
| `arith.select(cond, t, f)` | `cond.select(t, f)` |
| `arith.cmpi(slt, a, b)` | `a < b` |
| `arith.maximumf/minimumf(a,b)` | `fx.max(a, b)` / `fx.min(a, b)` |
| `arith.maxsi/maxui/minsi/minui(a,b)` | `fx.max(a, b)` / `fx.min(a, b)` |
| `arith.maxnumf(a,b)` | `fx.maxnumf(a, b)` — different NaN semantics from `fx.max` |
| `arith.ceildivsi/ceildivui(a,b)` | `fx.ceildiv(a, b)` |

`arith.cmpf` and explicit `*FOp` forms remain warranted where no operator exists or fastmath is needed.

### `scf`
| Raw | Preferred |
|---|---|
| `scf.ForOp` | `range_constexpr(N)` (unrolled) or `range(lo, hi, step, init=[...])` (runtime, loop-carried) |
| `scf.IfOp(_raw(cond))` | Python `if cond:` (runtime) / `if const_expr(flag):` (compile-time) |

The rewriter's loop contract is version-specific. In FlyDSL 0.3.2:

- `range_constexpr` requests Python unrolling.
- `range(..., init=[...])` emits an `scf.for` with explicit carried state and converts its bounds to
  index, including Python integer bounds. It does not discard `init` merely because the bounds are
  static.
- Ordinary `range` without `init` uses automatic carried-state dispatch and expects `i32` bounds;
  index bounds are converted to `i32`, and `i64` is rejected.

The selected loop form, its supported bounds, and its state types are all part of the contract. See §5
for runtime branches inside helper functions.

### `vector`
| Raw | Preferred |
|---|---|
| `vector.extract(v, static_position=[i])` | `fx.Vector(v)[i]` |
| `vector.bitcast(ty, v)` | `fx.Vector(v).bitcast(fx.Float32)` |
| `vector.splat` / const vector | `fx.Vector.filled(width, val, fx.Float32)` |
| build from scalars | `fx.Vector.from_elements(...)` |
| reg-memref load/store | `fx.memref_load_vec(r)` / `fx.memref_store_vec(v, r)` |

### `llvm` / `memref` / `math`
- `llvm.*` pointer math, load/store and constants → layout views (`fx.make_view`, `fx.get_iter`),
  `fx.Array` + `SharedAllocator`, `fx` constants. Existing intrinsic wrappers cover most equivalents;
  an unsupported boundary stays local to aiter rather than being fixed by extending FlyDSL.
- `memref.*` → layout tensors/views + copy atoms.
- `math.*` → `fx` math helpers (`expr/math.py`). `math_dialect.fma` and friends stay where no wrapper
  exists.

### 3b. `fly.ptr` → `!llvm.ptr` (backend-resolved address space)

Given an `fx` pointer (`fly.ptr`) that must become a raw `!llvm.ptr` at a hard boundary, the DSL
primitive maps the pointer's semantic address space to the backend's LLVM address-space number.
Hand-building one with a hardcoded `<1>` / `<3>` via `IntToPtrOp` re-encodes that mapping.

```python
# Before (hardcoded address space)
p = buffer_ops.create_llvm_ptr(lds_addr, address_space=3)
p = mem_ops._create_llvm_ptr(val, address_space=1)   # a.k.a. mem_ops.to_llvm_ptr
# After
p = ptr.llvm_ptr          # property on an fx pointer
p = fx.to_llvm_ptr(ptr)   # equivalent free function; backend resolves the AS
```

- Applies only when a `fly.ptr` is already in hand. A raw int/index address (an LDS byte offset with
  no pointer form) still needs manual construction.
- `mem_ops.get_llvm_ptr` / `element_ptr` also fold in `+ offset*dtype_bytes` arithmetic. Keep the
  offset math (layout views / `get_element_ptr`) and swap only the final pointer cast.
- Byte-versus-element GEPs and alignment provenance are load-bearing. An equal numeric address does
  not imply equal memory instructions — compare the generated loads and stores when replacing an
  epilogue pointer path.

### 3c. Manual `s_waitcnt` bitfields → `fx.rocdl.s_waitcnt(vmcnt=/lgkmcnt=/expcnt=)`

Hand-encoding a wait-counter bitfield (or calling `rocdl.s_waitcnt(magic)` with a raw number) is
arch-fragile: the field widths differ per architecture (CDNA3 `lgkmcnt` max 15 versus RDNA 63). The
keyword form of `fx.rocdl.s_waitcnt` (`expr/rocdl/universal.py`) is arch-dispatched across
gfx942 / gfx950 / gfx11xx / gfx120x and packs the correct bitfield.

```python
# Before
rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))      # per-kernel encoder
rocdl.s_waitcnt(0)                               # raw "wait for everything"
_s_waitcnt(0xC07F)                               # magic LGKMCNT_0_ONLY bitfield
# After
fx.rocdl.s_waitcnt(lgkmcnt=0)                    # wait for LDS/SMEM only
fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0) # matches raw s_waitcnt(0)
```

- Unset fields default to "no wait" (their per-arch max), so only the needed counters are named.
- Per-kernel `_encode_waitcnt` / `_s_waitcnt` shims and magic `*CNT_*` constants become dead once
  their last caller is converted.
- Use the public `fx.rocdl.sched_barrier` / `fx.rocdl.sched_group_barrier` wrappers when the pinned
  version exposes them. The legacy wait form remains available as positional
  `fx.rocdl.s_waitcnt(bitfield)` for a boundary the keyword form cannot express.
- **Scheduler-sensitive.** `s_waitcnt` placement drives hot-loop pipelining in tuned attention and
  GEMM kernels, so an op-identical swap can still shift the schedule. This is the clearest case where
  a mechanical migration can cost performance: verify median-based timing, not just correctness, and
  leave pervasively tuned kernels alone.

---

## 4. `SmemAllocator` / `SmemPtr` → `SharedAllocator`

The legacy LDS path uses a manual base pointer, byte offsets, and `finalize()`. The current form
declares an `@fx.struct` of `fx.Array` fields and allocates via `fx.SharedAllocator`; the compiler
sizes the LDS global and there is **no finalize**.

```python
# Before
allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
base = allocator.get_base()
smem_a = SmemPtr(base, 0, dtype_, shape=(BLOCK_M * BLOCK_K,))
smem_b = SmemPtr(base, a_bytes, dtype_, shape=(BLOCK_K * BLOCK_N,))
allocator.finalize()
# After
@fx.struct
class SharedStorage:
    a: fx.Array[fx.Float16, BLOCK_M * BLOCK_K]
    b: fx.Array[fx.Float16, BLOCK_K * BLOCK_N]

lds   = fx.SharedAllocator().allocate(SharedStorage).peek()
lds_a = lds.a.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
lds_b = lds.b.view(fx.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))
```

- Default `static=True` leaves `launch(smem=...)` unset; only `static=False` auto-infers `smem` from
  `allocated_bytes`.
- `SmemPtr.get()` caches its view, so reusing it in an epilogue after a `scf.for` produces a dominance
  error. `SharedAllocator` avoids this (the view is taken per use). Legacy code works around it by
  clearing `ptr._view_cache = None`.
- This is a structural change: a kernel's whole LDS moves at once, and its test re-runs.

> Several operator cards in this base still describe `SmemAllocator` as the FlyDSL LDS knob. Treat
> those as descriptions of the kernels as written, not as a recommendation for new code.

---

## 5. Runtime branches inside helper functions

Inside a rewritten `@flyc.kernel` or `@flyc.jit` function, an ordinary Python `if` supports side
effects and carried scalar/list/tuple state. Carried values are initialized before the branch with
matching types; `None` cannot become an SSA result. A branch that produces values does not by itself
require raw SCF.

A plain helper that executes outside the rewriter's scope needs a local `@flyc.jit` boundary for its
runtime `if` — the public decorator, not a direct call to `ReplaceIfWithDispatch.scf_if_dispatch`.

```python
# Before
with _if_then(_scf.IfOp(_raw(ArithValue(q_start < seqlen_q)))):
    ...
# After
def then_path(): ...
def else_path(): ...

@flyc.jit
def dispatch():
    if q_start < seqlen_q:      # typed fx compare → scf.if
        then_path()
    else:
        else_path()

dispatch()
```

- A bare `if cond:` is enough for a simple guarded side effect; no helper needed.
- `const_expr(flag)` marks compile-time branches. Runtime SSA (`gpu.thread_id`, `lane`) must never be
  wrapped in `const_expr`.
- For branches that return or update values, confirm the rewriter preserves their structure and types.
  A manual `scf.IfOp` stays only for a *demonstrated* unsupported control-flow contract, kept local.

---

## 6. Raw `rocdl.mfma_*` → MMA atom + `fx.gemm`

Raw intrinsics hardcode fragment types, the `[a, b, c, 0, 0, 0]` operand tuple, and the instruction.
Building an atom and issuing it handles fragment layouts and packing, and the atom family follows the
target: `MFMA` for CDNA3/CDNA4, `WMMA` for gfx11/gfx1250.

```python
# Before
c_frag = rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0])
# After
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))   # → f32 acc
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)                    # d, a, b, c (prefer this)
fx.mma_atom_call(mma, frag_C, frag_A, frag_B, frag_C)           # single tile — see §7b
```

- `fx.rocdl.MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None)` picks the intrinsic from shape + dtype.
  Scaled variants: `fx.rocdl.cdna4.MFMA_Scale`; gfx1250/gfx11: `fx.rocdl.WMMA` / `WMMAScale`.
- Build fragments with `fx.make_fragment_like` / `make_fragment_{A,B,C}`, not raw `T.vec(...)`.
- Operand order is **d, a, b, c** (accumulator first).
- Structural: a complete supported MMA path converts at once, with a numerics diff. Raw calls stay for
  instructions or operand forms the builders cannot express, and wherever the alternative changes
  required semantics or scheduling.

---

## 7. Tiled copy / MMA: build from a TV layout, iterate with `fx.copy` / `fx.gemm`

### 7a. Building the tiled copy (TV layout)

A tiled copy is a copy atom laid over a **thread-value (TV) layout** plus a tiler. The TV layout is
built from separate thread/value layouts with `fx.make_layout_tv` (returning `(tile_mn, tv_layout)`),
both are passed to `fx.make_tiled_copy`, sliced per-thread with `.get_slice(tid)`, and then used to
partition the tensor. See `examples/02-tiledCopy.py` in a FlyDSL source checkout.

```python
# thread + value layouts -> (tile_mn, tv_layout) -> tiled copy
thr_layout = fx.make_layout((4, 1), (1, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
copy_atom  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
thr_copy   = tiled_copy.get_slice(tid)
part_src   = thr_copy.partition_S(bA)   # bA = fx.slice(fx.zipped_divide(A, tile), (None, bid))
part_dst   = thr_copy.partition_D(bB)
frag       = fx.make_fragment_like(part_src)
```

- `fx.make_tiled_copy_tv(atom, thr_layout, val_layout)` is the one-call shortcut for the
  `make_layout_tv` + `make_tiled_copy` pair. Either it or the explicit two-liner beats hand-building a
  TV layout inline in `make_tiled_copy`.
- For copies matched to an MMA operand layout, do **not** hand-build a TV layout — use
  `fx.make_tiled_copy_A/B/C(copy_atom, tiled_mma)` (they read the atom's `tv_layout_{A,B,C}_tiled`),
  then `.get_slice(tid)` + `partition_S` / `.retile(frag)`. See `examples/03-tiledMma.py` in a FlyDSL
  source checkout.
- Build the MMA with `fx.make_tiled_mma(mma_atom, atom_layout)`; slice with `.thr_slice(tid)` /
  `.get_slice(tid)` and make fragments via `make_fragment_{A,B,C}`.
- Layouts passed to `make_layout_tv` must be **static** — plain Python-int shapes/strides in
  `make_layout`.

### 7b. `fx.copy` / `fx.gemm` over `*_atom_call`, including single-atom sites

`fx.copy` / `fx.gemm` iterate the atom over a tiled/partitioned layout and take atom state as kwargs,
with no hand-written loop and no `atom_set_value`. They apply **not only to loops but to single-atom
sites**, where supported by the pinned API: `fx.copy(atom, src, dst)` can issue the same atom over a
single-tile partition. API-level equivalence does not prove identical ISA or performance — check the
generated code.

```python
# Before — loop
for k in range_constexpr(K_TILES):
    fx.copy_atom_call(copy_atom, part_src[k], frag[k])
for k in range_constexpr(K_TILES):
    fx.mma_atom_call(mma, frag_C, frag_A[k], frag_B[k], frag_C)
# Before — single atom (helpers, one tile)
fx.copy_atom_call(copy_atom, fx.slice(tiles, (None, idx)), r)
fx.mma_atom_call(mma, frag_C, frag_A, frag_B, frag_C)
# After — same in both cases
fx.copy(copy_atom, part_src, frag)                                     # loop or single
fx.copy(copy_atom, fx.slice(tiles, (None, idx)), r)                    # single-atom swap
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C, scale_a=sa, scale_b=sb)   # atom state as kwargs
```

- `fx.copy` for partitioned tensors (`partition_S` / `partition_D` / tiled divide); `fx.gemm` for the
  MMA loop (accumulator-first order).
- A supported single-atom swap is one-for-one (`fx.copy_atom_call(a, s, d)` → `fx.copy(a, s, d)`); no
  new TV layout is needed. Manufacturing a TV layout for a degenerate single-tile load whose
  thread→data mapping is a mandatory swizzle is the wrong move — pass the existing single-tile slice.
- **Keep** `copy_atom_call_ssa` / `mma_atom_call_ssa`: the SSA-*returning* variants are a different
  primitive. Keep any raw atom call whose operands have no tensor/partition form to pass.
- Diff numerics and ISA. For scheduler-sensitive hot loops, compare repeated paired graph timings —
  unchanged register/LDS resources do not prove unchanged time.

---

## 8. Comments and dead code

Low-value comments and dead code within the requested cleanup scope go — including pre-existing dead
code, not only what the migration itself made redundant. A line-count reduction is context, not
evidence of correctness or of complete cleanup.

**Removable:** comments that restate the code; commented-out or dead blocks; per-line step narration;
ASCII banners (one concise header per section is enough); stale comments that contradict the code;
locals, imports and helpers the migration made unused; runs of two or more blank lines.

**Keep:** the *why* — non-obvious layout/stride math, swizzle rationale, ABI quirks, offset-unit
gotchas, invariants, spec/ISA references.

- A broad comment-only pass belongs in its own commit, separate from executable changes; comparing
  ASTs (ignoring docstrings where appropriate) is what verifies that claim.
- Callers, re-exports, generated/JIT lookup and import side effects all need checking before a helper
  or argument is deleted. Text hits alone do not establish that code is dead.
- Similar formulas are not necessarily interchangeable. Activation rounding, batch scheduling,
  signed/unsigned extrema and packed bit widths survive deduplication: extract the common operation
  and keep the meaningful variants explicit.

---

## 9. Launch overhead: `_run_compiled`

Calling a `@flyc.jit` wrapper directly re-runs per-call dispatch (DLPack, argument marshalling, cache
lookup). On hot paths, `_run_compiled` (`aiter/ops/flydsl/kernels/tensor_shim.py`) compiles once,
caches the `CompiledFunction`, and fast-dispatches afterwards.

```python
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

compiled = compile_my_kernel(...)          # {"launch": <exe>, ...}
_run_compiled(compiled["launch"],
              ptr_arg(out), ptr_arg(a), ptr_arg(b),
              a.stride(0), M, N, K, stream)
```

- At the pinned `b0ced008` revision, `tensor_shim._run_compiled` documents
  `flyc.compile(exe, *args)` as **compile-and-execute** on the first call, then invokes the cached
  `  CompiledFunction` on later calls. Preserve that helper's contract. For no-dispatch
  materialization use the existing preload helper; never pass placeholder pointers to a compiling
  launch path without that guard.

> **Version conflict — inspect the implementation before writing a replacement.** The later
> `04c7b808` review rules say `flyc.compile()` only compiles, and report a cache-miss path that
> returned an uninitialized `torch.empty` without calling `cf(*args)` (ROCm/aiter#3987). These two
> contracts cannot be made safe by always calling the returned function: on a compile-and-execute
> version that would launch twice. Use the `_run_compiled` shipped with the pinned runtime, and test
> the first invocation with a sentinel output before introducing a custom cache shim.
- Pass flat scalars and pointers (`ptr_arg(t)`, `data_ptr()`, `stride(i)`, sizes, `stream`) to bypass
  DLPack. Existing callers named by the upstream skill: `mla_reduce_kernels.py`,
  `linear_attention_prefill_kernels.py`, `splitk_hgemm.py`.
- Reuse `tensor_shim._run_compiled` rather than adding a second copy in a wrapper module — but note
  the distinct tuple-argument `moe_kernels._run_compiled` contract described above.
- Worth it for small kernels in tight loops, not for cold one-shot launches. Argument order and types
  must match the compiled signature.

---

## 10. Procedure

1. **Find** legacy usage. Run from an aiter checkout, scoped to `aiter/ops/flydsl/`:
   ```bash
   rg -n 'ArithValue|_to_raw|arith\.(unwrap|index|index_cast)|fx\.Index\(' <file>
   rg -n 'buffer_ops\.(create_buffer_resource|buffer_load|buffer_store)' <file>
   rg -n '_mlir\.dialects|from flydsl\.expr import' <file>
   rg -n '\b(scf\.(For|If)Op|vector\.(extract|bitcast|splat)|llvm\.(load|store|mlir))' <file>
   rg -n 'SmemPtr|SmemAllocator|\.finalize\(\)' <file>
   rg -n 'fx\.(Int32|Int64|Float32)\(fx\.(Int32|Int64|Float32)\(' <file>
   rg -n 'rocdl\.mfma_|\bmfma_(f32|i32)_|copy_atom_call|mma_atom_call' <file>
   rg -n 'create_llvm_ptr|_create_llvm_ptr|get_llvm_ptr|IntToPtrOp' <file>
   rg -n 's_waitcnt\(|_encode_waitcnt|_s_waitcnt|CNT_[0-9A-Z_]*=|0x[Cc]07[Ff]' <file>
   rg -n 'LOG2E|log2e|def .*sigmoid|def .*tanh|def .*ceildiv|def .*ptr' <file>
   ```
   Resolve import aliases and inspect nested definitions and callers. Text hits are candidates, not
   proof of duplication or of dead code.
2. **Triage.** Mechanical swaps first (operators, casts, `vector.extract` / `bitcast`), structural ones
   second (control flow, `buffer_ops` offsets, MMA loops).
3. **Migrate in small commits**, one family at a time, matching local style.
4. **Verify:**
   ```bash
   black --check <changed-files> && ruff check <changed-files>
   FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest op_tests/test_flydsl_<kernel>.py -v
   # or: op_tests/flydsl_tests/test_flydsl_<kernel>.py
   ```
   Some FlyDSL tests are scripts rather than pytest modules — use the test's own CLI. The exit code is
   not sufficient: check asserted comparisons and dispatch logs, and compare numerical results, ISA
   and performance for the changed paths.
5. **Review the merge-base diff and the remaining candidates.** Callers of shared helpers, excluded
   paths that import them, and the final tree after any upstream merge all matter. `git diff --stat`
   and `git diff --check` catch the mechanical residue.
6. **Report coverage and residuals.** Separate native correctness/performance results from
   compile/ISA-only cases, skips and pre-existing baseline failures. Record each retained low-level
   boundary's reason and the tested commit, and inspect CI failures before calling the work done.
   Rerun any check affected by a later code change; an earlier green run does not cover it.

---

## Quick reference

| Legacy | Current |
|---|---|
| `ArithValue(x) + y` | `x + y` (typed `fx`) |
| `arith.unwrap(v)` / `_to_raw(v)` | `v.ir_value()` (boundary only) |
| index-typed arithmetic | explicit `fx.Int64/Int32(...)` where supported; retain `fx.Index` at index-typed boundaries |
| `arith.mulf/addf/trunc_f/select` | `*`, `+`, `.to(ty)`, `.select(...)` |
| raw integer min/max or ceil-div | `fx.max` / `fx.min` / `fx.ceildiv` when signedness and overflow behavior match |
| `vector.extract/bitcast/splat` | `fx.Vector(v)[i]` / `.bitcast(ty)` / `.filled(...)` |
| `scf.ForOp` / `scf.IfOp` | `range_constexpr` / `range(..., init=)` / Python `if` / `const_expr` |
| `buffer_ops.*` + offsets | `fx.rocdl.make_buffer_tensor` + layout + `fx.copy` |
| raw `llvm` / `memref` access | `fx.make_view` / `fx.get_iter` / `SharedAllocator` |
| `create_llvm_ptr(v, address_space=N)` / manual `IntToPtrOp` | `ptr.llvm_ptr` / `fx.to_llvm_ptr(ptr)` (backend-resolved AS) |
| `rocdl.s_waitcnt(_encode_waitcnt(...))` / magic bitfield | `fx.rocdl.s_waitcnt(vmcnt=/lgkmcnt=/expcnt=)` (arch-dispatched) |
| `SmemAllocator` / `SmemPtr` + `finalize()` | `@fx.struct` + `fx.SharedAllocator().allocate(...).peek().view(...)` |
| raw SCF or AST-rewriter calls in a plain helper | local `@flyc.jit` with Python `if` and typed carried state |
| `fx.Int32(fx.Int32(x))` / wrapping const ints | plain Python int; wrap once |
| `rocdl.mfma_*` raw intrinsic | `fx.make_mma_atom(fx.rocdl.MFMA(...))` + `fx.gemm` for supported operands and semantics |
| hand-built TV layout in `make_tiled_copy` | `fx.make_layout_tv` + `fx.make_tiled_copy` / `make_tiled_copy_tv`; `_A/_B/_C` for MMA operands |
| `*_atom_call` (loop *or* single atom) | `fx.copy` / `fx.gemm` where equivalent; retain raw SSA and unsupported operand contracts |
| restated / dead / stale comments, blank runs | delete within the requested scope; retain invariant and rationale comments |
| per-call `@flyc.jit` on a hot path | `_run_compiled(exe, *args)` fast dispatch |

## Sources
- ROCm/aiter@b0ced008:.claude/skills/flydsl-kernel-code-cleanup/SKILL.md — origin (ingested as reference).
- ROCm/aiter@b0ced008:requirements.txt — `flydsl==0.3.2`, the pinned version every mapping is stated against.
- ROCm/aiter@b0ced008:aiter/ops/flydsl/kernels/tensor_shim.py — `ptr_arg` (L255), `_run_compiled` (L266).
- ROCm/aiter@b0ced008:aiter/ops/flydsl/moe_kernels.py — tuple-argument `_run_compiled` (L1001).
- ROCm/aiter@b0ced008:aiter/ops/flydsl/kernels/kernels_common.py — `LOG2E` (L20), `ceildiv` (L23).
- ROCm/aiter@b0ced008:aiter/ops/flydsl/kernels/buffer_ops.py — legacy buffer-resource path still shipped.
- ROCm/aiter@04c7b808:.claude/skills/review-pr/rules.md — the three safety amendments added 2026-09-21:
  the `max_size=True` runtime-ragged bound hazard in §2 (aiter#4151 → #4546), the version-dependent
  `flyc.compile` contract in §9 (aiter#3987), and the `arith.bitcast` / `arith.unwrap` boundary in §1
  (aiter#3944). These are review findings against aiter kernels, not part of the cleanup skill.
- AMD CDNA3 ISA (wait-counter field widths, MFMA shapes): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD CDNA4 ISA (scaled MFMA, gfx950 memory ops): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- Cross-refs: [`authoring_tile_programming.md`](authoring_tile_programming.md) (tile/TV model for §7) · [`authoring_optimization.md`](authoring_optimization.md) (performance work, which this is not) · [`debugging.md`](debugging.md) (post-migration triage) · [`deep.md`](deep.md) (FLIR / ROCDL surface) · [`patterns.md`](patterns.md) (library-side FlyDSL GEMM usage)
