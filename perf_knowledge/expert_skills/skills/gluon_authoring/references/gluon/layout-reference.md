# Gluon Layout Reference (gfx950 / gfx942, wave64)

Companions: `imports-and-launching.md` (imports, host-built layouts passed as `gl.constexpr`),
`matrix-reference.md` (MFMA operand/result layouts), `../tile-programming/layout-recipes.md` (tile
recipe, TTGIR -> Gluon recovery). A layout change is a cost change; its effect shows in the ISA and the timing, not in the spelling.

## BlockedLayout Constraints (wave64)

- `product(threads_per_warp) == 64` (gfx950 and gfx942 are wave64); `size_per_thread` entries are
  powers of two.
- **`product(warps_per_cta) != num_warps` is a compile failure, not a slower number.** A `num_warps`
  sweep over a hand-authored kernel returns build errors, so plan it as one layout redesign per
  point. Internally inconsistent layouts give an unattributed assert instead
  (`../pitfalls/negative-patterns.md ## warps_per_cta is not an independently tunable knob`).
- **Over-coverage is a cost, not a legality escape hatch.** A layout whose coverage product exceeds
  the tile and masks correctly still re-fetches the same bytes within a lane; one measured instance
  fetched that tensor twice, and cutting coverage to equal the tile was a clear kernel-level win there,
  while a streaming microbenchmark tied at twofold coverage (the duplicates hit in cache) — the cost
  ranges from none to large, so A/B it (GPU-measured, gfx950;
  `../pitfalls/negative-patterns.md ## Instruction count is not the objective function`).
- Load width is set by **`bytes_per_lane = size_per_thread[contig] * dtype_bytes`**: 8 x bf16 =
  16 B (dwordx4 reachable), 8 x fp8 = 8 B (not). A narrower dtype moves you *away* from wide access
  unless `size_per_thread` grows. Coalescing is judged on the wave's footprint, width on the lane's
  (`../tile-programming/memory-path.md ## A load's width is issue-side; a transaction's width is access-side`).
  The async direct-to-LDS width set is owned by
  `memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`.
- Raising `size_per_thread` also moves the register budget and (with a `gl.gather` on the tile) the
  shuffle count, so its effect reaches beyond the one line.

## DistributedLinearLayout Basis Design

Basis vectors map register/lane/warp bits to tensor offsets and **silently freeze tile sizes**: a
basis `[128, 0]` needs a row extent > 128, so it is incompatible with `BLOCK_SIZE_M=128` or smaller.
Shrinking a tile means redesigning the highest basis bit for that dim, not only changing the
launcher. If a tile-size change produces an LLVM or layout-surjectivity error, read the bases before
widening config search. An async-copy offset tensor that fails as `BlockedLayout` usually needs a
`DistributedLinearLayout` matched to the shared consumer (`memory-reference.md ## Deriving the
producer layout for a padded async destination`) — layout-contract evidence, not a toolchain ceiling.

## Shared Layouts: The Three Constructors

Build on the host, pass as `gl.constexpr`.

| Constructor (3.8.0 spelling) | Reach for it when |
| --- | --- |
| `SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])` | The usual default. No capacity cost; lowers on the async direct-to-LDS path at any legal width (`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`). |
| `PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape)` | A matrix-operand buffer; on gfx950 + 3.8 derive it with `## Let 3.8 Compute The Padded Layout For You (CDNA4)`. Costs LDS capacity. As an **async** destination it lowers only when the interval covers one wave's copy (64 lanes x per-lane bytes: 1 KiB at 128 bit) **and** the producer is derived from `.offset_bases`. gfx942: padding is the recorded fix for the v3 transpose-B translation failure (`../hardware/cdna3-gfx942.md`). |
| `SharedLinearLayout(offset_bases, block_bases=[], alignment=16)` | Recovering an explicit linear layout from TTGIR verbatim (`../method/transcribe.md`). |

**Constructor asserts (fail at trace time, not silently):**

- `PaddedSharedLayout`: >= 1 `[interval, padding]` pair; every interval and padding a power of two;
  intervals **distinct** (a repeated interval raises, it does not sum); every `offset_bases` /
  `cga_layout` basis the rank of `shape`. Where a position falls in several intervals the paddings
  **add**.
- `SharedLinearLayout`: `offset_bases` non-empty, one rank, `alignment` a positive power of two.
- **`PaddedSharedLayout.with_identity_for(interval_padding_pairs, shape, order, cga_layout=[])`**
  (`@constexpr_function`, host side) derives identity `offset_bases` from `order`; needs
  `len(shape) == len(order)` and power-of-two `shape`. It is **not** the async default: as an async
  destination it lowers only under the same two conditions (interval >= one wave's copy; producer
  derived from `.offset_bases` — for this identity layout an exact-cover row-major `BlockedLayout`
  is that producer). Compile-verified 3.8.0.

Padding is an occupancy decision: LDS caps resident waves independently of VGPRs
(`smem-lds-reference.md ## LDS is a second, independent occupancy limiter`).

### Checking the choice at compile time

**`gl.bank_conflicts(distr_ty, shared_ty)` exists but is unusable on the 3.8.0 ROCm wheel: it aborts
the Python process (SIGABRT) at trace time for every layout pair tried** — Blocked, DotOperand and
SharedLinear against `SwizzledSharedLayout` (`Assertion "inDim not found in layout"`,
`LinearLayout.cpp`) and anything against `PaddedSharedLayout` (`Assertion "unknown layout"`), on
gfx950 and gfx942 (compile-verified). So the `gl.static_assert(gl.bank_conflicts(...) == 0)` idiom
from upstream tutorials kills the compile process instead of checking. Settle conflicts with the ISA
(`ds_read` widths, `amd_isa.py --arch gfx950 lds`) and the `SQ_LDS_BANK_CONFLICT` counter; what the
number would mean is in `appendix-api.md ## The compile-time oracle worth knowing about`.

## Let 3.8 Compute The Padded Layout For You (CDNA4)

**`gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True)`**
returns a `PaddedSharedLayout` that avoids read-side bank conflicts for that dot operand, or
**`None`**. `@constexpr_function` (host factory). 3.8.0+ only; on 3.7.1 and earlier hand-compute
(`../tile-programming/layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`). This page is
the home of its contract and its `None` thresholds.

### Which buffers get it — the selection rule, decided per buffer

**Only a buffer read back as an MFMA dot operand can use it** (API: the first argument is a
`DotOperandLayout`). fp32 scale vectors, lookup tables, reduction/exchange scratch and the epilogue
tile take `SwizzledSharedLayout`. One kernel usually makes both choices on adjacent
`allocate_shared_memory` calls — the unit of decision is the buffer, not the kernel.

### The selection rule is a starting point, not a result — the write side can reverse it

**The factory prices the read, not the fill.** Measured on gfx950 / 3.8.0 (one operand shape,
only the shared layout changed): the padded destination emitted **96 `ds_write_b8` + 32
`ds_write_b8_d16_hi` per K step** (the padding breaks the store into scalars); the swizzled one
emitted none, and swapping to `SwizzledSharedLayout` was faster at every shape measured there, before any async
copy. **Re-checked on gfx950 / 3.8.0 with other register producers:** the scalar fill appears when the producer's
per-thread contiguous run does not match the padded layout's contiguous run (an order mismatch gave narrow
`ds_write_b16`); with a matched producer every variant filled with `ds_write_b128`, and padded tied swizzled or
won (on an MN-contiguous B operand the factory padded layout was the faster one). So neither layout wins in
general.

- Signal: non-zero narrow `ds_write` (`_b8` / `_b8_d16_hi` / `_b16`) on a staged operand = scalar fill;
  first check that the producer's contiguous run matches the destination. Then A/B padded vs swizzled and
  **time it** — the count locates, it does not price.
- Counter-example (GPU-measured): cutting `ds_write_b8` 16 -> 4 on a 16-byte *scale* tile's staging
  measured a **tie**; the gain first credited to it came from the global side of the same edit
  (`global_load_ubyte` 148 -> 132, 1 -> 4 B/lane). Separate the two sides into two arms.
- Scope: one shape/dtype/kernel. What it shows is that "padded is the efficient default for matrix
  operands" does not hold in general — not that swizzle wins in general.

### Contract points — places this goes wrong quietly

1. **Pass the `DotOperandLayout`, not the MFMA layout itself** — the helper reads `.parent`, so an
   `AMDMFMALayout` raises `AttributeError: 'AMDMFMALayout' object has no attribute 'parent'`
   (checked 3.8.0). **Its parent must be `AMDMFMALayout` with `version == 4`** (asserted; re-checked 3.8.0). No gfx942
   form; on v3 hand-compute (`../hardware/cdna3-gfx942.md`).
2. **`shape` is the shared-memory tile, not the problem tile**: `[BM, BK]` for A, `[BK, BN]` for B.
3. **Rank-2 only.** For a ring `allocate_shared_memory(dtype, [STAGES, BM, BK], shared_a)` pass the
   single-stage `[BM, BK]`; the stage dim belongs at the allocation. A rank-3 `shape` is not
   diagnosed for you.
4. **Element bit-width must be in {4, 8, 16}.** Packed fp4 is documented as `uint8` (reuses the
   8-bit pattern) — no compile-verified packed-fp4 instance exists, so check the returned layout
   against your allocation. `uint8` is also used for byte-transparent fp8 staging, so it is not
   evidence of an fp4 path.
5. **`None` is a return value, not an error.** Causes (docstring): `k_width` not in {4, 8, 16};
   bit-width not in {4, 8, 16}; an MFMA `instr_shape` / `k_width` combination it does not handle.
   **The third fires in practice and turns on the non-K tile length.** Measured on gfx950 / 3.8.0,
   `AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4])`,
   `dtype=uint8`, default `is_k_contig=True`, K extent <= 128, shape `[BM, BK]` for operand 0 and
   `[BK, BN]` for operand 1, sweeping operand x `k_width` x non-K extent (compile-free host sweep):

   | operand | `k_width` | non-`None` when non-K extent >= |
   | --- | --- | --- |
   | 0 or 1 | 16 | 32 |
   | 0 or 1 | 4 or 8 | 64 (32 once K = 256) |
   | any | 32 | never (cause 1; re-checked 3.8.0) |

   The two operands behave the same once each shape is passed in its own orientation (an operand-1 shape
   passed as `[BN, BK]` sweeps K instead and looks non-`None` from 16). At `instr_shape=[32, 32, 64]` every
   `k_width` in {4, 8, 16} needs a non-K extent >= 64 for both operands (`k_width` 4 / 8: >= 32 at K = 256).
   `is_k_contig=False` lowers the thresholds (all >= 16 at `[16, 16, 128]`). The threshold is a function of
   **`(instr_shape, k_width, K extent, is_k_contig, tile)`**, so a gate on `BLOCK_M` alone is not enough.
   Decode-band (small) tiles get `None`; prefill tiles do not. Probe your own combination in a host loop
   (no compile needed; some `is_k_contig=False` points crash the process, so run one point per subprocess).

   Handling: if every argument is statically pinned, the result is fixed at first compile and a
   fallback branch is dead code. If any argument varies at compile time (a native-fp8 flag switching
   `instr_shape`/`k_width`/`dtype`, a `k_width` sweep, a `BLOCK_M`/`BLOCK_N` sweep), gate the
   **whole LDS path**:

   ```python
   A_SHARED: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_a, [BM, BK], DOT_TYPE)
   B_SHARED: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_b, [BK, BN], DOT_TYPE)
   USE_LDS: gl.constexpr = A_SHARED is not None and B_SHARED is not None   # else: register-staged arm
   ```

   Clamping `k_width` (`gl.DotOperandLayout(0, mma, min(K_WIDTH, 16))`) covers cause 1 only, not a
   short tile. Passing `None` into `allocate_shared_memory` fails downstream with a message about
   `None`, not about coverage.

### `is_k_contig` — the default fits GEMM A/B; attention second operands differ

`shape` is in the operand's logical order (`[M, K]` op 0, `[K, N]` op 1). `is_k_contig` says
whether K is the fast axis **in the allocation you make**. For GEMM A/B the default is right. The
exception is the **second dot operand of attention / KV-cache** (V, or a key tile stored
key-index-major): write the tile as allocated, set `is_k_contig` to whether the dot's K axis is the
fast one, and if the allocated axis order differs from `shape`, rebuild with swapped bases
(`.interval_padding_pairs` / `.offset_bases` are readable attributes):

```python
shared = gl.amd.cdna4.compute_efficient_padded_shared_layout(operand_b, [head_dim, block], dtype, is_k_contig=True)
key_layout = gl.PaddedSharedLayout(shared.interval_padding_pairs,
                                   [[b[1], b[0]] for b in shared.offset_bases], [], [block, head_dim])
# V tile allocated [block, head_dim] with K=block outer: pass shape [block, head_dim], is_k_contig=False
```

Wrong flag = no error, a padding pattern for the transposed tile (bank conflicts). Check the
`ds_read` pattern in the ISA. On the async path, a factory operand B with default `is_k_contig=True`
over a row-major source also fails to lower (non-contiguous per-lane run) — pass `is_k_contig` to
match the source.

### You are only half done when the layout comes back

The factory gives the consumer side. An async fill also needs the producer layout derived from the
returned `.offset_bases` — a row-major `BlockedLayout` producer fails to lower (the factory permutes
row bases); the derived one lowers and copies back exact (compile-verified 3.8.0). Recipe and
helper: `memory-reference.md ## Deriving the producer layout for a padded async destination`.

## The Scale Operand's Layout For mfma_scaled

fp4 on CDNA4 has no regular MFMA; fp8 is a shape question (regular `16x16x32` / `32x32x16`, scaled
`16x16x128` / `32x32x64`, `matrix-reference.md ## Matrix-Family Details`). The scale operand's
layout is derived:

```python
scale_layout = gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)
```

- `@constexpr_function`, returns a `DistributedLinearLayout`. `shape` is the **scale tensor's** shape.
- **`scale_factor` is asserted to be 32** (3.8.0: "Only scale factor 32 is supported for CDNA4 Scaled MFMA") — the
  hardware group size. A group of 128 is a software convention; NVFP4's 16 is another vendor's.
- **A quant producer must emit exactly one E8M0 scale per 32 elements of the quantized dim** (scale
  group == 32, the value `get_mfma_scale_layout` asserts). A producer that computes one scale per tile
  must tile at exactly 32: at 16, 64 or 128 it compiled, ran, gave no NaN and was not bit-exact (scales and
  codes; Triton MXFP8 producer vs a reference quantizer, gfx950 / Triton 3.8.0). A producer that reshapes
  to `[.., BLOCK/32, 32]` and reduces per group was bit-exact at 32, 64 and 128 — so the tile may be any
  multiple of 32 as long as it emits `tile / 32` scales; assert the group, not the tile:
  `gl.static_assert(GROUP == 32)` turns a silent mismatch into a trace error (producer side:
  `../workloads/prologue.md ## The quant epilogue's scale layout is a contract`).
- Parent must be `AMDMFMALayout` (asserted); the result depends on `instr_shape`, `tiles_per_warp`,
  `warps_per_cta` and operand index, so **recompute it wherever `tiles_per_warp` changes**. In a
  fused producer/consumer pair only the consumer can derive it — generate it there, pass it over.
- **Suggestion: treat the producer (staging) layout of the scale tile as a 2-3 candidate sweep, not a
  formula** — signal: the scale path shows replicated loads or spills, or a width change moved time.
  Hard constraints (each matches an observed failure): `size_per_thread * element_bits in {32, 128}` if
  async-staged (a 2-byte async scale copy failed, 4 B/lane lowers); `threads_per_warp[contig] *
  size_per_thread[contig] == extent`; `threads_per_warp[other] <= other extent` (a layout replicated
  across warps cost registers and spilled). The same nominal change won on one kernel and tied on
  another. Time the candidates; "the one you had" is a valid answer.

### The scale operand is register-resident, so staging it through LDS is a choice

`mfma_scaled` takes `a_scale` as a **register** tensor; LDS staging is optional. Build indices
directly in the scale layout, e.g.
`gl.arange(0, BM, gl.SliceLayout(1, get_mfma_scale_layout(dA, [BM, NG])))`. Bit-exact spellings:

| spelling | compiled LDS | VGPR | spills |
| --- | --- | --- | --- |
| stage: global -> regs -> LDS -> scale layout | 43520 B | 124 | 0 |
| convert: global -> regs -> `convert_layout` | 41984 B | 118 | 0 |
| direct: indices built in the scale layout | 40960 B | 124 | 0 |

**One CDNA4 MoE gemm1 measured the direct form slower** at its three shapes (upstream-reported, no local
record; not reproduced here) despite fewer `ds_read_u8`, no spills. Compile-verified mechanism (gfx950,
3.8.0): the direct form turns dword scale loads into per-byte `global_load_ubyte` and drops the
`ds_read_u8`; it also loses any ring-stage prefetch of the scales. **No default sign:** the aiter Gluon MXFP4
GEMM measured the direct form faster at every shape tested (gfx950 / 3.8.0) although occupancy was not
LDS-bound, and an a16w4 MoE gemm1 measured a tie. A/B it per kernel; check the per-byte scale loads, the lost
LDS read path and spills.

Traps when measuring:

- The `convert_layout` spelling returns only part of the LDS: cross-lane conversion lowers through
  shared scratch.
- **`metadata.shared` is the peak of simultaneously live buffers** (compile-checked, gfx950 / Triton
  3.8.0): conversion scratch overlaps an explicit allocation only when that allocation is dead at the
  convert (then `shared` is their max) and adds to it when the allocation is live across the convert.
  Scratch size depends on the source and target layouts (a `[128, 256]` fp32 epilogue convert took
  8-32 KiB across the layouts tried) and can hide a small allocation change; a probe measuring an
  allocation must store in the MMA layout so no convert scratch is involved.
- **Operand 1 must be read as `[K, N]`.** `deduce_scale_factor` reads K off dim -2 for
  `operand_index=1`; a `[N, K]` buffer loaded without `.permute([1, 0])` fails with
  a scale-factor error naming neither the operand nor the permute — the text depends on the shapes:
  **`scale factor must be 16 or 32. Got 128`** (reported), or **`Operands must have the same scale
  factor; (lhs: 32 vs rhs: 16)`** (gfx950 / Triton 3.8.0, `[N=64, K=128]` operand 1).

## Shape APIs And Layout Propagation

- A 2D `DotOperandLayout` is not an `arange` layout: derive a 1D `SliceLayout` of the parent, or
  build the 2D offsets in the parent layout.
- `reshape(..., can_reorder=True)` is not generally supported. `reshape` / `split` may infer a
  `DistributedLinearLayout` where the store expects `BlockedLayout` — plan the store layout.
- `gl.expand_dims(x, axis=d)` requires `x.layout == SliceLayout(d, parent)`; broadcast indices derived
  from the exact parent of the expression need no conversion, while re-parenting 1D tensors with
  `convert_layout` adds a conversion.
- `convert_layout(..., assert_trivial=True)` fails early if a conversion is not a reinterpretation.
  Cross-lane conversions go through LDS scratch (counted in `metadata.shared`); on gfx950 a register
  `convert_layout` can beat store-to-shared + reload for small hot-loop conversions — A/B both.

### The two layouts that carry no shape

`AutoLayout` and `CoalescedLayout` raise `ValueError` on `.rank`.

- **Hand-written index math left on `AutoLayout` with nothing to resolve it does not compile**
  (`'tt.make_range' op Failed to infer return type`; compile-checked 3.6.0, 3.7.0, 3.8.0). Resolve it with
  `gl.set_auto_layout(value, layout)`. The silent slow case is different: a `triton_to_gluon` translator
  skeleton falls back to a scalar default `BlockedLayout` (size_per_thread 1), which compiles, is bit-exact,
  passes the oracle and can be several times slower on load-bound tiles (GPU-checked gfx950) — apply the
  recovered layouts to every `arange` / load (`../method/transcribe.md`).
- `CoalescedLayout` asks for "some coalesced arrangement" — fine for bring-up, a hole once measuring.

## gl.gather — a lane exchange only when the layouts allow it

`gl.gather(src, index, axis)`: `result[I] = src[I[0], ..., index[I], ..., I[n]]`, shape/layout of
`index`. Free function on all versions 3.6.0-3.8.0. Cost is decided by layouts:

| Layouts | Lowering | Cost |
| --- | --- | --- |
| warp-local | warp shuffles only | zero scratch; shuffle count quadratic |
| anything else | store `src` to LDS, index out | scratch = **whole `src`** + round trip |

Warp-local needs all three: (1) moving along the gather axis never changes the warp, for `src` and
`index`; (2) identical `(block, warp)` mapping on every non-gather dim; (3) identical `lane` mapping
on every non-gather dim. **The fallback is silent** and charges the full `src` size to LDS
(occupancy). Derive the index layout from the source layout. Warp-local emits one shuffle per
(index element, candidate source register) pair — widening `size_per_thread` on the gather axis
grows shuffles faster than work.

`smem.gather` is a different mechanism (an LDS read by construction): `smem-lds-reference.md ##
Indexed access to LDS — .gather / .scatter`.

Symptom -> fix routing for layout/broadcast failures: `../method/triage.md`.
