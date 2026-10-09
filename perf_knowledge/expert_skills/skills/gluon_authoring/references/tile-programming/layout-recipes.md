# Gluon Layout Recipes (gfx950)

Read this for the layout chain and for recovering compiler-inferred layouts from
a plain-Triton TTGIR during transcription. For full API rules see the owned
`../gluon/index.md` (mechanism router), `../gluon/layout-reference.md`, and
`../gluon/matrix-reference.md`.

Imports:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

## Layout families

| Family | Gluon symbol | Key parameters |
| --- | --- | --- |
| Blocked (global load/store) | `gl.BlockedLayout` | `size_per_thread`, `threads_per_warp`, `warps_per_cta`, `order` |
| Linear / XOR (async offsets) | `gl.DistributedLinearLayout` | `reg_bases`, `lane_bases`, `warp_bases`, `block_bases`, `shape` |
| Slice (1D broadcast index) | `gl.SliceLayout` | `dim`, `parent` |
| Shared padded | `gl.PaddedSharedLayout` | `[[pad_interval, pad_amount], ...]`, swizzle bases, block bases, shape |
| Shared swizzled | `gl.SwizzledSharedLayout` | `vec`, `per_phase`, `max_phase`, `order` |
| Dot operand | `gl.DotOperandLayout` | `operand_index`, `parent` (mfma), `k_width` |
| MFMA result | `gl.amd.AMDMFMALayout` | `version`, `instr_shape=[M,N,K]`, `transposed`, `warps_per_cta` |

`BlockedLayout` constraint (wave64): `threads_per_warp[0] * threads_per_warp[1]
== 64`. `CTAShape = [s0*t0*w0, s1*t1*w1]`.

MFMA instruction K-dim (not a free knob):

```text
kDim = (waveSize / nonKDim) * kWidth * kGroup     # wave64, nonKDim=16 -> 4*kWidth*kGroup
```

gfx950 defaults (nonKDim=16, wave64): fp16/bf16/fp8/bf8 -> `(kWidth=8, kGroup=1)`;
i8 -> `(16,1)`; f4/fp6/bf6 -> `(32,1)`. Result layout = `AMDMFMALayout(version=4)`.
16x16 is the gfx950 default instruction shape (`instr_shape=[16, 16, 32]` for fp16/bf16).

**gfx942 downgrade:** `AMDMFMALayout(version=3)`; the fp16/bf16 16x16 instruction is
`[16, 16, 16]` (half the K per issue, so `k_width=4`), fp8 is the FNUZ encoding, and there is no
f4/fp6/bf6 row. Per-dtype `instr_shape` / `k_width` cells for both arches:
`../hardware/capability-matrix.md` and `../gluon/matrix-reference.md`.

## Standard GEMM chain (gfx950, FP16)

```python
# 1. global load layout
blk = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 16],
                       warps_per_cta=[4, 1], order=[1, 0])
# 2. shared (LDS) layout: padded OR swizzled (conflict-free ds_read)
sh  = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])     # vec, perPhase, maxPhase
# 3. MFMA result + operands
mfma = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32],
                            transposed=True, warps_per_cta=[2, 2])
a_op = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
b_op = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
# 4. store: BlockedLayout (often back through convert_layout from mfma)
```

Construct layouts on the **host** and pass them as `gl.constexpr`; never build a
layout object inside the `@gluon.jit` body.

**gfx942 downgrade of this chain:** `AMDMFMALayout(version=3, instr_shape=[16, 16, 16], ...)` with
`k_width=4` operands; the swizzle period is sized for **32 banks** (64 on gfx950), so a
`max_phase` copied from a gfx950 layout over-rotates; and the LDS budget the shared buffers must fit
is 64 KiB per CU instead of 160 KiB. A gfx942 async destination additionally needs `order=[1, 0]`
(`memory-path.md ## The memory-path ladder`).

## Padding vs swizzle (LDS bank conflicts)

Copy one of the two shared-layout constructors (host-side `gl.constexpr`), then specialize:

```python
# Padding: change the row stride so consecutive rows land on different banks.
# [[pad_every, pad_by]] — pad `pad_by` elems every `pad_every`. Costs LDS capacity.
sh = gl.PaddedSharedLayout.with_identity_for([[512, 16]], shape, order)

# Swizzle (preferred, zero extra LDS): XOR-remap the bank index per row.
# phase = (row // per_phase) mod max_phase ; new_vec = XOR(vec_id, phase)
sh = gl.SwizzledSharedLayout(vec, per_phase, max_phase, order)   # may need ds_bpermute
```

`PaddedSharedLayout` has **no short constructor**: the class takes four positional fields
(`interval_padding_pairs`, `offset_bases`, `block/cga bases`, `shape`) on all four versions
(checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0), so `gl.PaddedSharedLayout([[512, 16]])` raises rather
than defaulting the rest. Use `with_identity_for(interval_padding_pairs, shape, order)` when you
want plain padding with no element remapping, and the full constructor when you are transcribing
recovered `offset_bases` out of a TTGIR dump (`## TTGIR -> Gluon recovery map (for transcription)`,
which emits the four-argument form).

- **Padding** costs LDS capacity but is simplest.
- **Swizzle** adds no capacity; start `per_phase`/`max_phase` from the `ds_read_b128`
  interval diagnostic below and widen the period until the conflict clears.
- **Apply the permutation on the source addresses instead, and the shared layout goes away.**
  The XOR the swizzled layout would perform on the bank index can be folded into the *offsets the
  copy reads with*, so the data arrives in shared memory already permuted and the descriptor can be
  the plain identity layout:

  ```python
  # phase from the row, XORed into the word index of the GLOBAL read
  offsets = row[:, None] * (row_stride // PACK) + (
      word[None, :] ^ ((row[:, None] % MAX_PHASE) * (VEC // PACK))
  )
  ```

  What it buys is not bank behaviour — that is the same permutation either way — it is that no
  padding bytes are spent and no non-identity shared layout has to be carried through the
  consumers. What it costs is that the permutation now lives in the producer's index arithmetic,
  so **every** consumer of that buffer must agree on it: the invariant moves from a descriptor the
  compiler checks into an index expression nothing checks. Prefer the layout constructors while
  the buffer has one consumer; this is the variant for a buffer whose LDS budget is already spent
  and whose readers you own.

  **Fold the XOR into the offsets, at the address computation.** That part is the technique: the
  swizzle is expressed once where the addresses are formed, so the shared layout the consumer reads
  can stay trivial. What is **not** established is the ordering claim that often travels with it —
  that masking or permuting a tile after a reshape, versus before, changes the addressing the
  backend generates. The two describe the same values, and this pack has no lowering guarantee
  about which expression survives. Read the emitted form if a decision depends on it; if a rewrite
  rests on the ordering alone, take it through `../method/triage.md`
  (`missing_lowering_behavior`) rather than treating it as a rule.

Diagnostic: conflict-free `ds_read_b128` issues every 16 cycles; 32/64 means
2/4-way conflict (`../hardware/roofline-models.md ## LDS throughput (conflict diagnostic)`; the
machine-readable interval is `thresholds.json confirm.ds_read_b128_interval_cyc` in
`perf_knowledge/hardware/data/`). Bank index is `(addr/4) % 64` on gfx950 and `(addr/4) % 32` on
gfx942 (`../hardware/planning-constants.md`). Verify a candidate layout with
the `layout_plot` tool (see `## Layout self-check tool` below) before committing.

### Transpose-read + dual-orientation conflict

To feed one LDS tile to two dots in **different operand orientations** (instead of
storing the tile twice), store it once in natural order and read the transposed
operand via `smem.permute((1, 0)).load(dot_layout)` (transpose-on-read /
`ds_read_b64_tr` on gfx950). **gfx942 downgrade:** there is no `ds_read_tr`; the permuted load
still lowers, but as ordinary `ds_read`s plus in-register shuffles, so price it against storing
the tile twice. The catch: a tile read in **both** orientations from one buffer
conflicts on the transposed read — a row that spans all banks becomes a many-way
conflict on the transposed column read. Mitigate with a **higher-period swizzle**
(`per_phase = 1`, `max_phase ~= banks / vec`) or padding; reuse the same
`ds_read_b128` interval diagnostic to confirm the conflict dropped. This is the
layout-dependency side of the async transpose-on-read path
(`../gluon/memory-reference.md ## Shared-layout family + transpose-on-read`).

Copy this transpose-read skeleton, then specialize the dot layout and swizzle:

```python
# Store one tile once; specialize the consumer orientation.
transposed_operand = smem.permute((1, 0)).load(dot_layout)
# Feed transposed_operand to the second dot and verify ds_read_tr conflicts.
```

## Compute the layout instead of tabulating it (constexpr_function)

`triton.constexpr_function` wraps an ordinary Python function so it can be called with `constexpr`
arguments from a `@gluon.jit` body and hand back a `constexpr` result. It runs at **trace time**,
so a layout it builds is a compile-time constant — this is not the runtime layout construction the
host-factory rule above forbids. Present on all four versions (checked 3.6.0 / 3.7.0 / 3.7.1 /
3.8.0), and upstream leans on it for exactly this purpose: `PaddedSharedLayout.with_identity_for`
is itself a `@constexpr_function` that computes `offset_bases` from `shape` and `order`.

Called from host code the same function returns a plain Python value; called from a JIT body it
returns a constexpr. One definition therefore serves both sides, which is what stops the host's
layout and the body's assumption about it from drifting apart.

The payoff case is the **producer side of a shared buffer**. A padded or linear shared layout
states its bit-to-coordinate mapping explicitly in `offset_bases`; the async-copy producer needs a
`DistributedLinearLayout` whose register / lane / warp bases agree with that mapping
(`../gluon/layout-reference.md ## DistributedLinearLayout Basis Design`). Deriving the second from
the first by hand is the step that silently rots when a tile dimension changes, because nothing
ties the two tables together. Derive it instead:

```python
@triton.constexpr_function
def producer_ll_for(shared_layout, vec, num_warps):
    """Split a shared layout's offset bases across reg / lane / warp for its writer."""
    bases  = list(shared_layout.offset_bases)          # low offset bit first
    n_reg  = int(math.log2(vec))                       # per-thread vector width
    n_lane = 6                                         # wave64
    n_warp = int(math.log2(num_warps))
    assert len(bases) >= n_reg + n_lane + n_warp       # one copy per thread; loop if not
    return gl.DistributedLinearLayout(
        reg_bases=bases[:n_reg],
        lane_bases=bases[n_reg:n_reg + n_lane],
        warp_bases=bases[n_reg + n_lane:n_reg + n_lane + n_warp],
        block_bases=[],
        shape=shared_layout.shape)
```

Read what this does and does not buy. **Where you cut** the basis list is a design decision, not a
derivation — it sets per-thread vector width and which bits vary across lanes, so it decides
coalescing, and a different producer wants a different cut. What the function removes is the
retyping: the bases themselves come from the consumer, so a shape change cannot leave the two
sides describing different buffers. Confirm the result with `layout_plot`
(`## Layout self-check tool (layout_plot)`) before committing it.

### Put a gate on the factory, because its failure mode is a wrong layout rather than an error

A factory like the one above carries **constants that are not parameters**, and the `n_lane = 6`
line is the one to look at: it is `log2(warpSize)` written as a literal with a comment. On the
wave64 targets this pack is aimed at, that is correct and stays correct. The hazard is that the
same function is exactly what gets copied when a kernel is retargeted, and on a wave32 part the
literal is silently off by one — the cut lands in the wrong place, the bases still have the right
*length*, `DistributedLinearLayout` still constructs, and what you get is a layout that describes a
different buffer than the shared side does. Nothing raises. Two rules follow, and both cost one
line each:

- **Derive the wave-size term rather than writing it** (`n_lane = int(math.log2(warp_size))`, with
  `warp_size` passed in alongside `num_warps`), and derive anything else that is a function of the
  target. A constant in a factory is a portability claim the factory does not check.
- **Make the illegal combination fail loudly, at the earliest point that can see it.** Two forms
  are in use and they compose: build the per-tile pieces as a **dict lookup keyed by the tile
  dimension**, so an unsupported tile raises `KeyError` at trace time instead of falling through to
  a default branch; and pair the result with `gl.static_assert` in the body stating the premises the
  bases were baked for (`gl.static_assert(BLOCK_K == 128 and NUM_WARPS == 4, "...")`). The
  `assert len(bases) >= n_reg + n_lane + n_warp` above is the minimum, not the set — it catches a
  basis list that is too *short* and nothing about whether the cut means what you intended.

The reason to spend the two lines: a hand-written basis factory with no assertions is the exact
construct that produces a silently wrong layout when `num_warps` or the tile changes, and the
symptom (wrong numbers, or a slow inner loop) does not point back at it. Production source that
writes the same factories guards them this way, and the unguarded version is on record producing
a pass-manager crash with no attribution to a layout or a line
(`../pitfalls/negative-patterns.md`).

Two scoping limits:

- `offset_bases` exists on `PaddedSharedLayout` and `SharedLinearLayout`. `SwizzledSharedLayout`
  is parameterized by `vec` / `per_phase` / `max_phase` / `order` instead, so there is no basis
  list to read and this recipe does not apply to it.
- `PaddedSharedLayout`'s third field was renamed `block_bases` -> `cga_layout` at 3.7.0.
  Positional construction survives the rename; keyword construction does not. `offset_bases` and
  `shape` keep their names on all four versions.

## TTGIR -> Gluon recovery map (for transcription)

This table is **automated**: transcription runs primarily through `scripts/ttgir_bridge.py`
(builds the layouts through the installed compiler's own bindings, and `ttgir_bridge verify` is the
layout-diff leg of the equivalence check), with `scripts/ttgir_to_gluon.py` as the text-mapping
fallback and `scripts/recover_gluon.py` / `dump_ir.sh --emit-gluon` only assembling the anchor —
all gluon pack. The table is the spec they implement -- read it to review the emitted layouts, not
to hand-map them. From the triton pack the table is still the spec, but the automation runs after
the champion handoff.

> **Version note.** The attribute/class spellings below are this build's instance of a
> version-stable rule (1:1 transcription of the lowered IR's compiler-chosen
> layout/memory/pipeline decisions into explicit Gluon). When a spelling or the IR
> format drifts and the script breaks, do not hand-map from this table from memory --
> follow `../method/transcribe.md ## 7. Version-agnostic recovery` (discover this build's
> names from the IR + the installed Gluon API by meaning, then run the four-gate check). Dump the
plain-Triton `.ttgir` (`scripts/dump_ir.sh`) and map its inferred layouts to explicit
Gluon objects 1:1:

| TTGIR attribute | Gluon object |
| --- | --- |
| `#blocked<{sizePerThread, threadsPerWarp, warpsPerCTA, order}>` | `gl.BlockedLayout(...)` with the same fields |
| `#mma` / `#amd_mfma<{version, instrShape, ...}>` | `gl.amd.AMDMFMALayout(version, instr_shape, transposed, warps_per_cta)` |
| `#shared<{...}>` (padded/swizzled) | `gl.PaddedSharedLayout(...)` or `gl.SwizzledSharedLayout(...)` |
| `#linear<{register, lane, warp bases}>` | `gl.DistributedLinearLayout(...)` |
| operand of `tt.dot` with `#dot_operand<{opIdx, kWidth, parent}>` | `gl.DotOperandLayout(operand_index, parent, k_width)` |
| `ttg.convert_layout` placement | explicit `gl.convert_layout(...)` at the same point |
| loop `num_stages = N` (pipeliner) | **no Gluon object.** Record N in the champion record and the budget (it sizes the plain overlap the anchor no longer has). `num_stages` is dead on the Gluon path in 3.8.0; the authored ring depth is chosen at the pipeline layer (`pipeline.md ## Budget before deepening`), not copied from N |

Preserve logical tiles, masks, dtype, launch config, and the measured boundary;
only the layout / memory / pipeline expression becomes explicit. The result is
the **equivalence anchor** (`../method/transcribe.md`). A lost vectorization in the anchor is a
`lost_layout` case (the suspects are exactly `lost_pipeline` / `lost_layout` / `lost_RA`): the fix
is to re-recover the layout from the IR, not to hand-push a wider vector.

## Wide / non-pow2 dim: recognize the pad-or-split decision (cue, not a recipe)

A load/dot/store dim wider than the efficient matrix / `ds_read` tile, or non-pow2
(surfaces as the pow2 shape-assert, `../hardware/capability-matrix.md`, or as
padding/compute waste), is a **decision point**, not a fixed recipe. On that
signature: **recall** the two strategy families — **pad to pow2** vs **split into
pow2 sub-tiles** — and **decide** by

- **bound class**: an LDS-bound sub-kernel prefers **pad** (fewer, wider `ds_read`s —
  each `.load()` is a separate read and a narrow tile under-fills the transfer, so
  chunking multiplies the read count; `../hardware/roofline-models.md` LDS-operand-reread
  amplification); an MMA-bound one prefers native **chunk** (less wasted compute);
- **per-sub-tile layout availability**: reuse a proven layout, or recover it
  (`../method/transcribe.md` — never hand-derive), or use the
  **hybrid** (fast path for the clean sub-tiles, sync `convert_layout` for the one
  awkward sub-tile).

The implementation (which axis is chunked → accumulate vs per-chunk output
accumulator; async vs sync per sub-tile) is **derived** from those two decisions, not
templated. Fires for attention head-dim, GEMM wide-K/N, MoE/GQA group dims, conv
channels — recognize the situation, recall pad-or-split, then decide.

## Epilogue store convert: permlane vs LDS (fidelity, not speed)

The output-store `convert_layout` lowers to an **in-register cross-lane shuffle
(permlane)** when the store uses the recovered `#linear` layout that matches the
matrix-core output, vs an **LDS round-trip** (`ds_write` + barrier) for a `blocked`
store layout. Recognize that this is the **epilogue** (runs once) → **perf-neutral**
→ a **fidelity** recovery, not a speed lever; do not spend a perf budget on it. Two
correctness traps (general, beyond attention):

- **asm-match != correct.** The recovered `#linear` is matrix-warp-arrangement
  specific; reusing one kernel's store layout on a kernel with a different warp /
  split arrangement gives **numerically wrong** output even though the asm shows a
  perfect permlane. Never accept a layout on asm shape — verify numerically +
  determinism (`../method/benchmark-hygiene.md ## Determinism race-test`).
- **preserve every offset term.** When rebuilding store/load index tensors in the new
  layout, keep the block-row base **and** the intra-block `arange`; dropping the block
  base makes all blocks alias the same rows (silent wrong output, rel ~= 1).

## Reuse, do not duplicate

For broadcast-safe `gl.arange` (SliceLayout), `convert_layout` rules, shared /
AOT layouts, and full per-target validity tables, read the owned
`../gluon/{layout-reference,matrix-reference,memory-reference,shared-aot-reference}.md`
and `../hardware/capability-matrix.md` rather than restating them here.

## Layout self-check tool (layout_plot)

Before compiling a layout chain, sanity-check it with the layout visualizer from
[ROCm/gfx950-gluon-tutorials `layout_plot/`](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/layout_plot). It
re-implements Triton/Gluon layout semantics (no torch/triton import needed) and
renders thread/lane/data assignment, MFMA operand fragments, and LDS bank
patterns, so you can confirm coalescing and conflict-freedom from the budget
before a compile-profile cycle.

```bash
git clone https://github.com/ROCm/gfx950-gluon-tutorials.git
cd gfx950-gluon-tutorials/layout_plot
# global BlockedLayout (gfx950 wave64)
python3 plot_layout.py blocked --gfx 950 --sizePerThread 1 8 --threadsPerWarp 16 4 --warpsPerCTA 1 2
# MFMA dot operand+result layout
python3 plot_layout.py dot --gfx 950 --dotShape 128 128 128 --warpsPerCTA 2 4 --dtypeA fp16 --kWidth 8
# LDS swizzle + ds_read bank-conflict overlay
python3 plot_layout.py lds --gfx 950 --layout swizzle --access read --tensorShape 128 128 --kWidth 8 --dtype fp16
```

Source: `ROCm/gfx950-gluon-tutorials`; the visualizer's `--gfx {942,950,1250}`
selects wave size, LDS banks, and default `kWidth/kGroup` matching the target.
Pin the same Triton build when comparing against checked-in IR dumps, and record which
tutorial tag that was — the checked-in dumps are only a control for the tag they were
generated on (`compiler-contract.md ## Toolchain identity`).

