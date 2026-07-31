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

## Padding vs swizzle (LDS bank conflicts)

Copy one of the two shared-layout constructors (host-side `gl.constexpr`), then specialize:

```python
# Padding: change the row stride so consecutive rows land on different banks.
# [[pad_every, pad_by]] — pad `pad_by` elems every `pad_every`. Costs LDS capacity.
sh = gl.PaddedSharedLayout([[512, 16]])

# Swizzle (preferred, zero extra LDS): XOR-remap the bank index per row.
# phase = (row // per_phase) mod max_phase ; new_vec = XOR(vec_id, phase)
sh = gl.SwizzledSharedLayout(vec, per_phase, max_phase, order)   # may need ds_bpermute
```

- **Padding** costs LDS capacity but is simplest.
- **Swizzle** adds no capacity; start `per_phase`/`max_phase` from the `ds_read_b128`
  interval diagnostic below and widen the period until the conflict clears.

Diagnostic: conflict-free `ds_read_b128` issues every 16 cycles; 32/64 means
2/4-way conflict (`../hardware/roofline-models.md`). Verify a candidate layout with
the `layout_plot` tool (see `## Layout self-check tool` below) before committing.

### Transpose-read + dual-orientation conflict

To feed one LDS tile to two dots in **different operand orientations** (instead of
storing the tile twice), store it once in natural order and read the transposed
operand via `smem.permute((1, 0)).load(dot_layout)` (transpose-on-read /
`ds_read_tr`). The catch: a tile read in **both** orientations from one buffer
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

## TTGIR -> Gluon recovery map (for transcription)

This table is **automated** by `scripts/ttgir_to_gluon.py` (driven by
`scripts/recover_gluon.py` / `dump_ir.sh --emit-gluon`); the table is the spec it
implements -- read it to review the emitted layouts, not to hand-map them.

> **Version note.** The attribute/class spellings below are this build's instance of a
> version-stable rule (1:1 transcription of the lowered IR's compiler-chosen
> layout/memory/pipeline decisions into explicit Gluon). When a spelling or the IR
> format drifts and the script breaks, do not hand-map from this table from memory --
> follow `../phases/transcribe.md ## Version-agnostic recovery` (discover this build's
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
| loop `num_stages = N` (pipeliner) | starting pipeline depth N in the Gluon body |

Preserve logical tiles, masks, dtype, launch config, and the measured boundary;
only the layout / memory / pipeline expression becomes explicit. The result is
the **equivalence anchor** (`../phases/transcribe.md`).

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
  (`../phases/transcribe.md` — never hand-derive), or use the **hybrid** (fast path for
  the clean sub-tiles, sync `convert_layout` for the one awkward sub-tile).

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
  determinism (`../benchmark-hygiene.md ## Determinism race-test`).
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
Pin the same Triton build (tutorial tag `gfx950-tutorial-v0.1`) when comparing
against checked-in IR dumps.

