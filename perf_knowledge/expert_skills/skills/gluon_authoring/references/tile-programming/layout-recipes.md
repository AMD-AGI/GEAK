# Gluon Layout Recipes (gfx950)

> **Lookup over recall:** the MFMA operand layout a recipe must match — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 layout <V_MFMA_…> -A` / `locate`; LDS banking — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 lds`.

Layout chain recipes and the TTGIR -> Gluon recovery map. Full API rules:
`../gluon/layout-reference.md`, `../gluon/matrix-reference.md`.

## Layout families

| Family | Gluon symbol | Key parameters |
| --- | --- | --- |
| Blocked (global load/store) | `gl.BlockedLayout` | `size_per_thread`, `threads_per_warp`, `warps_per_cta`, `order` |
| Linear / XOR (async offsets) | `gl.DistributedLinearLayout` | `reg_bases`, `lane_bases`, `warp_bases`, `block_bases`, `shape` |
| Slice (1D broadcast index) | `gl.SliceLayout` | `dim`, `parent` |
| Shared padded | `gl.PaddedSharedLayout` | `interval_padding_pairs` (`[[pad_interval, pad_amount], ...]`), `offset_bases`, `cga_layout`, `shape` |
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

A layout is a compile-time constant: build it on the host and pass a `gl.constexpr`, or bind
`L: gl.constexpr = gl.BlockedLayout(...)` in the body.

**gfx942 downgrade of this chain:** `AMDMFMALayout(version=3, instr_shape=[16, 16, 16], ...)` with
`k_width=4` operands; the swizzle period is sized for **32 banks** (64 on gfx950), so a
`max_phase` copied from a gfx950 layout over-rotates; and the LDS budget the shared buffers must fit
is 64 KiB per CU instead of 160 KiB. A gfx942 async destination additionally needs an `order` matching the
source's fast dim (`[1, 0]` for a row-major tile) (`memory-path.md ## The memory-path ladder`).

## Padding vs swizzle (LDS bank conflicts)

The two shared-layout constructors (host-side `gl.constexpr`), as starting points to specialize:

```python
# Padding: change the row stride so consecutive rows land on different banks.
# [[pad_every, pad_by]] — pad `pad_by` elems every `pad_every`. Costs LDS capacity.
sh = gl.PaddedSharedLayout.with_identity_for([[512, 16]], shape, order)

# Swizzle (zero extra LDS): XOR-remap the bank index per row.
# phase = (row // per_phase) mod max_phase ; new_vec = XOR(vec_id, phase)
sh = gl.SwizzledSharedLayout(vec, per_phase, max_phase, order)
```

**`PaddedSharedLayout` has no short constructor** (3.8.0): it takes four positional fields
(`interval_padding_pairs`, `offset_bases`, `cga_layout`, `shape`), so `gl.PaddedSharedLayout([[512, 16]])`
raises `TypeError`. `with_identity_for(interval_padding_pairs, shape, order)` covers plain padding, the full
constructor covers transcribing recovered `offset_bases` (`## TTGIR -> Gluon recovery map (for
transcription)`). (Third field was `block_bases` before 3.7.0: positional construction survives,
keyword does not.)

- **Padding** costs LDS capacity; **swizzle** does not. The `ds_read_b128` interval diagnostic below gives a
  starting `per_phase`/`max_phase`; widening the period is the knob until the conflict clears.
- **Variant: fold the XOR into the source offsets** so data lands pre-permuted and the shared layout
  is the identity:

  ```python
  # phase from the row, XORed into the word index of the GLOBAL read
  offsets = row[:, None] * (row_stride // PACK) + (
      word[None, :] ^ ((row[:, None] % MAX_PHASE) * (VEC // PACK))
  )
  ```

  Same bank behaviour, no padding, no non-identity layout to carry; but every consumer must agree on
  the permutation, which now lives in index arithmetic nothing checks. It fits when the LDS budget is
  spent and you own all readers. (Inferred: no lowering guarantee about masking/permuting before vs
  after a reshape; read the emitted form.)

**Diagnostic (gfx950, all four SIMDs streaming): conflict-free `ds_read_b128` issues every 16 cycles; 32/64
means 2/4-way conflict.** Under another load (one wave per CU, a SIMD pair, short bursts) the absolute
interval differs; read the ratio to a conflict-free baseline measured under the same load
(`../hardware/isa-mechanisms.md ## LDS instruction behavior (gfx950)`)
(gfx942 conflict-free interval unverified, 16 or 32: `../hardware/cdna3-gfx942.md`)
(`../hardware/roofline-models.md ## LDS throughput (conflict diagnostic)`). **Bank = `(addr/4) % 64` on
gfx950**, `(addr/4) % 32` on gfx942 (`amd_isa.py --arch gfx950 lds`). `layout_plot` (`## Layout self-check
tool (layout_plot)`) shows a candidate's bank pattern before a compile cycle.

### Transpose-read + dual-orientation conflict

Store the tile once and read the other orientation with `smem.permute((1, 0)).load(dot_layout)`
(`ds_read_b64_tr` on gfx950; gfx942 has no `ds_read_tr` — it lowers to `ds_read`s + shuffles, price it
against storing twice). A tile read in **both** orientations can conflict many-way on the transposed read;
a higher-period swizzle (`per_phase = 1`, `max_phase ~= banks / vec`) or padding addresses it, and re-checking the
`ds_read_b128` interval confirms it (`../gluon/memory-reference.md ## Shared-layout family + transpose-on-read`).

## Compute the layout instead of tabulating it (constexpr_function)

`@triton.constexpr_function` runs at trace time: called from a `@gluon.jit` body with constexpr args
it returns a constexpr (a compile-time layout), called from host code a plain value — one definition
serves both sides. Upstream's `PaddedSharedLayout.with_identity_for` is itself one.

**Payoff: derive the async-copy producer from the shared layout's `offset_bases`.** A padded async
destination lowers only when its producer `DistributedLinearLayout` is derived from those bases
(`../gluon/memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`);
hand-copying the bases rots when a tile dimension changes. Verified recipe (cited by `memory-path.md`):

```python
@triton.constexpr_function
def producer_ll_for(shared_layout, vec, num_warps):
    """Split a shared layout's offset bases across reg / lane / warp for its writer."""
    bases  = list(shared_layout.offset_bases)          # low offset bit first
    n_reg  = int(math.log2(vec))                       # per-thread vector width
    n_lane = 6                                         # wave64
    n_warp = int(math.log2(num_warps))
    assert len(bases) >= n_reg + n_lane + n_warp       # tile must cover every lane/warp once
    return gl.DistributedLinearLayout(
        # leftover high bits become per-thread repeats; dropping them is a non-surjective
        # layout (LLVM ERROR at compile) whenever the tile exceeds one copy per thread
        reg_bases=bases[:n_reg] + bases[n_reg + n_lane + n_warp:],
        lane_bases=bases[n_reg:n_reg + n_lane],
        warp_bases=bases[n_reg + n_lane:n_reg + n_lane + n_warp],
        block_bases=[],
        shape=shared_layout.shape)
```

**Where you cut** the basis list is a design choice (it sets per-thread width and coalescing); the
function only removes the retyping. `layout_plot` shows the result before a compile cycle.

### Put a gate on the factory, because its failure mode is a wrong layout rather than an error

`n_lane = 6` is `log2(warpSize)` for wave64; derive it (`int(math.log2(warp_size))`) if the factory may be
retargeted, since a wrong cut still constructs and silently describes another buffer. Fail loudly at
trace time: key per-tile pieces by a dict (unsupported tile -> `KeyError`) and add
`gl.static_assert(BLOCK_K == 128 and NUM_WARPS == 4, "...")` for the premises the bases assume; an
unguarded factory is on record producing a pass-manager crash with no layout attribution
(`../pitfalls/negative-patterns.md`). `offset_bases` exists on `PaddedSharedLayout` and
`SharedLinearLayout`, not on `SwizzledSharedLayout`.

## TTGIR -> Gluon recovery map (for transcription)

Automated by `scripts/ttgir_bridge.py` (uses the compiler's own bindings; `ttgir_bridge verify` diffs
layouts), fallback `scripts/ttgir_to_gluon.py`; `dump_ir.sh --emit-gluon` assembles the anchor. The
table is the spec to review their output against. If spellings drift, follow
`../method/transcribe.md ## 7. Version-agnostic recovery`.

| TTGIR attribute | Gluon object |
| --- | --- |
| `#blocked<{sizePerThread, threadsPerWarp, warpsPerCTA, order}>` | `gl.BlockedLayout(...)` with the same fields |
| `#mma` / `#amd_mfma<{version, instrShape, ...}>` | `gl.amd.AMDMFMALayout(version, instr_shape, transposed, warps_per_cta)` |
| `#shared<{...}>` (padded/swizzled) | `gl.PaddedSharedLayout(...)` or `gl.SwizzledSharedLayout(...)` |
| `#linear<{register, lane, warp bases}>` | `gl.DistributedLinearLayout(...)` |
| operand of `tt.dot` with `#dot_operand<{opIdx, kWidth, parent}>` | `gl.DotOperandLayout(operand_index, parent, k_width)` |
| `ttg.convert_layout` placement | explicit `gl.convert_layout(...)` at the same point |
| loop `num_stages = N` (pipeliner) | **no Gluon object** (`num_stages` has no effect on the Gluon path, 3.6.0–3.8.0); size the authored ring at `pipeline.md ## Budget before deepening` |

A lost vectorization in the anchor is restored by re-recovering the layout from the IR; hand-pushing a
wider vector risks a layout that no longer matches the IR.

## Wide / non-pow2 dim: recognize the pad-or-split decision (cue, not a recipe)

A dim wider than the efficient MFMA / `ds_read` tile, or non-pow2 (pow2 shape-assert): **pad to pow2**
for an LDS-bound sub-kernel (fewer, wider `ds_read`s; each `.load()` is a separate read) vs **split into
pow2 sub-tiles** for an MMA-bound one (less wasted compute); or hybrid (fast path for clean sub-tiles,
sync `convert_layout` for the awkward one).

## Epilogue store convert: permlane vs LDS (fidelity first; speed only on short K)

The store `convert_layout` lowers to an in-register permlane shuffle with the recovered `#linear`
layout, vs an LDS round-trip (`ds_write` + barrier) with a `blocked` one. It runs once per tile, so it is
perf-neutral only when the K loop dominates. Measured on gfx950 / Triton 3.8.0 (an fp8 Gluon GEMM, 8 warps):
at long K the store variants tied; on a short-K or many-tile GEMM a convert to a wider (16 B per lane) store
beat the MMA-layout store even through the LDS round-trip, and the LDS route beat the permlane route there.
A/B the store layout when the epilogue is a visible share (the epilogue-store section of `memory-path.md`
covers the store width). **Overrides KB** for Triton / Gluon kernels on Triton 3.8.0 (gfx942 and gfx950):
every unlabelled `OPTIMIZE_EPILOGUE` recommendation for a Triton kernel in `perf_knowledge/`, among them
`perf_knowledge/languages/triton_amd/deep_codegen.md` (the epilogue convert is "killed by
`OPTIMIZE_EPILOGUE=1`"), `languages/triton_amd/knobs.md` ("drop epilogue convert (GEMM → 1)"),
`languages/triton_amd/patterns.md`, `languages/triton_amd/pitfalls.md`, `optimization/mfma_scheduling.md`
("to avoid the **512-byte Tagram (CShuffle) hotspot**") and the Triton operator cards (e.g.
`operators/dense_gemm/backends/triton.md`, `operators/gemm_epilogue_fused/backends/triton.md`). Basis: the
variable occurs nowhere in the installed 3.8.0 tree (string search of the Python sources and
`libtriton.so`), so it changes nothing in a Triton 3.8.0 compile. Pages about CK, hipBLASLt or FlyDSL
describe other libraries and are not covered. Two correctness traps:

- **asm-match != correct.** The recovered `#linear` is matrix-warp-arrangement
  specific; reusing one kernel's store layout on a kernel with a different warp /
  split arrangement gives **numerically wrong** output even though the asm shows a
  perfect permlane. Never accept a layout on asm shape — verify numerically +
  determinism (`../method/benchmark-hygiene.md ## Determinism race-test`).
- **preserve every offset term.** When rebuilding store/load index tensors in the new
  layout, keep the block-row base **and** the intra-block `arange`; dropping the block
  base makes all blocks alias the same rows (silent wrong output, rel ~= 1).

## Layout self-check tool (layout_plot)

Offline visualizer from
[ROCm/gfx950-gluon-tutorials `layout_plot/`](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/layout_plot)
(needs a network clone and a LaTeX / `pdflatex` install; no torch/triton import): renders blocked, MFMA operand
and LDS layouts (lane/register assignment, bank patterns) for gfx942 / gfx950 / gfx1250 before a compile cycle.

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

`--gfx {942,950}` selects wave size, LDS banks and default `kWidth/kGroup`.

