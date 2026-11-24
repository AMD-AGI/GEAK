# Slicing / Register Layer (gfx950)

Fit tile + prefetch + accumulator in the joint VGPR+AGPR budget and the occupancy budget with zero
hot-loop spill, keeping the AC/LR/DOT overlap. Tile / block / split sizes are the cheapest knob (a smaller
tile that fits has usually beaten a larger one that spills; halving the key tile of an attention-bwd kernel
removed all spills and ran faster). Hot-loop spills (`scratch_load`) cut MFMA efficiency sharply in the cases
recorded; slicing the output tile keeps the overlap, where reverting it gives the overlap up.

## Slice-N

Split N: `smemB_left` / `smemB_right`, half-N accumulators, staggered B loads, still unroll-x2; block-level
register budget ~512 -> ~448 (operands + accumulator; an upstream design budget, not an allocator count —
compiled on stock 3.8.0 the allocator still reported close to the full file, while the hot-loop spills and most
in-loop accumulator copies of the unrolled pre-slice kernel disappeared: read `.vgpr_spill_count` and in-loop
`v_accvgpr_*` / `scratch_*`, not `vgpr_count`). Upstream pairs it with an **AGPR accumulator hint** to shrink in-loop
`v_accvgpr_mov` copies — on 3.8.0 that is `llvm_fn_attrs=[["amdgpu-agpr-alloc", "256"]]`, usable **only at
1 wave/SIMD** (a ~512-register body), where it asks for the 256/256 split the default already gives (and the
accumulator was measured in VGPRs regardless, `llvm-fn-attrs.md`), so ISA-diff it against no attribute; at 2+ waves
`"256"` leaves no arch VGPRs and the binary fails at launch (`llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`) (verify `.agpr_count`; `amdgpu-mfma-vgpr-form` is an `llc` command-line option, not a function attribute — as an `llvm_fn_attrs` entry it is inert (identical asm for 0/1))
(`compiler-contract.md ## What upstream 3.8.0 actually gives you`; the attribute mechanism and its
assembly-diff verification are in `llvm-fn-attrs.md`). Downgrade: `llvm_fn_attrs` does not exist
before 3.8.0, so on 3.6/3.7 the AGPR rung is unreachable per compile.

## Slice-MN

2x2 quadrants, load order `B_left -> A_top -> A_bot -> B_right`, 4 regions per K-tile: spreads the loads
(16 `buffer_load`s over ~1000 cycles -> 4 per region over ~1500 cycles of MFMA), relieving TCP/FIFO
stalls at large K; block-level register budget ~448 -> ~384 (upstream design figures for the source kernel, not
an allocator count: the allocator total may not move while the in-loop accumulator copies do). It is the final
design for fp16, a8w8 and a4w4 in the gfx950 Gluon tutorial, the likely upstream source (`low-precision.md`).
Measured on gfx950 / Triton 3.8.0 on a plain (not warp-pipelined) bf16 ring: the
2x2 quadrant split of the LDS reads + MFMA alone gave a small gain and freed registers at 4 warps; the load
order above was not tested there.

## Budget to compute first

```text
R_acc      ~= BM * BN * acc_dwords * sharing / (num_warps * 64)
R_total     = R_acc + R_operand + R_prefetch + R_offsets + R_side + R_epilogue + R_compiler_tmp
constraint  : R_total <= per-wave budget (zero hot-loop spill): 512 at 1 wave/SIMD, 256 at 2 (e.g. 8 warps)
```

If `R_total` exceeds the per-wave budget after the pipeline layer, added prefetch depth only raises it further; slicing lowers it.

### Joint VGPR + AGPR budget + capacity walls (P5)

**ArchVGPR + AGPR share one 512-entry-per-SIMD file** on gfx950 and gfx942 (`next_free_vgpr` reports the
combined figure); re-splitting values between VGPR and AGPR moves demand, it adds no capacity. When the
combined budget overflows, only fewer live values remove the spill (arithmetic). Remat / streaming a
loop-invariant operand needs LDS room (160 KiB/CU gfx950, 64 KiB gfx942). A second live result tile (scaled accumulation,
split reductions) is the largest cost after the accumulator; a rewrite that carries one tile
(`../workloads/gemm.md ## Preshuffled / block-scaled signals`) is a lever of slicing's order.

### Occupancy budget (P8)

For memory/latency-bound kernels waves/CU is often a large lever: occupancy = f(LDS/wg, VGPR/wg), the
capping resource wins. The capping resource is often LDS (dedup buffers: one LDS copy of q and dO read in
both layouts instead of one per consumer took an attention-bwd kernel from 1 to 2 workgroups/CU where LDS
was the binding limiter — upstream-reported as its largest single win, mechanism reproduced on gfx950 /
Triton 3.8.0; where VGPR binds, the same dedup only saves the duplicate writes — check which limiter binds
before expecting a WG/CU step). Not always the top lever: where LDS binds first, more workgroups/CU can give nothing (a
linear-attention case).

**When waves are VGPR/LDS-capped, software prefetch (register prefetch or LDS double-buffer) has regressed
in the cases recorded**: it spends the capping resource. Waves/CU is the number to read before adding
prefetch depth, including the first two rungs of `pipeline.md ## Where the overlap comes from, and it is not
the same question per tier`. Tri-lemma: cross-stage overlap (VGPR), 2+ waves/SIMD, deep-unroll ILP (LDS
buffers) — the budget usually covers two (`../hardware/bound-class-signals.md ## Lever gating laws`). A
prediction of post-change VGPR/LDS and waves/CU prices a register-buffered pipeline before it is built. (A two-stream ping-pong sharing LDS/VGPR needs a multi-stream
budget that is not written in this pack; derive it and say so.)

## Slice recipe (ttgl.amd.slice)

**`ttgl.amd.slice` is a register-only view (3.8.0; absent from the 3.7.0 `gluon/language/amd` exports)**: layout preserved, no
cross-thread data movement, so it cuts the live-VGPR set for slice-N / slice-MN. For K, stream sub-tiles
from a shared descriptor (`.slice(...)`, also on older builds).

```python
# register view: cut the M/N register tile — no data movement, layout preserved
a_sub = ttgl.amd.slice(a_reg, [BLOCK_M, SUBTILE_K], [0, k0])
# or bound live VGPRs by streaming K sub-tiles from a shared desc into MFMA:
for k0 in gl.static_range(0, BLOCK_K, SUBTILE_K):   # static: .slice start must be a Python int (3.8.0 rejects int32)
    a_sub = a_smem.slice(k0, SUBTILE_K, dim=1).load(a_layout)   # one sub-tile resident at a time
```

Extent/offset must align to the layout's CTA tiling (3.8.0 docstring). Verify: spill -> 0, waves/CU up.

## Hoist address arithmetic (cut IOPs)

When VALU integer/address IOPs dominate, loop-invariant offset halves built once outside the K-loop and
`buffer_load(ptr, offsets)` (scalar base in SGPR) in place of a tensor of pointers cut them.

```python
offs_am = ttgl.arange(0, BLOCK_M, layout=am_layout)   # once, outside the K-loop
offs_a  = offs_am[:, None] * stride_am                 # invariant half, hoisted
for k in range(0, K, BLOCK_K):
    a = gl.amd.cdna4.buffer_load(a_ptr,                # scalar base -> SGPR, fewer per-lane VALU
                                 offs_a + (k + offs_ak[None, :]) * stride_ak)
# gfx942 downgrade: identical, spelled gl.amd.cdna3.buffer_load
```

Verify: IOPs down, address math gone from the loop in the ISA.

## LDS dedup (carve reuse from one allocation)

When LDS caps waves, reuse sub-views carved from ONE `allocate_shared_memory` with `smem.slice(...)` and a
right-sized padded layout lower LDS/workgroup.

```python
smem = ttgl.allocate_shared_memory(dtype, [2 * XBLOCK, K], layout)
buf0 = smem.slice(0, XBLOCK, dim=0)          # two reuse regions from one allocation
buf1 = smem.slice(XBLOCK, XBLOCK, dim=0)     # the 2*XBLOCK extent is dim 0, so slice dim 0
# gfx950: minimal padded layout to avoid over-allocation / bank conflicts.
# Takes the DOT-OPERAND layout (it reads `.parent` internally), the shape, and the dtype --
# passing the mfma layout raises AttributeError. May return None when no padding helps.
sh = gl.amd.cdna4.compute_efficient_padded_shared_layout(a_operand_layout, [XBLOCK, K], dtype)
```

**`gl.amd.cdna4.compute_efficient_padded_shared_layout` (3.8.0, gfx950, asserts MFMA v4) takes the
dot-operand layout — passing the mfma layout raises `AttributeError` — and may return `None`.** As an
async-copy destination a padded layout has extra conditions (`memory-path.md ### Async copy: smoke-test
before wiring (mandatory)`). gfx942: `gl.PaddedSharedLayout.with_identity_for` is the padded form
(`layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`); with 64 KiB LDS, dedup often decides
whether a second workgroup fits. Verify: LDS/workgroup down, waves/CU up.

## IR / asm acceptance signals

| Change | Confirm in IR/asm |
| --- | --- |
| slice-N + the AGPR hint | fewer in-loop `v_accvgpr_mov`; no `scratch_load` (spill) in the hot loop |
| slice-MN | four-region AC pattern (distributed `buffer_load_to_lds`, not a burst) |
| budget | observed VGPR (from `.amdgcn`/compile stats) <= 512, spill count = 0 |

## Reprofile signal

After slicing lands, reclassify: at large K the next bound may be L2/XCD traffic (`../workloads/gemm.md`).
