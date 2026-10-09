# Slicing / Register Layer (gfx950)

Read this for the slicing layer (after pipeline, before beyond-hot-loop). Goal:
fit the tile + prefetch + accumulator inside the register budget (the **joint
VGPR+AGPR** file, see Budget below) and the occupancy budget, with zero hot-loop
spill, while keeping the AC/LR/DOT overlap.

## Sweep granularity first (cheapest knob, P4)

Before spending rounds on slicing / RA hints / compiler co-design, **sweep the
tile / block / split sizes** -- it is the cheapest knob and often the largest
lever. A smaller tile that *fits* (no spill, higher occupancy) usually beats a
larger tile that spills or caps occupancy. Only after the granularity is chosen
do slicing and the compiler knobs earn their cost. [On one attention-bwd kernel,
halving the key tile removed all spills and ran markedly faster than the larger,
spilling tile; gfx950/MI350, one build.]

## Why slicing (the over-unroll spill lesson)

Naive unroll-x2 plus deep pipelining pushes `R_total` past the arch ceiling -> spills
(`scratch_load` + `s_waitcnt`) -> matrix-issue efficiency **collapses**: once the hot loop is
spilling, the interleave you paid for is gone and the reading looks like a much earlier layer
failed. The fix is not to revert the overlap but to **slice the output tile** so each half needs
fewer registers. Read a spill count > 0 in the hot loop as a hard gate, not as a cost to trade.

## Slice-N

Split the N dimension: keep `smemB_left` / `smemB_right`, half-N accumulators,
and staggered B loads; still unroll-x2. Peak register pressure drops (e.g.
~512 -> ~448). Pair with the **AGPR accumulator hint** so the accumulator stays in AGPR and
in-loop `v_accvgpr_mov` copies shrink — on 3.8.0 that is
`llvm_fn_attrs="amdgpu-agpr-alloc=256,amdgpu-mfma-vgpr-form=0"`
(`compiler-contract.md ## What upstream 3.8.0 actually gives you`; the attribute mechanism and its
assembly-diff verification are in `llvm-fn-attrs.md`). Downgrade: `llvm_fn_attrs` does not exist
before 3.8.0, so on 3.6/3.7 the AGPR rung is unreachable per compile.

## Slice-MN

Split both M and N into 2x2 quadrants; load order `B_left -> A_top -> A_bot ->
B_right`, 4 regions per K-tile. This spreads the loads (e.g. 16 `buffer_load`s
over ~1000 cycles becomes 4 loads/region over ~1500 cycles of MFMA), relieving
TCP/FIFO stalls at large K, and drops the register budget further (~448 -> ~384).
Upstream has no post-assembly peephole to pair with this, so the residual scalar work in
the loop has to be reduced at source level instead
(`compiler-contract.md ## What upstream 3.8.0 actually gives you`).

## Budget to compute first

```text
R_acc      ~= BM * BN * acc_dwords * sharing / (num_warps * 64)
R_total     = R_acc + R_operand + R_prefetch + R_offsets + R_side + R_epilogue + R_compiler_tmp
constraint  : R_total <= 512  (zero hot-loop spill)
```

If `R_total > 512` after the pipeline layer, slice before adding any more
prefetch depth. Slicing N halves `R_acc` and `R_operand` for the N half;
slicing M+N quarters the accumulator footprint per region.

### Joint VGPR + AGPR budget + capacity walls (P5)

The register wall is the **combined** file, not VGPR alone: on CDNA the budget is
`ArchVGPR + AGPR` sharing **one 512-entry-per-SIMD file** on both gfx950 and gfx942
(`perf_knowledge/hardware/data/hw_constants.json` `vgpr_file_per_simd`, and `next_free_vgpr` in a
compiled kernel reports that
combined figure), and MFMA accumulators may
live in AGPR. So **re-splitting** values between VGPR and AGPR does not add
capacity -- it moves the same demand. When the combined budget overflows, the fix
is **fewer live values**, not a better split.

Every live value must be re-sourced from one of **registers / LDS / global**, and
each tier is a hard wall:

- registers: the joint VGPR+AGPR budget above;
- LDS: the per-CU shared capacity (160 KiB on gfx950; **gfx942 downgrade: 64 KiB**, so a
  remat/streaming plan sized on gfx950 rarely fits there), already partly spent on buffers;
- global: cheap to store, expensive to re-read.

So **remat / streaming** of a loop-invariant operand helps only if a *cheaper*
tier has room for it. [On one attention-bwd kernel, streaming the loop-invariant
operands was blocked because they did not fit the free LDS, so the register wall
could not be relieved that way; gfx950/MI350, one build.]

### Occupancy budget (P8)

> The overlap/occupancy/ILP **tri-lemma** derived here is indexed in the lever-applicability
> single source of truth (`../hardware/bound-class-signals.md ## Lever gating laws`).

Compute occupancy next to the register budget -- for memory/latency-bound kernels
**waves/CU is often the dominant lever**:

```text
occupancy (waves/CU) = f(LDS_per_wg, VGPR_per_wg)   # the resource that caps waves wins
```

If the kernel is occupancy-bound, attack the **wave-capping resource first**
(usually LDS: dedup / reuse buffers to drop LDS/wg and raise waves/CU) **before**
adding pipeline depth or slicing. [On one attention-bwd kernel, deduplicating the
q/dO shared buffers dropped LDS/wg enough to go from 1 to 2 workgroups/CU and was
the single largest win; gfx950/MI350, one build.]

When waves are VGPR/LDS-capped, **software prefetch (register prefetch OR LDS
double-buffer) regresses** — it adds the wave-capping resource and drops waves/CU.
Check waves/CU before adding any prefetch depth: rungs 1 and 2 of the overlap order
(`pipeline.md ## Where the overlap comes from, and it is not the same question per tier`) are
first in *order*, not exempt from this gate. This is the standing prior for
**occupancy-latency-hidden** access patterns (e.g. gather / decode where the
gathered operand is L2-resident and its latency is already hidden by occupancy):
the lever is more occupancy, not more prefetch.

#### The overlap / occupancy / ILP tri-lemma

Generalize the prefetch-regression prior into the governing tension for **any**
instruction-level overlap on an occupancy-latency-hidden architecture:

> **Cross-stage overlap costs VGPR; 2+ waves/SIMD occupancy needs few VGPR;
> deep-unroll ILP costs LDS buffers.** On a kernel whose latency is already hidden
> by occupancy you cannot have all three — buying overlap or ILP by spending
> VGPR/LDS lowers waves/CU, and the occupancy loss usually exceeds the overlap gain.

```text
tri-lemma: { cross-stage overlap (VGPR) , 2+ waves/SIMD (<= VGPR cap) , deep-unroll ILP (>= N LDS buffers) }  -- pick two
```

Before adopting any register-buffered pipeline (cross-iteration software pipeline,
multi-stage GEMM-softmax overlap, register prefetch), **predict the post-change
VGPR/LDS and the resulting waves/CU**; if occupancy drops a tier, expect a net
regression and verify against the no-overlap baseline first.

> **Dual-stream ping-pong sharing LDS/VGPR** (two streams share one buffer set) needs the
> multi-stream extension of this budget — the peak-non-stacking / stagger rule and the per-workgroup
> LDS formula. **That write-up is not in this pack.** It was cited from the LLVM co-design handbook,
> which no longer carries it, and no other page picked it up; treat the rule as unwritten rather
> than as somewhere you have not looked yet (`../method/triage.md`). Until it is restored,
> derive the two-stream peak from the single-stream budget above and **state that you derived it**.

## Slice recipe (ttgl.amd.slice)

`ttgl.amd.slice` is a **register-only view** — it preserves the layout and moves no
data across threads, so it is the primitive that cuts the live-VGPR set to cross an
occupancy cliff (slice-N / slice-MN). For K, stream sub-tiles from a shared descriptor.
**Version gate: `slice` enters `gl.amd.__all__` on 3.8.0 and is absent on 3.6.0 /
3.7.0 / 3.7.1** — on an older build the descriptor-side `.slice(...)` form in the
second example below is the one that exists.

```python
# register view: cut the M/N register tile — no data movement, layout preserved
a_sub = ttgl.amd.slice(a_reg, [BLOCK_M, SUBTILE_K], [0, k0])
# or bound live VGPRs by streaming K sub-tiles from a shared desc into MFMA:
for k0 in range(0, BLOCK_K, SUBTILE_K):
    a_sub = a_smem.slice(k0, SUBTILE_K, dim=1).load(a_layout)   # one sub-tile resident at a time
```

Extent/offset must align to the layout's CTA tiling. Ref: upstream
`triton/experimental/gluon/language/amd/slice.py`; `lds_subtile_load` in the AMD
gfx1250 f16-gemm example. Verify: spill -> 0 / waves/CU up (`## Occupancy budget (P8)`).

## Hoist address arithmetic (cut IOPs)

When VALU integer/address IOPs (`2.1.1`) dominate FLOPs (`2.1.0`), the per-lane index
math is being recomputed in the loop. Build the loop-invariant offset halves **once**
outside the K-loop and advance only the varying axis; prefer `buffer_load(ptr, offsets)`
(scalar base in SGPR + tensor offsets) over a tensor-of-pointers `tl.load`.

```python
offs_am = ttgl.arange(0, BLOCK_M, layout=am_layout)   # once, outside the K-loop
offs_a  = offs_am[:, None] * stride_am                 # invariant half, hoisted
for k in range(0, K, BLOCK_K):
    a = gl.amd.cdna4.buffer_load(a_ptr,                # scalar base -> SGPR, fewer per-lane VALU
                                 offs_a + (k + offs_ak[None, :]) * stride_ak)
# gfx942 downgrade: identical, spelled gl.amd.cdna3.buffer_load
```

Ref: `test_amd_mfma` offset construction + `cdna3/cdna4.buffer_load` scalar-base contract
(upstream `python/test/gluon/test_core.py`). Side benefit: the dead index math frees
VGPRs. Verify: IOPs down, scalar% down, isa shows the address math gone.

## LDS dedup (carve reuse from one allocation)

When LDS is the wave-capping resource (SPI limiter), carve reuse sub-views out of ONE
`allocate_shared_memory` with `smem.slice(...)` instead of allocating twice, and
right-size the padded layout so LDS is not over-allocated.

```python
smem = ttgl.allocate_shared_memory(dtype, [2 * XBLOCK, K], layout)
buf0 = smem.slice(0, XBLOCK, dim=0)          # two reuse regions from one allocation
buf1 = smem.slice(XBLOCK, XBLOCK, dim=0)     # the 2*XBLOCK extent is dim 0, so slice dim 0
# gfx950: minimal padded layout to avoid over-allocation / bank conflicts.
# Takes the DOT-OPERAND layout (it reads `.parent` internally), the shape, and the dtype --
# passing the mfma layout raises AttributeError. May return None when no padding helps.
sh = gl.amd.cdna4.compute_efficient_padded_shared_layout(a_operand_layout, [XBLOCK, K], dtype)
```

Ref: `smem.slice` aliasing (upstream `test_consan.py`) +
`cdna4.compute_efficient_padded_shared_layout` (gfx950; asserts MFMA v4, and the constructor itself
is **3.8.0-only** — probe the build before writing it in). A padded layout is for a buffer filled by
`smem.store`: as an **async-copy destination** it has failed to translate on both padded
constructors, so a `buffer_load_to_shared` target stays swizzled
(`memory-path.md ### Async copy: smoke-test before wiring (mandatory)`). **gfx942 downgrade:** no
`compute_efficient_padded_shared_layout` (MFMA v3) — use `gl.PaddedSharedLayout.with_identity_for`
(`layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`); with 64 KiB LDS the dedup is
usually what decides whether a second workgroup fits at all. Verify: LDS/workgroup down, waves/CU up.

## IR / asm acceptance signals

| Change | Confirm in IR/asm |
| --- | --- |
| slice-N + the AGPR hint | fewer in-loop `v_accvgpr_mov`; no `scratch_load` (spill) in the hot loop |
| slice-MN | four-region AC pattern (distributed `buffer_load_to_lds`, not a burst) |
| budget | observed VGPR (from `.amdgcn`/compile stats) <= 512, spill count = 0 |

## Reprofile signal

Slicing should recover MFMA efficiency lost to spills and keep climbing toward
the budget target; measure it at the pinned boundary rather than carrying a figure.
After it lands, reclassify — at large K the next bound may be L2/XCD traffic
(`beyond hot loop`, see `../workloads/gemm.md`).
