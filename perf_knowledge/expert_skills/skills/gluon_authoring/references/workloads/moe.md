# Mixture-of-Experts On The Gluon Path (gfx950 / gfx942)

> **Low-precision weights (MXFP4 / fp8 + E8M0) on gfx950: [`moe-lowprec.md`](moe-lowprec.md)** holds the short verified page — the levers measured on one a16w4 MoE, the `mfma_scaled` `k_width` table, counter facts, combine choice.

MoE is a GEMM with a data-dependent operand assignment: `gemm.md` owns the tile and the layer order;
this page owns what the routing adds (expert-set bound, routing metadata, bucketing, schedule, combine,
and which `gl` primitives exist). "Surveyed production" below = production Gluon MoE source read at
named sites (what production writes, not what is faster); "measured" = GPU-measured on gfx950 /
Triton 3.8.0. The upstream Gluon MoE example (`python/examples/gluon/05-moe-bmm1-fused-gather.py`)
exists on 3.8.0 only and targets Blackwell (`## What does not port from the upstream example`).

**gfx942 downgrade (recurring):** the list in `index.md`; here Axis 2 form 3 and fp8 route 1 need
`mfma_scaled`, which gfx942 does not have (no `cdna3.mfma_scaled` on Triton 3.8.0), and direct-to-LDS
async is 32-bit only; its LDS destination order has to match the source's contiguous dim
(`order=[1,0]` for a row-major tile — a mismatched order failed to lower on gfx942 and gfx950 alike,
compile-checked on 3.8.0).

## Top correctness gate

### `k_width` at a scaled site is a correctness gate, not a performance knob

**At an `mfma_scaled` site use operand `k_width=16` whenever a scale is driven (real E8M0) or the
formats are mixed (e4m3 x e2m1): in the shapes measured (Triton 3.8.0 / gfx950) other values compiled,
ran and returned O(1) relative error silently** (no assert, no NaN). The per-(shape, scales, format) table and the one-GPU check script
(`scripts/mfma_scaled_check.py`) live in
`moe-lowprec.md ### mfma_scaled k_width: use 16 whenever a scale is driven or formats are mixed`.

- No derivation is known here (from element width, "elements per lane" or the unscaled `k_width`
  formula), only the measured table: 16 is the value correct in every condition tested, while other values are
  correct only with `None` scales and one format (and at `[32,32,64]` e4m3 with real scales 4/8/16).
  That is why two shipping MoE families can use different `k_width` at the same instruction shape and
  why a value copied from an unscaled kernel breaks once scales are driven. Wrong values gave relative
  error of order 1 (about 0.9-1.2), the right one about 1e-5 (GPU-verified gfx950 / 3.8.0).
- **Verify each candidate numerically on a single tile before it enters a kernel.** "It compiled and
  ran" and "`compute_efficient_padded_shared_layout` still returns a layout" are not correctness
  evidence (`## Version gates`). Disassembly evidence: `../gluon/matrix-reference.md`.

## Type the kernel before advising it — three questions

Read the answers from the entry signature and the source, **not the file name** (in the production
source surveyed, the name recorded the shape the file was tuned for and did not match either fusion
boundary or band).

| Axis | Question | Answers seen |
| --- | --- | --- |
| fusion boundary | what goes in / comes out? | router-only · complete-fused · expert-GEMM-only |
| quantization form | how does a low-precision operand reach the MFMA? | bf16 · fp8 + software block scale · fp8/fp4 + hardware E8M0 scale operand |
| shape band | which M band, what selects the file? | decode (M ≲ few hundred) · middle (~10³) · prefill (~10⁴) |

### Axis 1 — where the fusion boundary falls

- **router-only**: logits in, `(topk_ids, topk_weights)` (+ route metadata) out; bf16 router
  projection, no scaled MFMA. Governed by `## Router and top-k: four implementation generations` and
  `## Which tier the routing kernel belongs on`.
- **complete-fused**: hidden + weights in, combined output out; still several *launches*
  (`### The launch boundary is the grid-level synchronization`). Majority shape.
- **expert-GEMM-only**: takes `topk_ids`/`topk_weights` as arguments; inherits a slot ordering it did
  not produce (`## Reduction order is part of the contract, not an implementation detail`).
- In a shape-dispatch table, **signature arity + fixed tail** distinguishes types sharing M values
  (5-arg tail with hidden/experts/route-slots/top-k = fused op; 4-arg hidden/experts/top-k = its
  router): two operators, not competing specializations.
- Weight dtype / TP degree (pack directory) and preshuffled-vs-standard weight layout are contracts
  that shape/dtype validation cannot catch — wrong choice = wrong numbers, not a refusal.

### Axis 2 — how the quantization reaches the matrix core

1. **bf16 throughout** (routers, small-M expert GEMMs).
2. **fp8 + software block scale** (dominant): fp32 scale per 128-wide group applied on the
   accumulator (`gl.fma`). gfx942: regular `cdna3.mfma` fp8 on FNUZ operands; fp4 has no matrix
   instruction in the CDNA3 ISA.
3. **Hardware block-scale operand driven** — zero surveyed files (production checkpoints carry fp32
   scales per 128; the instruction wants **E8M0 per 32**), but the mechanism works.

**`mfma_scaled` with `a_scale=None, b_scale=None` still emits the SCALED MFMA.** Production calls it
with `None` scales, `e4m3`, fresh zero accumulator — i.e. as a native fp8 op. The frontend splats a
constant scale; no canonicalizer folds it back to an unscaled instruction. A kernel that needs the
unscaled form has to bypass the wrapper. (Verified on Triton 3.8.0 / gfx950; plain `tl.dot(fp8)` and the
K=64/128 shapes: `../gluon/matrix-reference.md`.)

**Group-32 E8M0 scales + fp8/fp4 activations: drive them as real scales** (`get_mfma_scale_layout` ->
`v_mfma_scale_f32_16x16x128_f8f6f4`, verified correct; on Triton 3.8.0 `mfma_scaled` asserts its formats
are `e2m1` / `e4m3` / `e5m2`, so bf16 operands are rejected). Layout
contract: `../gluon/layout-reference.md ## The Scale Operand's Layout For mfma_scaled`.

**`gl.amd.cdna4.scaled_upcast` (3.8.0; absent in v3.7.0 / v3.6.0)** fuses fp8/fp4 upcast + block scale -> bf16 in one op
(native packed convert-with-scale on CDNA4; software-emulated on CDNA3). It takes the **raw E8M0
payload** (`int8`/`uint8`). Its **fp4 branch asserts one E8M0 byte per fp4 element** (scale shape ==
source shape); the expansion can stay register-only (a linear-layout broadcast, checked with
`convert_layout(..., assert_trivial=True)`); the fp8 branch has scale shape == source shape.

On fp4 with low-precision activations, `mfma_scaled` avoids the convert VALU that `scaled_upcast` -> bf16
`mfma` pays (reported bit-identical; on the dense-tile microbench measured on gfx950 the scaled route
ran ahead at prefill and tied at memory-bound decode). `scaled_upcast` is the route when something
forces a bf16 matmul (bf16 activations: `moe-lowprec.md`).

**Counter caliber:** busy cycles / `MfmaUtil` count scaled MFMA (rocprofv3 1.3.5, gfx950, measured); the
op counters split by operand format (`../hardware/bound-class-signals.md ## Rules (necessary-not-sufficient)`).

### Axis 3 — the shape band, and what actually selects a file

- Production dispatch = exact M points + fallback ranges, **exact wins**; ranges interleave with
  points, so M±1 around a tuned point can select a different file. Treat "which file runs at M" as a
  lookup.
- In the surveyed production families the band selected most mechanical choices: combine placement,
  row tile, sort or not, register-resident vs async-ring overlap, fp8 route, scheduler strategy.
- **Unit of transfer is (model, band), not the operator**: two production families solving the same
  fused fp8 MoE use disjoint overlap mechanisms. A technique transfers together with its band; a
  family's absence of a knob is not evidence the knob is inert.

## The three real causes of a non-uniform epilogue

In the surveyed MoE families each worktile owns a complete K loop; a separate reduce is over **route
slots**, not K. Causes: (1) ragged tail block per
expert (data-dependent mask), (2) scatter-back through the routing table, (3) top-k combine as a
variable-length reduction outside the matmul. (1)/(2) are absorbed by the metadata
(`## Routing metadata: precompute it once, look it up in O(1)`). **Upstream `triton_kernels/reduce.py`
enables its `row_idxs` bitmap path (and heavy-block subtiling) only for `is_cuda()` with CC >= 9 and
`K <= 32`** (source-checked): on gfx942/950 it silently takes the plain path. aiter's `moe/reduce.py`
has no such gate.

### Separate or fused combine: fuse when the shared-expert GEMM owns the output tile

- Separate: a `contributions` buffer `(tokens, route_slots, hidden)` summed by a later launch. Fused:
  the **shared expert's down-projection kernel** folds the routed slots into its accumulator (buffer
  has one slot fewer; host tell e.g. `8 if <cond on M> else 9`).
- **Criterion (surveyed production): production fuses when the shared-expert down GEMM already owns
  the same output tile** the reduce would write; otherwise re-tiling is needed and production reduces
  separately. Same-band siblings disagree, so take the switchability, not a lookup table. Where the
  GEMM owned the tile, the fused combine measured faster than a separate combine kernel at decode,
  middle and prefill token counts (gfx950 / Triton 3.8.0, one bf16 shared-expert down GEMM + top-k
  combine).
- **Fused and separate are bit-identical when the shared-expert output enters the sum at the same
  rounding point** — round it to bf16 before the add in the fused path, as the separate path (which
  stores it bf16) does. Measured on gfx950 / Triton 3.8.0, this held both for a per-add bf16 round trip
  (`acc = (acc + part.to(f32)).to(bf16).to(f32)`, one surveyed family) and for fp32 accumulate +
  round once (aiter's reduce); keeping the shared expert in fp32 in the fused path breaks identity
  under either contract. The slot order stays static in both. Under a tolerance oracle either rounding
  is acceptable (`## Reduction order is part of the contract, not an implementation detail`).

### The combine epilogue can dominate the kernel, and it is a write-amplification problem

- **A bf16 `atomic_add` combine writes `M·TOPK·H` values for `M·H` output**; at prefill it can be a large
  share of a MoE kernel (screen it below before sizing it).
- **Atomic vs plain store of the same bytes (gfx950 / Triton 3.8.0, unique addresses, so no
  contention):** a tie at decode-size grids — but the atomic form also needs a zero-fill dispatch — and
  several times slower once the epilogue is throughput-bound (mid / large-M prefill, more so from an
  MFMA-layout epilogue). The gap is atomicity itself, not contention. fp32 atomics are native
  (`*_atomic_add_f32`) and bf16 lowers to `*_atomic_pk_add_bf16` (2 elements/instr); whether the packed
  form beats fp32 depends on the memory semantics, the per-lane layout and the shape
  (`../tile-programming/memory-path.md ## Output / reduction path (atomics)`).
- **The gfx950 ISA has no wide (128-bit) atomic** (ISA database): 16 values = 4 `global_store_dwordx4` vs
  16 separate atomics.
- **The alternative that won where measured removes the atomic rather than tuning it.** Stage 2 stores
  **slot-major** (one contiguous region per route slot, plain store), then a second pass gathers each
  token's `TOPK` slots and reduces in registers (static order, so the rounding contract holds). **The
  atomic combine also lost at decode** in the graph-replayed a16w4 / a8w4 MoE ops measured (gfx950;
  `moe-lowprec.md ## Combine structure (down projection with scatter + router weights)`); A/B it where
  the zero-init dispatch is free (eager, or an already-zeroed output). Cost: an `M·TOPK·H` buffer ->
  blocked-N strips if memory-limited.
- A cheap screen: delete the epilogue, keep the rest, read the share (one compile). At decode the
  epilogue share was small where measured.

## Routing metadata: precompute it once, look it up in O(1)

Precomputed per-block table instead of a per-CTA group walk: an exclusive prefix sum over per-expert
block counts gives each expert's first block, and `(expert, block)` is packed into one int32 per entry
(if aiter is available: `moe_routing/expt_data.py`). Upstream builder
(`triton_kernels/tensor_details/ragged_tensor.py`) is two `@triton.jit` launches (memset + fill) — on
device, separate from the matmul; its `torch` twin is the oracle. The table linearization is the
L2/XCD scheduling lever (`gemm.md ### Worktile scheduling for load imbalance (LPT/SPT makespan)`).
Production prefill kernels compose an arithmetic pid remap **and** a table lookup on adjacent lines —
read both before reasoning about which expert a program touches.

## Bucketing: three shapes, and what selects one

| Shape | How counts are formed | Cost / needs |
| --- | --- | --- |
| one workgroup, LDS atomics | one CTA counts in LDS, prefix-sums in place | LDS `atomic_scatter_*` (**3.8.0**, `## Version gates`); one CTA's parallelism |
| global atomics + lazy publish | blocks bump global per-expert counters | any version; reservation order ≠ token order |
| communication-free count + scan | per-block counts, separate scan | second pass + scratch |
| no table: consumer recomputes routing | each computing CTA re-runs the top-k from keys | `O(E log E)` register work per consumer instead of a launch + table (decode) |

Pick by availability: LDS atomic surface present? token count large enough that one CTA serializes?
does any consumer fold in slot order (then atomic reservation order is a reproducibility bug)?
**Histogram over expert ids fed a sentinel id (one past the range) must size bins above it** (or pass
an explicit mask) — mixing conventions is silently wrong (correctness).

### The launch boundary is the grid-level synchronization

The surveyed production fused MoE (incl. aiter) uses **no spin loops, `atomic_cas` polling or
cooperative launch**: grid-level
ordering comes from launch boundaries, plus at most a non-spinning **last-arriver ticket** (aiter's
routing folds its scan into the histogram launch: release atomics + an `acq_rel` ticket, the last CTA
does the scan). Ways to save a launch: build a table, let the consumer recompute, use a last-arriver
ticket, or **concatenate independent phases behind one 1-D grid** (e.g. grid
`224 + M*groups + 1`, `if pid < 224: router K-split / elif: quantize / else: zero scratch + seed
queue`). Not the same as linearizing a grid against the small-axis-0 2D-grid slowdown
(`prologue.md`).
**When a payload has no reader inside this launch, `relaxed` integer atomics with no `scope=` are
correct** — the boundary orders the next launch; raising `sem` buys a writeback nobody waits for
(`collective.md ### Who issues the maintenance`). Adding a same-launch consumer silently invalidates
this.

## Router and top-k: four implementation generations

| Gen | Band | Mechanism | Cost |
| --- | --- | --- | --- |
| A sortless selection | all | max, write `-inf` at winner, repeat k (hierarchical over groups) | `O(k·E)` VALU, no cross-lane, no atomics; `num_warps=1` |
| B ballot + leader | decode (M ≲ 256) | 64-bit ballot, find-first-set, `v_readlane`; ballot + `bcnt`/`mbcnt` = group total + lane prefix | inline asm; `num_warps=1`, lane-pinned layout |
| C bitonic in registers | middle (M ≳ 1024) | butterfly over monotone int key, partner `gl.gather(keys, lane ^ (1 << step), 0)` | `O(log²)` shuffles while the gather is warp-local (else LDS fallback) |
| C′ ballot match-any | prefill | one ballot per expert-id bit (9 for 512 experts), AND-reduce, `bcnt`/`mbcnt` -> group size + rank | chunk must fit one wave |

- **B's payoff: ballot + `mbcnt` turns k per-token slot reservations into one atomic per wave.**
- C's key = expert id high bits + lane id low bits; run starts via `associative_scan` (max). The
  `gl.gather` lane shuffle silently falls back to LDS scratch sized by the whole source when not
  warp-local (Triton 3.8.0 / gfx950, compile-checked; `../gluon/layout-reference.md ## gl.gather — a lane exchange only when the layouts allow it`).
- C′ needs only contiguity + rank, not a sort (inferred intent).
- In the surveyed families, inline-asm generations appear only in decode files. The smallest router GEMM is an FMA-path dot
  (`gl.dot_fma`, VALU, invisible to the matrix counter).

### The monotone integer key has four incompatible spellings

`gl.sort` / `gl.topk` / `gl.cumsum` are absent, so encoders are hand-written. Four correct spellings in
one family that **cannot be mixed**:

| Spelling | Order-preserving map | Dead sentinel | Tie-break |
| --- | --- | --- | --- |
| bf16 xor | `~bits` if sign else `bits ^ 0x8000` | 0 | `(ordered << 9) \| (511 - e)` |
| bf16 biased | `bits + 0x8000` if pos else `0x10000 - bits` | 0 | `(ordered << 9) \| (511 - e)` |
| bf16 signed-magnitude int32 | `±(bits & 0x7fff)` | **int32 min** | `(ordered << 9) \| (511 - e)` |
| fp32 bit-level | `bits ^ 0x7FFFFFFF` if neg else `bits` | 0 | `(ordered & -512) \| (511 - e)` (mask, not shift) |

Encoder, decoder and sentinel are one contract: take all three from the same kernel (a 0-sentinel fed
to a signed key makes padding win; a shifted key into a masking decoder garbles the expert id).
Canonicalize ±0.

## Which tier the routing kernel belongs on

**`gl.sort`, `gl.topk`, `gl.cumsum` do not exist in Gluon** (tags v3.6.0, v3.7.0, v3.8.0, whole
`triton/experimental/gluon/language/` tree; checked against the source and the installed 3.8.0); `tl.sort`, `tl.topk`, `tl.bitonic_merge`, `tl.cumsum`
exist in plain Triton.

- **Top-k expressible with `tl.sort`/`tl.topk`** -> a plain Triton kernel covers it (upstream
  `triton_kernels/topk_details/` packs value+index into one int key).
- **Built from wave-level algorithms** (ballot, ffs, `v_readlane`, prefix popcount) -> Gluon, which pins
  the lane map: `num_warps=1` + `BlockedLayout([1], [64], [1], [0])` + `gl.static_assert(x.shape[0] == 64)`
  (`../gluon/inline-asm-reference.md ## Class 5 — cross-lane and wave-collective`). All surveyed
  production routers are on this branch.
- In-kernel expert-offset prefix sum = `gl.associative_scan` + combine fn (no cumsum). It takes **no
  `layout=` kwarg** (re-export of the plain builtin); **`gl.histogram` does take `layout=`** (signatures
  checked on 3.8.0).
- Fused router softmax: production sums the denominator over selected experts but takes the max over
  all, so NaN propagates like the reference.

## Load balance is a schedule, not a tile

**LPT (longest-first) needs unequal block cost.** In MoE kernels that compute their padded rows,
prefill blocks are close to equal-cost (padding programs were nearly free under graph replay, and
skipping MFMA on padded sub-tiles paid little on the gfx950 a8w4 MoE recorded), so ordering has little to
gain. A block-fill sweep (time vs fraction of real rows per block) is the cheap check before
implementing an ordering — it is not on record, so run it; if the kernel skips padded sub-tiles, or
block cost tracks content (tiny tiles, variable K), cost becomes fill-dependent and the sweep decides. Other gates: jobs
per CU (single wave -> order irrelevant; differs decode vs prefill); same-expert grouping shares the
weight read.

### Two reuse axes, and this page used to carry only one

- **Weight reuse**: permute pids so same-expert jobs land together (metadata cost, no registers).
- **Activation reuse**: one program holds an A block + scales in registers and sweeps `REUSE`
  consecutive N tiles of the down-projection (`for step in range(REUSE)`, A loaded outside); host
  picks REUSE by band (surveyed production values, source not in this repo — not a target: 1 below
  middle, single-digit in middle, 14 or 28 at prefill); grid divided by it. Competes for registers with weight reuse.
  The band shape held on a gfx950 / Triton 3.8.0 a16w4 down-projection sweep (flat at decode, a small value
  in the middle band, about half the N-tile count at prefill; a full sweep — one program per m-block —
  did not beat half), and single values can be outliers: sweep REUSE per band.
- High reuse is what makes an **async ring** worth building (enough MFMA per weight block to hide the
  next transfer) — production's async double-buffer appears only at its top band.
- **Do not skip the reuse sweep because the epilogue is combine-heavy.** On the gfx950 / Triton 3.8.0
  a16w4 down-projection sweep, the atomic-combine arm gained the most from activation reuse (fewer
  programs also cut atomic traffic), and reuse paid least at decode, where weight traffic dominates. An
  ablation (delete the epilogue / the weight load) names the bound; then sweep reuse anyway. (An earlier
  record of a flat sweep on a combine-bound kernel did not reproduce.)

### XCD-aware pid swizzle: the layer above longest-first

The remap gives each XCD a contiguous run of experts; surveyed production **stripes each job class
(routed vs shared) separately**. Production formula (bijective, no clamp, works on a runtime count):
`stripe = pid % STRIPES; new = stripe * (n // STRIPES) + min(stripe, n % STRIPES) + pid // STRIPES`.
STRIPES is tuned (one file uses twice the XCD count below a token threshold). Gate as `gemm.md ## Beyond hot loop (L2 / XCD locality)`.
Expect a gain only when expert-weight re-reads are a visible share of time: on one measured gfx950 a8w4
MoE an expert-to-XCD remap was within noise because a generic XCD swizzle already captured the reuse.
Production then often applies `pid = load(order_table + pid)` on the next line.

### Out-of-range programs: early return on a uniform predicate, or clamp

A metadata-driven MoE launches a constexpr worst-case grid and learns the live job count at runtime.
**Default: early-`return` on a workgroup-uniform predicate** (`program_id` + scalar loads — aiter's
gfx950 a16w4 MoE does this). On that kernel (gfx950 / Triton 3.8.0, async copies and `wait_group` in the
K loop) the early return passed the task oracle and was never slower than clamping. Clamp the
out-of-range pid to the static bound and mask its work (one surveyed family:
`pid = gl.where(pid < ctas, remapped, max_ctas)`) only when uniformity cannot be shown, and then also
skip the K loop (zero trips): masking only the store makes every padding program do full work, which
measured slower wherever the worst-case grid carried padding. Reason for the clamp: `gl.barrier()` is a whole-workgroup
rendezvous and needs a fixed participant set; an exit predicate built from a runtime count must be
proven workgroup-uniform before an early return next to a barrier is safe, and the clamp removes that
proof at the cost of one fully masked program. Not a compiler requirement: a runtime early `return`
before `gl.barrier()` compiles (3.8.0, gfx950, compile-checked), and a predicate on `program_id` alone
is uniform.

## Getting fp8 into the matrix core: four routes, all in production

1. **`mfma_scaled` with both scales `None` + fp32 descale per K group** (dominant; K block ==
   quant group == MFMA K, one `gl.fma` per trip; if aiter is available: its gfx950
   `gemm_a8w8_blockscale.py`). gfx942: `cdna3.mfma` fp8 FNUZ, same structure.
2. **Upcast to bf16 + regular MFMA** (exact; moves the kernel onto the generic counter).
3. **Upcast to fp16 + regular MFMA.**
4. **Three-MFMA exact fp8 (Karatsuba)**: `x_hi = (x + 1536) - 1536` (1536 = 1.5·2^10, fp16 ULP 1 —
   valid because e4m3 max 448 < 512), `x_lo = (x - x_hi)*512`; `high = a_hi·b_hi`, `low = a_lo·b_lo`,
   `cross = (a_hi+a_lo)(b_hi+b_lo) - high - low`; only `high` promoted to int32; recombine with weights
   1, 1/512, 1/512² in fp64. Re-derive for `e4m3fnuz`.

Routes 2-4 coexist in one family at adjacent bands: the fp8 route is a shape-tuned knob, not a dtype
property. The fp8->wide upcast is per-element VALU the FLOP roofline does not show
(`../hardware/optimization-gotchas.md`).

## Reduction order is part of the contract, not an implementation detail

- **Execution order nondeterministic, reduction order static.** Slots reserved with `relaxed` int32
  atomics (the surveyed production families use **no float atomics**); the combine walks slots in a compile-time unrolled
  range, so the sum order is fixed by route slot index. Shared-expert contribution added last.
- **Match the reference's rounding points only when the oracle is bit-exact.** If a bit-exact reference
  rounds per add, keep the bf16 round trip per add (`acc = (acc + part.to(f32)).to(bf16).to(f32)`) —
  deleting it changed a large share of output elements (gfx950 / Triton 3.8.0). It is not needed for
  determinism (static order alone gave identical reruns) or for fused/separate identity, and against an
  fp64 reference fp32 accumulate + round once (aiter's reduce) was the more accurate contract.
- **`enable_fp_fusion=False`** on launches aligning bit-exactly (contraction rounds differently).
- Decide first: *any correct order* or *bit-exact vs reference*.
- **Reference ambiguity:** a library can ship two same-signature quant ops differing in scale
  rounding (one steps at `f >= 0.75`, the other at `f > 0.5`, differing only for `f ∈ (0.5, 0.75]`) —
  it looks exactly like a reduction-order bug. Resolve the callee from the dispatch path and record
  it with the tolerance before debugging numerics.

## Gather: separate the index layer from the transport layer

- **Index layer ports:** out-of-range tokens get **sentinel row index = row count** (OOB access
  discards it); optional index-vector reuse across N is off by default upstream (per-shape flag).
  Recompute threads-per-warp for wave64.
- `make_tensor_descriptor` is rewritten to pointer arithmetic on CDNA (no TDM): gather with per-row
  pointer loads staged into LDS.
- **In the surveyed families pointer tables carry all the indirection:** expert weight bases, gathered rows and scatter
  destinations arrive as integers and become pointers via `gl.pointer_type`
  (`../gluon/imports-and-launching.md ## Pointers as data (gl.pointer_type)`). Output store =
  ordinary masked store at `slice_offset + row_offsets`.

## Overlap in a MoE loop: what this archetype actually reaches for

Numbered for reference, not an order; the signal decides.

1. **Register-resident prefetch is the most common form in the surveyed families** (weight-streaming
   decode loops). Under VGPR pressure — wide gate-up tiles — a VGPR-free async LDS ring won on one gfx950 MoE
   (Triton 3.8.0) even without extra activation reuse, while register prefetch paid on a narrower tile of the
   same family; A/B both: a Python tuple of in-flight bundles (A tile, both
   weight tiles, both scales) rotated by a statically unrolled loop (issue loads for `i+depth`, MFMA
   for `i`); depth pinned by `gl.static_assert(K_groups % depth == 0)` so no remainder loop; depth a
   launch arg chosen by band (deep at small M).
2. **Async-copy ring** (`async_copy` + `commit_group`/`wait_group`; gfx942 sync-staged): production
   pairs it with high activation reuse (`### Two reuse axes, and this page used to carry only one`).
3. **`gl.amd.warp_pipeline_stage` at `num_warps=1` has no phase shift** (Gate 0 `num_warps >= 8`; per the
   pass source the group size is `warpSize*4` threads, independent of `num_warps`): the surveyed MoE
   sites are 1-warp, and at `num_warps=1` the compiled loop keeps the borders' `sched_barrier` and
   `s_setprio` but no `s_barrier` (Triton 3.8.0 / gfx950, compile-checked) — a gain there is not a
   cross-wave phase effect (`../tile-programming/warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`).

Cross-wave LDS reads after `wait_group` need a barrier; on Triton 3.8.0 / gfx950 the compiler inserts
it (`s_barrier` after the wait, compile-checked;
`../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier placement that differs between them`).

## Version gates

Language-level (not MoE-specific); per-build check: `../../scripts/probe_levers.py --all`; marker /
pipeline symbols: `../gluon/pipeline/marker-and-version-gates.md`.

| Capability | 3.8.0 | 3.7.x | 3.6.0 |
| --- | --- | --- | --- |
| `gl.gather`, `gl.pointer_type`, `gl.associative_scan` | ✓ | ✓ | ✓ |
| shared-descriptor `.gather` / `.scatter` | ✓ | ✓ | absent |
| shared-descriptor `.atomic_scatter_add/_max/_min/_and/_or/_xor/_xchg` | ✓ | absent | absent |
| `gl.amd.cdna4.scaled_upcast` | ✓ | absent | absent |
| `gl.amd.cdna4.compute_efficient_padded_shared_layout` | ✓ | absent | absent |
| `gl.amd.slice` (register sub-tile, layout preserved) | ✓ | absent | absent |
| `gl.sort` / `gl.topk` / `gl.cumsum` | absent | absent | absent |

Table checked against the v3.6.0 / v3.7.0 / v3.8.0 tagged sources (later 3.7 patch releases not checked).

**`compute_efficient_padded_shared_layout` returns `None` silently** (asserts an MFMA v4 parent, so
unavailable on gfx942). `None` when `k_width ∉ {4,8,16}`, element bit-width ∉ {4,8,16}, or an
unhandled instr-shape/`k_width` combination — checked on 3.8.0 at `[16,16,128]`: a **short non-K tile
(16 rows) returns `None` for every legal `k_width`**, so `None` is the normal outcome in the **decode
band**; at `[32,32,64]` every legal `k_width` needs a non-K extent >= 64 for both operands (thresholds move
with K and `is_k_contig`; table in the contract below). Gate on the return value per operand
(`../gluon/layout-reference.md ### Contract points — places this goes wrong quietly`); hand-built
`gl.PaddedSharedLayout` stays valid. Never pick `k_width` for helper availability — at a scaled site it
is a correctness choice
(`### \`k_width\` at a scaled site is a correctness gate, not a performance knob`).

**`atomic_scatter_*`** (`../gluon/smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family`):
- methods on a shared-memory descriptor (LDS atomics, workgroup scope), lowered on 3.8.0 for gfx942
  and gfx950 (compile-checked); no pointer-form `smem.atomic_add` exists;
- **`add` and `xchg` accept int and float; `max`/`min`/`and`/`or`/`xor` are integer-only** (compile
  error otherwise; float `add` / `xchg` lower and float `max` fails on both arches, 3.8.0) -> float max goes through a monotone int key; `max`/`min` are signedness-aware.

## Deciding the bound before sizing a prize

### First branch: is this phase bound by the host, not by the device?

- Host-dispatch bound -> only the number of dispatches matters (device-side ceiling = 0). Count by
  dependency edges, not source order. Discriminator: time flat across a shape sweep
  (`../method/benchmark-hygiene.md ## Measurement basis + benchmark-artifact pitfalls`).
- Per-node cost = slope of time vs N inserted empty graph nodes; an optimization that adds a kernel must
  beat it at decode, else fold the reduce into a neighbour.

**Byte model knobs** (`kernel_workflow/scripts/kernel_tools/hw_budget.py --workload moe`):
- `E_touched` default `min(E, top_k*M)` is an upper bound: exact at M=1 (a token's top-k experts are
  distinct), over-counting as routing collisions grow with M (and with routing skew). Over-counting inflates
  the bytes, so the HBM floor can land *above* the measured time — read that breach as the model not
  applying, and pass a measured `--shapes E_touched=...` or `--measured-dram-mb`.
- `stage`: 1 = w1 gate+up `2*I*H` (default), 2 = w2 `I*H`, 0 = fused `3*I*H` — 1.5x error either way;
  budget per launch.
- `scale_overhead`: E8M0 bytes per element in `perf_knowledge/hardware/cdna4_mi350/fp4_fp6_microscaling.md`;
  production fp32/128 scales differ — the checkpoint wins.
- Weight dedup can hit optimum (measured ratio 1.0000) and still trail the reference -> next question
  is achieved bandwidth, not bytes.

### The mnemonic histogram: a cheap issue-slot ceiling

A mnemonic histogram of the asm bounds the issue-slot saving of any change that removes one
instruction class (class = 1.2 % of issue -> ≤1.2 % from issue slots on an issue-bound kernel; a
removed wait or a shortened dependency chain can gain more through latency); read the whole histogram. Filter `.` directives, `;` comments, labels,
`s_nop`. It is not an identity check (order-independent).

**`CompiledKernel.n_regs` raises `AttributeError` before the first launch** (set only in
`_init_handles()`) and afterwards is a HIP runtime query. Use the static figures: regex
`\.vgpr_count:` (and `.sgpr_count`, `.vgpr_spill_count`, `.sgpr_spill_count`, `ScratchSize`,
`codeLenInByte`) from `kernel.asm['amdgcn']`. Verified on Triton 3.8.0 (compile-only). Its `; Occupancy`
line is the register term only (no LDS term); compute occupancy as `min(register term, LDS term)` with
`metadata.shared` (`../hardware/planning-constants.md ## The emitted ; Occupancy: N is a register-term answer`).

## Tile choice: tokens-per-expert is the bucketing variable

Tokens-per-expert selects `BLOCK_M` (~16 decode, 32/64 middle, per-launch at prefill; one program per
token at the smallest M). Idle MFMA rows at decode cost little while the tile is memory bound; only
`gl.dot_fma` has no M waste
(`../gluon/matrix-reference.md ## When the matrix core is the wrong instrument (gl.dot_fma)`).

### `BLOCK_M` at prefill is a register-allocation decision first, a padding decision second

`BLOCK_M` and `num_warps` are coupled: sweep the pair and read `.vgpr_spill_count`. At `BLOCK_M >= 128`
more warps kept the tile spill-free without halving `BLOCK_M` (which re-reads the weight); at a
routing-fixed `BLOCK_M=64` the measured best ran fewer warps, so the direction depends on the band; accumulator = `BLOCK_M*BLOCK_N / (64*num_warps)` VGPRs/lane
(`../tile-programming/slicing.md`). A prefill `BLOCK_M=128` tile that spilled at 4 warps ran spill-free
at 8; tuned AMD fused-MoE tables pair `BLOCK_M` >= 128 with 8 warps (**Overrides KB**: the "num_warps 4,
not 8" advice in `perf_knowledge/operators/fused_moe_grouped_gemm/backends/triton.md`, `.../tuning.md` and
`perf_knowledge/operators/shared_expert_fusion/backends/triton.md`; verified by spill counts and the tuned
tables: couple `num_warps` to `BLOCK_M` and read `.vgpr_spill_count`).
On plain Triton the MFMA shape is part of that register budget: on gfx950 / Triton 3.8.0 the default
`matrix_instr_nonkdim=0` picks `32x32x16` for `BLOCK_M >= 32`, and a 256x256x64 bf16 GEMM tile at 8 warps
spilled under it but compiled spill-free with `matrix_instr_nonkdim=16` (`16x16x32`). Read the spill count
with both `nonkdim` values before attributing an 8-warp spill to `num_warps`.

#### What raising `num_warps` does and does not buy you

Convert waves/SIMD to workgroups/CU (a workgroup is indivisible;
`../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it`):

```text
wg_per_CU = min( waves_per_SIMD_by_VGPR * simd_per_cu // num_warps ,
                 LDS_per_CU // roundup(lds_bytes_per_wg, 1280 on gfx950 / 512 on gfx942) )
```

- At `num_warps=8` only VGPR thresholds 64 / 80 / 128 / 256 change `wg/CU` (4 / 3 / 2 / 1;
  `$KT/amd_occupancy.py --vgpr N`): a register cut below the next threshold changes nothing in `wg/CU`.
- A `vgpr_count` ceiling check cannot fail at `num_warps >= 8` (a spilling kernel still reports
  `vgpr_count <= 256`) — read `.vgpr_spill_count` in the same row
  (`../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it`).
- A computed occupancy loss is a strong warning, not a proof of slowdown (lower-occupancy builds have
  won where they bought registers or ILP — A/B it); a computed gain is not a predicted win.
- The binding resource (VGPR vs LDS) is per (kernel, config, source version); re-check after edits.

### Padding's cost changes sign across the bands

The spill term usually dominates padding, so it is priced first. Decode: padding costs latency;
surveyed production removes M tiling
(one program per token). Middle: ignorable (weight bytes dominate). Prefill: wasted matrix work;
production reserves a worst-case job capacity (one padded block per expert) above a token threshold.

### Three different things are called "padding" here

1. Route slots padded to a power of two so a `gl.split` reduction tree is expressible (zero weight).
2. Each expert's token run padded to the row tile so every block has one expert (no in-loop boundary
   tests; at most one partial block per expert).
3. Padded route entries: either point at **token 0** + zero row scale + masked store, or a **sentinel
   expert id one past the range** + mask (histograms must size above it). Pick one per kernel and check
   every consumer.

## Implementation-level facts that decide rounds, and are not in the layer ladder

1. **Block-scaled expert GEMM puts VALU between MFMAs; the fork-era GEMM throughput-pairing scheduler
   rungs fail there** (`../hardware/optimization-gotchas.md`): rebuilt from the gfx950-tutorial-v0.1 fork
   (in-tree LLIR scheduler), the pass gave invalid IR on attention and an assert or a large slowdown on
   block-scale, and the AGPR accumulator hint regressed — treat those rungs as unavailable on such loops.
   Stock 3.8.0 has neither. Upstream MFMA scheduling (`schedule_hint="mfma-schedule"`, Triton main from
   gfx950-tutorial-v3.1) is a different pass that compiles these loops — A/B it there. The per-compile `llvm_fn_attrs` strategy is separate and production selects it
   per launch and per M regime (`../tile-programming/instruction-scheduling.md`).
2. **Scaled-MFMA counters:** busy cycles / `MfmaUtil` count it; op counters split by format (fp8 x fp4
   -> `MOPS_F8`, fp4 x fp4 -> `MOPS_F6F4`). Use `../hardware/bound-class-signals.md` and state the route.
3. Surveyed production runs `num_warps=1` for router/gate launches, 2-8 for down-projection and slot
   reduce (per launch); with a 2D/3D grid see the hazard in `prologue.md`.
4. Inline asm here: motion boundaries, scalarizing warp-uniform values, wave collectives, protocol
   blocks — classify per `../gluon/inline-asm-reference.md ## Classifying a site: mechanism × intent`;
   a hand-written atomic does not inherit the cache maintenance that `release` / `acq_rel` language
   atomics emit (`buffer_wbl2` / `buffer_inv`; Triton 3.8.0 / gfx950, compile-checked).

## What does not port from the upstream example

Generic Blackwell -> CDNA primitive map: `perf_knowledge/languages/hipkittens/primitives.md`; the Gluon
spellings:

| Upstream (Blackwell) | gfx950 replacement |
| --- | --- |
| tensor-descriptor async gather, multicast | pointer-tensor per-row load + explicit LDS staging |
| `mbarrier` ring with phase tracking | `gl.barrier` + hand-authored multi-buffering (`../gluon/pipeline-reference.md`) |
| `tcgen05` MMA into tensor memory | MFMA into VGPR accumulators (register budget) |
| `gl.warp_specialize` partitions | one persistent loop with software pipelining |
| 2-CTA cluster layouts | nothing (CDNA has no CTA clusters); delete the axis |
| packed-f32x2 intrinsics | CDNA packed-VALU equivalents |

What ports: routing metadata layout, packed O(1) schedule lookup, sentinel-index gather, index reuse
across N, per-slice tile selection, pointer-tensor epilogue.
