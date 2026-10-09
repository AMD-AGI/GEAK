# Workload: Reduction / Elementwise / Vector

Memory-bound tile patterns. Often the escalation gate says **stay plain** here —
explicit Gluon layout rarely beats the compiler unless there is a concrete
coalescing / bank-conflict / vector-width mechanism.

## Logical tiles

- Elementwise / vector: a 1D or 2D `BlockedLayout` tile per program; one
  index/mask/load-store path.
- Reduction / norm / softmax: one accumulator or reduction-axis layout; partial
  sums per block, then a combine stage.

## Usual bound class

**memory-bound** almost always (`intensity << ridge`). The metric is a byte one rather than an
MFMA one — but **which** byte metric is a question this archetype has to settle before quoting
anything, and more often than the others, because its footprints are the smallest in the set.

- **Settle the level the bytes actually came from, first.** A reduction or a fused norm at decode
  shapes can have a working set of tens of kilobytes. Anything that fits inside the memory-side
  last level never reached DRAM, so "% of HBM peak" is not the denominator — the floor is
  **inapplicable, not merely imprecise**, and a kernel can measure faster than its own HBM floor
  without being superhuman. `Verify:` price the footprint (`--footprint-mb`) against the
  memory-side cache before quoting a fraction of peak, per
  `../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`.
  The same check, with the GEMM side's full reasoning, is in
  `gemm.md ## Usual bound class`.
- **Once it does reach DRAM**, the metric is achieved HBM bandwidth against an **in-shape probed**
  ceiling, plus effective in-flight bytes against the 32 KiB/CU cap. The datasheet figure is the
  wrong denominator for a mixed read/write stream at this archetype's program counts; probe it
  rather than quoting it (`../hardware/roofline-models.md`). Record the result with its
  `numerator_basis` and `denominator_basis` — a datasheet denominator (from
  `perf_knowledge/hardware/data/sku.json`) ranks only; a probed one may gate or close.
- **At small shapes neither may bind.** A byte account still returns a number when the kernel is
  actually gated on launch and dispatch, and then the correct action depends on the production
  boundary rather than on the bytes: fusing dispatches pays only while the launch is exposed, and
  is neutral-to-negative once a graph has amortized it
  (`../hardware/roofline-models.md ## Kernel time decomposition`, the launch-fusion law). Settle
  which boundary is in production before opening a round at these shapes — the two answers point
  opposite ways.

## What helps (in order)

1. coalescing: contiguous dim aligned with `size_per_thread` on the fast axis;
   wide (128-bit) aligned transactions.
2. vector width: maximize bytes per load/store; minimize mask cost on the hot
   path.
3. minimize LDS: only stage through LDS when the tile is actually reused; a
   single-use tile through LDS is just an extra read/write.
4. fuse producer/consumer to remove a temporary tensor / extra launch.

## Group-wise quantization: the three-stage shape

The reduction that feeds a low-precision GEMM lives here, not on the matrix pages.
`../tile-programming/low-precision.md` owns the **consumption** of a scale tensor — staging it so it
reaches the matrix op in time. It never produces one. This section owns the production side (group
absmax -> scale -> cast) and the dequant epilogue that closes the round trip. **Where** the
quantization runs relative to the GEMM is a dispatch decision with a known trap, and it belongs to
neither page: see the quantize-once vs fused-per-tile bullet in
`gemm.md ## Small-M (decode / GEMV) regime — memory/occupancy-bound`.

The invariant that governs the tile: **`BLOCK % GROUP == 0`**. A tile never holds a fraction of a
group, because a group's scale cannot be finished by two programs. A block size that is not a
multiple of the group is not handled by shrinking the group; the ragged edge is masked up to a whole
number of groups.

## Stage 1 — the group absmax reduction

The canonical form is not a separate kernel. Reshape the tile so the group becomes its own axis —
`[rows, BLOCK/GROUP, GROUP]` — and reduce the innermost axis with `keep_dims=True`, which leaves the
result already shaped for the broadcast back.

Two masks, and both are required for different reasons:

- **On the absmax input, the masked identity is a negative finite value, not `-inf` and not zero.**
  Padding lanes must be unable to win the maximum; feeding them zero makes a fully-padded group
  produce a zero scale.
- **After the cast, padding is forced to zero** in a second, separate `where`. A block-scaled format
  has no representation for "absent", so whatever the padding lane happened to hold becomes data.

**The portable absmax is an integer operation.** Bitcast the fp32 tile to `uint32`, clear the sign
bit with `& 0x7FFFFFFF`, and take an integer maximum. The saving is in the *max*, not in the abs —
a float `abs` is itself a free source modifier on CDNA VALU encodings — and what the integer form
buys is that a plain integer max is already NaN-propagating, because NaN sorts above `+inf` in that
order. It replaces a NaN-propagating float max with an ordinary one.

Upstream's preferred branch is different and does not port: it reduces with an inline
`max.NaN.xorsign.abs.f32`, a PTX instruction gated on CUDA compute capability 8.6+, and falls back
to the integer form otherwise. So the integer form is not an exotic alternative — it is upstream's
own portable path, and it is the one to take here. Where a NaN-propagating float max is wanted
directly instead, `tl.maximum` accepts a NaN-propagation mode on all four versions.

**Whether stage 1 costs a second pass over the data is a residency question, not a style question.**
If the group is wholly resident in one program's registers, the same values serve the absmax and the
scaling with no extra traffic. If the group spans a K-loop, the reduction forces either a second
read or a split into two kernels. The classic layer-norm tutorial shows the shape of that cost: it
re-reads `X` from HBM three times — once for the mean, once for the variance, once to normalize —
rather than holding the row in registers. Note it does this *unconditionally*, and refuses the
row-does-not-fit case outright rather than degrading; the three reads come from the two-pass
mean/variance structure, not from a capacity fallback.

## Stage 2 — scale dtype, and the group size that is not the hardware's

The most expensive misconception in this area:

> **The hardware scale group on CDNA4 is 32, not 128.** One 8-bit scale covers 32 elements of an
> operand — that is the OCP MX block size, and what `mfma_scaled` consumes. A **group of 128 is a
> software convention** (per-token blockwise activation scaling) that the matrix instruction has no
> notion of. NVFP4's group is 16, a third distinct number.

**gfx942 downgrade:** there is no scaled matrix path at all, so every group size — 32, 128 or
anything else — is a software contract between producer and consumer, applied as explicit
VALU on the consumer's accumulator; and the fp8 target is the FNUZ encoding, whose finite maximum
differs (`e4m3fnuz` 240 against OCP e4m3's 448; `e5m2fnuz` 57344 like e5m2), so the clamp
constants below are per-arch.

So on gfx950 a group-128 scale tensor **cannot be handed to the scaled matrix path directly**. Two ways out,
and they have different costs: expand each group-128 scale into four identical 32-element scales
(exact only when the scale is a power of two), or keep the 128 grouping and rescale explicitly on
the fp32 accumulator once per 128 of K. Decide this before choosing the group size, not after.

Scale dtype is a real fork:

- **e8m0** — an 8-bit pure exponent, stored in a `uint8`. Extract it by masking the fp32 exponent
  field (`& 0x7F800000`) and shifting down by 23; rounding up means adding the full mantissa mask
  first, so the scale is never too small. This is the MX path.
- **A small float scale** (fp8 e4m3) carries mantissa bits and is the NVFP4 path. Upstream accepts
  only the ROUND_UP *mode flag* here, and says in the same breath that the stored scale is actually
  rounded by the fp8 cast rather than forced upward. Read that carefully: the flag's name does not
  carry the e8m0 path's guarantee, and assuming it does is what makes the next paragraph bite.

Clamping: **keep the saturating conversion.** With an e8m0 round-*up* scale the quantized values
genuinely cannot exceed the target's finite maximum (448.0 for e4m3, 57344.0 for e5m2, 6.0 for
e2m1), so the clamp looks removable. With an fp8 scale they can exceed it, because that scale is
rounded to nearest and may land below `absmax / max_finite`. Upstream saturates unconditionally on
both paths — its torch reference clamps explicitly to emulate the saturating hardware conversion,
and its fp4 path ends in an explicit `minimum` — so the overflow is caught by the conversion, not
prevented by the scale. When the scale comes from a global or stale tensor instead — a
delayed-scaling scheme — the values can exceed the range by an arbitrary factor and an explicit
clip is mandatory on every path.

## Stage 3 — dequant + residual + norm

Argue this fusion in **bytes, not speed**. Each unfused step is one full HBM round trip of the
activation tensor, and the pre-norm tensor the residual add needs is the one the dequant just
produced and still holds in registers. That is a structural statement and it does not require a
measurement to make.

Two rules that decide correctness rather than throughput:

- **Accumulate in fp32 regardless of the storage dtype.** Upstream's own block-scaled quantization
  kernel opens by casting its bf16 input to fp32 and says why in a comment: most operations are not
  supported on bf16 in the first place, so the promotion happens whether you write it or not. The
  reason to write it explicitly is to control *where* — one promotion at the top beats a promotion
  and a demotion around every intermediate.
- **The residual must be added before the variance reduction, in fp32.** Add it afterwards and the
  tensor you computed statistics over is not the tensor you stored. That is a correctness statement
  about which tensor the statistics describe; this pack has not characterized what the resulting
  error looks like downstream, so treat it as a structural rule and not as a symptom to recognize.

- **A deliberate `.to(bf16).to(fp32)` round trip is a correctness contract, not dead code.** The
  previous rule says compute in fp32; the consequence nobody expects is that a fused kernel which
  *stays* in fp32 across a point where the unfused reference materialized bf16 will be **more
  accurate than the reference**, and against a bit-exact or very tight oracle more accurate also
  fails. The dtype of an intermediate tensor is a contract **between** kernels, and fusing two
  kernels deletes the rounding point that used to sit on the boundary. So when you fuse across a
  materialization, re-insert it by hand at the same place. Production names this explicitly — one
  such site is labelled a "BF16 materialization boundary" in its own source, and another carries a
  host-side rounding policy argument for it — and the lines look removable to every reader who did
  not know that.
  **Before deleting a narrowing round trip, find out whether the oracle was generated with it.**
  This is the mirror of the risk in `linear-attention.md`, where the failure is losing precision
  you needed; here the failure is keeping precision you were contractually supposed to drop.

One diagnostic worth knowing: `enable_fp_fusion` (on by default) controls whether LLVM is allowed to
contract multiply-add pairs. Read its scope carefully before using it, because two plausible
readings are both wrong. It is **not AMD-specific** — it is a compile option on the AMD and NVIDIA
backends alike, defaulting from the same language-level knob. And it is **not per-op or
per-region**: when enabled it sets the fusion mode on the `TargetMachine` that compiles the whole
module, so turning it off changes the rounding of *every* multiply-add in the kernel, not just the
dequant's `x * scale` and the norm's `x_hat * w + b`.

That still leaves it useful, but as a coarse instrument: flipping it tells you whether an oracle
mismatch is contraction-related at all. If the answer is yes, the next step is to localize which
multiply-add moved, which the flag itself cannot do for you.

## The group reduction and the broadcast back (explicit-layout path)

**This section is about the explicit-layout path only.** On plain Triton the reshape-and-reduce
form above is not merely acceptable, it is what upstream's own block-scaled quantization kernel
does: reshape to `[rows, groups, GROUP]`, `tl.max(axis=2, keep_dims=True)`, broadcast back. Plain
readers should take that shape and stop here.

On the explicit-tile path the round trip still works, and it is worth being precise about why,
because the obvious worry is the wrong one. Reducing a rank-3 tile over its innermost axis yields a
slice layout, and expanding that slice back along the same axis is recognized by the frontend: it
reconstructs the parent rather than inventing a new layout. So reduce-then-broadcast along one axis
is self-consistent and does not need rescuing.

The real cost is upstream of that. **The 2D-to-3D reshape produces a rank-3 layout you did not
author**, and every subsequent tile in that chain inherits it. When the broadcast result has to meet
a tile whose layout you *did* author — the operand you are about to scale, or a rank-2 scale tensor
you intend to store — the layouts do not match and a `convert_layout` appears to reconcile them. It
compiles and it is correct; it is also a full shuffle through LDS that you did not plan for and
will not see without reading the TTGIR.

Two ways to avoid it, in order of preference:

- **Let the block width equal the group** where the shape permits. The group reduction becomes an
  ordinary reduction over the last axis of a 2D tile you authored, and no rank-3 layout is ever
  created. This is a fallback rather than a default — it forces `BLOCK == GROUP`, which may be a
  worse tile than the workload wants, so take it only when the tile is acceptable on its own terms.
- **Otherwise author the rank-3 layout explicitly** rather than letting the reshape infer it, so the
  broadcast back lands on a parent you chose. Confining the reshape to shared memory is the third
  option — reshaping an allocation and reshaping a distributed tile are different operations with
  different rules.

Either way the acceptance signal is the same: look for an unplanned `convert_layout` between the
broadcast and its consumer.

## Scoping a quantized-GEMM measurement

A standalone low-precision GEMM number is a **scoped** measurement: it presumes the scales already
exist, in the layout the matrix op wants. Before quoting one, name which kernel produced them and
which kernel consumes the output, and make the comparator cover the same span.

Upstream shows both halves of why that matters, and they point in opposite directions:

- **The production side is a real kernel, and you can put it where you like.** `downcast_to_mxfp` is
  a `@triton.jit` kernel on all four versions — it computes the group absmax, derives the scale, and
  stores scale and payload in the same launch. The matmul path can also fold that work in as a fused
  epilogue, selected by a constexpr on the kernel. So "quantize as a separate pass" and "quantize
  inside the producer's epilogue" are both shapes upstream ships, which is exactly the dispatch
  decision `gemm.md ## Small-M (decode / GEMV) regime — memory/occupancy-bound` sends you to.
- **The benchmark side deliberately skips it.** The block-scaled matmul tutorial synthesizes its
  scale tensors directly with `torch.randint` rather than running the quantizer. Note the bounds it
  picks — a narrow band of e8m0 exponents near unity, not random bytes, because a uniform random
  e8m0 would span a dynamic range no real tensor has and the result would not be numerically
  meaningful. That is a legitimate choice for timing the GEMM in isolation; it is also precisely why
  the tutorial's number is not an end-to-end number.

The trap is reading the second bullet as evidence for a claim about the first: the harness omitting
the quantizer is a scoping decision, not a statement that the quantizer is cheap or that it does not
exist.

The group size is a contract among three kernels — producer, GEMM, consumer — and changing it in one
invalidates the other two. The scale **layout** is a stronger contract still: the host-side shuffle
and the in-kernel un-shuffle must be exact inverses, and the permutation depends on the MFMA
non-K dimension, so a scale layout is not portable across instruction shapes on the same chip.

## When Gluon is justified here

Only when an **explicit** mechanism the compiler will not choose is needed: a
specific `BlockedLayout` to fix coalescing, a `PaddedSharedLayout` to kill a
bank conflict in a reused staging buffer, or `buffer_load` to remove mask
branches. Otherwise this is a Stage-Plain-Triton win (config / fusion /
dispatch) — record the budget gap and stay plain.

## Layer roadmap (if escalated)

1. anchor: explicit `BlockedLayout` matching the inferred TTGIR layout.
2. memory path: `buffer_load`/`buffer_store` for mask-branch removal; ensure
   coalesced offsets (identical on gfx950 and gfx942; only the async direct-to-LDS widths differ,
   and this archetype rarely stages through LDS).
3. LDS layout: only if a reused staging buffer has measured bank conflicts.

There is usually no pipeline / slicing layer for pure memory-bound kernels.
Buffer-op rules: `../gluon/memory-reference.md`.

## Family signals

- **Elementwise / vector**: the safest early Gluon candidates, but tiny vector
  stages often lose to layout + launch overhead — a slow correct result is
  evidence, not permission to broaden scope. Preserve launcher / masks / dtype;
  recover the logical tile from `tl.arange` + launch constants; derive a wave64
  `BlockedLayout`; use generic `gl.load` / `gl.store` first.
- **Softmax / reduction / norm**: choose one logical reduction axis; keep the
  identity value and accumulator/output dtype explicit; derive reduction state from
  the compute parent layout it broadcasts into; split layout fixes from matrix /
  wrapper fixes; avoid hardcoded shape literals in multi-shape paths. Tiny 1D
  reductions can be whole-helper execution anchors but are not automatically good
  performance candidates (padding / mask / launch overhead can dominate).
