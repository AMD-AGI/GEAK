# Mixture-of-Experts On The Gluon Path (gfx950 / gfx942)

MoE is not a new archetype — it is a GEMM whose operand assignment is data-dependent. That framing
is still correct and still the reason the variable framework in
`gemm.md ## Applying the framework to GEMM variants` reads this workload without modification: that
page owns the tile and the layer order, and this one owns what the data-dependent assignment adds —
the bound the expert set decides, the routing metadata, the bucketing and schedule, and which of the
routing / gather / scatter-back primitives exist in `gl` on which versions.

**But "it is a data-dependent GEMM" can no longer be the only organizing principle of this page,
because it does not predict a single one of the decisions below.** Surveyed production source puts
four models and roughly five dozen MoE files on this target, and kernels that agree on the archetype
disagree on the combine placement, on how fp8 enters the matrix core, on whether a router exists in
the file at all, and on which overlap mechanism is used — including two families from the same
authors that solve the same operator with disjoint mechanism sets. What predicts those is the
kernel's **type**, so type first and advise second.

**What backs which part of this page.** The API existence claims and the version-gate table are
**source-proven against 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0**. The bound reasoning is not a source claim
at all — it comes from the archetype's own byte model and the caveats recorded with it
(`perf_knowledge/hardware/data/workload_models.json`), which is why it can tell you the default `E_touched` is wrong
in both directions without any kernel having been run. The launch-geometry hazard is cited at the
grading `../hardware/capability-matrix.md` gives it, which is that page's local evidence rather
than this one's. Everything attributed to **surveyed production source** below is `source-proven`
(a construct read at a named site) or `static-census` (a count over the surveyed files) — it is
evidence about *what production writes*, never about what is faster. **Nothing on this page has
been timed here**, and the new production material does not change that: where a graded claim
appears, the grade belongs to the page it came from, and promoting any of it to measured belongs in
the build's record with the probe beside it (`../method/triage.md`).

**Arch scope.** Written gfx950-first (CDNA4, MI350X / MI355X), which is where the surveyed
production MoE kernels run. Where gfx942 (CDNA3, MI300X / MI325X) changes an answer it is marked
*gfx942 downgrade*; the recurring ones are no scaled MFMA (so axis 2's form 3 and route 1 below do
not exist), fp8 in the FNUZ encoding, direct-to-LDS async only at 32-bit with `order=[1,0]`, and
64 KiB / 32-bank LDS.

**One scope caveat that governs the upstream material.** The API existence claims and the
version-gate table below really are checked across all four tags. The *upstream Gluon MoE example*
is not four-version material: `python/examples/gluon/05-moe-bmm1-fused-gather.py` appears on
**3.8.0 only** — 3.6.0 / 3.7.0 / 3.7.1 ship exactly one Gluon example
(`01-attention-forward.py`) and nothing MoE-shaped. So every sentence that begins "the upstream
Gluon MoE example" is a statement about 3.8.0 and about a Blackwell kernel; on an older build there
is no such file to read. It is **one data point, and a mis-targeted one** — read it for the
structural ideas listed in the last section, not as the reference implementation. The plain-Triton
machinery it calls into (`triton_kernels/`) *is* present on all four, and is flagged separately
where it matters.

## Type the kernel before advising it — three questions

Three answers locate a MoE kernel, and every recommendation later on this page hangs off one of
them. Answer them from the **entry signature and the source**, not from the file name: in surveyed
production source the file name is the shape the file was *tuned* for, and it is wrong about both
the fusion boundary and the band it accepts.

| Axis | The question | The answers seen in production |
| --- | --- | --- |
| **fusion boundary** | what does the entry take, and what does it return? | router-only · complete-fused · expert-GEMM-only |
| **quantization form** | how does a low-precision operand reach the matrix core? | bf16 · fp8/fp4 with software block scaling · fp8/fp4 with the hardware block-scale operand (**absent from the surveyed checkpoints — a census result, not a property of the mechanism; the mechanism works**) |
| **shape band** | which M band was this file tuned for, and what selects it? | the decode / middle / prefill bucketing this page already uses |

The axes are close to independent: a router-only kernel is usually bf16 at every band, a
complete-fused kernel usually carries the fp8 expert GEMM *and* the router in one file, and the
band then decides the mechanisms inside. What the axes are **not** is a taxonomy for its own sake —
each one changes a different set of the answers below, and the last section of this typing block is
the reason the combination matters more than any single axis.

### Axis 1 — where the fusion boundary falls

- **router-only** — logits in, `(topk_ids, topk_weights)` and usually the route metadata out. No
  expert GEMM, therefore no low-precision expert path: across the surveyed router-only files the
  scaled-MFMA count is **zero**, and the matrix work that is there is a bf16 router projection.
  Roughly a third of the surveyed MoE files are this shape, and one family ships fifteen of them,
  one per band, doing nothing else.
- **complete-fused** — hidden states and weights in, the combined output out; the router, the
  metadata build, both expert GEMMs and the combine all live behind one entry point (which still
  means several *launches* — see `### The launch boundary is the grid-level synchronization`).
  This is the majority shape, about three fifths of the surveyed files.
- **expert-GEMM-only** — takes `topk_ids` / `topk_weights` as arguments and computes only the
  expert matmuls and the combine. Rare, and the one place the file name is actively misleading:
  four surveyed files are named as the complete operator and are this shape, discoverable only from
  their argument list.

**Why this is the first question and not a label.** It decides which of the later sections even
applies. A router-only kernel is governed by `## Router and top-k: four implementation generations`
and `## Which tier the routing kernel belongs on`, and nothing in the fp8 or the reuse sections
touches it. An expert-GEMM-only kernel is the inverse: the routing sections are somebody else's
contract, and its own risk is that it inherits a slot ordering it did not produce
(`## Reduction order is part of the contract, not an implementation detail`). Only the
complete-fused shape owns the whole page — and it is also the only shape where "how few launches
can I express this in" is a live question.

**How to read it off the dispatch layer.** Where a pack dispatches by shape, the entry's
**signature arity plus the fixed tail of the signature** is what distinguishes the types, and it
does so even when two types share a file of dispatch records and share M values. A five-argument
entry whose tail carries the hidden size, the expert count, the route slot count and the top-k is a
complete-fused op; a four-argument entry from the same table carrying only hidden size, expert
count and top-k is the router for it. Those are **two different operators**, not two candidates for
the same call — reading them as competing specializations of one op is the failure mode here. At
the other extreme, a family can bake every type constant into the file and dispatch on M alone; the
types are then not in the signature at all and have to be read from the source.

**dtype outranks all of this, and it is not in the signature.** Where the pack directory encodes
the weight dtype and the tensor-parallel degree, choosing the wrong one is choosing the wrong
*pack*, not the wrong kernel — nothing in the per-kernel typing can recover from it, and the failure
surfaces as wrong numbers rather than a refusal. The same holds for the weight layout: a preshuffled
and a standard weight tensor are mutually exclusive contracts asserted by name, and in surveyed
dispatch code the shuffle flag can be discarded and re-asserted from a hardcoded string, so it is a
precondition the shape/dtype validation structurally cannot catch.

### Axis 2 — how the quantization reaches the matrix core

Three forms are conceivable, and **all three work; only two occur in the surveyed checkpoints**:

1. **bf16 throughout.** Router projections and the small-M expert GEMMs. No scale plumbing, no
   descale on the accumulator, and the generic matrix-utilization counter reads normally.
2. **fp8 operands with the scaling done in software.** The dominant form. Weights are block-scaled
   (a per-block fp32 scale over a 128-wide group is the shape surveyed production uses), and the
   scale is applied on the **accumulator** afterwards, as an fp32 multiply or `gl.fma`.
3. **fp8 or fp4 operands with the hardware block-scale operand actually driven.** **Zero surveyed
   files, and a working mechanism.** Keeping those two statements apart is the single most useful
   thing in this section, because the older wording — "the empty set" — reads as a property of the
   instruction and it is not one.

**gfx942 downgrade:** only forms (1) and (2) exist. There is no scaled matrix op, so form (2) is
the regular `cdna3.mfma` fp8 instruction on **FNUZ** operands (`e4m3fnuz`, max 240 rather than
448) — an OCP-e4m3 checkpoint has to be re-encoded, not just reinterpreted — and fp4 has no matrix
path at all.

**Read (3) with its scope attached, because the scope is the whole content of the claim.** The
emptiness is a **census result about the checkpoints these kernels were written for**: their scales
are fp32 over 128-element groups, and the instruction wants E8M0 over 32. It is not a statement
that the path is unreachable, untested or slow. Driving both scale slots with **real `uint8` E8M0
operands** routed through `gl.amd.cdna4.get_mfma_scale_layout` is measured correct on gfx950 /
`v3.8.0` at `a_format="e2m1", b_format="e2m1"` — relative error `8.8e-08` and `1.9e-07` on two tile
shapes, holding down to `num_warps=1`, with a second independent implementation reaching `1.7e-03`
and passing a production kernel's numerical oracle. The disassembly is
`v_mfma_scale_f32_16x16x128_f8f6f4` with **no** fp32 compensating multiply, **no** `scaled_upcast`
and **no** hand-written dequant. So:

> **If your model's scales are group-32 E8M0, this is the preferred route** — it deletes the
> descale from the accumulator chain rather than making it cheaper, and `get_mfma_scale_layout`
> writes the scale operand's layout for you. Reach for it first and fall back to route (2) only if
> the checkpoint format rules it out. `../gluon/matrix-reference.md ## Matrix-Family Details` has
> the measurement and the disassembly evidence; `../gluon/layout-reference.md ## The Scale
> Operand's Layout For mfma_scaled` has the layout contract.

**The census behind (3).** Across every `mfma_scaled` call site in the surveyed MoE files — and,
widening to every surveyed kernel in those four families, across **hundreds** of call sites — the
`a_scale` and `b_scale` arguments are `None` **without a single exception**, the operand formats are
`e4m3`, and the accumulator is a fresh zero rather than a carried chain. The scaled MFMA is being
called as a native fp8 matrix op with its scaling path deliberately unused. (A regex will not
establish this: most call sites span several lines. The count above comes from an AST walk over the
call nodes, which is the form to reproduce if you need it re-checked.)

**The reason is a granularity and format mismatch, and it is a property of the model, not of the
kernel.** The hardware scale operand is one E8M0 — a pure exponent, no mantissa — per 32 elements.
The production models quantize with one **fp32** scale per **128** elements. Neither the format nor
the group size fits, so the hardware path is not merely unused, it is unusable *for these weights*.
Do not read the emptiness as an oversight to be fixed; read it as the reason the fp32 compensating
multiply exists, and expect it in any kernel inheriting the same checkpoint format.

**And the upstream API agrees, which is a stronger argument than the census.** The paragraph above
is a claim about the *weights*; this one is a claim about the *entry point*, and it closes the door
from the other side. On `3.8.0` the scale-layout helper for scaled MFMA grew a group-size argument —
`gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)` — which reads like
the granularity became configurable. It did not: the first statement in the body is
`assert scale_factor == 32`, so the parameter exists and admits exactly one value. **Do not spend a
round trying to reach the hardware path with a 128-wide group**; it is not a matter of finding the
right spelling. **The same assertion read forwards is the adoption rule**: 32 is exactly OCP MXFP4's
group size, so a checkpoint quantized to the OCP block-scale spec lands on this entry point without
any reshaping, and the wall is between the *model's* format and the hardware, never between the
toolchain and the hardware. The companion op is worth knowing for the same reason —
`gl.amd.cdna4.scaled_upcast`
(**`3.8.0` only**: absent on 3.6.0 / 3.7.0 / 3.7.1) fuses an fp8-or-fp4 upcast with its block scale
in one op and has no plain-`tl` standalone equivalent, so it is the natural way to delete a manual
dequant sequence while keeping a bf16 matmul. It takes the **raw E8M0 payload** in `int8` / `uint8`,
which is the same format wall — across every surveyed MoE file its use count is **zero**. Grade it
as available-but-unreachable *for this checkpoint format* rather than unused
(`../hardware/capability-matrix.md`; `../tile-programming/low-precision.md`).

**Where it belongs once a model does ship E8M0 scales is route 2 of
`## Getting fp8 into the matrix core: four routes, all in production`, not route 1.** It is the op
that makes "upcast to bf16, then issue a regular MFMA" a single op instead of a hand-written
dequant sequence — that is its job, and it is a good one. It is **not** the way to reach the
hardware block-scale path: that is `mfma_scaled` with the scale slots driven, which keeps the
operands narrow and never materializes a bf16 tile at all. So with E8M0 scales in hand, try
`mfma_scaled` first and reach for `scaled_upcast` when something else forces you onto a bf16
matmul (a cosine gate against fp8-MMA is the usual reason).

**That ordering is now measured, and the price of getting it backwards is 2.46x.** Same MoE stage-1
kernel, same tiles, same epilogue, on a checkpoint that does ship group-32 E8M0 so the format wall
is absent; the matrix-core path is the only thing that varies:

| route | path | ratio |
| --- | --- | --- |
| 1 | `mfma_scaled`, hardware block-scale operand, `[16,16,128]` f8f6f4 | **1.00x** |
| 2 | `scaled_upcast` → bf16 `mfma`, `[16,16,32]` | **0.4067x** (2.46x slower) |

Stable at 0.398–0.439 across a 7x range of M, and the **outputs are bit-identical** — every e2m1
code matches, zero E8M0 scale mismatches, at every shape. So this is a pure cost comparison with no
accuracy argument on either side. Three things are worth carrying out of it:

- **The loss is not the lowering quality of `scaled_upcast`.** On CDNA4 it is native: it lowers to a
  real packed convert-with-scale instruction, not a software sequence. (CDNA3's own docstring says
  CDNA3 takes the software-emulated path; CDNA4's does not, and the ISA agrees.) Route 2 also used
  *fewer* `ds_read`, *fewer* `buffer_load`, *less* LDS and *fewer* VGPRs than route 1 and still lost.
- **The loss is VALU issue, and it is structural.** Materializing a bf16 tile means touching every
  element with the vector unit — 192 converts feeding 32 matrix instructions, six per MFMA. The
  block-scale operand never materializes the tile, which is the whole point of it. No tile or
  occupancy change moves this; it is the cost of the route.
- **Packed fp4 carries a hidden format cost on top.** The upcast op asserts that the scale tensor
  has one E8M0 byte **per fp4 element**, not one per group of 32 — a `[16,16]` scale tile becomes
  `[16,512]`, a 32x register replication — and the E8M0→bf16 conversion then runs as ordinary VALU
  *over the expanded tensor*. Only the upcast-and-multiply is fused; the expansion is not.
  `mfma_scaled` takes the compact `get_mfma_scale_layout` tile and needs none of it.

**Scope: decode shapes, fp4 only.** The op's fp8 branch is different code — no expansion, scale
shape equal to source shape — so this ratio must not be carried to fp8.

**Two traps sit on this axis.**

- **Passing `None` does not get you an unscaled instruction.** The frontend materializes the absent
  scale into a splat constant, so the lowering sees two scales present and emits the *scaled* MFMA
  anyway; no canonicalizer folds it back. A kernel that really needs the unscaled form has to
  bypass the wrapper. "I passed `None`, therefore the ISA op is unscaled" is surface intuition and
  false — including where an author's own comment says otherwise.
- **The matrix-utilization counter is dtype-aware here.** A kernel on form (2) can saturate the
  matrix engine while the generic counter reads near zero; a kernel that upcasts fp8 to bf16 and
  issues a regular MFMA is on the *other* caliber in the same arch. Which applies is a per-kernel
  reading of the op actually issued (`## Implementation-level facts that decide rounds, and are not
  in the layer ladder`, item 2).

### Axis 3 — the shape band, and what actually selects a file

The banding this page already uses — decode (M up to a few hundred), middle (M of order 10³), and
prefill (M in the 10⁴s) — is retained unchanged and is the axis the rest of the page is written
against. Two things about it are routinely gotten wrong.

**The file name is not the band.** In one surveyed family, five separate files each `assert` the
**same** decode set of M values, and five more each accept the whole `M >= 1024` half-line. The name
records the shape the file was tuned for; the assert records what it will run. If you are reading a
kernel to learn "what does production do at M=64", you have to check which file the dispatcher
actually selects — the one whose name contains 64 may not be it.

**A production dispatch table is exact points plus fallback ranges, and exact points win.** One
surveyed family dispatches its MoE on roughly two dozen **exact** M values plus two catch-all
ranges, with the catalog resolving exact before range. The consequence is that the ranges interleave
with the points: a file registered only at exact points sits *inside* another file's range, so
stepping M by one across a tuned point moves you to a different file — and, because the names lie,
to a file whose name suggests a band you are nowhere near. Treat "which file runs at this M" as a
lookup, never as an inference from the name.

**What the band decides.** Nearly everything mechanical on this page: the combine placement, the row
tile, whether a sort exists, whether the overlap is register-resident or an async ring, how fp8
enters the matrix core, and whether a per-compile scheduler strategy is set at all. That is why the
band is an axis and not a footnote.

### The unit of specialization is (model, band), not "the operator"

This is the structural conclusion the three axes exist to support, and it is the one that should
change how you read the rest of the page.

Two surveyed families implement the *same* operator class — a complete-fused, block-scaled-fp8 MoE
on the same chip, from authors working to the same contract — and they overlap their loops with
**disjoint** mechanism sets. One builds its overlap on the device, in registers: a tuple of
in-flight operand bundles rotated by a statically-unrolled loop, with the depth pinned by a
`static_assert` and chosen per band at the launch. It uses the async-copy path in only a small
minority of its files. The other builds its overlap around an async-copy ring with explicit
`wait_group` drains, hundreds of sites' worth, paced from the host by a per-M cascade of constexpr
choices and — in some of its files — a per-compile scheduler strategy that is itself selected by the
row count. Neither family's mechanism appears in the other's mainstay position.

**Neither is wrong, and no archetype-level rule would have produced either.** What the two agree on
is the *type*: both are complete-fused, both are form (2) on axis 2. What they disagree on is
everything the band and the family's own authoring conventions decide. So:

> **Advice given per operator class is given at the wrong level.** "MoE kernels should use X" is a
> sentence this corpus falsifies by construction, because two MoE families chose disjoint Xs and
> both shipped. The transferable unit is **(model, band)** — a mechanism read out of a kernel at
> one band of one model transfers to the next band of that model far more reliably than to the same
> band of a different model.

Three practical consequences:

- **When you lift a technique from a surveyed kernel, lift its band with it.** A register-prefetch
  depth, a row tile, a combine placement and a reuse factor are all band-keyed in the source
  itself — usually visibly, as a host-side conditional on the row count.
- **Do not read a mechanism's absence as a verdict on the mechanism.** One family writing zero
  occurrences of a loop-level pipelining knob is evidence about that family's authoring style, not
  evidence the knob is inert. (Where a knob *is* inert on this path, this tree says so at the knob's
  own page, and for a different reason — see `## Overlap in a MoE loop: what this archetype actually
  reaches for`.)
- **Count at the right scope, and say which scope you counted.** A census over one family and a
  census over the whole surveyed set give different answers on almost every question in this
  section, and a number quoted without its scope and its match pattern is not evidence.

## Which variables MoE flips, and which it does not

Two of that page's variables, no more: **data access pattern** (the addressing is indirect, which
puts an address chain in the loop) and **does the loop body's composition vary per iteration**
(yes, which is what removes the static-unroll premise). Both are already written there.

The one thing worth stating separately, because getting it backwards sends you to the wrong fix:

> **The reduction structure is *not* what MoE changes.** `gemm.md` says that
> **Stream-K** "moves the reduction *boundary* per worktile, so the epilogue is no longer
> uniform" — that sentence is about
> Stream-K, and the MoE sentence is the next one. In the upstream MoE example each worktile owns a
> complete K-loop; there is no cross-workgroup K split, so there is no moving reduction boundary.
> Reading MoE's non-uniform epilogue as Stream-K's leads to split-K atomics or a separate reduce
> over K, neither of which addresses any of the three real causes below. Surveyed production agrees:
> where a separate reduce kernel exists it reduces over **route slots**, never over K.

## The three real causes of a non-uniform epilogue

1. **Ragged tail blocks.** The token count per expert is a runtime value loaded inside the kernel,
   so the last M-block of every expert is a partial tile and its mask is data-dependent. What
   varies per worktile is the number of *live lanes*, not the reduction extent.
2. **Scatter-back through the routing table.** The output row address comes from the routing map,
   so store coalescing is a property of the index distribution rather than of anything you can
   express at the store.
3. **Top-k combine is a genuinely variable-length reduction**, and it lives *outside* the matmul.
   Each token is processed by k experts and their weighted sum spans worktiles.

Cause (3) is the one with a real design decision attached, and it is the subject of the next two
sections — one on *where* the combine runs, one on *what it costs*, and at prefill the cost section
is the one that changes the plan. Causes (1) and (2) are absorbed by the metadata stage — see
`## Routing metadata: precompute it once, look it up in O(1)` and
`### Three different things are called "padding" here`.

The upstream answer to (3) is worth knowing but not worth generalizing from: its reduce kernel
(`python/triton_kernels/triton_kernels/reduce.py`, present on all four versions) runs the combine as
a **separate kernel**, not as atomics folded into the matmul. `gemm.md` frames this as an
atomic-or-separate-reduce choice, and that one kernel picked separate-reduce. It is one data point.

Its row-bucketing refinement — count the active experts per row and group rows by that count so
every group reduces a uniform length — is the part that does **not** arrive for free, and the reason
is a host-side gate rather than a missing primitive. On 3.8.0 that path is selected by a flag whose
predicate includes `target_info.is_cuda()` **and** a CUDA compute capability of 9 or above, because
the masking it depends on is built on a 32-bit bitmap and `libdevice.ffs()`. On gfx942 or gfx950 the
predicate is false before any of the numerics are consulted, so the kernel silently takes its
unbucketed path. Read the uniform-length idea as a structure to re-derive for CDNA, not as
behaviour you inherit by calling upstream's reducer.

### Separate or fused combine is a per-band switch, not a design choice

Reading "upstream picked separate-reduce" as the answer is the wrong pedagogy even though the
sentence is true. **Surveyed production switches between the two repeatedly across the bands of a
single family**, and in two of its files it switches at *runtime* on M within one file.

The mechanism to recognize: a separate-reduce writes a `contributions` buffer of shape
(tokens × route-slots × hidden) that a later launch sums; a fused combine has the **shared expert's
down-projection kernel** fold the routed slots into its own accumulator and store the final row, so
`contributions` is allocated with one slot fewer and the reduce launch disappears. In the surveyed
family the tell is visible on the host in one line — the slot count of the scratch buffer — and in
two files it is literally `8 if <condition on M> else 9`.

**The criterion is not the band and not the token count. It is whether the shared expert's down
GEMM already owns the same output tile the reduce would write.** When it does, the reduce is free:
the accumulator, the row addresses and the store mask are all already in hand, and folding the
routed slots in costs one load per slot. When the shared path is tiled differently — a different
row block, a different panel decomposition, a separate launch geometry — fusing would mean
re-tiling the shared GEMM to match, and production takes the separate reduce instead. That is why
the switch tracks the band *indirectly*: the band moves the tiles, and the tiles decide whether the
two work partitions coincide.

**Two cautions on how strong a pattern this is.** Across the surveyed decode files, sibling kernels
in the *same* band disagree — some fuse the combine into the down kernel, others launch a distinct
slot-reduce — so "at band X production does Y" does not survive contact with the files. And the
prefill fused path is conditional on disjoint M intervals (one surveyed file fuses at
`m < 14336 or 24576 <= m < 28672` and reduces separately elsewhere), which is not a threshold at all
but a tuned set. Take from this the *switchability*, not a lookup table.

**What makes switching legal is a numerical contract, and that is the transferable part.** Both
paths must produce bit-identical results or the switch is a correctness change disguised as a
schedule change. Surveyed production buys that with a per-slot rounding discipline: each partial is
added in fp32 and immediately **rounded back through bf16** before the next add
(`acc = (acc + part.to(f32)).to(bf16).to(f32)`), in a statically-unrolled slot order. Because the
rounding sequence is fixed by the static slot index and not by which kernel performed the add, the
fused and the separate path agree bit for bit and are interchangeable. See
`## Reduction order is part of the contract, not an implementation detail` — **without that
contract, this whole section is unavailable to you.**

### The combine epilogue can be half the kernel, and it is a write-amplification problem

The section above decides **where** the combine runs. This one is about **what it costs**, which
this page did not previously say and which is the larger of the two facts: at prefill the combine
can be a bigger line item than the expert GEMM it follows.

Measured ablation on one prefill MoE kernel (top-k 8, at the top of its M band), varying only the
stage-2 epilogue and leaving the matmul identical:

| stage-2 epilogue | time |
| --- | --- |
| bf16 `atomic_add` into the token-major output | 10571 µs |
| fp32 `atomic_add` into the token-major output — **not a dtype comparison, see below** | 44183 µs |
| plain bf16 `store` (no accumulation — an ablation, not a correct kernel) | 1368 µs |
| no epilogue at all (accumulator discarded) | 454 µs |

The bf16 atomic alone is **9.2 ms of a 17.9 ms kernel — 52 %**. The mechanisms below are not
specific to the kernel that produced these numbers. **The ratios are** — every one of them was
measured at the top of one prefill M band. One of them has since been **retracted outright** (the
arms were not isolating what they were read as isolating) and another moves by two orders of
magnitude between bands. Read each with its band attached, and read the next two paragraphs before
carrying any of them anywhere.

**The mechanism is top-k fold write amplification, and it is structural.** A routed combine writes
`M · TOPK · H` elements to produce `M · H` of output: every token is written once per expert that
selected it, and the fold happens *in memory*. At top-k 8 that is an 8x write multiplier on the
largest tensor the kernel touches. No epilogue-side tuning removes it; only moving the fold out of
memory does. What the amplification does **not** explain is the gap between the `store` row and the
`atomic_add` row — those two issue the same number of writes and differ 7.7x. That gap is a
separate question, and an earlier revision of this page answered it wrongly.

**The fp32 row is a retracted measurement, and the retraction is worth more than the number was.**
Two earlier revisions of this page read the 44183 µs row as the price of widening the accumulator —
first as a flat 4.2x rule, then as a band-local 4.2x observation. **Both readings are withdrawn.**
The two arms that produced those rows replaced the **whole epilogue**: the addressing and the loop
structure moved together with the dtype, at `n = 1`, on a single shape. The dtype was never
isolated, so the row is not evidence about fp32. It is kept in the table only because it is part of
the epilogue-share ablation, where the comparison being made is epilogue-vs-no-epilogue.

**What retired the number did not require a re-run, and that is the reusable part.** The ISA puts a
ceiling on how large a dtype effect can be here: bf16 lowers to `buffer_atomic_pk_add_bf16`, **two
elements per instruction**; fp32 lowers to `buffer_atomic_add_f32`, **one**. For the same element
count that is 16 instructions against 32, so the widest time ratio the dtype mechanism can produce
is **2.0x**. A 4.2x observation is *outside the ceiling of the mechanism it was being used to
demonstrate* — which settles that it was measuring something else, before anyone re-runs anything.
The general form of that check is
`../pitfalls/negative-patterns.md ## Give a ratio a mechanism ceiling before you believe it`.

**Isolated, the dtype effect is small, and the two bands agree.** Re-measured with the epilogue held
fixed — same grid, same addresses, the same 1.61e9 element updates, a single statement differing
between arms:

| band | isolated f32 / bf16 | against the 2.0x ceiling |
| --- | --- | --- |
| large-M (prefill shapes) | **1.26x** | under it |
| decode-scale shapes | **0.947–1.029x** | under it |

Two independent bands, both inside the ceiling, neither anywhere near 4.2x. `cmpswap` is 0 on every
arm: fp32 atomics are **native** on this part (`buffer_atomic_add_f32` / non-returning
`global_atomic_add_f32`, no `vdst`, no `sc0`), not emulated — so the emulation story was never the
explanation either. **Widening the accumulator is a small cost, not a catastrophic one.**

The dtype is therefore not the lever. Neither are the other two candidates. A differential
experiment on the decode-scale shapes — one source text, one statement changed per arm, 8 arms
across 7 sizes — isolated them one at a time:

| quantity isolated | ratio | reading |
| --- | --- | --- |
| contention (many programs on one address vs. disjoint addresses) | 0.997–1.004x | **not a lever** |
| address shape / addressing mode | 0.984–1.010x | not a lever |
| dtype, fp32 vs bf16 | 0.947–1.029x | **not a lever** |
| atomicity itself (same address, same dtype, atomic vs. plain store) | **3.89–8.85x** | the whole cost |

The entire penalty is the **atomicity**; the width, the sharing and the addressing are free. It is
not raw bandwidth either: warm, the plain store gains 2.4–3.0x while the atomic gains only 1.3–1.5x,
and the atomic's throughput stays flat at roughly 3 G/s across a 7x span in element count. The
instruction count also runs the opposite way from any "wide access" story — for the same 16 values
the store path emits 4 `global_store_dwordx4` and the atomic path emits 16 separate atomics.
**There is no wide atomic.**

> **The rule that holds in both bands: remove the atomic, do not tune it.** What does *not* hold
> across bands is the multiple, so carry the rule and re-measure the number: at decode-scale shapes
> an atomic epilogue costs about **6.4x** a plain store of the same data (3.89–8.85x across the
> sizes swept), and at large-M prefill shapes the same comparison is **86–113x**. Two orders of
> magnitude apart, same instruction, same verdict — change the *schedule* so the accumulate stops
> happening in memory. In neither band is the dtype worth a revision.

So "widen the accumulator to be safe" is a **1.26x-or-less** decision, and "keep the accumulate in
memory" is a 6x-to-100x one. Those are the two edits that look equally innocuous in the source, and
they are not remotely the same size. If only one thing survives this section, it should be that
ordering rather than any of the numbers.

One thing here is **open and stays open**: on bf16 the packed form halves the emitted instruction
count and is *not* faster. The 2:1 ceiling above says what the packing could have bought; it does
not say why nothing was collected. No mechanism has been established, and none is offered.

**The remedy is to stop folding in memory.** Have stage 2 store **slot-major** — one contiguous
region per route slot, an ordinary non-atomic store — and then run a second pass that gathers a
token's `TOPK` slots and reduces them in registers. The fold moves from scattered atomic traffic to
one coalesced read per slot plus one write per token. On the kernel above, `gemm2 + combine` went
from `10571 + 73` µs to `1254 + 808` µs: **5.3x on the stage, 3.9x end to end**. The reduction
order is static in both forms, so the numerical contract of
`## Reduction order is part of the contract, not an implementation detail` is preserved — which is
what makes this a schedule change rather than a correctness change. Note also that this *is* the
separate-reduce arm of `### Separate or fused combine is a per-band switch, not a design choice`,
reached from the cost side rather than from the tiling side: the `contributions` buffer that
section describes is the slot-major buffer here.

**Write down its cost before adopting it.** The slot-major buffer is `M · TOPK · H` live at once,
which at the top of the prefill band was about **3.6 GB**. That is a real constraint on a
small-VRAM part and on any deployment already near its memory budget, and the answer is a
**blocked-N loop** — process a strip of the hidden dimension at a time, so the buffer is
`M · TOPK · H_block` — at the cost of re-reading the routing metadata per strip. Decide the strip
width from the memory you actually have, not from the timing.

**And check the ratio before assuming this applies to you.** The epilogue's share rises with
`TOPK`, with `H`, and with M; at decode the whole kernel is small and the fold is a handful of
rows. The one-line screen is the ablation in the table above — delete the epilogue, keep
everything else, and read the fraction off directly. It costs one compile, and it is also the
ablation that tells you whether the reuse axis of
`### Two reuse axes, and this page used to carry only one` is worth sweeping at all.

## Routing metadata: precompute it once, look it up in O(1)

The plain-Triton grouped-GEMM tutorial (`python/tutorials/08-grouped-gemm.py`) has every persistent
CTA walk the group list linearly to find which problem its tile belongs to. That is O(G) of serial
scanning per CTA, and it is a **reasonable shape only while G is small**. At the expert counts
production MoE runs, it is not.

The upstream Gluon MoE example (`python/examples/gluon/05-moe-bmm1-fused-gather.py`, 3.8.0 only)
replaces it with a table built once, ahead of the matmul — a prefix sum over the per-expert token
counts yields the slice offsets and block offsets, and the `(expert index, block index)` pair for
every worktile is **packed into a single int32**. The kernel's scheduling step is then an unpack of
one loaded word. Surveyed production builds the same object by the same route, which is why this
idea is worth keeping from a kernel that otherwise does not port.

**"Ahead of the matmul" is not "on the host", and the difference is the part worth copying.** The
builder (`triton_kernels/tensor_details/ragged_tensor.py`, present on all four versions) is a Python
function, but its body is two `@triton.jit` launches: a memset kernel that zeroes the offset and
schedule arrays, then a compute kernel that fills them. The prefix sum and the block schedule are
produced **on the device**. Upstream does also ship a `torch` implementation of the same metadata
right beside it, and that one is labelled a reference — it is the oracle, not the path taken. So the
lever here is not "move work to the host"; it is **separate the metadata kernel from the matmul
kernel**, which is available to you on CDNA unchanged because both halves are ordinary Triton.

Two further properties of that table matter more than the packing trick:

- **The linearization is a scheduling lever, not bookkeeping.** The example walks blocks in a banded
  order rather than plain row-major, which is the same L2/XCD-locality decision
  `gemm.md ### Worktile scheduling for load imbalance (LPT/SPT makespan)` already
  covers. MoE is one instance of that lever, not a separate mechanism — and
  `### XCD-aware pid swizzle: the layer above longest-first` is where production spends it.
- **The tail-block problem is solved in the metadata stage or not at all.** Once the schedule is a
  table, a partial tile is a table entry like any other.

**What production adds that the example does not show: the table is a place to put a permutation.**
Surveyed prefill kernels compose *two* indirections at adjacent source lines — an arithmetic
remap of the program id, then a table lookup that replaces it outright with an entry the metadata
kernel wrote. Reading either one alone gives a wrong account of which expert a program is about to
touch, which matters as soon as you are reasoning about weight residency.

## Bucketing: three shapes, and what selects one

The metadata above has to be *built*, and that build is a counting sort over `(token, slot)` pairs
keyed by expert. Three shapes of it are viable on this target, and they differ in what they spend
rather than in what they produce — all three can emit the same offset table:

| Shape | How the counts are formed | What it costs / needs |
| --- | --- | --- |
| **One workgroup, LDS atomics** | a single CTA accumulates per-expert counts in shared memory, then prefix-sums them in place | needs the LDS atomic-scatter surface (**3.8.0**, see `## Version gates`), and the whole sort runs at one CTA's parallelism — fine when the token count is small relative to launch overhead, a serialization when it is not |
| **Global atomics + lazy publish** | every block bumps a global per-expert counter to reserve slots, and job records are published as the counts settle | available on every version; the reservation order is not the token order, so anything downstream that assumes stable ordering must be re-checked |
| **Communication-free parallel count** | each block counts its own tokens independently, then a separate scan combines | no atomics and no inter-block traffic in the counting phase, at the price of a second pass and the scratch to hold per-block counts |
| **No table at all — the consumer recomputes the routing** | nothing is published for this launch to read; each computing block re-derives the routing it needs from the keys | costs `O(E log E)` of register-resident work **per consumer**, and buys the removal of a grid-level dependency rather than of a table. The test is the fourth availability question below |

**The fourth shape deserves its own sentence, because it does not look like the other three.** The
first three all produce a table and differ in how. The fourth **declines to produce one**: a
surveyed decode kernel has its low-`pid` programs write a route table that only the *next* launch
will read, while the computing programs in the same launch ignore that table entirely and re-run
the rank-k selection from the keys themselves. Every CTA repeats a selection network over the same
key set. That is not redundancy for its own sake — it is trading `O(E log E)` of register work for
a grid-level synchronization that would otherwise have to exist, and at decode shapes, where M is
small and the top-k fits in registers, production takes that trade in several files. **The
criterion is direct: is recomputing the routing cheaper than one more launch plus the table?**

**What picks one is not performance in the abstract — it is three availability questions.** Does the
build have the LDS atomic surface; is the token count large enough that one CTA is a serialization;
and does anything downstream depend on slot order being token order. Answer those and usually one
shape is left. Where more than one survives, they are A/B-able against each other because the
output contract is identical — which is the useful property of having built the table as data.

> **The reservation-order question is a correctness question, not a tuning one.** An atomic
> reservation hands out slots in arrival order, which is nondeterministic across runs. That is
> harmless if the combine is an ordered reduction over a stable index, and it is a reproducibility
> bug if any consumer folds partials in slot order. Decide which you have before choosing shape two.
> Surveyed production resolves it the first way and says so in its own docstrings; the discipline
> that makes that safe is in `## Reduction order is part of the contract, not an implementation
> detail`.

One more sentence on the counting primitive, because it produces wrong answers silently: if the
counting histogram is fed a **padded or sentinel expert id**, the bin count must cover the sentinel.
A surveyed family pads with an expert id one past the real range and therefore sizes its histogram
above it; sibling sites that size the histogram to the real expert count instead pass an explicit
mask and account for the sentinel with a separate reduction. The two are equivalent; mixing them —
a real expert count of bins with an out-of-range sentinel flowing in — is not a slowdown.

### The launch boundary is the grid-level synchronization

The fourth shape above only makes sense once this is stated explicitly, because it is the thing
that makes a fused MoE buildable without any grid-wide primitive at all.

**A kernel launch boundary is a full grid-level barrier, and production treats it as the
synchronization contract rather than as overhead to be eliminated.** Surveyed fused-MoE kernels say
so in their own docstrings — "launch boundaries provide all producer-consumer synchronization", "all
scratch is invocation-local; no atomics or cross-workgroup dependencies", "metadata published by
the same launch is consumed only by the following finish launch". Across that survey, spin loops,
`atomic_cas`, flag polling and cooperative-grid launches are all **absent**. The design question is
therefore not "which grid-sync primitive do I reach for" but "how few launches can I express this
in", and the two ways to spend a launch you did not want are the ones above: build a table, or let
the consumer recompute.

**The third way to spend fewer launches is to concatenate independent phases behind one 1-D grid,
and it is a signature technique of this archetype.** A surveyed decode kernel launches a single 1-D
grid sized `224 + M * groups + 1` and splits it inside the kernel with `if pid < 224 / elif ... /
elif ...`: the low range runs the router GEMM as a 14-way K-split over 16 expert panels, the middle
range block-quantizes the hidden state, and one trailing program zeroes the route scratch and seeds
the job-queue sentinels. The same idiom appears in a second family as "N router tiles plus one
independent queue-reset CTA". The payoff is **one dispatch instead of three** for phases that are
genuinely independent, with no host round trip between them; the cost is that the grid is a
hand-maintained sum and every phase's geometry is now encoded in a comparison.

> **Do not confuse this with linearizing a grid to avoid the launch-geometry hazard** (item 3 of
> `## Implementation-level facts that decide rounds, and are not in the layer ladder`). They produce
> the same *shape* — a 1-D grid — for unrelated reasons. Phase concatenation is about saving launch
> boundaries and is chosen at design time; hazard-driven linearization is about grid rank
> interacting with warp count and is a timed probe. A kernel can want one, both, or neither.

**The consequence that is easy to get wrong sits in the memory model, not the schedule.** When a
payload has **no reader inside this launch**, the ordering it needs is already supplied by the
boundary. So a `relaxed` integer atomic — reserving disjoint slots, bumping a counter — is *correct*
there, and the absent `scope=` is a decision rather than an oversight: the readers are on this
device, and the ones in the next launch are ordered by the boundary. Raising `sem` on those buys a
writeback nobody is waiting for (`collective.md ### Who issues the maintenance`, row 4). Note what
this does *not* license: it holds because no reader exists inside the launch, so introducing one —
a second consumer in the same grid — invalidates it silently.

## Router and top-k: four implementation generations

"Write the top-k" is not one decision. Surveyed production contains four distinct skeletons for it,
they are ordered by band, and each one is a different answer to *how much structure the downstream
consumer actually needs*. This is the part of the workload with the most production variety and the
least coverage anywhere else in this tree, so it is written out here.

| Generation | Band it is used at | Mechanism | What it costs |
| --- | --- | --- | --- |
| **A — sortless selection network** | every band; the baseline | take the max, write `-inf` into the winner's position, repeat k times; a hierarchical variant does it over expert groups first | `O(k·E)` VALU, **zero** cross-lane traffic, **zero** atomics; runs under `num_warps=1` |
| **B — ballot + leader election** | decode (roughly M ≤ 256) | compare into a 64-bit ballot, find-first-set to elect a lane, `v_readlane` to extract its value; a companion ballot plus `bcnt`/`mbcnt` yields the group total *and* this lane's prefix in one step | inline asm, wave-shaped; ties the kernel to `num_warps=1` and a lane-pinned layout |
| **C — in-register bitonic sort** | middle band (M ≳ 1024) | a butterfly exchange over a monotone integer key, with the partner read as `gl.gather(keys, lane ^ (1 << step), 0)` | `O(log²)` shuffles; no LDS and no inline asm, but the gather must stay warp-local |
| **C′ — ballot match-any instead of a sort** | prefill only | one ballot per bit of the expert id (nine of them for a 512-expert space), AND-reduced into "the lanes holding my expert", then one `bcnt`/`mbcnt` pair for the group size and this lane's rank | the chunk must fit one wave, because the match is wave-local |

**Generation B's real payoff is not the election, it is the atomic count.** Publishing k route slots
per token naively costs k atomic increments. One ballot plus a bit-count collapses them into
**one** atomic for the whole wave, with each lane's slot derived from its own prefix. That is the
reason the idiom survives at all: the election is cheap, the atomic traffic it removes is not.

**Generation C's key is the same packing trick the upstream router uses, written for wave64 without
a sort primitive.** The key places the expert id in the high bits and the lane id in the low bits,
so one integer encodes both the sort key and its stability, and after the sort the run starts fall
out of an `associative_scan` with a max combine. Note what makes this expressible in Gluon at all:
the butterfly partner is fetched with `gl.gather` used as a **lane shuffle**. That is only free
while the gather is warp-local, and the fallback is silent and allocates scratch for the whole
source tensor — read `../gluon/layout-reference.md ## gl.gather — a lane exchange only when the
layouts allow it` before assuming this is a shuffle.

**Generation C′ is the one to understand, because it changes the problem rather than the
implementation.** Sorted-token MoE does not actually need a *sort*. What the expert GEMM needs is
that same-expert tokens be **contiguous**, plus each token's rank within its expert's run. Match-any
gives exactly that: it produces the group and the rank directly, the key never has to be ordered,
and the entire sorting network disappears — trading `O(log²)` shuffles for one ballot per id bit. It
is available only where the run can be confined to a wave, which is why the surveyed use is prefill,
with the chunking cut to match. *(Attribution note: the surveyed source carries no comment stating
this rationale. The reading above is inferred from the code's shape and is **unverified** as the
author's intent; the mechanism and the sites are source-proven.)*

**The implicit selection rule, stated as what it is — reverse-engineered from the switch points, not
quoted from any source.** Fewer routes than a wave holds → recompute with A. Routes on the order of
a wave, needing a cross-program total → B's ballot plus one atomic. Routes far exceeding a wave and
needing a global order → C's sort. Routes far exceeding a wave but needing only a within-group rank
→ C′. Read that as a hypothesis with four consistent data points, not as a law.

**Two band facts that go with this section.** The inline-asm generations are confined to the decode
files — in one surveyed family every `inline_asm_elementwise` site is at M ≤ 256 and the middle and
prefill files contain none, so the whole ballot idiom is *retired* above the decode band rather than
carried forward. And at the very bottom of the band, the router projection itself stops using the
matrix core: production writes the tiny router GEMM as an FMA-path dot with a blocked-layout
operand, because the tile cannot fill an MFMA
(`../gluon/matrix-reference.md`, and note that the FMA dot is a VALU op that is mutually exclusive
with MFMA — it will not show up on a matrix counter).

### The monotone integer key has four incompatible spellings

Generations C and C′, and every ordered top-k in the surveyed corpus, run on the same idea: encode
the float logit into an integer that compares in the same order, and spend the low bits on the
expert id so ties break to the lowest id deterministically. `gl.sort` / `gl.topk` / `gl.cumsum` do
not exist on this path (`## Version gates`), so **every one of these kernels writes the encoding by
hand** — and in one surveyed family there are four spellings of it that are each correct and
**cannot be copy-pasted into one another**:

| Spelling | Order-preserving map | Dead-slot sentinel | Tie-break |
| --- | --- | --- | --- |
| bf16, xor form | `~bits` if sign set else `bits ^ 0x8000` | **0** | `(ordered << 9) \| (511 - e)` |
| bf16, biased form | `bits + 0x8000` if positive else `0x10000 - bits` | **0** | `(ordered << 9) \| (511 - e)` |
| bf16, signed-magnitude int32 | `±(bits & 0x7fff)` by sign | **int32 minimum** | `(ordered << 9) \| (511 - e)` |
| fp32, bit-level | `bits ^ 0x7FFFFFFF` if negative else `bits` | **0** | `(ordered & -512) \| (511 - e)` — **masks** the low bits instead of shifting |

Three independent axes of incompatibility, each of which fails silently:

- **The sentinel convention flips.** The signed-magnitude spelling produces a *signed* key whose
  "loses to everything" value is the int32 minimum; the others produce keys whose dead value is `0`.
  Move an encoder across that line and the padding slot becomes the **winner**.
- **Shift versus mask.** Three spellings widen and shift left by 9; the fp32 one does not widen at
  all and instead clears the low 9 bits of the key, sacrificing nine mantissa bits. Feed a shifted
  key to a masking selector and it overflows; feed a masked key to a `>> 9` decoder and the expert
  id is garbage.
- **The source width.** Three encode a bf16 bit pattern, one bitcasts fp32, and the inverse mapping
  one of them ships inverts only its own form.

**The lesson is not "here are four encodings", it is that the encode step is a contract with the
decoder and the sentinel, and all three have to move together.** The surveyed authors annotate this
in their own comments — canonicalizing signed zero so `+0` and `-0` compare equal, and choosing the
sentinel so it ranks below every legal negative key — which is the checklist to re-derive rather
than the code to lift. If you take an encoder from one kernel, take its sentinel and its decoder
from the same kernel.

## Which tier the routing kernel belongs on

**The tier is decided by how the top-k is expressed, not by the size of the kernel** — and the two
answers below are equally real. An earlier version of this page named only the first one in its
heading, which sent agents navigating by title to the wrong default; the version gate underneath it
is unchanged and still correct.

The gate first, because it is the hard constraint on the whole workload and it is not in any
capability-matrix cell:

> **`gl.sort`, `gl.topk` and `gl.cumsum` do not exist on any of the four versions.** Checked
> 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 across the entire `triton/experimental/gluon/language/` tree — zero
> occurrences, in `__init__.py` and in `_standard.py` alike. Their plain counterparts `tl.sort`,
> `tl.topk`, `tl.bitonic_merge` and `tl.cumsum` are all present in `triton/language/standard.py` on
> all four.

**Branch 1 — if the top-k is expressible with `tl.sort` / `tl.topk`, write it in plain Triton and
leave it there.** It is a small latency-bound kernel with no tile structure worth authoring
explicitly; escalating buys nothing and costs you exactly the two primitives it is built on.
Upstream's own routing kernels (under `python/triton_kernels/triton_kernels/topk_details/`) are this
shape, and pack the value and the index into one integer key so that a single sort orders both.
**The gate above is what makes this branch decisive**: on the Gluon tier those two primitives are
gone, so a router that was only ever a sort has nothing to gain and one thing to lose.

**Branch 2 — if it is built out of wave-level algorithms, Gluon is the only tier that can state the
precondition.** Ballot, find-first-set, `v_readlane` / `v_writelane`, prefix popcount — a
register-resident selection network, a leader election, a compaction — **are missing from plain
Triton too**, so the gate above is not what decides this case. What decides it is that the immediate
operand in `v_writelane_b32 $0, $1, <slot>` only has a meaning if you know which lane holds which
element, and under compiler-inferred layouts nothing pins that. Gluon can pin it: `num_warps=1` with
`BlockedLayout([1], [64], [1], [0])` over 64 elements makes element order and lane order the same
map, and `gl.static_assert(x.shape[0] == 64)` turns the assumption into a trace-time error. **Those
are the preconditions of that one 64-lane bijection construction, not of wave-collectives in
general** — production correctly builds wider predicate tensors under other layouts when the tensor
is an operand rather than something read back per lane, so check which construction you are in
before importing the checklist
(`../gluon/inline-asm-reference.md ## Class 5 — cross-lane and wave-collective`).

**Which branch production is on: branch 2, and not marginally.** In surveyed production source every
file carrying routing logic is Gluon — including a family that ships fifteen router-only kernels,
one per band, all of them Gluon — because the ballot-shaped top-k of
`## Router and top-k: four implementation generations` depends on `num_warps=1` and on the tile
being the physical wave. That is a fact about the *routers people ship*, not a repeal of branch 1:
a router whose top-k is a sort still belongs in plain Triton, and generation A's selection network
is expressible on either tier. Read the two branches together and pick by expressibility.

Three consequences that follow regardless of branch:

- **An expert-offset prefix sum inside a `@gluon.jit` body must be built from
  `gl.associative_scan`** plus an explicit combine function. There is no `cumsum` shortcut. The
  scan itself is a re-export of the plain builtin and takes **no Gluon-specific `layout=` kwarg** —
  unlike `gl.histogram`, which does.
- Naming `tl.sort` through an imported `triton.language` module inside a `@gluon.jit` body is legal
  (the frontend admits module objects), but a sort's internal shuffles are exactly the layout-bound
  operation this pack tells you to author rather than inherit. Prefer the separate plain kernel.
- A router fused into the expert kernel drags its softmax into your body, and the base-2
  exponential fold applies to it exactly as in attention
  (`attention.md ## Online softmax (preserve, do not rewrite first)`), with the same caveat: the
  saving is instruction count, visible in time only where the loop is VALU-bound. One surveyed
  detail is worth carrying because it is a correctness choice and not a performance one — the
  softmax **denominator** sums only the selected experts while the **maximum** is taken over all of
  them, deliberately, so that a NaN anywhere in the logits propagates the way the reference
  implementation propagates it.

## Load balance is a schedule, not a tile

Tokens-per-expert is a distribution, so the tile question (`## Tile choice: tokens-per-expert is the
bucketing variable`) and the *order the work is issued in* are separate decisions, and only the
second one addresses a long tail. The makespan argument and the LPT/SPT recipe are already written
for the general case in `gemm.md ### Worktile scheduling for load imbalance (LPT/SPT makespan)`;
MoE's instance adds three things, and the **first one is a gate that this section used to leave
out**:

- **The precondition is unequal block cost, and in this archetype it usually does not hold.** Every
  makespan argument assumes the jobs cost different amounts; if they cost the same, *every*
  permutation has the same makespan and the ordering cannot pay, however long the tail of the token
  distribution looks. Measured on one prefill MoE kernel, holding the number of active blocks fixed
  at 256 and sweeping how full each block was from **12.5 % to 100 %**: the useful work rose
  **8.00x** and the gemm1 time rose **1.0859x**. A block that does 8x the arithmetic for 8.6 % more
  time is a block whose cost is set by something other than its occupancy — so the blocks are
  effectively equal-cost, and **longest-first is inert here for a reason that has nothing to do
  with grid size.** Check this first: it is one sweep, it is cheaper than implementing an ordering,
  and it subsumes the jobs-per-CU gate below when it fails.

  This is a statement about the kernel as measured, not about the algorithm. **If a later change
  makes block cost track block content again** — a tile small enough that the padding is no longer
  amortized, a variable-K path, per-block work that scales with the token count — **the lever
  reopens and this gate has to be re-run.** Record which side of it you are on rather than
  inheriting the answer.
- **The grouping is by expert weight-residency, not by token count alone.** Two jobs on the same
  expert share its weight read; two jobs on different experts do not. So an ordering that finishes
  the heaviest experts first also keeps their weights hot, and the two effects point the same way.
- **The small-grid gate applies, and MoE reaches it often.** Longest-first ordering only helps when
  there are enough jobs per CU for the ordering to matter; below that the grid is a single wave and
  the order changes nothing while the metadata pass costs a scan. Check jobs-per-CU before adopting
  it, and expect the answer to differ between decode and prefill on the same kernel.

The three subsections below are the parts surveyed production actually spends its scheduling effort
on, and the ordering is deliberate: **the XCD remap outranks longest-first**, and both outrank the
tile.

### Two reuse axes, and this page used to carry only one

The bullet above — group same-expert jobs so they share the weight read — is **weight** reuse, and
it is only half of what production does. The other half runs in the opposite direction: hold **one
activation block in registers and sweep it across several consecutive N tiles of the
down-projection**, re-fetching only the weight. In surveyed source this is an explicit loop
`for step in range(REUSE)` with the A operand and its scales loaded *outside* it, and a reuse factor
chosen on the host by band — 1 below the middle band, a single-digit factor across the middle band,
and 14 or 28 at prefill. The grid is divided by the same factor, so the choice is visible at the
launch.

**These are two different axes and they compete for the same registers.** Weight reuse is bought by
permuting program ids so same-expert jobs land together; activation reuse is bought by making one
program do more N tiles. The first costs a metadata pass and no registers; the second costs the
registers holding A and its scales for the whole sweep, and it lengthens the program, which is
exactly what shrinks the grid. At decode neither pays — there is one token block and the weights are
read once regardless.

**The second-order effect is the important one, because it is what makes an async ring worth
building.** A weight block used once or a handful of times leaves no slack: the weight movement *is*
the critical path, and there is no compute to prefetch behind. Push the reuse factor to 14 or 28 and
the MFMA work per weight block becomes long enough to hide the next block's transfer — which is
precisely where the surveyed family's async-copy double buffering first appears, at its highest
band and nowhere below it. **So "should this loop have an async ring" is not answered by the loop's
shape; it is answered by the reuse factor**, and raising the reuse factor is the enabling edit, not
the ring. `## Overlap in a MoE loop: what this archetype actually reaches for` is where that lands.

**The precondition, which is easy to skip and expensive to skip.** All of the above is conditional
on the kernel being **weight-transport-bound**. Sweeping the reuse factor on a kernel that is not
moves nothing, and measuring that is cheaper than reasoning about it: on one prefill kernel here,
`NN` swept over `{1, 2, 3, 6, 12, 24}` produced 650–711 µs at the smallest M and 10554–10686 µs at
the largest — **flat, and slightly negative at the top**, across a 24x change in the factor. The
bottleneck in that kernel was the combine epilogue
(`### The combine epilogue can be half the kernel, and it is a write-amplification problem`), so
the weight reads it was amortizing were not what the clock was being spent on.

So **confirm the bound side by ablation before spending a round on this axis**: delete or
short-circuit the epilogue, or the weight load, and see which one the time follows
(`## Deciding the bound before sizing a prize`). A flat reuse sweep is not evidence that reuse does
not work — it is evidence that you measured the wrong lever, and the sweep cost is the price of
not having asked first. Do not delete this axis from a kernel on that basis either; it pays where
the precondition holds, which is exactly where the surveyed family uses it.

### XCD-aware pid swizzle: the layer above longest-first

`../hardware/planning-constants.md` gives the XCD count, and `gemm.md ## Beyond hot loop (L2 / XCD
locality)` gives the general remap and — importantly — the gate on when a remap is inert. What
neither says is that **MoE is the archetype where this matters most, and that surveyed production
reaches for it before it reaches for longest-first ordering.**

The reason is specific to this workload. The dispatcher hands program ids to XCDs round-robin, so
consecutive pids land on *different* XCDs. In a MoE the expert weight block is the large reused
object, and consecutive pids are exactly the programs most likely to share one. Round-robin
therefore splits every expert's weight across all XCDs' L2 slices by construction: each slice holds
a fraction of every expert instead of all of a few. The remap inverts that — it maps pid to a
position **within** its own stripe's contiguous block, so each XCD receives a contiguous run of
experts and the weights it needs stay in its own slice.

The arithmetic surveyed production writes is one line and worth recognizing on sight: `stripe = pid
% STRIPES`, then `stripe * (programs // STRIPES) + min(stripe, programs % STRIPES) + pid //
STRIPES`. The middle term hands the remainder programs one each to the lowest-numbered stripes, so
the map is **bijective with no gaps** — which is why it needs no clamp of its own and why it can be
applied to a runtime job count rather than a constexpr one.

Three things to carry across, all of which are easy to miss:

- **The stripe count is a parameter, not the XCD count.** One surveyed file uses twice the XCD count
  below a token threshold and the XCD count above it. Hardcoding 8 discards a tuned knob.
- **When there are several job *classes*, stripe each class.** A surveyed kernel computes separate
  quotas for routed jobs and shared-expert jobs so every XCD gets the same *mix*, not merely the
  same count. Striping a concatenated job list instead gives one XCD all the expensive class.
- **Composed with a table lookup, this is two layers, and they are adjacent source lines.** An
  arithmetic remap followed by `pid = load(order_table + pid)` is not one mechanism written twice;
  the first spreads by XCD, the second applies the metadata kernel's ordering. A second family's
  interleave has to be paired with a **local bound** or the interleaved index leaves the live range.

The gate from `gemm.md` still applies unchanged: a remap is inert unless the kernel is actually
L2/HBM-bound with well-coalesced access, and it is inert again if the grid is already XCD-aligned.
Check that before spending a round here — the argument above says *why MoE is the likely case*, not
that your kernel is it.

### Clamp the out-of-range program instead of returning it

A metadata-driven MoE launches a **constexpr worst-case** grid and learns the live job count at
runtime, so some programs have no work. The obvious spelling is an early `return`. Surveyed
production writes the other one: clamp the out-of-range pid to the static upper bound and let the
program run every instruction with all of its accesses masked off.

```
max_ctas: gl.constexpr = <static worst case>        # compile-time upper bound
ctas = <runtime live count, read from the counters>  # what actually exists this call
pid = gl.where(pid < ctas, <remapped pid>, max_ctas) # clamp, do not return
```

**The benefit is that no wave exits early, so `gl.barrier()`'s participant set is static.** A
barrier is a whole-workgroup rendezvous; a program that returned before it is a participant that
will never arrive. Early-exit and barriers are safe only when the exit is uniform over the
workgroup, and in a metadata-driven MoE the exit predicate is a runtime count — exactly the thing
that is not obviously uniform. Clamping removes the question instead of answering it, at the cost of
one workgroup's worth of fully-masked work.

Note the clamp target is the **constexpr** bound, not the runtime count, so the subsequent
`gl.load(... , job < JOBS, other=-1)` masks cleanly rather than aliasing a live job. In the surveyed
source this is fused into the same expression as the XCD stripe above — one `gl.where` doing both.
*(The rationale stated here is the reading of an uncommented construct: the code shape is
source-proven, the authors' reason for it is **unverified**. It is written up because the
alternative — an early return in a barrier-carrying kernel — is a real hazard regardless of why
this particular author avoided it.)*

## Getting fp8 into the matrix core: four routes, all in production

Axis 2 said the hardware block-scale operand is never driven. That leaves the question of how the
fp8 operands reach the matrix core at all, and surveyed production answers it **four different
ways** — three of them inside a single family, selected by band.

1. **Scaled MFMA with both scale operands `None`, descaled afterwards.** The dominant route by a
   wide margin. The instruction consumes `e4m3` operands directly; the block scales are applied to
   the accumulator as one fp32 `gl.fma` per K group. This is cheap **because the three granularities
   are made to coincide**: the K block, the quantization group and the MFMA's K extent are all the
   same width, so "apply the scale" degenerates to a single fused multiply-add per trip and the
   accumulator is never rescaled mid-chain.
2. **Upcast to bf16 and issue a regular MFMA.** Exact — every fp8 value is representable in bf16 —
   and it moves the kernel onto the generic matrix counter, which is a diagnostic difference as much
   as a performance one.
3. **Upcast to fp16 and issue a regular MFMA.** Same idea, different target dtype, different
   register cost and a different accumulate path.
4. **No upcast at all: split the product and issue three MFMAs for an exact fp8 GEMM.** The
   interesting one, described below.

**gfx942 downgrade:** route 1 does not exist (no scaled MFMA); the native-fp8 route there is the
regular `cdna3.mfma` fp8 instruction on FNUZ operands, with the same "descale once per K group on
the accumulator" structure. Routes 2 and 3 are unchanged in shape, and `scaled_upcast` on CDNA3 is
the software-emulated path its docstring describes. Route 4's magic-constant split is keyed to
OCP e4m3's 448 maximum; re-derive its binade argument for `e4m3fnuz` before reusing it.

**Route 4, because it is the one that is not obvious.** Each fp8 value is decomposed exactly into a
whole part and a fraction: add a magic constant, subtract it back, and what survives is the
round-to-nearest integer with no branch and no `floor`. With a bias of `1536 = 1.5 · 2^10` the sum
lands in the binade where the fp16 ULP is exactly 1, so the add-subtract *is* a rounding; the trick
is safe precisely because e4m3's largest finite magnitude is 448, so the sum can never leave that
binade — **it is dtype-specific and breaks for any format reaching ±512**. The residual is an exact
multiple of `2^-9`, so scaling it by 512 gives a small exact integer. Writing `a = a_hi + a_lo/512`
and likewise for `b`, the exact product needs four cross terms, and the kernel gets them from
**three** MFMAs the Karatsuba way: `high = a_hi·b_hi`, `low = a_lo·b_lo`, `mixed = (a_hi+a_lo)
(b_hi+b_lo)`, with `cross = mixed − high − low`. Every operand is a small integer, so each MFMA
accumulates integers that fp32 represents exactly; only the `high` partial can outgrow fp32's exact
integer range across a group, so **only that one** is promoted to int32, and the final recombination
weights the three by 1, 1/512 and 1/512² in fp64. The activation halves are precomputed by the
quantize kernel; the weight halves are split in the loop.

**The conclusion is the reason this section exists.** Routes 2, 3 and 4 coexist in one surveyed
family, at adjacent bands, on the same dtype and the same weight contract. So:

> **"How fp8 enters the matrix core" is a shape-tuned knob, not a property of the dtype.** Any
> sentence of the form "fp8 on this target must go through the scaled path" is a dtype-level claim
> where the truth is per-(version, shape) — the regular fp8 matrix intrinsic does exist on this
> architecture, and production uses it deliberately in several files.

### `k_width` at a scaled site is a correctness gate, not a performance knob

This belongs here rather than in a list of adjacent facts, because it sits on every one of the four
routes above and because the way this page used to frame it points at the wrong test.

**The starting fact is unchanged and still right: there is no formula.** At a scaled site the
operand `k_width` is an authoring choice rather than a compiler-enforced contract. Two surveyed MoE
families write *different* `k_width` values at the **same** instruction shape on the same scaled
instruction, both ship, and one file selects between them on the host by row count.
`../gluon/matrix-reference.md` refuses to give a rule in either direction, and that refusal is
correct — every candidate rule offered so far is falsified by a production site. What changes is
what you are supposed to do with the gap.

> **A wrong `k_width` is not a forfeited optimization. It is a wrong answer that compiles cleanly
> and benchmarks normally.** Measured at `instr_shape=[16, 16, 128]` with everything else held
> fixed, sweeping the operand `k_width` and comparing against the fp32 reference:
>
> | `k_width` | 4 | 8 | 16 | 32 | 64 |
> | --- | --- | --- | --- | --- | --- |
> | max relative error | 1.057 | 1.052 | **5.6e-05** | 1.030 | 0.937 |
>
> One value is right. The other **four are wrong by order 1.0 — a ~100 % relative error** — and all
> four **compiled, issued, ran to completion and returned plausible-looking numbers**: no assert,
> no NaN, no lowering failure, no layout-verifier complaint, because nothing in the layout contract
> is violated.

Three consequences, and the first is the one that replaces the old advice:

- **The test is a numerical one, per value.** "Pick a `k_width` and see whether
  `compute_efficient_padded_shared_layout` is still available" is a *performance* test, and it
  passes on values that are numerically wrong — the helper's `None` set and the correct set are
  different sets. **Validate each candidate against a numerical reference on a single tile before
  it enters a kernel**, and read "it compiled and the kernel ran" as no evidence in either
  direction (`../gluon/matrix-reference.md` has the same finding at the instruction's own page,
  with the disassembly).
- **Deriving the value is exactly the trap the absent formula leaves open.** Reasoning from "how
  many elements does one lane hold" is the reasoning that lands on the wrong values here; it is a
  plausible derivation and the page owes you the warning that plausibility is not the gate.
- **Two shipping implementations disagreeing is the situation, not an anomaly to resolve.** The
  value is a property of the operand tile you built, not of the instruction shape — so a value that
  is right in somebody else's kernel at your `instr_shape` is not thereby right in yours. Do not
  carry it across.

One adjacent fact that belongs to another page but changes decisions here:

- **The fp8→wide upcast is a per-element VALU op that the FLOP roofline has no axis for.** A kernel
  on routes 2 or 3 can read as "N% of the MFMA roofline" while the actual bound is the convert
  (`../hardware/optimization-gotchas.md`, row 5). Route 1 is what removes the convert entirely.

## Reduction order is part of the contract, not an implementation detail

A MoE combine sums k routed contributions plus, usually, a shared expert's. Float addition is not
associative, so **the order of that sum is an observable**, and surveyed production treats it as a
published contract rather than as an implementation choice. One kernel's docstring states it
outright: *integer reservations affect execution order, never the reduction order* — the finish adds
the routed slots in index order in fp32, then adds the shared contribution last.

**The separation that makes this work is the thing to learn.** Two orderings exist in these kernels
and they are deliberately decoupled:

- **Execution order is nondeterministic and that is fine.** Slots are reserved with `relaxed` int32
  atomics on counters; which program gets which ticket varies run to run. Across the surveyed
  corpus there are **no float atomics at all** — every atomic is an integer reservation.
- **Reduction order is static.** The combine walks slots with a compile-time unrolled range, so the
  summation sequence is fixed by the **route slot index**, which is a property of the routing, not
  of the schedule. Nothing about who arrived first can reach the arithmetic.

This is why the atomics can be `relaxed` and why the combine can move between kernels
(`### Separate or fused combine is a per-band switch, not a design choice`) without changing a bit
of the output. It is also the direct answer to the reproducibility caveat raised under
`## Bucketing: three shapes, and what selects one`: reservation order being nondeterministic is
harmless **exactly when** the consumer folds by static index rather than by slot arrival.

**Two constructs enforce it, and both look like redundancy until you see what they are for.**

- **The per-add round trip through bf16.** Surveyed source writes `acc = (acc +
  part.to(f32)).to(bf16).to(f32)` on every step — in one family, dozens of times across every file.
  It does not save storage and it does not change the register width; the *only* effect is clearing
  mantissa bits at a specified point in the sequence. That is the point: it reproduces the reference
  implementation's per-element rounding sequence exactly, which is what makes two differently-tiled
  kernels bit-identical. Deleting it as redundant is a silent numerics change, and it is the
  construct in this archetype most often misread that way.
- **Disabling floating-point contraction at the launch.** Where a family is aligning bit-exactly to
  a reference, `enable_fp_fusion=False` appears on its launches — in one surveyed family on every
  file — because a contracted multiply-add rounds differently from a separate multiply and add. This
  tree otherwise mentions the flag only as a diagnostic; here it is half of a numerical contract
  whose other half is the rounding discipline above.

**The operational rule.** Before you move, fuse, split or reorder a MoE combine, decide which of the
two contracts you are under: *any correct order* (then all of this is optional and the reordering is
free) or *bit-exact against a reference* (then the slot order, the per-step rounding and the
contraction setting are all load-bearing, and a schedule change that touches any of them is a
correctness change). Surveyed production is overwhelmingly in the second regime and says so in its
own docstrings — check the kernel's before assuming the first.

### The reference side has a contract too, and it can be ambiguous

Everything above is about pinning the order on **your** side of the comparison. It quietly assumes
the other side is a single, known thing. That assumption has cost a full round.

The shape of the failure: a library ships **two ops with the same purpose and the same signature**
— here, two activation-quantization entry points — that differ in their **scale convention**. The
divergence is small and structured enough to look exactly like a rounding-order bug in the kernel
under test: the two agree everywhere except on one band of the mantissa fraction `f`, where one
rounds up at `f >= 0.75` and the other steps at `f > 0.5`, so they differ by exactly one step for
`f` in `(0.5, 0.75]` and nowhere else. A sparse, structured, one-ulp-scale mismatch is precisely
the signature you would expect from a reduction-order defect, which is why this misleads rather
than merely confusing.

> **Before judging whether two implementations agree, establish that they are the same op.**
> Resolve the callee from the **dispatch path** — follow what the call actually binds to at run
> time — rather than picking the most prominent symbol of that name and kind out of the module.

Three practical consequences:

- **Do this before the numerical debugging, not after.** It is a lookup, and it is cheaper than a
  single bisect over reduction order. The round that was lost here was lost to debugging a
  correctly-ordered kernel against a reference that was not the one the pipeline calls.
- **"Same name, same signature, same module" is not identification.** Convenience wrappers,
  deprecated siblings and dispatch-table variants coexist routinely; the one that is easiest to
  import is not necessarily the one in the path.
- **Record the resolved callee alongside the tolerance**, so the comparison stays reproducible.
  `## Deciding the bound before sizing a prize` makes the same point about error bars: a threshold
  is meaningless without the quantity it is a threshold on, and a bit-exactness claim is meaningless
  without the identity of what it was exact against.

## Gather: separate the index layer from the transport layer

The upstream example does both in one place and only one half of it ports.

**The index layer ports unchanged**, and contains the one idiom worth lifting: out-of-range tokens
are given a **sentinel row index equal to the tensor's row count** rather than a second mask, so the
out-of-bounds row is discarded by the access itself. The example can also cache the loaded index
vector across the N dimension so that the same M block does not re-issue the indirect load — the
concrete mitigation for the address chain `gemm.md` warns about in its access-pattern
variable. Note how upstream treats it: that reuse is a constexpr flag that defaults to **off**, and
only two of its tuned configurations turn it on. It carries a cost — a predicate comparing the
current `(pid_m, slice_idx)` against the cached pair, and the cached vector's own registers — so
upstream treats it as a per-shape tuning decision, not as an always-on improvement. Adopt it the
same way. **Surveyed production reaches the same conclusion from the other direction**: there, the
index-reuse question is subsumed by the activation-reuse factor of
`### Two reuse axes, and this page used to carry only one`, which is likewise band-keyed and
likewise off at the low bands.

**The transport layer is where the port actually breaks, but not for the reason usually given.**
The example moves the gathered rows with a tensor-descriptor asynchronous gather. The descriptor
*spelling* is not the obstacle: `make_tensor_descriptor` is a plain `tl` builtin on all four
versions, not an NVIDIA-only symbol, and a kernel using it compiles for gfx942 and gfx950 — the AMD
backend queries whether the target has TDM hardware and, when it does not, runs a pass that rewrites
every descriptor op back into ordinary pointer arithmetic. gfx950 and gfx942 take that rewrite path.

So what is missing is not the API but the engine behind it, and treating the successful compile as
evidence the mechanism transferred is the trap. Two consequences follow. The descriptor buys you
nothing on CDNA that a pointer tensor does not already give you, because a pointer tensor is what it
becomes. And the **asynchronous gather** form specifically — the descriptor-driven gather the
example depends on — has no CDNA equivalent at all; the replacement is a per-row global load through
a pointer tensor, staged into LDS explicitly, where you regain the addresses but you own the
staging. Consistent with that, surveyed production on this target contains **no tensor descriptors
at all**; its indirection is pointer arithmetic and reinterpretation throughout.

Whether the register-level form is even available is a layout question with a hard test, already
written up: `../gluon/layout-reference.md ## gl.gather — a lane exchange only when the layouts allow it`.
Read it before assuming the gather is cheap; the fallback is silent and allocates scratch for the
whole source tensor. One thing to carry across when you do: the upstream index layout is written for
a 32-lane wave, and gfx950/gfx942 are wave64, so the threads-per-warp component has to be recomputed
rather than copied.

## Pointer tables are the only indirection CDNA has

Since the descriptor degrades to pointer arithmetic anyway, write the pointers. Every indirect
operand — the per-expert weight base addresses, the gathered activation rows, the scatter-back
destinations — arrives as integers and becomes dereferenceable through `gl.pointer_type`. The mechanics and the three ways the cast bites (element
type vs address width, silent truncation from a 32-bit table, faults landing far from the cast) are
in `../gluon/imports-and-launching.md ## Pointers as data (gl.pointer_type)`.

The grouped-GEMM tutorial shows the plain-Triton convention for the table itself, and the Gluon MoE
example shows the epilogue side: the output store is an ordinary store through a pointer tensor at
`slice_offset + row_offsets`, with a runtime mask from the expert's token count. Neither end needs a
descriptor.

## Overlap in a MoE loop: what this archetype actually reaches for

`../gluon/pipeline-reference.md` owns the mechanisms and the version gates; this section says only
which of them a MoE loop reaches for, and in what order, because the ordering an agent infers from
mechanism-page word counts is close to the reverse of what production writes.

**First: register-resident prefetch, and in surveyed production this is the mainstay, not a
fallback.** The construct is a Python tuple of in-flight operand bundles — activation tile, both
weight tiles, both scales — rotated by a statically-unrolled loop: issue the loads for iteration
`i + depth`, issue the matrix ops for iteration `i`, consume `i`'s results, rotate. No LDS, no
`async_copy`, no marker. Two details make it work and both are worth copying: the depth is pinned by
a `gl.static_assert` that the K-group count is divisible by it, so the prologue/steady/epilogue
split is exact and the unroll leaves no remainder loop; and the depth itself is a launch argument
chosen by band — deep at the smallest M, shallow as M grows and the registers are wanted elsewhere.
This is the form to try first on a MoE expert GEMM.

This is the same hand-written-first order `../tile-programming/pipeline.md` defines for every
Gluon loop, instantiated for MoE; re-injecting plain's pipeliner sits below all four items here
(a below-parity diagnostic or last resort, numbers labelled *injected*, never on an incumbent).

**Second: an async-copy ring, but only when the reuse factor has earned it.** On gfx950 that is
`async_copy` with `commit_group` / `wait_group`; on the gfx942 downgrade direct-to-LDS async is
32-bit only with `order=[1,0]`, so the ring is usually sync-staged through registers. The surveyed family
that prefetches in registers uses async copy in only a small minority of its files, all at its top
band — and `### Two reuse axes, and this page used to carry only one` explains why: below a high
activation-reuse factor there is not enough matrix work per weight block to hide a transfer behind.
A second surveyed family builds its overlap this way throughout, which is the (model, band) point
again. Treat "add an async ring" as **downstream of raising reuse**, not as an independent lever.

**Third: `gl.amd.warp_pipeline_stage`, and in this archetype it is almost always the scheduling
wall rather than the phase offset.** Its Gate 0 is `num_warps >= 8` — the pass computes its group
from `warpSize * 4` and never reads `num_warps`, so below that the warp index is identically zero
and the phase offset silently does not exist (`../tile-programming/warp-pipeline.md ## Gate 0`).
Every surveyed MoE site sits in a kernel launched at one warp, so **no MoE site in the survey has a
live phase shift**. What survives is the border marker's `sched_barrier`, an intra-wave scheduling
wall, which is a real and sometimes wanted effect — but it is a different effect, and reporting a
timing change from it as "the warp pipeline helped" is the misattribution to avoid.

**Fourth, and only to close it out: `num_stages` is not a lever on this path** — it is dead on the
Gluon path in 3.8.0 (no pass consumes it). One surveyed family
writes it zero times across its entire MoE tree; the other writes it three times, all with the value
that means *off*. Its absence is not evidence about pipelining behaviour in either direction
(`../gluon/pipeline-reference.md`).

**The barrier question, because two surveyed siblings disagree on it.** Two prefill kernels with the
same double-buffered LDS weight panel and the same cross-wave read pattern do different things: one
follows **every** `wait_group` with `gl.barrier()`, the other follows **none** of them, keeping only
the fence before the refill. The criterion that resolves which is required is not about the depth:

> `wait_group` retires **this wave's** outstanding copies. It is not a workgroup rendezvous and it
> carries no LDS-write visibility for anyone else. So the post-wait barrier is **mandatory whenever
> a wave reads an LDS region another wave filled** — which is the normal case for a shared weight
> panel at `num_warps > 1` — and unnecessary only when each wave reads exactly what it itself
> issued. The mirror-image barrier before the refill guards the other hazard and is a separate
> decision.

Applied to the pair above, the file without the consume-side barrier is relying on something the
source does not state. **This page does not adjudicate it** — both ship, and "it ships" is weak
evidence about a race, which is not obliged to fail every run. What is settled is the criterion;
the two-hazard framing and the three concrete barrier placements are owned by
`../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier
placement that differs between them`, and if you cannot say which hazard a barrier of yours is
guarding, resolve that before tuning the depth.

## Version gates

**Scope signpost, because this table is not MoE-specific.** The rows below are language-level
facts and are true for any workload; they live on this page for historical reasons and are cited
from four other files by this heading, so they stay here. What *is* MoE-specific is the grading
underneath each row — which of these capabilities this archetype actually reaches for. If you want
the version question and not the MoE reading, the four-version authority for pipeline and marker
symbols is `../gluon/pipeline/marker-and-version-gates.md`, per-target capability is
`../hardware/capability-matrix.md`, and `../../scripts/probe_levers.py --all` answers it for the
build in front of you. Read the table here, read the paragraphs after it as MoE advice.

Columns run newest first: 3.8.0 is the target, the older tags are downgrade notes. The `cdna4`
rows are the gfx950 namespace. On gfx942 the padded-layout helper is unavailable (it asserts an
MFMA v4 parent), and whether a `cdna3` counterpart of `scaled_upcast` exists on your build is a
probe (`../../scripts/probe_levers.py`) — where it does, it is the software-emulated path.

| Capability | 3.8.0 | 3.7.1 | 3.7.0 | 3.6.0 |
| --- | --- | --- | --- | --- |
| `gl.gather` (register-level lane exchange) | ✓ | ✓ | ✓ | ✓ |
| `gl.pointer_type` | ✓ | ✓ | ✓ | ✓ |
| `gl.associative_scan` | ✓ | ✓ | ✓ | ✓ |
| shared-descriptor `.gather` / `.scatter` | ✓ | ✓ | ✓ | absent |
| shared-descriptor `.atomic_scatter_add` / `_max` / `_min` / `_and` / `_or` / `_xor` / `_xchg` | ✓ | absent | absent | absent |
| `gl.amd.cdna4.scaled_upcast` (fp8/fp4 + E8M0 -> bf16 in one op) | ✓ | absent | absent | absent |
| `gl.amd.cdna4.compute_efficient_padded_shared_layout` | ✓ | absent | absent | absent |
| `gl.amd.slice` (register-only sub-tile, layout preserved) | ✓ | absent | absent | absent |
| `gl.sort` / `gl.topk` / `gl.cumsum` | absent | absent | absent | absent |

**The three `3.8.0`-only AMD rows are not equally live for this workload.** `scaled_upcast` is
graded in `### Axis 2 — how the quantization reaches the matrix core`: reachable only from E8M0
scales, therefore unused across every surveyed MoE file. `gl.amd.slice` extracts a sub-tile in
registers while **keeping the source's distributed layout**, so it costs no cross-lane movement —
but the slice extent and offsets must align to that layout's tiling, and surveyed MoE source does
not use it even though other surveyed workloads do; treat it as an untaken option on this archetype
rather than as established practice. The padded-layout helper is the one with real MoE uptake and
it has a trap, below.

**`compute_efficient_padded_shared_layout` returns `None` instead of raising, and the condition is
coupled to a decision you made earlier.** It derives a bank-conflict-avoiding `PaddedSharedLayout`
from the dot-operand layout, the shared tile shape and the element type, so it replaces a hand-built
padding pattern with one call. One assert and three `None` conditions. The assert: the operand's
parent must be an **MFMA v4** layout. The three `None` conditions, per its own documented contract,
are `k_width` outside `{4, 8, 16}`, **element bit-width outside `{4, 8, 16}`**, and **an MFMA
instruction-shape / `k_width` combination the algorithm does not handle** — an earlier revision of
this paragraph listed only the first and the third, and the omission matters because the third is
the one that fires most. All of them are silent: an allocation built from a `None` layout fails
later and elsewhere, not at the call.

The coupling to watch is that `k_width` at a scaled site is **not** pinned by any formula
(`../gluon/matrix-reference.md`), so the value you pick for unrelated reasons decides whether this
helper is available at all — a `k_width` of `32` is outside the set and loses it without saying so.
**Do not let that be the reason you choose one.** Availability of this helper is a performance
consequence; the *first* consequence of the same choice is numerical, and it is silent
(`### k_width at a scaled site is a correctness gate, not a performance knob`). Settle the value
against a numerical reference, then read this paragraph to find out what it cost you.
**And the coupling that actually bites this archetype is the tile, not the `k_width`.** Measured
at `instr_shape=[16, 16, 128]`, the third condition turns on the **non-K tile length**: inside the
legal `{4, 8, 16}`, a short non-K extent returns `None` at every one of those values, and a long
one returns a layout at all of them. That makes `None` the ordinary outcome in the **decode band**,
where the row tile is small by design, and a non-event at prefill tiles — so a decode kernel that
gets `None` here should not go looking for a `k_width` mistake. Do not turn that into a
`BLOCK_M` threshold, though: the cut-off moves with the **instruction shape and the operand index**
as well as with the tile, and at `[32, 32, 64]` the two operands land on opposite sides of it — so
**gate on the returned value, per operand**, not on a tile size
(`../gluon/layout-reference.md ### Contract points — places this goes wrong quietly` has the sweep
and its scope). In surveyed production source both paths ship: the files that call the helper are bf16
router-projection kernels in the decode band at `k_width` `8`, while the prefill-band fused kernels
build `gl.PaddedSharedLayout` by hand. Hand-building stays correct and stays version-portable; the
helper is the shorter road when you are inside its supported set. **Check the return value for
`None` before allocating.**

Four notes on the atomic-scatter row, because it is the one most likely to be misread:

- **The spelling is the whole surface, and it is why this row gets read as dead.** There is no
  pointer-form `smem.atomic_add`; the scatter form is all there is. A search for the atomic-add
  name on a descriptor therefore comes back empty whether or not the mechanism is in use, so
  "nobody uses this" is a conclusion that spelling alone can manufacture. Uptake of this family is
  small and concentrated in exactly the bucketing shape above — small is not zero. The mechanism
  page is `../gluon/smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family`,
  which carries the signature, the barrier discipline, the determinism argument you have to own,
  and the downgrade.
- **These are methods on a shared-memory descriptor — LDS atomics, not global ones.** They are the
  natural spelling for accumulating per-expert partials in LDS before a single global write-out.
  They are unrelated to the cross-device atomics in `collective.md`, which are global and
  carry `sem` / `scope`.
- **The AMD backend does lower them on 3.8.0**, for both gfx942 and gfx950, to a workgroup-scoped
  atomic on the LDS address space. This is not an NVIDIA-only 3.8.0 feature.
- **`add` and `xchg` accept integer and floating dtypes; `max`, `min`, `and`, `or` and `xor` are
  integer-only** and raise at compile time otherwise. So an expert-wise *accumulate* and an
  expert-wise *claim* both work directly on fp32 partials, while an expert-wise **maximum over
  floats does not** — it has to go through an order-preserving integer key, the same trick the
  routing kernel uses for its sort, with the four incompatible spellings of
  `### The monotone integer key has four incompatible spellings`. `max` and `min` are also
  signedness-aware: an unsigned tile dispatches to the unsigned comparison, which is exactly what
  an integer-key scheme wants.

The `gl.sort` / `gl.topk` / `gl.cumsum` row is the load-bearing one for the whole workload, and
`## Which tier the routing kernel belongs on` is where its consequences are worked out. Its
practical meaning is visible in production: **every ordered top-k in the surveyed corpus is
hand-written**, which is why that page has four generations of them and four key encodings rather
than one call.

## Deciding the bound before sizing a prize

This archetype has a real byte/FLOP model (`scripts/hw_budget.py --workload moe`), it defaults to
`memory`, and its own metadata says the default answer should not be acted on until one variable is
measured. That caveat is in the model and not in any prose, which is how a MoE round ends up
optimizing the side that was never binding.

### First branch: is this phase bound by the host, not by the device?

Everything below prices bytes and FLOPs, which presumes the phase is device-bound. That is a
question, not a given. A MoE step is assembled from many small dispatches, so a phase of it can
be **host-dispatch bound** instead — and whether it actually is depends on the serving contract,
not on the kernels. For such a phase every quantity the rest of this section computes is
irrelevant:

> **When a phase is host-dispatch bound, the only lever is the *number* of dispatches. The
> ceiling on every device-side optimization is zero.** Not small — zero. Making each kernel
> faster shortens a span the wall clock is not waiting on.

The count that matters is **not** the number of launches in source order: **it is set by the
dependency edges.** Two adjacent launches with no dependency between them do not have to be two
serial dispatches, so reading the source top-to-bottom systematically over-states the serial
launch count — and therefore over-states the prize available from removing any one of them.

**Sizing that prize needs a per-dispatch cost, and this page deliberately does not give you a
number for it**, because a per-dispatch cost is not a constant of the machine:

- **Declare the contract first.** A per-call launch in eager execution and a node inside a
  captured-graph replay are different things by an order of magnitude, not by a correction
  factor. **A per-dispatch cost measured under one contract cannot be carried to the other**, and
  an unlabelled absolute figure is the form this error takes — the number looks like a hardware
  property, so nobody re-derives it. Whichever contract production uses is the one to measure in.
- **Measure a slope, not an absolute.** Insert `N` empty nodes into a **real** multi-node graph
  of the shape you actually run, sweep `N`, and regress time on `N`. The **slope** is the
  marginal cost of one dispatch; the **intercept absorbs your instrument's own floor**, which is
  exactly the term that corrupts a single absolute reading. Report the `r²` alongside the slope,
  and run a null control, so the fit is falsifiable rather than decorative.
- **Suspect the floor whenever a measured per-launch cost lands near your tool's own floor.**
  Timing harnesses have a fixed cost of their own, and a one-launch measurement cannot separate
  it from the launch. If the two are the same order of magnitude, the default reading is that you
  measured the floor. The way out is the sweep above — the slope is insensitive to a constant
  floor, and `N = 1` can never be.

Getting this wrong is not a rounding error in the prize, it is a reversal of the verdict: an
inflated per-dispatch cost turns a modest host share into an apparent host wall, and the phase
gets closed as hopeless when the device side was the lever all along.

**At the decode band, carry that per-dispatch cost into the budget of every optimization, even
when the phase is device-bound.** A decode call is O(100 µs) of wall time over kernels doing
O(10 µs) of work each, so one extra graph node is not overhead to be amortized — it is a
first-order line item, and it belongs in the budget table next to the bandwidth or FLOPs the
optimization is meant to save. Measured by the slope method above on a real five-node decode
graph: one node was **~1.4 % of a call-weighted decode call, and ~2.4 % of the smallest shape's
call** — and that shape carried a quarter of the call weight. The consequence is qualitative
rather than quantitative:

| split-K arm | call-weighted net |
| --- | --- |
| with the one extra node it needs | **+1.38 %** |
| the same arm, node cost excluded | +2.80 % |

**The node ate half the lever**, and at the smallest shape it turned a gain into a small loss. So
the question "should this kernel do a split-K reduction" was decided not by the split factor but
by **whether the reduction can be folded into an adjacent kernel's prologue or into the last
split** — i.e. by whether a node can be avoided at all. State the criterion that way:

> **An optimization that requires adding a kernel must beat roughly 1.5–2 % before it is net
> positive at this band.**

This is also the reading of why production decode MoE crowds routing, quantization and reduction
into neighbouring kernels: that is **not** a bandwidth argument, it is a node-count argument.

So before sizing anything: establish the phase is device-bound. The discriminator is free and is
written up in `../method/benchmark-hygiene.md ## Measurement basis + benchmark-artifact pitfalls` —
host cost does not scale with problem size, so a segment whose
time is flat against a shape sweep is reporting issue cost. Note also that event timing does not
raise an error in this situation; it silently reports the host interval under a device-time
label, so a plausible-looking "kernel time" is not evidence that the phase is device-bound.

**Weight bytes dominate, so the expert *count* is the whole model** — not the tile, not the token
count. That makes `E_touched` (the experts actually routed to) the one variable the budget turns
on, and it is a property of the runtime routing distribution rather than a shape you can read off
the launch.

- The default is `E_touched = min(E, top_k * M)`, and it **over-counts**: routing collisions put
  several of those (token, slot) pairs on the same expert.
- At decode `M` the over-count is about `E / E_touched`, which is large enough to push the model's
  HBM floor *below* the measured time. The tool reports that as a breached floor and names this
  cause first. **Read it as the model announcing it does not apply to your shape** — not as a
  several-fold prize sitting on the table.
- Passing a measured value (`--shapes E_touched=...`) does not make the model exact; it narrows the
  caveat to "a lower bound at a measured `E_touched`", because the activation term still
  under-counts cross-block re-reads. `--measured-dram-mb` is what takes the byte model out of the
  path altogether.

**And the byte account has an end, so say when you have reached it.** Deduplicating the expert
weight reads is the lever this section points at, and it terminates: measured on a decode band, a
deduplicated kernel moved **470.2 MB against a 470.2 MB optimum — a ratio of 1.0000** (n = 64
routing draws × 7 shapes, against 842.3 MB naive), and was **still behind the reference at all 7
shapes**. At that point there are no bytes left to remove and every further question is about
**achieved bandwidth**, which is a different measurement with a different instrument: it needs a
one-variable twin probe on the same access shape, and it cannot be inferred from a spec number
(`## Deciding the bound before sizing a prize`). The cost of not writing that sentence down is
concrete — a re-plan was built on the byte framing after the byte ratio had already been closed,
and attributed a gap to dedup bytes that dedup could no longer supply. **Scope: uniform routing
over a large expert count.** Concentrated routing raises the token count per expert and makes the
per-expert chunk loop run more than one trip, which is the one regime that could reopen the byte
question; it was not measured.

**`stage` is the second thing to set, and it is a 1.5x error in either direction.** It selects which
GEMM you are budgeting and scales **both** terms: `stage=1` (w1, gate+up) is `2*I*H` per expert and
is the **default**; `stage=2` (w2, down) is `I*H`; `stage=0` (one fused kernel doing both) is
`3*I*H`. So a two-kernel MoE left on the default has budgeted one of its two kernels, and the same
kernel budgeted at `stage=0` is over-counted by 1.5x. Mixing one stage's FLOPs with both stages'
bytes is worse than either: it does not err conservatively, it produces an arithmetic intensity that
means nothing. **Axis 1 is what tells you which `stage` to pass** — a complete-fused entry is not
automatically `stage=0`, because it is usually several launches behind one Python function, and
what you are budgeting is a launch.

**A low-precision weight carries a scale tax the raw weight count misses.** `scale_overhead` adds
the micro-scaling scale bytes — a block-32 e8m0 scale is `1/(32*b)` of the payload, so **6.25 % at
fp4 and 3.125 % at fp8**. It is derived from the dtype; override it only against a measurement. On a
workload whose model *is* the weight bytes, that is a line item rather than a rounding error. Note
the interaction with axis 2: production's scales are **fp32 over a 128-wide group**, which is a
different byte count from the block-32 e8m0 default — this is a place where the dtype-derived
default and the checkpoint disagree, and the checkpoint wins.

> The model's note also quotes a tokens-per-expert threshold for the memory→compute crossover. That
> figure is for a different part and a different precision. The ridge is a SKU property: recompute
> it for the chip you are on (`../hardware/roofline-models.md`) instead of carrying the number
> across. The same applies to any per-expert-shape overhead claim — grouped GEMM pays one, and its
> size is a measurement, not a constant.

### Two independent gates: readable is statistics, meaningful is physics

Sizing a prize needs an effect to clear **two** thresholds, and they are not the same threshold:

1. **Readable** — is the effect larger than the run-to-run noise of the measurement? A statistics
   question, answered with standard errors.
2. **Meaningful** — is the effect large enough to matter against the quantity it is supposed to
   move? A physics question, answered against the scale of that quantity.

> **Passing one says nothing about the other, and the common failure is to check only one — usually
> the wrong one.** An observation that has been retracted from this page cleared the first gate by a
> wide margin (about **39 standard errors**) and failed the second one completely: the effect was
> **0.011 %** of the common radius of the two things being compared. Thirty-nine sigma of something
> that cannot matter is still something that cannot matter.

**And the error bar has to be on the statistic you are comparing.** The trap that produces those
39 sigma is picking the wrong quantity to put the error bar on, and it is subtle enough that it has
been got wrong inside this campaign:

> **Run-to-run *distance* is not the error bar for a *difference of distances*.** If you are
> comparing "how far is A from the oracle" against "how far is B from the oracle", the noise figure
> you need is the variation the perturbation induces **in that difference** — not the magnitude of
> the perturbation itself.

The two are not close. When the perturbation is near-orthogonal to the direction being measured,
almost all of its magnitude cancels out of the difference: measured here, the perturbation's own
size was **0.001363** while the induced change in the compared statistic was **4.38e-07** — the
naive error bar overstates the real one by roughly **3100x**. Used as a threshold, it hides every
effect below it; used as a denominator, it manufactures significance. **Derive the error bar by
perturbing and re-computing the statistic you actually report**, rather than by measuring how big
the perturbation was.

### The mnemonic histogram is the cheapest ceiling tool there is

Before writing an instruction-level optimization, count the instructions. A histogram of mnemonics
from the compiled assembly gives an **absolute upper bound** on any change that only removes or
cheapens one class of instruction: if that class is 1.2 % of the issue mix, no rewrite of it can
return more than 1.2 %. One candidate direction here was closed on a **1.172 %** bound without a
line of optimization being written — which is the whole point, because the alternative was a round.

> **Read the whole histogram, not only the row you came to count.** The direction that actually
> shipped on that kernel, worth about **2 %**, was a division sequence two rows below the entry
> being investigated — nobody was looking for it, and it was visible in the same output. The
> histogram is a survey instrument that happens to also answer your question; treating it as a
> single-row lookup is how the larger item gets walked past.

One thing the histogram is **not**: a check that two builds are the same program. Counts are
order-independent, so a reordered program passes an identical histogram — and adding more tallies
(registers, LDS bytes, code length) does not fix that, because none of them are order-dependent
either. Use it to bound a prize; use a normalised line-by-line diff to establish identity.

Two mechanical notes, both checked against `v3.8.0`, because getting either wrong makes the counts
meaningless:

- **Take the register figure from the compiled metadata, not from a `CompiledKernel` attribute.**
  `CompiledKernel.__init__` does **not** set `n_regs`; it is assigned only inside `_init_handles()`,
  which runs on first launch (via the `run` property, `__getitem__`, or `launch_metadata`), and the
  class has no `__getattr__` fallback — so on a compiled-but-never-launched kernel, `.n_regs` raises
  `AttributeError`. Even after a launch it is a **HIP runtime query**
  (`hipFuncGetAttribute(..., HIP_FUNC_ATTRIBUTE_NUM_REGS, ...)`, inside the AMD backend's
  `load_binary`), which is
  a different provenance from the compiler's static count and should not be assumed interchangeable
  with it. The static figure is in the assembly metadata: regex `\.vgpr_count:` out of
  `kernel.asm['amdgcn']`. Upstream's own AMD Gluon example at that tag does exactly this, and the
  same pass picks up `.sgpr_count`, `.sgpr_spill_count`, `.vgpr_spill_count`, `ScratchSize`,
  `codeLenInByte` and `Occupancy` — take all of them while the text is open.
- **Filter the assembly before counting.** Skip `.`-prefixed directives, `;` comments, labels, and
  `s_nop`. Directives and comments are not instructions, labels are not instructions, and `s_nop`
  is padding whose count says nothing about the work; leaving any of them in moves the
  denominator, and the denominator is the entire value of the bound.

## Tile choice: tokens-per-expert is the bucketing variable

`gemm.md` organizes by M regime; MoE is the case where **M is a runtime distribution
rather than a number**, and the expected tokens-per-expert is the thing that selects the bucket. The
upstream example picks its M-tile by expected slice size across **four** sizes — 16, 32 and 64 from
its tuned branches, falling back to 128 for shapes no branch claims. Note where the spread sits: the
default is the ordinary GEMM tile, and the tuned sizes run an order of magnitude below it, so the
tuning exists entirely to make the tile *smaller* as the per-expert token count shrinks. Treat a single tuned `BLOCK_M`
for all experts as a bug, not a simplification, and read the budget as a distribution — the three
consequences in `attention.md ### The dynamic-body variable (sparse / paged / varlen)`
apply here without modification.

**Surveyed production spans the same range and resolves the bottom of it differently from the
upstream example.** Its row tiles climb with the band — of order 16 through the decode band, 32
below the middle of the middle band and 64 above it, and at prefill a router tile and a
down-projection tile that are separately chosen and separately doubled at a token threshold. What it
does *not* do is shrink the row tile below the matrix instruction's own M extent at decode. At the
smallest M there is no row tile at all: the kernel issues one program per token and the M dimension
disappears from the **tiling question**. That is the shape to compare against before concluding a
tiny `BLOCK_M` is the decode answer.

**"M disappears" is a statement about the tiling, not about the instruction, and the difference
matters on the scaled path.** `gl.amd.cdna4.mfma_scaled`'s M extent is hard-wired by its
`instr_shape` — 16 at `[16, 16, 128]`, the shape the f8f6f4 entries are registered at
(`../gluon/matrix-reference.md ## Matrix-Family Details`). So at decode, one program per token
still issues a 16-row matrix instruction of which **1 row is real**; you cannot tile M below 16 on
this path and there is no `warps_per_cta` setting that recovers it.

**That is acceptable, and the reason is worth stating in full rather than filed as a known
inefficiency.** The decode band is memory bound: the weight block is read in full regardless of how
many token rows consume it, and the 15 idle accumulator rows generate **no traffic** — they cost
matrix slots the kernel was not going to be limited by. Paying registers or restructuring to
recover them is optimizing the resource that is not scarce. What *would* change the answer is
finding independent work to put in those rows, which is a different edit and is written up at
`../gluon/matrix-reference.md ## When the matrix core is the wrong instrument (gl.dot_fma)`.

**The one path where M genuinely disappears is `gl.dot_fma`**, which has no instruction shape at
all and issues only the work it has. That is the trade to weigh at decode — matrix core with 15/16
of the instruction idle but no VALU chain, versus an FMA tree with no waste but no matrix
throughput — and it is a measurement, not a derivation.

### `BLOCK_M` at prefill is a register-allocation decision first, a padding decision second

**Read this before the padding argument below it, because it predicts the opposite winner.** The
padding reasoning says a bigger row tile costs more wasted matrix slots at prefill, so it pushes
you toward the smaller tile. Measured, the largest tile wins — once the launch geometry lets it.

One prefill expert GEMM, at `M = 1936`, sweeping `BLOCK_M` at two warp counts:

| | `BLOCK_M=32` | `BLOCK_M=64` | `BLOCK_M=128` |
| --- | --- | --- | --- |
| `num_warps=4` | 423 µs | **344 µs** | 662 µs |
| `num_warps=8` | 439 µs | 419 µs | **237 µs** |

At `num_warps=4` the `BLOCK_M=128` cell is not paying for padding — it is **spilling**. That
configuration puts 128 VGPRs per lane into the accumulator alone and the compiled kernel reports
`.vgpr_spill_count: 394`. Doubling the warps halves the per-lane accumulator, the spill count goes
to **zero**, and `BLOCK_M=128` then wins at **every** M across the band. On the same kernel that
took the largest-M case from 5729 µs to 1902 µs — **3.0x** — from a two-parameter change with no
structural edit.

> **The methodology rule, which is the transferable part: `BLOCK_M` and `num_warps` are coupled,
> and sweeping either one alone produces a spill-contaminated conclusion.** The `num_warps=4` row
> read on its own says "128 is too big"; that is the register allocator talking, not the tile. If
> you take one thing from this section, sweep the pair, and read `.vgpr_spill_count` out of the
> compiled kernel before you believe any ranking of tile sizes
> (`../pitfalls/negative-patterns.md ## gfx942 / CDNA3 negative-result signatures (mechanism + error
> text)`, item 2, is the same rule reached from the pipeline side;
> `../tile-programming/slicing.md` owns the combined VGPR+AGPR budget the cliff is against).

The accumulator's register cost is `BLOCK_M * BLOCK_N / (64 * num_warps)` VGPRs per lane, so it is
the only term in the tile choice that `num_warps` moves. That is the whole mechanism on the
too-big side: there is a cliff, it is not smooth, and the tile that is one step past it looks like
a bad tile rather than like a spilled one.

#### What raising `num_warps` does and does not buy you

`num_warps` is where the occupancy ladder is actually spent in a Gluon kernel, and it is the one
knob that changes the *units* the ladder is written in — so read this before motivating a warp-count
change by occupancy. The 3.0x above came from **deleting a spill**, not from adding residency, and
those are different claims with different evidence.

**The published ladder is in waves per SIMD; the residency that feeds bandwidth is workgroups per
CU, and a workgroup is indivisible** — leftover waves in the register file cannot be filled by half
a workgroup. Convert before spending a round
(`../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a
round on it` owns this):

```text
wg_per_CU = min( waves_per_SIMD_by_VGPR * simd_per_cu // num_warps ,
                 LDS_per_CU // lds_bytes_per_wg )
```

The two forms coincide **only** when `num_warps == simd_per_cu`, so carry `num_warps` as an
explicit column on any occupancy table you write here. Three consequences land directly on this
section's knob:

- **Raising `num_warps` deletes most of the ladder.** At `num_warps=8` on a 4-SIMD CU the only VGPR
  thresholds that still move `wg/CU` are **64, 128 and 256** — 129–256 is one flat plateau, and a
  rung like `<= 168 → 3 waves/SIMD` is worse than useless there, because it allocates a third
  wave's registers per SIMD that can never be occupied. Measured: an arm that cut `next_free_vgpr`
  from **240 to 146** (two rungs, −39 %) bought **exactly zero** additional workgroups per CU.
  Before spending a round on register relief at `num_warps=8`, compute the *next* threshold that
  changes `wg/CU` and check the tile can reach it.
- **The VGPR half of a "fits 2 wg/CU" entry condition goes vacuous exactly where this section
  operates.** The ceiling is `512 * simd_per_cu / (2 * num_warps)` — 128 at 8 warps, **256 at 4**,
  512 at 2 — and on gfx950 256 is also the architectural per-thread maximum, so at
  `num_warps >= 4` the condition **can never fail**. One configuration read `vgpr_count: 256`,
  "passed", and carried **345 spill slots**: the register demand did not shrink, it changed address
  space. This is why the coupled-pair rule (sweep `BLOCK_M` and `num_warps` together) says read
  `.vgpr_spill_count` from the same artifact, in the same row — at these warp counts it is the only
  half of the pair that can still say no.
- **Which resource binds is a property of the `(kernel, configuration, source version)` triple, not
  of the kernel.** In one four-cell sweep the binding resource read register / register / LDS / LDS,
  and a single load-packing change moved it from the register side to the LDS side. Re-determine it
  after every structural edit and **retire the old answer in place** rather than leaving both
  standing.

> **And keep the criterion one-sided: a computed occupancy *loss* is a legitimate veto; a computed
> occupancy *gain* is not an argument that a change will win.** On CDNA4 a **+28.6 %** computed-
> occupancy arm measured **+0.16 %** call-weighted and carried a **−2.9 %** regression at exactly
> the size that should have been most occupancy-limited; a deliberate **−50 %** measured
> **−2.81 %** overall and was *faster* at the large-M buckets; **+16.7 %** measured 1.038x. Two
> directions, labelled points, and the arithmetic right every time while the prediction was wrong
> every time — **do not average them into an exchange rate, two contradictory points are not a
> curve**, and do not write the opposite over-reach ("occupancy is worthless here") either. What
> survives is the compile-side feasibility use: does the configuration fit, does it spill, what is
> the next threshold that actually moves `wg/CU`. If you intend to spend on this axis, measure your
> own kernel's Δtime per Δoccupancy once first, and pre-register a floor — `+0.2 %` is the size of
> answer this axis returns when it returns nothing.

**The too-small side loses for an unrelated reason, and a zero spill count does not clear a tile.**
Halving the row tile from the winner cost **0.8379x** on the same kernel with
`.vgpr_spill_count: 0` — nothing had spilled, so the register story says nothing about that cell.
What it bought instead was twice as many blocks, each of which re-reads the expert's **entire
1.57 MB weight panel**: the weight traffic scales with the block count, not with the useful work,
so halving the tile doubles the bytes moved for the same arithmetic. Two different mechanisms
therefore bound `BLOCK_M` from the two sides — registers above, weight re-reads below — and
**neither one is visible in the other's diagnostic**. A sweep that only watches the spill count
will conclude a too-small tile is healthy.

*Scope:* one kernel, one band, one architecture. The coupling, the spill-cliff shape and the
two-sided bound are the general claims; the specific winning cell, the `0.8379x` and the panel size
are not.

### Padding's cost changes sign across the bands

An earlier version of this page said tile padding is "real and usually irrelevant, check which
before optimizing it". The check is right and the default conclusion misleads, because **padding is
not one cost — it changes sign across the bands**, and the band is knowable before any measurement.
It is, however, the **second** thing to price at prefill and not the first: the register/spill term
in `### BLOCK_M at prefill is a register-allocation decision first, a padding decision second` is
the term that dominated when both were measured on the same kernel, and a padding argument built
on a spill-contaminated sweep will point the wrong way.

- **At decode, padding costs latency.** The weights are read in full regardless, so the padded
  matrix slots are not the constraint; what *is* the constraint is the dependent chain through one
  token's tiny GEMM. Production's response is not a smaller padded tile but **no M tiling at all** —
  one program per token — which removes the padding by removing the dimension.
- **In the middle band, padding really is ignorable.** Weight bytes still dominate the budget by a
  wide margin and the padded fraction is small once an expert holds tens of rows. This is the band
  the old sentence was true for.
- **At prefill, padding becomes wasted compute, and production starts paying to bound it.** Once the
  activation-reuse factor has amortized the weight reads, the kernel is on the compute side and the
  padded rows are real matrix work. The tell in surveyed source is a host-side worst-case **job
  capacity** — a reservation sized for one padded block per expert — that is computed *only* above a
  token threshold and is otherwise zero, on the line next to the one that doubles the
  down-projection row tile at the same threshold. A bigger tile means more residual padding per
  expert, so the two decisions are the same decision.

**So the instruction is: price padding against the bound, and read the bound off the band.** The
question "is this kernel on the weight-bytes side or the matrix-slots side" is the one that decides,
and `## Deciding the bound before sizing a prize` is how to answer it rather than assume it.

### Three different things are called "padding" here

They have different purposes, different failure modes, and no relationship to each other. Confusing
them is a live source of wrong reasoning because all three are visible in the same file.

1. **Route slots padded to a power of two.** A top-k that is not a power of two is padded up so a
   `gl.split`-style reduction tree over the slots is expressible at all. This is a *layout*
   requirement of the reduction primitive; it has nothing to do with tiles, and the extra slots
   carry zero weight.
2. **Tile padding of each expert's token run.** Each expert's run is rounded up to the row tile so
   that **every block belongs to exactly one expert** — which is what lets the expert GEMM skip all
   cross-expert boundary handling inside its loop. The cost is bounded and computable: at most one
   partial block per expert, which is exactly the term a production host-side capacity bound
   reserves. Far from being an inefficiency to remove, in the surveyed family this is the
   foundation the whole schedule rests on; removing it puts a boundary test back in the hot loop.
3. **Padded route entries that have to be made harmless.** The two surveyed families take
   **opposite** approaches and both are correct. One points the padded routes at **token 0** — a
   valid gather source, so no branch and no fault — and annihilates their contribution with a zero
   row scale plus a masked store. The other writes a **sentinel expert id one past the real range**
   and masks on it. The opposition has a downstream consequence: the sentinel convention forces
   every histogram over expert ids to size its bins above the sentinel (or pass an explicit mask and
   account for the sentinel separately), while the token-0 convention leaves the id space clean but
   requires that *every* consumer of a padded row be scale-and-mask safe. Pick one per kernel and
   check every consumer against it; mixing them is how a padded row survives into the output.

## Implementation-level facts that decide rounds, and are not in the layer ladder

Four things bite on this archetype specifically. None of them is a layer to climb; each is a check
to run once, early, because each can invalidate a round's premise rather than merely cost time.

**1. The compiler scheduling ladder is excluded here, and the failure is not a slowdown.** A
block-scaled expert GEMM puts vector work *between* matrix ops — the scale multiply-add sits at
every K-group boundary. On that shape the in-tree scheduler does not produce a worse schedule, it
produces **invalid IR or trips a verifier assertion** (`../hardware/optimization-gotchas.md`). So
treat the scheduler rungs as unavailable on the scaled path rather than untried, and do not read a
crash there as a bug in your kernel. The portable per-compile attribute is a separate mechanism
with its own gate (`../tile-programming/instruction-scheduling.md`); it is not the same rung, and it
is the one surveyed production does use — selected **per launch and per M regime** out of the same
source file, which is the granularity a process-wide environment variable cannot express.

**2. Matrix engagement needs a dtype-aware read, and this archetype is where it is always needed.**
A scaled matrix pipeline does not necessarily increment the generic matrix-utilization counter. On
a fused MoE whose expert GEMM is scaled throughout, that counter reads near zero while the matrix
engine is saturated — and the natural conclusion ("not compute-bound, go find bytes") is exactly
wrong. Use the unified matrix-engagement test in `../hardware/bound-class-signals.md` and state
which path the measured entry took. This is the single most likely way to misclassify a MoE kernel,
and axis 2 is what tells you which caliber to expect: route 1 of
`## Getting fp8 into the matrix core: four routes, all in production` is invisible to the generic
counter, routes 2 and 3 are not.

**3. A single-warp workgroup is the decode band's default, and only becomes a hazard in
combination.** Start from what production writes rather than from the hazard: across the surveyed
decode files, the prepare, router, select and gate stages are launched at **one warp**, uniformly,
because the ballot-shaped top-k needs the tile to *be* the physical wave
(`## Which tier the routing kernel belongs on`) and because a body with no cross-wave data has
nothing to gain from more. That is a design choice, not an oversight, and it is not a file-level
property either: within one file the same author escalates the down-projection and the slot-reduce
launches to two, four or eight warps. **Warp count is a per-launch decision, and reading it off one
launch as "this kernel uses one warp" is wrong.**

It becomes a recorded hazard when the metadata-driven launch also wants a dimension per axis
(expert, token block, channel shard) and you end up with a **multi-dimensional grid together with a
single-warp workgroup** (`../hardware/capability-matrix.md`) — and three-dimensional grids do occur
in these files. Three things go with it: it requires a timing gate rather than being a certain
regression; the single-warp half is frequently intentional as above; and **grid rank and warp count
must be tested separately** — changing both at once tells you nothing about either. Linearizing the
grid is a zero-resource change when it wins, which makes it a cheap first probe rather than a late
one. Keep it distinct from the 1-D *phase concatenation* of
`### The launch boundary is the grid-level synchronization`: that one produces the same grid rank
for an unrelated reason and is not a response to this hazard.

**4. Padding is not one cost and the band tells you its sign.** It used to be written here as a
single "usually irrelevant" check; it is now `### Padding's cost changes sign across the bands`,
together with the three distinct things this workload calls padding. Read that before pricing any
of them.

> **Inline assembly has several recurring legitimate uses on this path, and they are not one kind
> of thing.** The two that show up first are a motion boundary the language does not offer and
> moving a warp-uniform value (an expert id, a base pointer, a shared scale) into a scalar register
> so the per-lane address arithmetic stops recomputing it — but the routing and bucketing bodies
> also reach wave-collectives (ballot, find-first-set, lane extract) and, on the fused-collective
> variants, whole protocol blocks.
> `../gluon/inline-asm-reference.md ## Classifying a site: mechanism × intent` bins a site on two
> axes — what the call is made of, and what deleting it would change — and each class section
> states what that class gives up. Do not carry a bin across kernels: what is keyed on the tile is
> re-derived, not transplanted (`## Shape-keying: inline asm is a per-M specialization`). Expect
> the bins to fit imperfectly here — the wave-collective routers of
> `## Router and top-k: four implementation generations` are the population that classification
> covers least well, so classify by what deleting the site would change rather than by matching a
> template. Note the one trap that matters when a MoE combine uses atomics: a
> hand-written atomic does **not** inherit the cache maintenance the language's atomic intrinsics
> emit, because the exemption is attached to the intrinsic and not to the operation.

## What does not port from the upstream example

This table is why the upstream Gluon MoE example is one data point and not the spine of this page.
It is written for Blackwell, and the mechanisms below have no gfx950 equivalent. Recognize them on
sight so the structural logic can be lifted without the substrate:

| Upstream mechanism | gfx950 replacement |
| --- | --- |
| tensor-descriptor async gather, multicast | pointer-tensor per-row load + explicit LDS staging |
| `mbarrier` ring with phase tracking | `gl.barrier` plus hand-authored multi-buffering (`../gluon/pipeline-reference.md`) |
| `tcgen05` MMA into tensor memory | MFMA into VGPR accumulators — the accumulator re-enters the register budget |
| four-way `gl.warp_specialize` partitions | a single persistent loop with software pipelining |
| 2-CTA cluster layouts | nothing; delete the axis |
| packed-f32x2 arithmetic intrinsics | the CDNA packed-VALU equivalents, re-derived |

The parts that *do* port are the ones worth the reading time: the routing metadata layout, the
packed O(1) schedule lookup, the sentinel-index gather, index reuse across N, per-slice tile
selection, and the pointer-tensor epilogue.

**And the parts it has no opinion on are most of this page.** It does not type itself on any of the
three axes, it has no band siblings to switch against, its quantization path is not the one the
production checkpoints force, and its overlap mechanism does not exist here. Where it and surveyed
production disagree, production is the evidence about this target and the example is the evidence
about a structure — use each for what it is.
