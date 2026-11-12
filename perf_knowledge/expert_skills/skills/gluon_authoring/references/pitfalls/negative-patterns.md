# Gluon Negative Patterns

Use this file when a Gluon route is plausible but may be the wrong layer for the
measured optimization direction. These are problem shapes and decision rules, not
kernel-specific recipes. gfx950 (CDNA4) is the main line; where an entry was measured on
gfx942 (CDNA3) it says so and states what carries to gfx950. Mechanisms are written for
upstream Triton 3.8.0; older minors appear only as downgrade notes.

**This file is also the escalation gate's backing**: the Quick Reject checklist
below = the gate's "skip transcription / stay plain" conditions
(`../method/entry.md`). Per-target support/evidence lives in
`../hardware/capability-matrix.md`. Target- and version-sensitive *platform* traps
(lowering, toolchain pins, harness-breaking aborts, profiler gaps) are the sibling page
`platform-known-issues.md`.

## Map of this page (by theme)

| theme | entries |
| --- | --- |
| **Admission** — is Gluon the right layer for this kernel? | `## How To Use This File`, `## Quick Reject Checklist (= gate skip-transcription conditions)`, `## No-Extra-Mechanism Decision`, `## Paged / Indirect Decode` (admission hint), `## Fused / Block-Scaled Matrix Kernels`, `## Migration Radius Too Large`, `## Stage-Gluon Body Route Guardrails`, `## Bandwidth-Ceiling Refinement` |
| **Layout and launch geometry** | `## Hot-Loop Layout Conversion`, ``## `warps_per_cta` is not an independently tunable knob``, `## A quantity the kernel pins and the hardware does not is invisible to the sweep`, `## A translation failure names the allocation, not the murderer` |
| **Measurement and attribution** | `## Instruction count is not the objective function`, `## Conditional rescale skip — measured negative at the occupancy its own path A asks for`, `## A ratio is a band-local measurement, not a property of the operation`, `## Give a ratio a mechanism ceiling before you believe it`, `## Slow-Correct Gluon Result` |
| **Inline asm and synchronization** | `## Inline asm: justify the reach before taking it`, `## Removing synchronization: a green run is not the evidence you need` |
| **Knobs that compile and then cost you** | `## Knobs and spellings that compile and then cost you` |
| **Target-specific (gfx950 first, gfx942 downgrade)** | `## Target-Specific Matrix Blockers (scoped)`, `## gfx942 / CDNA3 negative-result signatures (mechanism + error text)` |
| **Recording** | `## Negative Result Record` |

## How To Use This File

Before a non-trivial Gluon probe, ask:

```text
Direction / Triton + ROCm + target arch:
Minimal @gluon.jit smoke result / repo gate result:
Gluon-only mechanism:
Hot loop conversions / dot+reduction+atomic sites affected:
JIT boundary crossed / memory traffic changed:
Expected gain range / stop condition:
```

Reject, downgrade, or narrow the probe when the Gluon route would test too many
unrelated constraints at once. If the minimal smoke fails, classify as a
toolchain blocker before any mechanism conclusion; if only the repo production
gate fails, decide whether one forced-backend probe is worth the risk.

### Which entries here are prose, and which ones you should be copying into code

A page of measured-and-rejected findings does not prevent the findings from being re-proposed. Two
observed instances, same worker, same week: a configuration already recorded as silently wrong was
re-proposed **twenty minutes** after the comment recording it had been open in the editor, and a
knob's wrong default shipped four published runs under the wrong label for a week. The first was
caught instantly and at zero cost — by a `gl.static_assert` the same worker had written earlier in
the same round. The second was caught only when an assertion was finally added on the *reporting*
path. Prose lost both times; the machine won both times. So read each entry below for its
enforcement class:

- **Machine-enforceable at compile or launch** — a `gl.static_assert`, a shape check, a `constexpr`
  guard. **Copy the snippet, do not memorize the entry.** Most of the *silent numerical error*
  family has a real compile-time predicate available — a `BK` below the instruction's K extent, a
  tile extent that must equal the quantization group size, a drain depth against the outstanding
  group count. Write the predicate, not a list of known-good values: a whitelist passes the next
  variant that should have failed. Where no predicate exists — `k_width` at an `mfma_scaled` site
  is the standing example, because there is genuinely no formula — the enforceable form is to pin
  the value you *validated numerically* and assert against that, with the validation artefact named
  in the message. A reader who copies the guard is permanently immune; a reader who only reads the
  entry is not.
- **Enforceable on the reporting path** — assert the shipped configuration where the published
  number is produced. **Not in the A/B harness**: a harness *sets* the value it varies and is
  therefore structurally incapable of detecting a wrong default. That distinction is the one that
  cost four published runs.
- **Genuine judgement** — the Quick Reject checklist, "is this the wrong layer", bound
  classification. Prose is the right form for these and this page is already built for them.

When you write one of the first two, **put the number and the artefact path in the assertion
message.** The person it fires on is usually the person who wrote it, a few hours later, and they
will want to argue with it.

## Quick Reject Checklist (= gate skip-transcription conditions)

Reject or downgrade a Gluon probe (stay plain) when most of these are true:

- the hot path spans multiple `@triton.jit` helpers and tensor values cross the
  JIT boundary;
- converting one small idea requires changing the whole call chain to `@gluon.jit`;
- the inner loop has several dot products with different logical tile shapes;
- dot, reduction, and atomic constraints must all be solved in one patch;
- many constexpr branch combinations and no single measured subpath;
- the candidate requires a large rewrite before any output feeds the benchmark;
- the proposed Gluon version is the same algorithm with no explicit layout,
  memory-path, scheduling, or matrix mechanism;
- `../hardware/capability-matrix.md` marks the target/dtype/op path `wrong-result`,
  `version/API-blocker`, or a target-specific blocker;
- the kernel is launch/wrapper dominated and Gluon only changes device-body
  spelling;
- the path is partial migration through `tl.dot`, whose result will not carry a
  Gluon layout for later Gluon ops;
- plain Triton already lowers the hot dot to the desired `tt.dot/#mma` path and
  the remaining bottleneck is config, wrapper, dispatch, or memory traffic;
- the kernel is **decode-shaped or reads K/V through a page table** — this is an
  **admission hint**, not a reject on its own: read `## Paged / Indirect Decode` below
  before spending a transcription on it, and admit only if one of the mechanisms there
  (fewer bytes, a changed address chain, better locality, a removed conversion) is
  actually available;
- the plain backend stages an operand feed in `#ttg.amd_rotating_shared` (visible
  in the dumped TTGIR) **and the `gluon.language` build in front of you exposes no
  rotating-shared constructor** — check before concluding it, the surface moves:
  `scripts/ttgir_bridge.py` reports it as a named `UNRECOVERABLE` row (the text-only
  fallback `ttgir_to_gluon.py` emits `None  # NOT EMITTED: amd_rotating_shared ...`),
  and either message is a prompt to probe the build and re-dump at `ns=1`, not proof of
  a language gap. Never hand-roll a basis for it.
  (`amd_wmma` sat behind the same wording and turned out to be constructible as
  `AMDWMMALayout` all along — see `../gluon/rdna-wmma-reference.md`.) If the
  constructor really is absent, a faithful transcription is not expressible;
  `PaddedSharedLayout`/`SharedLinearLayout` can approximate the M-major
  staging but only to parity, so a Gluon arm gated on this layout stays plain
  (record it `deferred` or `structure_suspect`, bounded, not a proven wall —
  `../method/transcribe.md`).

A quick reject routes to `../method/front-end.md` (plain / config / dispatch) or a
smaller Gluon smoke. GEAK's kernel_workflow drives the rounds, so the deep_engineer
reports the reject in its result as the stay-plain verdict, and tech_lead hands the
work back to GEAK's plain rounds.

## No-Extra-Mechanism Decision

```text
Does plain Triton already lower the hot dot/load/store to the desired path?
Does Gluon remove real memory traffic, layout conversion, launches, or wrapper cost?
Is the Gluon-eligible phase a material fraction of the measured boundary?
Is the expected end-to-end gain above the noise/repeat threshold?
```

If the answers are `yes, no, no/low, no`, stop the Gluon route. Record
`no_extra_gluon_mechanism` and return to the open layer with larger headroom.

## Paged / Indirect Decode

**Admission hint** (the decode / paged row of `## Quick Reject Checklist (= gate
skip-transcription conditions)`): read it before admitting a decode or paged kernel to the
Gluon path, not as a verdict after one. Gluon is unlikely to help when K/V addresses depend on a page-table / gather
chain, address calc is load-dependent (`page_load -> kv_loc -> data_load`), the
per-iteration tile is small with high loop count, plain `tl.dot` already maps to
the desired matrix instruction, and latency is near the access-pattern ceiling.
Buffer ops / async copy / explicit layouts help only when they reduce bytes
moved, change the address chain, improve locality, or remove a measured
conversion. Otherwise keep the direction in plain Triton, split/reduce, dispatch,
or wrapper/boundary work.

One rule for the common case: when the gathered K/V is **L2-resident** (small,
re-read across queries) the bound is **load-latency / memory-level-parallelism, not HBM
bandwidth** — the signal is `MemUnitStalled ~ 0` with the busy counter well under 100%
(`../method/profile.md ## Derived metrics`). Then widening loads (already maxed) and
HBM-BW levers do nothing; only raising occupancy / MLP hides it, and software prefetch
regresses if VGPR/LDS already cap waves (`../tile-programming/pipeline.md ## Budget
before deepening`). Async direct-to-LDS also will not apply — on gfx950 or on the gfx942
downgrade: scattered per-token offsets are not pre-coalesced, so the Gluon path (no
`CoalesceAsyncCopy`) falls back to register staging or fails lowering.

## Fused / Block-Scaled Matrix Kernels

Gluon is a weak body-rewrite candidate when most hold: one body has multiple
matrix paths with different dtype/scale/instruction-shape contracts; the plain
path already lowers the hot dot; source is `dot(a,b)*scale` (scale-before-add,
which MFMA's accumulator cannot express directly); scaled-matrix scale
granularity does not match source granularity (forcing hot-loop scale
replication); the smallest executable rewrite is a serial anchor with repeated
operand `convert_layout`; the removing mechanism (direct-to-LDS staging, compiler
interleaving) is unavailable. Valid next: keep the plain comparator; split the
fused body by feature only if the ABI allows; test one executable subpath with a
mechanism that removes repeated conversion; or record a fused-architecture
negative. Build a phase map first (matrix phases / dtype per phase / scale phases
/ side-load placement / ideal tile per phase / accumulator lifetimes).

## Migration Radius Too Large

Negative signals: matrix operands, online state, masks, and stores all need
different parent layouts; atomic updates constrain layout/ordering in the same
loop as matrix work; a helper returns tensor values to another JIT helper (no
executable partial conversion); correctness depends on several dtype-narrowing
points. Responses: make a layout map before editing; select one executable
subpath; keep the direction plain if the smallest Gluon patch is a whole-kernel
rewrite; record a negative if the migration radius exceeds the expected gain.

## Stage-Gluon Body Route Guardrails

Once escalated, Gluon still needs a mechanism. Keep the plain-Triton comparator
(the target line) while classifying the body bottleneck:

```text
memory API changes bytes or address pressure:
layout changes remove conversion or enable a consumer:
shared memory removes traffic or conversion:
side paths move off the critical path:
epilogue/store layout removes movement:
compiler realization has source-level independence:
```

Reject broadening when the only evidence is "same body in Gluon". If matrix-core
work is absent or not the hot stage, stay on the measured memory/layout/side-path/
reduction/store bottleneck.

## Bandwidth-Ceiling Refinement

Stop device-side compute tuning when measured bandwidth is near the practical
ceiling, the change reduces arithmetic/instruction-count but not memory traffic,
repeats are inside the noise band, or the only gain is `<1-2%`. Exception:
register spill is hidden memory traffic — reducing `waves_per_eu` or changing
launch/config can cut HBM traffic with no visible load/store change; do not
quick-reject that as "no traffic change" until register pressure is considered.
Then: reduce bytes moved, improve locality/reuse, change dispatch/bucketing, or
report a bandwidth-ceiling negative.

## Hot-Loop Layout Conversion

Repeated `convert_layout` inside the innermost loop is a paid operation. It is a
negative signal when the conversion cannot be hoisted to a host-created layout
contract, a phase boundary, a matrix-operand boundary, or a precomputed metadata
path.

Rules: count conversions per loop iteration before tuning launch parameters;
check whether the converted value is loop-invariant; require a hoist point before
treating launch knobs as the fix; keep plain Triton when `tl.dot`/ordinary ops
avoid the movement; only continue with Gluon if the explicit matrix/memory path
removes more work than the conversions add.

### K-Proportional Conversion Cost

```text
fixed host/wrapper overhead / per-K conversion count / per-K scale conversion
K-loop iterations / large-shape comparator result / removable mechanism
```

Fixed launch/wrapper/host-layout costs shrink as a share of larger problems;
hot-loop conversion costs usually scale with `K / BLOCK_K`. If a larger-K or
compute-bound probe does not close the gap, stop the serial Gluon route unless the
next change reduces conversion count, loads in the consumer layout, uses shared
staging, or moves conversion to a phase boundary.

## `warps_per_cta` is not an independently tunable knob

It reads like a config field, so it invites a one-line A/B. It is not one: in a
hand-authored Gluon kernel the warp-to-tile mapping is **restated in every layout in
the file**, and changing `mfma_layout.warps_per_cta` alone makes the module
incoherent. (This entry is Gluon-specific — in plain Triton the mapping is the
compiler's, so there is nothing to keep in sync.)

Measured, gfx950, triton `3.7.0+amd.rocm7.2.0.git89002410` — editing only
`warps_per_cta=[4, 1] -> [2, 2]` on an MLA forward:

```
llvm/ADT/Sequence.h:275: iota_range::iota_range(T, T, bool):
  Assertion `Begin <= End && "Begin must be less or equal to End."' failed.
```

The process dumps core with **no attribution to a layout or a line**, which reads
like a compiler bug and is not one: that kernel pins **14 hardcoded `warp_bases`
tuples** across its blocked, linear and dot-operand layouts, every one written for a
4-warps-tile-M mapping — one of them the degenerate all-warps-see-everything form
`warp_bases=((0, 0), (0, 0))`, which is only meaningful when warps do **not** tile
the reduced dimension. `[2, 2]` invalidates all of them at once, and the assert is
the pass manager meeting that incoherence.

Consequences, both of which have cost a run:

- **Do not read the crash as "this arch/build cannot do `[2,2]`."** Nothing about
  the target was tested. Grep the source for `warp_bases` first; the count tells you
  the real size of the change.
- **Scope it as a coupled layout rewrite, not a knob.** Every `warp_bases` in the
  file moves together, plus whatever cross-warp exchange the new mapping introduces
  (splitting warps across the reduced dimension means the two dots no longer see the
  same operand rows). That is a multi-round direction; attempting it as a one-lever
  round produces this crash and a false negative. See
  [`../method/entry.md`](../method/entry.md) `## The depth contract`.

Why it keeps getting attempted: `warps_per_cta=[N, 1]` broadcasts the B operand to
all N warps, so an MLA-shaped kernel reads its whole K and V tile N times out of LDS.
Halving that replication is a real prize — it is just not a one-line one.

**The same applies to sweeping `num_warps` on a transcribed kernel.** The bridge emits
the IR's *literal* `warps_per_cta`, so any other warp count disagrees with the layouts —
a correctness bug or a compile failure, not a slow config
(`../gluon/layout-reference.md ## BlockedLayout Constraints (wave64)`). `triton.autotune`
itself works on `gluon.jit`, and upstream's own Gluon examples use it — but they sweep tile
shapes and stage counts with `num_warps` pinned. A tile retune is not a knob here either: it
re-recovers every layout, so it is a `resweep_request` back to the front end (in the
deep_engineer's result; tech_lead hands it to GEAK's plain tuning). Only the warp count is off the table inside this pack.

## A quantity the kernel pins and the hardware does not is invisible to the sweep

*`warps_per_cta` is not an independently tunable knob* is one instance of a general shape, and the
general shape is worth stating on its own because it decides when a direction may be called
finished. **A sweep can only be exhaustive over the axes it can name.** A value written as a
literal in the kernel — a warp-to-tile mapping restated across every layout, a per-thread width, a
stage count, a shared-layout parameter — is not a point any grid visits, and it leaves no gap in a
results table for anyone to notice afterwards.

> **Before writing "exhausted", list the quantities the kernel pins that the hardware
> does not, and mark each one swept or fixed-for-a-stated-reason.** A constant in the
> source is not a hardware constraint. It is a choice that was never put on the table.

This is why a clean grid is not evidence of completeness: every point being distinct
and legal is exactly what a search missing an axis looks like from inside. The same
blind spot appears in `## A translation failure names the allocation, not the murderer`
as the third possibility a layout sweep cannot find — there the un-named axis is the op,
here it is any pinned constant — and the answer is the same in both places, to move an
axis the sweep has no name for rather than to run the sweep again.

Two cautions that keep this usable. The list is written **before** the claim, not
produced on demand when someone challenges it; and an entry may perfectly well come back
`fixed` — a mapping that is a coupled rewrite rather than a knob is still a legitimate
`fixed`, as long as the reason is the coupling and not the fact that nobody tried.

## A translation failure names the allocation, not the murderer

`builtin.unrealized_conversion_cast` reaching LLVM translation is the most misread diagnostic on
this path, and the reason is that **the location it prints is the shared allocation, not the
operation that has no lowering.** The cast is a placeholder the conversion framework leaves behind
when some op producing or consuming that allocation could not be converted for the current
combination of layout, shape and dtype; by the time translation runs, the only thing still carrying
a source location is the allocation everyone pointed at.

So the rule is:

> **Bisect by deleting operations, not by swapping layouts.** Remove the ops around the allocation
> one at a time until the message goes away — the last one you removed is the one without a
> lowering. Swapping the layout changes the combination and can make the message move or vanish
> for a reason unrelated to the cause, which is how a layout gets recorded as "unsupported" when
> the actual gap was an op, a per-thread width, or an entry point.

Two live instances, both of which print the same text:
`buffer_load_to_shared` at a per-thread width outside the architecture's set
(`../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`), and
`buffer_load_to_shared` into a padded shared destination
(`../gluon/memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`).
The width is the cheaper of the two to rule out, so check it first. **A third possibility is
always open** — that the missing lowering belongs to an op you have not suspected — which is
exactly what the deletion bisect is for and what a layout sweep cannot find.

**Do not record this message as a build ceiling** on first sight. It is a
`(layout, shape, dtype, op)` coverage gap, and every one of those four is something you can move.

## Instruction count is not the objective function

Deleting work from the hot loop is the default instinct and it is wrong often enough
to need its own entry. **At low occupancy, whether VALU costs anything depends on
where it sits, not how much of it there is** — and a census cannot see the
difference, so an edit that improves every count you are looking at can be slower.

Four measured rounds on one gfx950 MLA forward, register- **and** LDS-bound at
1 wave/SIMD, all interleaved A/B in clean windows:

| edit | census | result |
| --- | --- | --- |
| transpose V on read instead of in registers | **−51** instr, `mfma`/`ds_read`/`shared` identical | **−1.4 %** |
| hoist the V `local_load` above the softmax | reorder only | **−1.0 %** (full drains 4→6, `s_nop` 7→11) |
| skip `acc *= re_scale` behind an exponent-margin test | **−64** `v_pk_mul_f32`/iter | **−6 %** (the branch + reduce cost more) |
| fuse the three back-to-back epilogue `acc` scales | −2 tile multiplies | **+0.6 %** |

**Carry the cell-to-cell spread next to each of those, and treat any delta under it
as `NOT RESOLVED`** — `ab_bench.py`'s own rule. Of the four, three are losses large
enough to act on; the **+0.6 %** is at the resolution limit and is recorded here as
the direction it pointed, not as a banked win.

The discriminator that predicts the three losses: **VALU the scheduler can hide
behind MFMA is free; VALU it cannot is exposed latency.** At 1 wave/SIMD there is no
second wave to interleave, so the only slack is intra-wave co-issue with the MFMA
pipe. Two places have none — the dependent chain between a dot and the reduce that
consumes it, and anything after the *last* MFMA in the epilogue — and the only edit
that did not lose was in the second of them. (The one edit aimed at the first, the
`re_scale` skip, still lost: what it removed was exposed, but a branch plus a reduce
cost more than the multiplies did. So "exposed" is a necessary condition for removal
to pay, not a sufficient one.) Everything the loop body could co-issue was already
free, so trading it for LDS pressure or for a branch is a straight loss.

How to use this rather than re-deriving it:

- **Locate before deleting.** Ask which side of the last MFMA the instruction is on,
  and whether anything is available to co-issue with it.
  `kernel_workflow/scripts/kernel_tools/probe.py measure --arch gfx950` gives the
  occupancy that decides whether co-issue slack exists at all.
- **A pipe at parity is not a pipe to feed.** On the same kernel MFMA and LDS were
  both ~2048 cyc/iter against a measured 6500–7700, i.e. the residual was neither
  pipe. Handing either one more slack (retiring global loads earlier, relocating LDS
  reads earlier) bought nothing, twice.
- **Report the count and the time separately.** A kept build that is 51 instructions
  *longer* and 1.4 % *faster* is a normal outcome here, not a measurement error.
- **A census that hits its target completely still has not priced it.** Removing every
  instance of a pattern from the emitted code, bit-exact and at unchanged occupancy, is a
  falsifier that came back positive — it establishes that the edit happened, not that it is
  worth anything (`../method/benchmark-hygiene.md ## A compile-only kill step is a falsifier, not a
  price`). And before crediting the timed leg that follows, confirm the edit moved only the
  quantity you are attributing to: an access-width change routinely moves a second one, and
  a null is then as consistent with cancellation as with no effect
  (`../method/benchmark-hygiene.md ## Before timing, list every quantity the edit moves`).
- **A width census counts issue slots, not bytes and not transactions — *when the redundancy is
  across lanes*.** "We emit narrow loads and the comparator does not" is not yet a finding:
  adjacent lanes on adjacent addresses are coalesced into wide transactions whatever each lane
  asked for, so the census can be entirely correct, the arm can eliminate every counted instance,
  and the access can have been costing the same all along. The discriminator is the lane-to-lane
  address delta and the contiguous run per instruction, not the histogram
  (`../tile-programming/memory-path.md ## A load's width is issue-side; a transaction's width
  is access-side`).
- **The carve-out that rule needs, because without it the page points away from a real win.**
  The justifying mechanism above is coalescing *across lanes*, and that argument has **no purchase
  on redundancy within a lane** — coalescing merges what different lanes ask for in the same cycle;
  it cannot merge one lane's two separate fetches of the same byte. So: **the census bounds issue
  slots when the redundancy is across lanes, and it bounds bytes too when the redundancy is within
  a lane. The discriminator is the layout's coverage product against the tile extent, per
  dimension.** Measured: a `BlockedLayout` whose coverage product on the fast dim
  (`size_per_thread * threads_per_warp * warps_per_cta` = 1·4·4 = 16) exceeded that dim's tile
  extent (8) fetched **every byte of that tensor twice in the same lane**. Cutting the coverage to
  equal the tile exactly moved bytes-per-lane 89 → 85 and sub-dword loads 25 → 17, and measured
  **+5.3–5.8 %** on the kernel and **+2.76 % call-weighted on the whole op, positive in 7 of 7
  shape buckets**. The census was correct, it *did* name bytes, and it priced a win. Note that
  none of the three across-lane instruments named in *A width census counts issue slots, not bytes
  and not transactions* (the histogram, the lane-to-lane address delta, the contiguous run per
  instruction) can detect over-coverage — read the layout, not the trace. And note the trap in the layout rule that owns this predicate
  (`../gluon/layout-reference.md ## BlockedLayout Constraints (wave64)`): it permits over-coverage
  when masks and ownership prove the extra lanes valid, which is a *legality* rule. A masked,
  functionally correct, over-covering layout passes it while paying 2x bytes on that tensor.
  Correct-but-wasteful is the case to look for.
- **Gate the edit on two numbers, not one, because a single knob can move them in opposite
  directions.** Widening a scale tile fixes over-coverage on one operand and *introduces* it on
  the other: one measured instance netted −1.1 % bytes-per-lane as (−4.5 % on one side) + (+3.4 %
  on the other), so the single-knob null said nothing about either. Report sub-dword count **and**
  bytes-per-lane, per tensor, and attribute counts to the pointer they load from rather than to
  the opcode. The over-covered dimension does **not** have to be the reduction axis — a broadcast
  operand over-covers the non-reduced dim just as easily, so do not check only K.

## Conditional rescale skip — measured negative at the occupancy its own path A asks for

`../workloads/flash-attention-structural-insights.md ### Conditional rescale skip` lists path A as
low occupancy (often one wave per SIMD), asm showing AGPR read-modify on the PV loop, and a running
max that is usually stable. On an attention **forward** that was register- and LDS-limited at one
wave per SIMD — path A's precondition satisfied, not violated — the lever measured **negative**.

The reason is a precondition the list states only under path B: **the skip has to be a real branch.**
In a hand-authored Gluon body the rescale condition is per **row**, so what the language gives you
is a lane mask, not a warp-uniform branch. A masked multiply still issues; the VALU work the lever
exists to remove is still in the instruction stream. What you added on top of it is a comparison, a
cross-lane reduce to form the predicate, and a second softmax path to maintain. At one wave per
SIMD there is no other wave to hide any of that behind — the same fact path A cites as the reason
the rescale is exposed makes the *replacement* exposed too.

So read path A's three conditions as necessary and not sufficient, and add the fourth before
spending a round: **can this condition be made warp-uniform?** If the answer is "only as a lane
mask", the arithmetic does not go away and the lever has nothing left to buy. Escalating to Gluon
does not change this — it is a property of the predicate's shape, not of the DSL. The general form
of that trap is `## Instruction count is not the objective function`.

## A ratio is a band-local measurement, not a property of the operation

The failure above is a mis-located diagnostic. This one is a mis-located *number*, and it is the
more common of the two: a ratio measured in one band gets quoted as though it described the
operation, and the reader applies it in a band where it does not hold — sometimes where it holds
with the opposite sign.

> **Before citing a measured ratio in a band you did not measure it in, reproduce it there.**
> If it does not reproduce, the ratio was a fact about that band's shapes, not about the op.

The worked instance is in `../workloads/moe.md ### The combine epilogue can be half the kernel, and it
is a write-amplification problem`: the cost of an atomic epilogue relative to a plain store of the
same data is about **6.4x** on decode-scale shapes and **86–113x** at the top of a prefill M band.
Same instruction, same verdict, **two orders of magnitude** between the multiples. Either number
quoted bare is misleading about the other band, and the larger one quoted as "the cost of atomics"
would have people rewriting kernels that stand to gain 6x.

What *does* transfer is a lever ranking obtained by isolation — change one quantity per arm from a
single source text and read which arm moves. In that same instance the ranking survived the band
change (atomicity is the entire cost; dtype, contention and addressing are not levers) while the
multiple did not. So when a finding has to travel, **carry the ranking and the mechanism, and leave
the number attached to the shapes that produced it.** A number without its band is how a mechanism
that is right gets applied to an instance that is invented.

**The same loss happens inside a single sweep whenever it is reported as one number.** An endpoint
ratio, a total speedup or a geometric mean collapses the curve, and what it discards is the shape —
so a quantity that degrades steadily across the sweep and one that steps once between two adjacent
points and is flat afterwards can yield the same summary while implying different mechanisms. Read
the per-point data before the aggregate, and say in advance which alternative the statistic is
supposed to exclude (`../method/benchmark-hygiene.md ## A pre-registered criterion fixes the threshold, not
the statistic's power`).

## Give a ratio a mechanism ceiling before you believe it

*A ratio is a band-local measurement, not a property of the operation* is about a ratio that was
real and got moved. This one catches the cheaper error
one step earlier: a ratio that **could not have been what it was labelled as**, decidable at the
desk, with no GPU.

> **Before accepting a measured ratio as evidence for a mechanism, compute what that mechanism can
> produce at most.** If the observation exceeds its own mechanism's ceiling, the arm was not
> isolating the variable it claimed to isolate. That is a *verdict*, not a doubt — and it costs one
> ISA reading, not a re-run.

The ceiling usually comes straight off the instruction selection. Worked instance: an epilogue was
reported **4.2x** slower in fp32 than in bf16, quoted as the cost of the wider accumulator. But
bf16 lowers to a packed atomic that retires **two** elements per instruction and fp32 lowers to one
that retires **one** — 16 instructions against 32 for the same data. A 2:1 instruction ratio cannot
buy more than **2.0x** of time, so a 4.2x observation is not a dtype effect no matter how carefully
it was timed. Re-measuring with the dtype actually isolated gave 1.26x and 0.947–1.029x in two
different bands — both, as required, under the ceiling.

Three practical notes:

- **The ceiling is an upper bound on the *effect*, not a prediction of it.** Landing well under it
  is normal and says nothing; landing *over* it is the signal.
- **It is worth computing before the run, not only after a suspicious result.** If the ceiling is
  2.0x and the decision needs 5x, the experiment is already answered and does not need to be run.
- **When the observation blows the ceiling, suspect the arm, not the hardware.** The usual cause is
  that the two arms differ in more than the named variable — here, both arms had replaced the whole
  epilogue, so addressing and loop structure moved with the dtype at `n = 1`.

An observation over its ceiling is the same failure as a mechanism applied to an invented instance
(`## A ratio is a band-local measurement, not a property of the operation`), approached from the
other end: there, a real number was attached to the wrong shapes; here, a real number was attached
to the wrong cause.

## Slow-Correct Gluon Result

A correct slower Gluon path is evidence, not failure to hide. Before changing more
code, name one overhead: layout conversion; layout padding; memory-path fallback/
mask cost; extra launch/dispatch; shared-memory staging; scheduling-barrier
mismatch; wrapper/artifact selection. If no single removable overhead is visible,
stop the Gluon search for that direction and return to the layer matching the
measured bottleneck.

## Inline asm: justify the reach before taking it

This section fires on **adding** an `inline_asm_elementwise` call that was not already in the
kernel. Reading or auditing one someone else wrote is a different task with a different route
(`../gluon/index.md`, the "reading, auditing or lifting" row).

**Why there is a gate at all.** Two structural costs, neither of which is a performance claim.

First, **the block is opaque to the optimizer in its whole region**: values entering it are
materialized, and the scheduler works around it rather than through it. That cost is paid per site
regardless of what the site buys.

Second, **the mechanism reached for most often is the one that is least durable.** The commonest
shape by a wide margin is an empty asm body whose constraint string is the entire instruction — that
is scheduling and register-class territory, and `../gluon/inline-asm-reference.md ## What it costs you,
on every class` records durability per cell: instruction selection outlives a compiler upgrade,
scheduling control usually does not, and its failure is silent because the asm still assembles.

Whether any given site pays is **`not established`** until you measure it on the kernel in front of
you. These are reasons the reach must be justified rather than assumed.

**Four questions. Write the answers into the round record before the edit, not after.**

```text
1. WHAT DOES THE LANGUAGE ALREADY EMIT?   Name the construct you checked and rejected.
   The recurring false alarms are enumerated at
   `../gluon/inline-asm-reference.md ## Before you reach for it: what the language already emits`
   (cache-scope bits on an atomic; an L2 writeback before a publish; priority around a
   matrix block; waiting on async copies). "I do not know the spelling yet" is not
   "the language has no spelling."

2. WHAT BREAKS IF THIS BLOCK IS DELETED?   Answer in the intent vocabulary:
   a wrong answer / a lost ordering / a different instruction selection / nothing visible
   (`../gluon/inline-asm-reference.md ## Classifying a site: mechanism × intent`). An answer of "nothing visible" is legal
   and common -- it is most of the empty-asm population -- but it is the answer that
   obliges you to say what you expect to move and which reading will show it, because the
   block itself is invisible to every dial in the four-dial set.

3. WHICH DIAL SHOWS IT LANDED?   Not a timing. The disassembly (the mnemonic, at the
   expected count per iteration; or, for an empty asm, the instruction mix AROUND it
   having moved), plus the class's own verify leg -- a determinism race-test for anything
   synchronization-shaped, a numeric case that actually exercises the mode bit for machine
   state, an input whose answer depends on the lane mapping for a wave-collective.

4. WHAT IS THE PRECONDITION, AND WHERE IS IT WRITTEN DOWN?   Per-mnemonic target
   constraint, per-class compiler-version durability, the tile constants the parameters
   are keyed on, and the paired site if there is one. Next to the kernel, not in your head.
```

**Stop, and stay with the language, when any of these is true:**

- the answer to (1) is one of the already-expressed constructs — that is the recorded mistake, not
  the trigger;
- the kernel is a **pure elementwise epilogue**. Surveyed `rmsnorm`, `fused_add_rmsnorm_*`,
  `topk_softmax` and `paged_attention_output_gate` sit at **0%** adoption, and structurally so:
  there is no ordering here the compiler cannot already see;
- the reason is that a production file you were reading had one. Adoption is a count of what was
  written, not a finding about what is right; the same survey records a tutorial tree that writes
  essentially none;
- the claim is "this will be faster." Nothing in this pack's evidence supports a speed claim about
  an asm site; open the round with a reading, not a prediction;
- the need is a **branch over Gluon-level code**. The door takes tensors and returns tensors; it
  cannot enclose the region you wanted skipped. Record the language ceiling
  (`../hardware/capability-matrix.md`) rather than writing the mnemonic;
- you are below the **parity gate** and the site is not repaying a named `lost_pipeline` /
  `lost_layout` / `lost_RA` debt. Recovery closes named debts with their own mechanisms; a new
  opaque block below the gate makes the residual harder to attribute for the rest of the run.

**Proceed, and record it as a named round, when:** (1) names a real absence, (2) is `wrong answer`
or `lost ordering` (or is `nothing visible` **with** (3) answered concretely), (3) names a
disassembly-level check you will actually run, and (4) is written next to the kernel. Then classify
the site you wrote on both axes and put the bin in the round record — a site whose bin nobody wrote
down is a site the next agent has to re-derive against
`../gluon/inline-asm-reference.md ## Deciding the bins`.

## Removing synchronization: a green run is not the evidence you need

The mirror image of *Inline asm: justify the reach before taking it*. Not adding a block, but
**taking an ordering constraint away** — eliding a barrier, weakening a scope, dropping an
acquire — because the kernel got faster without it. The win is frequently real, which is what
makes this direction worth naming here: the danger is not that it does not pay, it is that its
failure mode is the one no gate in this pack can report.

> **A gate that inspects values cannot see a missing acquire.** Bit-identical repeats, a tolerance
> against an oracle, and the determinism race-test all sample the interleavings that actually
> occurred. Removing an ordering constraint changes the result of no single interleaving — it only
> makes further interleavings **possible**. On the sufficiency of an ordering these gates are not
> weak evidence, they are **zero** evidence, and green stays green for as long as the window
> happens not to open.

So a relaxed sequence that passes every gate has established the arithmetic and nothing about the
ordering. What does establish it: the **semantic pairing** — which release pairs with which
acquire, at which scope — or the **emitted ISA**, where the maintenance either is or is not
present (`../gluon/inline-asm/classes-sync.md ### A gate on the value cannot see a missing acquire`;
for the multi-device shapes, `../workloads/collective.md ## How to verify any of this`). Record a
relaxation with that argument beside it, or record the speedup as unverified and keep the stronger
sequence in the shipping path.

## Knobs and spellings that compile and then cost you

A do-not-write list: each of these compiles, and then costs you a round, a wrong label, or a
wrong answer. Where a fact is owned by another page, the entry is the trap and the pointer;
the mechanism is not restated.

- **Deep pipelines by reflex.** Software prefetch regresses once waves are already VGPR- or
  LDS-capped, and aggressive unrolling can raise the `s_nop` count rather than lower it. Start an
  authored ring at 2 buffers and deepen only against a profile
  (`../gluon/pipeline/loop-knobs-and-targets.md ## Footguns`). `num_stages` is not the dial: it is
  dead on the Gluon path in 3.8.0 (`## Roll the loop (cut i-cache pressure)` on the same page), so
  a plain winner's value is a budget figure and a champion-record field, not a Gluon depth.
- **Runtime buffer indices — measure before paying for the unroll.** The rule and its narrow
  scope (identical ISA over sync staging; shipped async loops with a runtime modulo) are at
  `../gluon/pipeline/authored-overlap.md`, under the A5 worked example. Unroll when the ISA says it
  bought something, not by rule.
- **Async copy that "does not exist" on this arch.** It almost always does; the traps — one access
  per lane, a native width split across layout repetitions, varying the tiling before recording an
  arch ceiling, a coalescing pass that legalizes by adding a bounce, and a padded async destination
  that runs fast and returns NaN — are collected in
  `platform-known-issues.md ### Async copy: available on both generations, and easy to misdiagnose as unavailable`.
- **`sched_barrier` / `sched_group_barrier` / `set_prio` imported with no-op stubs.** Absent from
  `gl.amd.cdna3` and `.cdna4` on all four versions; production Gluon that imports them inside
  `try/except` is running dead stubs that still read like scheduling control. Do not copy the
  pattern (`../gluon/pipeline/authored-overlap.md ### What is actually on offer, and which
  architecture has it`, row B2, which also names the reachable form one layer down).
- **`gl.warp_specialize` read as available because it imports.** Present in core `gl` on every
  version and fails the pass manager on gfx942. `gl.amd.warp_pipeline_stage` *does* work on both
  gfx950 and gfx942 and emits `s_setprio` — but it is a scheduling hint, gated on
  `num_warps >= 8` for the inter-wave offset, and whether it pays is a measurement (rows B1 / B3 of
  the same table).
- **Fork-only environment variables.** `TRITON_GLUON_SWP_PIPELINE`, `TRITON_GLUON_COOP_LDS`,
  `TRITON_GLUON_PINGPONG` name nothing on any checked tree, and `TRITON_ENABLE_LLIR_SCHED`,
  `TRITON_ENABLE_AMDGCN_AS`, `TRITON_ENABLE_AMDGPU_RA_HINTS` (`dump_ir.sh --knobs
  LLIR_SCHED|AMDGCN_AS|RA_HINTS`, and the `gemm_compiler_stack` probe's `llir` / `ra` rungs) exist
  only on the fork lineage — 3.6, 3.7 and 3.8 upstream alike lack them. On a stock build each is an
  env var nobody reads: a silent no-op, not an error. Confirm with `probe_levers.py --all` before
  attributing any delta to one; the upstream route for each capability is
  `../tile-programming/non-upstream-reserve.md` (for matrix/VALU co-execution it is the 3.8.0
  coexec scheduler, on by default).
- **The (fork-only) LLIR scheduler toggle on anything with VALU between the matmuls** (softmax,
  scale, dequant). It assumes a pure MFMA→MFMA accumulator chain and emits **invalid IR**, not a
  slowdown. Default-skip it on attention shapes; check the stock `coexec` strategy instead
  (`../hardware/optimization-gotchas.md`, row 7).
- **`disableSched`** — costs occupancy outright. Never on this path.
- **`waves_per_eu` copied across with the champion's config.** It caps occupancy rather than hinting
  it; the measured case and the rule are at
  `../gluon/imports-and-launching.md ### waves_per_eu is a per-launch decision, and 0 is not a value`.
- **Re-injection-route traps (diagnostic / last resort only).** A hand register-prefetch next to a
  re-injected pipeliner is not additive; `convert_layout` scratch allocated after the pipeliner pass
  is outside hazard analysis and fails silently; and ping-pong fires only in a narrow measured
  window and never on hand-authored staging. All three are in `../gluon/pipeline/reinjection.md`,
  and every number from that route is labelled `injected`.
- **`amd_rotating_shared`** — has no `gluon.language` constructor; do not hand-roll a basis for it
  (`## Quick Reject Checklist (= gate skip-transcription conditions)`).
- **CDNA formulas on RDNA.** RDNA (gfx1201; GEAK calibration R9700-only) is a downgrade target with
  its own WMMA layouts and its own page, `../gluon/rdna-wmma-reference.md`; no MFMA formula on this
  page or in the matrix reference applies there.

## Target-Specific Matrix Blockers (scoped)

If `../hardware/capability-matrix.md` marks a target/dtype/op cell `wrong-result` or
`version/API-blocker`, do not spend rounds tuning around it. Example:

```text
target: gfx942 / dtype-op: FP8 Gluon MFMA
status: wrong-result or version/API-blocker
valid next action: plain Triton comparator, config dispatch, or gfx950-local probe
invalid next action: keep tuning the same gfx942 Gluon MFMA path
```

Keep the conclusion scoped to the target/dtype. A gfx942 FP8 blocker is not a
gfx950 blocker unless a gfx950-local probe proves the same failure — gfx950 is the main
line, so probe there first and treat the gfx942 cell as the downgrade. On gfx950 the
equivalent trap is the opposite one: a `no matching matrix core intrinsic` error for fp8
is usually one unregistered `(M, N, K)`, not a dtype prohibition
(`../gluon/matrix-reference.md ## Matrix-Family Details`).

## gfx942 / CDNA3 negative-result signatures (mechanism + error text)

Documented signatures so a future run recognizes the pattern without re-deriving it.
These are **generic signatures** (condition + mechanism + exact error), not perf logs —
do not read them as absolute ceilings for every shape; A/B on your own kernel.

**Read them from gfx950 first.** They were recorded on gfx942; what carries to the main
line is: (1) the LDS-cap regime is much rarer on gfx950 (160 KiB/CU against 64 KiB), so
re-check it rather than carrying it; (2) and (3) are arch-neutral; (4) is tied to the
in-thread-transpose pass, which gfx950 replaces with `ds_read_*_tr_*` where it can — re-check
before carrying it; (5) applies
on both arches at their own cap; (6) is gfx942-only — gfx950 has scaled MFMA.

1. **Explicit tile loses to plain's compiler-managed pipeline (LDS-cap-bound regime).**
   On large-M bf16/int8 GEMM, when the per-CU LDS cap (arch-specific — gfx942 is the tight
   one; read `perf_knowledge/hardware/data/hw_constants.json` `lds_per_cu_kib`) forces
   small tiles, even a full re-injected (last-resort) + async + ping-pong explicit Gluon
   loop can trail the tuned-plain compiler-managed pipeline (the compiler's pipeline/RA is hard to beat by hand, and a
   ping-pong turns barrier-sync-bound). Signature: explicit tile at parity-or-below plain
   after the pipeline layer is exhausted. Action: keep plain (measured), record the gap
   decomposition (`../method/entry.md`), do not keep forcing explicit control. Any
   number from the re-injected arm is recorded as `injected`.
2. **A hand register double-buffer can BEAT the compiler pipeline — check VGPR/spill
   before deepening.** On some large-M GEMM the winner is a hand-written register
   double-buffer (low VGPR, zero spill, high waves), while the re-injected pipeline spills
   and async did not lower at the width tried (gfx942 has async only at 32 bits per thread —
   `../gluon/pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`). Rule: **before deepening the pipeline, read VGPR / spill from the `.amdgcn`
   KD** — if a register double-buffer fits without spill, try it before more pipeline depth.
3. **Scheduler-limited MFMA-continuity ceiling (asm_loop_audit signature).** Pipeline is
   ON (relaxed `s_waitcnt lgkmcnt(N>0)`, few full-drains, `s_nop=0`, prefetch present) yet
   `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py` shows **`MFMA↔VALU transitions = 0` + a long MFMA clump**. That is a
   legitimate *scheduler* ceiling — not a kernel bug. On attention it is where a
   throughput-pairing GEMM scheduler asserts; stop chasing it with kernel edits, and check the
   stock `coexec` scheduler strategy first.
   **Scope the ceiling to the tool, not to the problem** — check three things before
   recording it: whether the build can load a *region-classifying* scheduler instead (there,
   matrix+VALU regions get a co-execution model rather than an assert); whether the loop's
   matrix **shape is in that tool's cost model**, since an unpriced shape drops the region
   silently and produces this exact signature; and whether the vector demand even fits the
   available shadow, because if it does not, no scheduler wins and the work has to move
   (`../tile-programming/llir-codesign.md`).
4. **Zero-copy V-transpose LLVM assert.** A zero-copy V-transpose layout can fail the
   backend with `TritonAMDGPUInThreadTranspose.cpp:526 PassManager failed`. Signature:
   compile-time PassManager failure tied to the in-thread-transpose pass. Action: fall back
   to an explicit transpose/staging path; record the layout as a lowering ceiling.
5. **cshuffle epilogue LDS process-abort (non-catchable).** A cshuffle epilogue needs
   `2·tile_m·tile_n` LDS and can exceed the per-CU cap on either arch; the abort, its
   pre-screen and the subprocess-isolation rule are stated once in
   `platform-known-issues.md ## gfx942 / CDNA3 hard failures that affect benchmark validity`.
6. **scaled-MFMA cannot-select (gfx942 only).** a4w4 / a8w4 / mxfp8 have no scaled path on
   CDNA3 — a genuine `does-not-lower` hardware ceiling; the hard-abort signature and the
   `llvm-mc -mcpu=gfx942` confirmation are in `platform-known-issues.md ## gfx942 / CDNA3 hard
   failures that affect benchmark validity`. Defer to gfx950, where `cdna4.mfma_scaled`
   exists (`../hardware/cdna3-gfx942.md`).

## Negative Result Record

```text
Direction / Probe class / Triton + ROCm + target arch:
Smoke result / repo gate or forced-backend result:
bottleneck class / Gluon mechanism tested:
dependent mechanism blocked or unjustified:
Why the path executed / Correctness / Measured boundary:
Result / Dominant overhead / Failure class:
Why not continue / Scaffolding removed / Next valid direction:
```

Keep results in terms of problem shape and mechanism (the `failure_class` enum is
in `../method/records.md`). Avoid recording kernel-specific constants unless they
are part of the public workload contract.
