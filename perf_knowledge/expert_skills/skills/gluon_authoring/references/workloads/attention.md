# Workload: Attention / Fused (prefill / decode / extend)

Two coupled matrix phases with a softmax between them. The accumulator layout of
the first dot must match the operand layout of the second.

## Logical tiles

```text
Q [Br, D] , K [Bc, D] , V [Bc, D] , S [Br, Bc] , O [Br, D]
QK:  S = Q @ K^T          (dot 1, accumulator -> softmax operand)
PV:  O = softmax(S) @ V   (dot 2)
```

This is a `multi_phase_matrix` structure: the QK accumulator layout
(`AMDMFMALayout`) must feed the PV operand (`DotOperandLayout`) without an
expensive `convert_layout` in the hot loop.

## Structural insights

Attention is two matmuls with softmax wedged between them — not a repeated GEMM.
Before importing FA-4 (or any published non-MMA trick), read the structural model,
bound-class porting filter, and **AMD validated strategies** in
`flash-attention-structural-insights.md`. Full FA-4 catalogue lives in
`tile-programming-cutedsl` (B200 only).

## Online softmax (preserve, do not rewrite first)

Maintain running `m` (max), `l` (sum), `acc`; per K/V tile:

```text
m_new   = max(m, rowmax(S))
alpha   = exp(m - m_new)
acc     = acc * alpha + exp(S - m_new) @ V
l       = l * alpha + rowsum(exp(S - m_new))
```

instance (Gluon MFMA): write the `acc` update as `acc = mfma(p, v, acc * alpha)`,
**not** `acc * alpha + mfma(p, v, acc)` (which double-counts `acc` and fails at
>= 2 K-tiles) — see `../gluon/matrix-reference.md ## Accumulator + per-step
rescale (online normalization)`.

When softmax is the confirmed VALU bottleneck, reduce its per-element VALU by
folding the scale off the tile: use `exp2(qk*(scale*log2e) - lse*log2e)` for
`exp(qk*scale - lse)`, moving the per-element `*log2e` onto the per-row `lse`
(`../method/profile.md ## Reducing compute-class VALU`). The `qk_scale` itself can
instead be folded into **`Q` at load** — one cast over the small Q tile, hoisted
out of the K-loop entirely (`q = (q * qk_scale).to(dtype)` once, before the loop) —
which removes the per-block `S * scale` and **lowers result-tile register
pressure** (helping the occupancy budget / tri-lemma). This is the general
"cheapest-carrier" fold: prefer the operand loaded once upstream over the
per-output-tile result; verify in asm the per-element scale is gone.

### Conditional rescaling (skip the per-block acc *= alpha)

The per-block rescale `acc *= alpha = exp(m_{j-1} - m_j)` is a vector multiply on
the **critical path** (it gates the PV accumulate). It is only needed when the
running max actually grows, and a bounded slack is tolerable: **skip the rescale
(and the running-max update) when `m_j - m_{j-1} <= tau`** — keep `m_{j-1}` and use
`exp(S - m_{j-1})`. Correctness holds because the final `Output = O_final /
l_final` renormalizes by the true max + sum at the end. A safe default is
`tau = log2(256) = 8` for base-2 exp (tolerates a 256x rescale factor before
forcing the update). To avoid warp divergence, rescale when **any** lane in the
warp needs it (predicate on the warp, not the lane). This removes most per-block
rescales, so it shortens the QK->softmax->PV critical path: it is a
**latency/dependency-bound** lever (`../method/profile.md ## Bound classification`),
not a VALU-throughput one.

**Two different mechanisms make this pay, and they have opposite occupancy preconditions** — so
the apply-when checks in `flash-attention-structural-insights.md ## Conditional rescale skip` are
a disjunction, not a single gate:

1. **Low-occupancy decode:** what is saved is the **accumulator round-trip**. The rescale is
   exposed because there is no other wave to hide it, and skipping it lets the value matmul
   accumulate in place.
2. **High-occupancy dense prefill with a co-execution budget:** what is saved is **budget**. The
   rescale is the largest per-tile item in one compute region, and removing it frees window
   capacity (`../tile-programming/llir-codesign.md ## Attention: the co-execution budget`).
   This is the case the first mechanism's checks explicitly warn *against*, and it is a real one.

For (2) the skip has to be a **real branch**: a row carrying a unit factor is numerically a no-op,
but its multiply still issues and still costs its slot, so eliding it needs control flow rather
than a multiply by one — and that branch belongs in a **memory** region, never in a compute
region, where it would be scheduled ahead of the first matrix op. Also: **freeing budget does not
by itself buy anything** — something downstream has to spend it, so attribute the two changes
separately (`../method/close.md ## Attributing a change that only creates headroom`).

**Check the branch is expressible before designing around it.** The useful granularity is the
wave — each wave owns its own rows and can decide independently, which is a *warp-uniform*
branch needing no cross-wave reduction and no barrier. Whether the DSL can say that is a
**build** question, not a design one: a per-wave predicate is a vendor-fork extension on this
stack and is absent from clean upstream builds, where the design is not writable at all rather
than merely slower. Establish this with the rest of the toolchain identity before spending a
round (`../tile-programming/compiler-contract.md ## Toolchain identity`), and if it is absent
record a version/API ceiling against the build — not a negative result about the technique.
Falling back to a lane-level mask does **not** substitute: a masked-off lane's multiply still
issues, which is the cost the skip exists to remove.

Why it pays beyond the multiply (the general trigger + the AGPR mechanism). The
generalizable trigger is **an MFMA accumulator that is READ-MODIFIED every iteration
by a VALU op** (online-softmax `acc *= alpha`, gating, a running scale) instead of
write-only-until-epilogue. On CDNA the MFMA accumulator lives in AGPR, so reading it
with a VALU op forces an `AGPR->VGPR` `v_accvgpr_read` round-trip every iteration
(`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`), which caps that
matmul's efficiency. When the modifier is the identity on the common path (running max
stable, gate open), the conditional skip lets the PV matmul accumulate **in place**
into the AGPR accumulator (`acc = mfma(p, v, acc)`), restoring the efficient
write-only-until-epilogue cadence; the rescale branch can be written
`mfma(p, v, acc*alpha)` to avoid a zero-accumulator temporary. Guard: per the occupancy
corollary (`../tile-programming/mental-model.md ## Latency vs throughput`) this is a
low-occupancy / latency-exposed win and adds a warp-uniform branch, so it is
neutral-to-negative on an occupancy-hidden multi-wave build — confirm waves/CU before
keeping it. This trigger generalizes past attention to any iterative
accumulator-with-rescale loop.

If instead the **transcendental (exp) unit** is the proven bound (its busy counter
saturated), raise softmax `exp` throughput by **partial FMA polynomial emulation**
(`../method/profile.md ## Raising exp throughput by partial FMA emulation`) — more
likely to bind on faster-MMA arch (`../hardware/roofline-models.md ## Asymmetric
scaling`).

Preserve logical shape, masks, split boundaries, and online state before any
layout / MFMA / scheduling work. A correct-but-slow transcription is the anchor.

## Usual bound class

- **prefill / extend**: compute-bound on QK+PV (MFMA continuity); also LDS-bound
  on the S buffer.
- **decode (split-KV)**: latency/memory-bound; the split count and the reduce/combine buffers are
  first-class directions. The count alone can be a Stage-Plain sweep, but **the structure around
  it is not** — split assignment, record geometry, the scratch round trip and the merge shape are
  separate decisions that interact, and they are the subject of
  `## Decode (split-KV): a structure, not a config knob` below.
  **Which** byte level the decode KV comes from is computed, not assumed: at short served context
  the cache footprint can sit inside the memory-side last level, and then a fraction of HBM peak
  is not the denominator and a byte-removal round has no DRAM traffic to remove
  (`../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`;
  the GEMM side's full reasoning is in `gemm.md ## Usual bound class`). Price the footprint against
  the served context range **before** reading the scratch account in
  `### The scratch round trip is the byte account decode adds` — the account is only actionable for the
  shapes that reach DRAM. Where the read is cache-resident the kernel is occupancy-latency-hidden
  rather than prefetch-limited: check waves/CU first, and note that software prefetch regresses if
  VGPR already caps waves (`../tile-programming/slicing.md ## Occupancy budget (P8)`).

## Layer roadmap

1. anchor: explicit layouts for Q/K/V loads, S, and the two dots (transcribe).
2. memory: K/V loads -> `buffer_load` -> `buffer_load_to_shared` (gfx950). **gfx942 downgrade:**
   direct-to-LDS async is 32-bit only and needs `order=[1,0]`, so K/V usually stage through
   registers (`buffer_load` + `ds_write`).
3. LDS layout: conflict-free `ds_read` for K/V and the S/softmax buffer — the V operand of PV can
   use the transposed `ds_read_b64_tr` read on gfx950; on the gfx942 downgrade there is no
   transposed LDS read and the 64 KiB / 32-bank LDS changes both the padding and how many K/V
   stages fit.
4. softmax/reduction state -> a `DistributedLinearLayout` consistent across the
   running state.
5. matrix subpath: QK and PV operand/accumulator layouts; keep the QK->PV layout
   match so no hot-loop `convert_layout`. instance: attention M = head-block is
   small, so keep M in one warp (`warps_per_cta=[1, num_warps]`, split N/D) — never
   split M, or the MFMA tile is half-empty
   (`../gluon/matrix-reference.md ## Matrix-Family Details`, small-dimension warp
   placement). This is one face of the matrix-tile-fill signature; the other (a large
   D-split head dim where a *too-small* BLOCK_M leaves the M-tile half-empty -> grow
   it) and the distinct under-amortized case are in
   `../method/profile.md ## Rule: low MfmaUtil has two distinct tile-size causes`.
6. pipeline the K/V tile loop (prefetch next K/V while computing current), hand-written in the
   order `../tile-programming/pipeline.md` defines — register prefetch, then an authored K/V ring,
   then `warp_pipeline_stage` (at `num_warps >= 8`). `num_stages` is dead on the Gluon path in
   3.8.0, so it is not how depth is set here.

**This roadmap is prefill-shaped, and step 6 is where decode forks.** A decode body's K/V walk is
short by construction, so the overlap that pays is not deeper prefetch inside one program — it is
the cross-split parallelism that produced the programs in the first place, plus what the partial
records cost to write and read back. Take steps 1-5 as written and replace step 6 with
`## Decode (split-KV): a structure, not a config knob`.

For FP8 attention (INT8 Q/K + FP8 V + per-block descale, `tl.dot(q,k) *
(q_descale * k_descale)`), treat the descale as a side path
(`../tile-programming/low-precision.md`) and verify the FP8 dtype gate — OCP fp8 on gfx950, the
FNUZ encoding (different max and bias) on the gfx942 downgrade, which also has no scaled MFMA. The side-path page is
GEMM-shaped and does **not** price the descale against the softmax's own budget, nor settle the
wave structure for a scaled path — that intersection is `attention-lowprec.md`, and it is the
required read before budgeting an fp8/fp4 attention kernel.

### Staged QK <-> softmax overlap (when it pays off)

A multi-stage pipeline that overlaps tile *i*'s softmax with tile *i+1*'s QK
(FA3-style) needs *k* live tiles, so it spends VGPR/LDS and **lowers waves/CU**. On
an occupancy-first arch the extra ILP from staging is often **already supplied by
the other resident waves**, so paying occupancy for it is a net loss. Add a stage
only when both hold: (a) you are **latency/ILP-bound at low occupancy** (busy
counter well under 100%, not throughput-bound — `../method/profile.md ## Rule: read
the busy/throughput counter`), and (b) the stage does **not** cross the occupancy
cliff (`../tile-programming/slicing.md ## Occupancy budget (P8)`). If you are
already at high occupancy the hardware overlaps for free and a hand-staged kernel
ties or loses to the leaner one. Note the **online-softmax recurrence blocks
symmetric ping-pong** (each step's rescale depends on the running max/sum, so two
warps cannot run identical-phase-offset softmax), and producer/consumer
warp-specialization is arch-gated (`../hardware/capability-matrix.md`).

Read "symmetric ping-pong" narrowly: what the recurrence blocks is two waves running the *same*
softmax step concurrently. It does **not** block the warp-pipeline shape, where the two waves are
a full stage apart and one is in a memory stage while the other is in a compute stage — that is
the structure the four-stage attention kernels are built on. Expressing it explicitly is a
Gluon-path mechanism (`warp_pipeline_stage`; the pass that groups the stages runs only in the
Gluon lowering), so the skeleton and its gates live in that pack's
`tile-programming/warp-pipeline.md`. On plain Triton the same shape is only reachable through
the automatic ping-pong pass, which you do not place.

On gfx950/942 the leaner non-staged kernel wins over hand-staged FA3-style overlap
(`../tile-programming/mental-model.md ## Porting a technique across architectures`).
Full FA-4 warp-spec / TMEM pipeline: `tile-programming-cutedsl` →
`flash-attention-structural-insights.md`.

**If the Gluon anchor is schedule-bound against a plain anchor, author the overlap first.** The
signal is equal VGPR+AGPR and occupancy, lower `MfmaUtil`, more full-drain `lgkmcnt(0)` — the
`lost_pipeline` debt. Repay it by hand, in the order
`../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`
gives: register prefetch of the next K/V tile, then an authored K/V ring (gfx950 `async_copy` +
`commit_group` / `wait_group`; gfx942 sync staging), then `warp_pipeline_stage` where Gate 0 holds.
Per-tensor depth and staggered chains are exactly what the auto-pipeliner cannot express, and on
this workload the authored ring is what production uses. Two loop-shape edits help either route
and are free: **split the causal mask into two loops** (branch-free hot region) and keep the K/V
loads where the chosen route wants them.

**Re-injecting plain's TTGIR software pipeliner is the lowest rung, not the cheap one.** It
(`add_schedule_loops` + `add_pipeline`, no rebuild) reproduces the cross-iteration overlap at
unchanged occupancy, and its ceiling is plain parity. Use it only (a) as a **diagnostic below the
parity gate**, to size the `lost_pipeline` debt the hand-written ring has to repay, or (b) as a
**last resort** when the hand-written ring cannot reach parity. Its numbers are labelled
*injected* and never counted as a win; it is never applied to an incumbent (already-Gluon) kernel;
and it is mutually exclusive with hand-written staging in the same loop. Recipe and gates:
`../tile-programming/pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`
and `../method/recover.md`; the loop shape it needs (in-body K/V loads, no hand register-prefetch):
`flash-attention-structural-insights.md ### Software-pipeline reproduction — the kernel shape re-injection needs`.

### Where the softmax goes, and why it rides with the matrix op

The GEMM models assume two instruction categories competing for a SIMD: matrix ops and the memory
ops that prepare their operands. Attention has a **third** — the softmax's vector math, which is
neither, and which the next matrix chain depends on. Everything about the loop's shape follows
from where that third category is put, so it is worth deriving rather than assuming.

Two issue rules decide it. A wave issues **at most one instruction per cycle**; the VALU and the
memory pipe are **separate ports**, so the SIMD can issue one of each in the same cycle — but only
from **different waves**. (LDS and VMEM share the memory port, so a shared-memory read and a
global load cannot pair with each other, only with a VALU.) Three candidate placements:

- **In the memory stage, with the loads.** Then the VALU and the shared-memory reads issue from the
  *same* wave, so by the one-per-cycle rule they take turns. The memory port idles on the vector
  cycles and the VALU port idles on the load cycles even though the hardware would have run both.
  Half the shadow is wasted.
- **In a stage of its own.** Three categories live at once needs **three waves per SIMD** — one per
  stage. The pairing would work, but the third wave has to come from a different workgroup, and
  nothing keeps three workgroups in step.
- **In the compute stage, with the matrix op.** Now the VALU issues from the **same wave as the
  matrix op** while the *other* wave supplies the memory traffic: different waves, different
  ports, so they pair in the same cycle. The same issue slots buy twice the work of the first
  option.

So the softmax rides with the matrix op. That is what makes an attention kernel
**inter-wave between memory and compute, and intra-wave inside each compute cluster** — the
`../tile-programming/scheduling-model.md` taxonomy applied per region rather than per kernel, and
the reason those clusters are handed to the compiler and have a co-execution budget at all
(`../tile-programming/llir-codesign.md ## Attention: the co-execution budget`).

**One ordering rule falls out of it and belongs to the kernel, not the compiler.** Work you know
will be *exposed* — anything the budget says will not fit a window — should be placed **before the
cluster's first matrix op**. A packed op cannot follow a matrix op back to back, so the first
uncovered packed op otherwise pays a hazard stall on top of already being outside the shadow. Same
instructions, same count, only the order differs; and only the kernel knows which work was never
going to be covered.

### What the authored path can express, and where it stops

The placement derivation above settles *where the softmax goes*. It does not settle *how much of
that placement you can actually author*, and the two halves of the loop answer differently.

**Authorable:** which stages exist, what goes in each, and which wave is in which stage at a given
time. That covers the whole matrix/memory side — the ping-pong that holds the two waves a stage
apart is a structure you build, and it works on a stock toolchain.

**Not authorable:** that a *particular* vector op lands inside a *particular* matrix op's shadow,
in a particular form. That is a request to the instruction scheduler, which is free to grant it,
ignore it, or move the op afterwards. So the softmax/matrix side is reachable only in part, and a
plan whose payoff depends on exact shadow placement carries a dependency the authored path does
not. Say which half your estimate came from before quoting it.

**Four author-side rules survive that boundary**, because they change what the scheduler is handed
rather than asking it for a placement:

- **Keep control flow out of the compute cluster.** A branch fragments the matrix stream that
  everything else is being hidden under.
- **Balance the clusters that share a shadow budget.** The overloaded one sets the pace for both,
  so equal demand beats a clever split.
- **Make the rebalancing free.** Moving work between clusters only pays if the move itself costs no
  instructions — a register-level view or a slice, never a copy.
- **Put known-exposed work before the cluster's first matrix op** (the ordering rule above).

> **Moving a vector op between clusters: take its producer with it.** Move the op but leave the
> arithmetic that feeds it behind, and the leftover op has its only consumer in the other cluster
> with nothing anchoring it — no chain edge, no barrier — so a scheduler may sink it somewhere
> neither cluster wanted. Carry the original slice across and do the arithmetic where it is
> consumed.

**Which arm implementations pick is not settled.** Explicit wave orchestration — phase-shifted
barriers, warp specialization, stage markers — fits the matrix/memory region, and that is where
kernels using it tend to use it: the GEMM-shaped bodies. Attention implementations are frequently
found with none of it, carrying the overlap in intra-wave levers instead: breaking the accumulator
chain into independent partial products, choosing which side of a stage boundary an operand
prefetch sits on, giving the row reduction a layout that keeps it wave-local. Both arms are real
and this page already records a case where the leaner non-staged kernel won. Treat it as a fork to
decide per kernel against the placement derivation above, not a ladder with ping-pong on top.

### The row reduction has three routes, and they are mutually exclusive

Named here because "give the row reduction a wave-local layout" above is only the first of three, and
picking among them changes the data flow and the register budget — a structural choice, not a lever.

**Count first: how many waves does the reduce axis currently span?** That number, not a preference,
narrows the set. A reduction already inside one wave has nothing to restructure.

- **Dedicated layout** — choose a layout that keeps the reduce axis wave-local. Cost: one extra
  `convert_layout` on the path that feeds it.
- **Two-level** — reduce within the wave first, then send only the per-wave scalars through LDS. Cost:
  a second reduction step; the LDS traffic becomes a handful of scalars per wave rather than a tile.
- **Defer the cross-lane step out of the loop** — keep the denominator as a fragment and finish it in
  the epilogue. Cost: registers, and the multiple is large enough to move the occupancy tier, so price
  it against A2 before writing it.

**Status: registered, not ranked.** Production carries instances of all three and does not converge on
one, which is the evidence that the choice is shape-dependent. No measured comparison is claimed here;
the deliverable of this entry is the wave-count precheck and the cost column, so a round picks one
deliberately instead of reaching for the first.

## Decode (split-KV): a structure, not a config knob

A decode step has one query row per head and a KV history long enough that the *only* parallelism
worth having comes from cutting the KV axis. That cut is not a single integer. It reaches slicing,
memory path, epilogue and layout at once, and the choices interact — which is why a decode kernel
that has been taken seriously tends to carry an explicit structure record rather than a tile
constant. Read this section before proposing a decode edit; the roadmap above is prefill-shaped.

### The dimensions that exist

A map of the decision space, not a ranking. Which one binds is a profile question.

| Dimension | What it decides |
| --- | --- |
| **split count** | KV-axis parallelism. The first-order knob, and the one the grid is built from. |
| **split assignment** | contiguous span per split vs cyclic interleave. Interleave is the more even of the two when served lengths are skewed, because every split then draws from the whole history instead of one region of it. |
| **split pitch** | the scratch stride *per split*, deliberately decoupled from the split count so records can be padded to an alignment the merge stage wants. |
| **grid dimension order** | whether the split index or the output-channel shard varies fastest. This is an L2/XCD locality choice, not a correctness one. |
| **record geometry** | how the partial record is grouped (channel group x head group). It sets both the merge stage's granularity and the write coalescing of the producer. |
| **output-channel sharding** | narrower shards buy programs but re-compute the QK product once per shard. The re-computation is the price of the parallelism, so it belongs in the budget rather than in the bug list. |
| **resident tail** | keep the final accumulation chunk in registers and never round-trip it. A register-vs-scratch trade (`../tile-programming/slicing.md ## Budget to compute first`). |
| **local checkpoints** | hold the first N partial accumulators in LDS instead of global scratch — the same trade one level down. |
| **guarded scalar loads** | skip scale loads entirely on a split that owns no work. |

### The scratch round trip is the byte account decode adds

Prefill's bytes are Q/K/V. Decode adds a second stream that prefill does not have, and it is easy
to leave out of the budget because no tensor in the signature names it:

```text
scratch_roundtrip = 2 * split_count * record_bytes      # written once, read once by the merge
kv_bytes          = history_len * head_dim * dtype_bytes
```

**At short served lengths the first term can exceed the second.** Both terms move with the split
count but in opposite directions — more splits shortens the per-split KV walk and lengthens the
scratch — so the split count cannot be chosen against latency alone. `Verify:` price both terms at
the *served* length distribution, not at the longest supported one; a split count tuned at maximum
context is tuned against the term that is smallest there.

Two independent amplifiers sit on top, and they are worth separating because their risk differs:

- **The producer writes full records for splits that own no work.** A split beyond the history
  length still has a record shaped for it, and the simple spelling fills it with zeros.
- **The merge stage reads every record back** when its mask is the split count rather than the
  live length.

**Fix the read side first.** Masking the merge against the live length (or bucketing by length)
touches no producer state and cannot change what a replay observes. The write side can: skipping
inactive records is only safe if the merge is *guaranteed* never to read them, and under graph
replay "never read" has to hold against the previous launch's residue as well. That is a property
of the scratch layout, so it has to be re-established per layout rather than inherited.

### The merge is a weighted sum, and it should not be an atomic

Rescaling each split's partial by the exponential of its own max against the global max and summing
is associative in whatever order the merge chooses, so the reduction can be deterministic at no
cost. On this target the no-atomic deterministic split is also the faster path
(`../tile-programming/memory-path.md ## Output / reduction path (atomics)`), so determinism here is
not a concession. Pack the per-split statistics that travel with each record — the running max and
the denominator move together and are read together.

> **The statistic has a base, and the folding above changes it.** Once `log2e` is folded onto the
> per-row term (`## Online softmax (preserve, do not rewrite first)`, the `exp2` rewrite), the
> `lse` this record carries is no longer in the base the unfolded form produced. That is invisible
> inside one kernel, where producer and consumer are the same code — and it is exactly what a
> split-KV record breaks, because the merge reads the statistic in a **different dispatch** that
> was written against whichever convention its author had in mind. Two things follow, and both are
> contract rather than tuning:
>
> - **State the base next to the record layout**, not in the kernel that writes it. A record is an
>   interface; its fields need units the way its offsets need strides.
> - **Fold on both sides or neither.** Rescaling one side "to match" at merge time re-introduces
>   the per-element multiply the fold existed to remove, so the two admissible states are both
>   folded or both not.
>
> This is the same class as the `acc = mfma(p, v, acc * alpha)` ordering above: a rewrite that is
> correct in isolation and silently wrong once a second reader shares the value. The difference is
> that the accumulator form fails loudly at two K-tiles, while a base mismatch returns plausible
> numbers — so it is found by reading the contract, not by running the kernel.

### LDS: stage once, then re-view

Decode bodies are frequently LDS-capped rather than register-capped, and two habits recover more
room than shrinking a tile does:

- **Re-view an allocation instead of allocating a second one.** One staged block can serve two
  consumers under different shapes or layouts — including a transposed read — when the second view
  is a re-interpretation of the same bytes rather than a copy. A transposed *view* costs no LDS
  traffic; a transposed *copy* costs a full round trip.
- **Allocate reusable planes once, not per tile.** An intermediate plane that every tile needs (a
  probability staging plane is the usual one) can be allocated once and reused across the loop.

Both are shared-memory-descriptor mechanics; the DSL's own shared-memory page owns the spelling,
routed from that DSL's mechanism index.

### Two places where the usual advice inverts

- **Below the occupancy floor, spending LDS is free.** Once the body is already at one workgroup
  per CU, more LDS does not cost a wave, so a lever that trades scratch traffic for LDS has no
  occupancy price left to pay (`../tile-programming/slicing.md ## Occupancy budget (P8)`). Check
  where you sit on that budget before rejecting an LDS-hungry option.
- **Streaming the scratch past L2 can be negative here.** A non-temporal store is the right default
  for output nothing on the device reads again — but the partial scratch has a consumer on the same
  device, immediately: the merge stage. Bypassing L2 on the write means the merge reads it from
  memory instead. `Verify:` the rule is whether the buffer has a near consumer on the same device,
  not whether the store is the kernel's last write.

### Grid fill when the body is LDS-capped

At one workgroup per CU the grid *is* the occupancy: if the number of workgroups does not exceed
the CU count, the whole launch is a single wave and `tail_efficiency`
(`../hardware/roofline-models.md ## Saturation / wave quantization`) degenerates into a device fill
fraction. A split count chosen for latency alone can leave CUs idle for the entire launch — cheap
to check statically, and worth checking before any deeper lever.

## Backward pass (general)

Attention backward has **three reductions on two different axes** -- a structural
choice problem (`../tile-programming/mental-model.md ## Reduction &
parallelization structure (decide before tiles)`), decided before any tile:

- `dk`, `dv` reduce over **queries** (per key tile);
- `dq` reduces over **keys** (per query tile).

`dk/dv` and `dq` therefore want *transposed* parallel axes -- you cannot make all
three clean in one grid (the transpose conflict). The recurring options:

- **Split (deterministic).** Separate query-parallel `dq` and key-parallel
  `dk/dv` kernels; each reduction is clean and atomic-free, but `S = softmax(QK)`
  is **recomputed** in both (a recompute tax).
- **Fused (atomic).** One key-parallel kernel computes `S` once and keeps `dk/dv`
  clean, but `dq` (cross-key) is **atomic-added** -> write amplification = #key
  tiles (`../tile-programming/memory-path.md ## Output / reduction path (atomics)`).
  The atomic write *volume* itself can be cut with packed bf16 atomics (same
  section: needs a coalesced/`vec>=2` `dq` layout), which narrows -- but does not
  remove -- the fused path's atomic disadvantage.

Pick by comparing **recompute cost vs atomic write-amplification cost** on the
target HW (P2/P3): when recompute is cheaper than the atomic traffic, the split
wins; the fused path needs few atomics (large key tile, no spill) to compete,
which on a fixed compiler often hits the register/occupancy walls
(`../tile-programming/slicing.md`). Read the vendor reference impl (e.g. CK)
first to set the realistic ceiling and see which structure it chose and why. [On
one MHA-bwd case the split path beat the fused-atomic path by ~1.5x, and adding
key-splits to a deterministic `dq` only ever increased atomics; gfx950/MI350, one
build.]

On gfx950 (and the gfx942 downgrade alike) there is no 2-CTA cluster MMA — the no-atomic deterministic *split* is
the fast backward path (atomics ~2x costlier than NVIDIA,
`../tile-programming/memory-path.md ## Output / reduction path (atomics)`). Blackwell
2-CTA + TMEM backward tricks: `tile-programming-cutedsl` →
`flash-attention-structural-insights.md`.

## Scheduling (load-imbalanced grids)

Causal masking and varlen make the attention grid **load-imbalanced**: per-`(head,
batch)` the masked-out blocks above the diagonal make worktiles range from short to
long, and a left-to-right grid order processes them shortest-to-longest, leaving a
long tail. The fix is a **longest-processing-time-first (LPT)** grid linearization
(general makespan rule: `../workloads/gemm.md ## Worktile scheduling for load
imbalance`), specialized for attention:

- **Causal forward/backward**: traverse `mblocks` in **reverse** within each
  `(head-section, batch)` so the heaviest (full-width) query blocks run first; keep
  the **batch axis outermost** and swizzle heads into **L2-capacity-sized sections**
  so the LPT order does not thrash L2 with KV from too many heads/batches at once.
- **Deterministic `dQ` (when using the fused-atomic path with lock-ordered
  reductions)**: order the `dQ` reductions by a **shortest-processing-time (SPT)**
  schedule so no CTA stalls on its first `dQ` write waiting for a slower peer —
  launch KV blocks descending, query blocks ascending from the diagonal. Read this as
  a makespan ordering and **not** as a progress guarantee: blocks of an ordinary launch
  have no forward-progress guarantee between them, so an ordering that merely makes the
  wait *short* is not what makes it *safe*. (On AMD the
  no-atomic deterministic *split* is usually the faster path anyway —
  `## Backward pass (general)`; this applies when atomics are kept.)
- **Varlen**: enforce LPT with a cheap **preprocessing kernel** that sorts batches
  by per-worktile time and writes a cached virtual->actual batch-index map the main
  kernel reads (no per-run sort cost) — **only once the uniformity gate below holds**.

Architecture-agnostic on AMD (gfx942/950); a `pid -> worktile` remap, no kernel-body
change. The gains are largest for causal. Evidence: `flash-attention-structural-insights.md`.

**LPT uniformity gate (varlen).** The makespan argument assumes worktile cost is
**monotone along the axis you reorder** — true for causal (cost grows with the
m-block index), and true for varlen only when the packing is uniform enough that
sequence length is what orders the tiles. Non-uniform packing (mixed prefill/decode,
a few long sequences among many short ones, or a packer that already interleaves for
balance) breaks it: the reorder then permutes an already-balanced dispatch and can be
**neutral or negative**, and it also stacks badly with a `pid` swizzle (`##
Scheduling` above; `flash-attention-structural-insights.md ## LPT causal`).

- **Gate**: apply LPT by default only on a **uniform** grid — single-sequence, equal-length
  packing, or causal-only imbalance. On non-uniform varlen packing it is a **hypothesis, not
  a default**: A/B it before keeping.
- **Knob**: the `pid -> worktile` remap (and its preprocessing/sort kernel) on/off; if a
  swizzle is also present, the A/B is three-way (LPT-only / swizzle-only / both).
- **Acceptance**: measure the **dispatch's own** worktile-duration spread (not the mean) —
  LPT should shrink the tail, so check the slowest-CTA / makespan proxy moved, then confirm
  end-to-end time on the *packing distribution you actually serve*. A win on a synthetic
  equal-length batch does not transfer to a skewed one.

### `waves_per_eu` on attention dispatches

`waves_per_eu` trades **per-wave register budget against resident waves**, so its best
value is set by the balance between *latency to hide inside one wave* and *how many waves
the grid can actually supply*. Attention moves both terms with seqlen: short seqlen means a
short K-loop and a small grid (occupancy-starved -> more resident waves help), long seqlen
means a long K-loop and a grid that already fills the machine (register headroom per wave
helps more). So the optimum is **not a constant across a seqlen range**, and the crossover
point depends on the kernel's register pressure and the SKU's register file
(`../hardware/planning-constants.md`, `scripts/calc_perf.py occ`).

- **Gate**: the dispatch serves a **range** of seqlens (or varlen with a wide length
  distribution) from one tuned config.
- **Knob**: `waves_per_eu` (include `0` = compiler's choice) — sweep it **across the seqlen
  range**, not at one shape, and check whether the winner flips somewhere inside the range.
- **Acceptance**: if a crossover exists, take **one shape on each side** and A/B the two
  candidate values there; keep per-range values (or the value that is least bad across the
  range) and record where the crossover sat. Never extrapolate a single-shape winner to the
  whole range. Occupancy actually achieved is the confirming signal, not the hint value:
  read it from the `.s` (`scripts/asm_loop_audit.py`, which prefers LLVM's own
  `; Occupancy:`), since the hint is a request the register allocator may not grant.

## Applying the framework to attention variants

This is the **attention fill** of a workload-neutral variable set; the set itself, the
archetype-mismatch check that precedes it, and the recompile-depth census that runs beside it are
in `intake.md`. Read that first for a variant you have not classified yet; read this
once you know it is attention.

Do not memorize per-variant recipes -- read a new variant through the **same
variables** the framework above uses, then apply the principles:

- **Which matmuls + which reductions and their axes** -> the structure decision
  (P1/P2) and where atomics/recompute land.
- **What sits between matmuls** (softmax / gating / smoothing / dequant) -> a VALU
  stage that breaks the pure-GEMM-accumulator assumption, so the GEMM-only compiler
  answers do not apply (P6: a throughput-pairing GEMM scheduler **asserts** here rather than
  regressing, and the AGPR accumulator hint
  regresses; the portable `schedule_hint` / `amdgpu-sched-strategy` scheduler option
  does apply for ILP-starved kernels (sweep + A/B), then interleave manually, load a
  region-classifying scheduler as a plugin, or author the policy under sanctioned co-design:
  `../tile-programming/compiler-contract.md ## VALU-between-matmul: manual interleave, or author a pass`)
  and may force AGPR<->VGPR round-trips.
- **Where that third instruction category has to ISSUE, and whether it fits** -> the
  **co-execution budget**. This is a separate variable from "what sits between matmuls": that
  one says the VALU stage exists, this one says whether it can be hidden. Per compute region,
  `capacity = (MFMA in region) x co-execution window` against `demand = sum of class issue
  costs`; `demand > capacity` means no ordering wins and the work must shrink or move to the
  other region. Two gates come first, and both are cheap: (a) **two resident waves per SIMD**,
  or the vector work has no memory work in the *other* wave to pair with and the whole
  argument collapses to the single-wave compiler-interleave model; (b) **the bound class is
  actually matrix-issue** — on a latency- or bandwidth-bound decode dispatch this budget is not
  the binding axis. Arithmetic and the balancing procedure:
  `../tile-programming/llir-codesign.md ## Attention: the co-execution budget`; the window and
  per-class costs (per shape, never carried) `../hardware/isa-mechanisms.md`.
- **Operand / latent footprint** -> the capacity-wall + occupancy budget (P5/P8);
  e.g. low-rank / compressed KV changes D and operand size, hence the LDS /
  register budget.
- **Precision side-paths** (quant / per-block descale) -> a side path
  (`../tile-programming/low-precision.md`) + P6.
- **KV access pattern** (dense vs sparse / indirect / paged) -> the memory path
  (gather / index chain) and possibly the parallel structure.
- **Does the loop body's composition vary per iteration?** (static vs data-dependent) -> whether
  a *schedule* is a well-posed question at all. See below — this is the one variable that can
  invalidate the others rather than just re-value them.

So MLA (compressed latent KV -> operand/latent footprint + LDS budget),
Sage-attention (quantized QK/PV + smoothing -> precision side-path + VALU gate),
and DSA (sparse / selected KV -> indirect memory path) are the *same* framework
with different values in these variables, not new recipes.

### The dynamic-body variable (sparse / paged / varlen)

The first six variables are all **static** properties of the kernel: given the shapes and
dtypes, you can compute their values on paper before compiling. Selected-KV, paged and varlen
attention add one that is not, and it deserves its own read because two of its consequences do
not show up anywhere in the other six.

- **The address chain is vector work, and it lands somewhere.** A gather / page-table indirection
  is VALU that a dense kernel does not have. Put it beside the matrix ops and the region now
  holds matrix + VALU + memory — **three categories, which is the combination that gets skipped
  rather than scheduled**, because the two scheduling models disagree about what the matrix op's
  shadow is *for* (`../tile-programming/llir-codesign.md ## Region routing`). The fix is
  placement, not tuning: keep the index chain in the memory region with the loads it addresses,
  so the compute regions stay two-category.
- **A data-dependent trip count removes the static-unroll premise.** Loop-carried buffers that
  swap places each tile are normally restored by unrolling by the swap period, so the same value
  lands in the same registers every iteration. That argument needs the period to be known at
  compile time. When the block count is data-dependent, either the unroll is not available or it
  needs a guarded remainder — and the remainder path is a *different* region structure from the
  main one, so a budget computed on the main body does not describe it.
- **The budget itself becomes a distribution, not a number.** `capacity` is per region and the
  regions still exist, but how many times each executes now varies per (query block, sequence).
  A ratio balanced for the average is over-full at one end of the distribution. Balance for the
  **common** case and check the tail separately, and read the acceptance signal on the worktile
  duration *spread* rather than the mean (`## Scheduling (load-imbalanced grids)`).

Practical consequence: on a dynamic body, prove the region composition is what you think it is by
reading the lowered code **before** computing any budget. A minimal dense body is not evidence
about the real kernel here — that gap has been measured, and it was large
(`../gluon/pipeline-reference.md`).

### Substituting into the co-execution variable (head dim, MLA)

Worth spelling out, because this variable moves in a way the others do not: **capacity and demand
both change with the tile, at different rates**, so nothing about a balanced schedule transfers
across shapes.

- **Head dim `D`.** Row-wise terms (the max/sum reduction tails, the cross-lane shuffle that
  closes them) are **independent of `D`**; per-element terms scale with the score tile's N;
  the accumulator rescale scales with `D`. So halving `D` halves the rescale while leaving the
  exponential burst untouched, and the split between the two compute regions has to be
  recomputed. **The balance ratio is a per-`(D, block-N, dtype)` quantity — never carry one
  shape's ratio to another**, and slice granularity is what makes a new ratio reachable at all.
  A `D` change usually also changes the MFMA shape, and the window is per shape: re-derive it
  from the MFMA the region actually contains rather than reusing the previous number.
- **MLA.** Read it as a **precondition check before the budget, not a recipe.** A large latent
  dim pushes the accumulator's per-lane register count up, which is exactly the resource that
  caps resident waves — and if occupancy drops to one wave per SIMD, gate (a) above fails and
  the co-execution framing no longer applies. **Do not read that as "so use
  `compiler_interleave`":** a
  read-modify-every-iteration accumulator also rules out that model's GEMM ladder, which
  asserts on VALU-between-matmul. That combination is its own cell with its own remaining
  levers — `../tile-programming/scheduling-model.md ## When steps 1 and 2 both say no — the
  neither-model cell`.
  Check the achieved occupancy first, then the bound class (decode MLA is usually
  memory/latency-bound), and only then compute a budget. The absorb / non-absorb forms differ in
  *which matmuls exist*, which is the first variable in the list above, not this one.
- **Beyond attention.** The generalizable form is "three instruction categories, two issue
  ports": any fused loop carrying matrix + memory + a third category (MoE gating, quantized-GEMM
  dequant unpack, a fused norm) is the same allocation problem. Note this **adds** a second
  prescription to the dequant-VALU reading in `../tile-programming/low-precision.md`: besides
  making the unpack cheaper, it can be *hidden* in the matrix shadow — provided it is not in a
  packed form and the budget admits it.

## Family signals (attention / decode)

Preserve logical shape, masks, split boundaries, online state, and output feeding
before layout/MFMA/scheduling. Check guarded fast paths, split/partition +
reduce/combine, and the wrapper/dispatch boundary before MFMA. Do not combine
RoPE, online softmax, QK/PV, split logic, K/V indirection, shared memory, and MFMA
in one patch.

- **Paged / indirect KV**: `kv_loc = page_table[offs//PAGE]*PAGE + offs%PAGE`. The
  dynamic tensor may have `BlockedLayout` (not `SliceLayout` lineage); if the next
  step needs `expand_dims(kv_loc, 1)` for a 2D address, that is a quick-reject
  signal for a direct Gluon stage migration — keep the indirect load plain, probe
  a later streaming/reduction stage, or design a source-proven gather. Explicit
  layout / buffer ops help only when they reduce the address chain, bytes moved, or
  a measured conversion.
- **Quantized attention**: decompose `quant/preprocess_us / attention_us /
  dequant_us / reduce_us` before body work; if quant/preprocess is material, tune
  that stage first (config wins from compute-bound attention do not transfer to
  memory-bound quant kernels). Treat per-block descale as a side path
  (`../tile-programming/low-precision.md`).
- **Atomic / non-deterministic correctness**: if baseline and candidate use
  nondeterministic atomic reductions, compare against a deterministic /
  high-precision reference with task-specific tolerance, not two atomic outputs
  (`../method/close.md`).

## Reuse (owned)

QK/PV layout-chain detail: `../gluon/matrix-reference.md`. Decode split/partition
and visible dispatch: `../method/front-end.md` (P0 split / P6 dispatch).
