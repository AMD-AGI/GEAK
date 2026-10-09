# Readings that point the wrong way

Part of `../inline-asm-reference.md`. Read this when a measurement disagrees with what you expected,
and before you act on a counter you have not calibrated on your own box.

Everything here is a *reading* problem rather than a mechanism problem, which is why it is filed
under no class. Each entry gives the reading, why it inverts, and how to recognise it without having
already been burned.

## The one that costs the most: "fewer is better"

Instruction count is not work, and "fewer of everything" is not a performance ordering. Two measured
shapes, in increasing order of how badly they break intuition.

### Fewer *and* packed, and slower anyway

One arm replaced a chain of single-width scale arithmetic with its packed two-at-a-time equivalent:
roughly half as many arithmetic instructions in that region, and materially fewer vector
instructions across the whole kernel. Same matrix-instruction count, same shape, zero spills and
zero shared memory in both arms. It ran **materially slower**, and reproduced across several timing
windows on more than one device.

The instruction saving was real and irrelevant. What actually changed was the register **class**:
without the tie the compiler moved dozens of accumulator values into the accumulation register file,
which pushed the unified footprint from just under an allocation boundary to just over it and
**halved the waves resident per SIMD**. A third arm that eliminated the arithmetic *algebraically*,
cutting the op count further still, remained slower than the original — which closes the "then cut
even more" escape hatch: the axis was never op count.

> **When you shrink a hot arithmetic chain, check the register footprint and the resulting
> residency before you check the clock.** Packing is a register-class decision wearing an
> instruction-count disguise.

### Leaner on every axis you can count, and still slower

A second arm had fewer disassembled lines, fewer static waits, fewer registers, thousands fewer
vector instructions and fewer issued instructions than its comparator — with matrix, memory and
shared-memory instruction counts *identical*, occupancy identical, spill counts identical — and ran
reproducibly slower.

> **A reader looking only at resource counts gets the opposite conclusion, with confidence.**

**Recognise it:** when every count is equal or better and the clock is worse, you have proven the
constraint is **timing, not capacity**, and no capacity counter will ever explain it. The reading
that settled this one was **dependency-wait cycles rising while active cycles stayed flat** — the
machine doing identical work and waiting longer for permission to do it — corroborated by the
stall-reason histogram from a sampling instrument.

**Be careful which timing instrument you reach for.** A per-instruction trace is the obvious choice
and it is the one most likely to mislead you here, because its stall columns can drift between two
captures of the *same* arm by more than the effect you are chasing — and which columns do that is
not predictable from one kernel to the next. If you use it, use it under the duplicate-capture
control described below, per class, and lean on its count columns rather than its stall columns.

**The general form:** resource counters answer *does it fit*. They do not answer *does it overlap*.

## "Which resource is over" is not "which site to remove"

A counter showing a register budget a few units over a step is a correct reading and an incorrect
instruction. Where several sites draw on one budget, removing the one that looks most expensive can
be several times worse than removing all of them. See `classifying.md ## BUDGET-GROUP` for the
worked table and the gate.

## A saturation metric whose direction is inverted

A shared-memory command-queue "full rate" can read near-saturated on the **fast** arm and near-empty
on the slow one. That metric counts commands *in flight* — pipeline depth — not back-pressure on a
consumer: pinning a value kept the reads issuing early and the queue populated, while removing the
pin let the compiler spread them out until the pipeline collapsed.

**Recognise it:** for any counter whose name contains "full", establish first whether it measures
occupancy-of-a-queue or refusal-of-a-producer. Two cross-checks that did not invert on the same
pair: the companion *data* queue read zero on both arms, and the median gap between matrix
instructions moved in the expected direction.

## Occupancy is not a goodness metric, in either direction

Two measured cases, opposite shapes:

- Higher occupancy on the **faster** arm, with summed wave-cycles also higher — because wave-cycles
  are summed over resident waves, so more residency inflates the total. **Never compare summed
  wave-cycles across arms with different occupancy; compare kernel time.**
- Occupancy **rising** by nearly half while the kernel got almost three times slower — the extra
  waves were resident because they were stalled in scratch traffic.

**Recognise it:** when occupancy moves, ask whether more work is in flight or more waves are
waiting. The reading alone does not distinguish them.

## Contention metrics falling because there is nothing left to contend for

On two separate slower arms, an arbitration-loss rate fell by an order of magnitude and a
"no instruction available" rate halved. Both look like improvements; both accompanied a regression,
because the pipeline had emptied.

**Recognise it:** check total issue volume before reading any contention metric as good news.

## Where register, spill and accumulation-register verdicts come from

On this target, the static kernel descriptor and the runtime profiler **disagree**, and the
disagreement is systematic rather than noisy:

| quantity | take it from | why not the other one |
| --- | --- | --- |
| register **demand** | the metadata vector-register count, cross-checked against the compiler's own total-registers comment — and **only when the spill count is zero** | the profiler's register column has been observed reporting a fixed fraction of the descriptor's, which makes any occupancy derived from it optimistic by that same factor |
| accumulation registers | the static accumulation-register count | the runtime column can read **structurally zero** on kernels whose descriptor plainly carries them — a zero there is not a measurement |
| spill | the static spill-slot count, the private-segment size, and a `scratch_*` instruction census | the profiler rows whose *names* promise a spill answer — an occupancy-limiter percentage, a scratch-stall rate, a spill/stack instruction count — have all been observed silent through a spill severe enough to multiply the kernel's runtime: the percentages reading zero in **both** arms, and the instruction count reporting an identical value for an arm with no scratch instructions and an arm with many |

**Three traps specific to these fields, all of which produce a confident wrong number:**

- **The allocation field is not the demand field.** The descriptor field the hardware allocates
  from is `max(actual demand, a floor implied by any declared waves-per-execution-unit hint)`. With
  no hint the two coincide, which reads as confirmation; with a hint the descriptor climbs to the
  hint's floor while the demand metadata stays flat. **For "is this kernel under register pressure"
  use demand; for "how many waves fit" use the allocation field** — and take that second derivation,
  with its rounding and its LDS term, from
  `../../hardware/planning-constants.md ## VGPR / occupancy thresholds`, not from here.
- **A spilling kernel has no demand figure.** Once the allocator is capped, the demand metadata
  collapses onto the cap and stays pinned there while the spill count climbs — so the number reads
  stable precisely while the situation worsens. The kernel's real demand is visible only at the
  loosest hint setting, or with the hint removed. **Spill count nonzero ⇒ the register count you
  are reading is a budget, not a demand.**
- **The accumulation-register base offset is not a register count.** It is the aligned start of the
  accumulation registers within the unified file. When the accumulation count is zero it is padding
  with nothing behind it, and because it is rounded up to the alignment it **can exceed total
  usage**. The tempting identity *demand − accumulation = offset* is therefore true **by
  construction whenever the accumulation count is positive** — where it cannot fail, so it confirms
  nothing — and a coincidence otherwise, holding only when the demand happens to land on the
  alignment. Never quote the offset as a register figure.

**Establish all of this on your own box the first time it matters** — dump one kernel you know
spills and compare the two sources. The recipe is the durable part; any figure you inherit is one
build's.

> **One conflict, resolved rather than left standing.** The shared profiling chapter used to
> instruct the opposite for accumulation registers — read them from the runtime profiler, not the
> static dump, on the reasoning that the split happens at run time. The split does happen at run
> time; what does not follow is that the profiler observes it. On the measured build the runtime
> column is structurally zero on kernels whose descriptor carries dozens of accumulation registers,
> so that rule has been corrected at its source. **Where a static and a runtime reading disagree
> about what the compiler allocated, the disassembly is the artifact you can re-read.**

A cheaper consequence worth taking: all three of these come out of the disassembly, so **every
question about what the compiler allocated is answerable with no GPU at all.**

## Reading a per-instruction trace: three things the tool does not enforce

### Find out what your source field actually is, before you parse it

An instruction-level trace attributes each record to source with a **space-separated chain of
`path:line` frames, outermost caller first and the leaf last**. But **whether the chains are there
at all is a property of your build, not of the instrument**: a build carrying inlining debug info
produces multi-frame records, and a build without produces records that are all depth one. Both
have been observed on this instrument.

**So measure the frame depth of your own trace first** — histogram the token count of the source
field across all records — and only then choose a parse. Each wrong choice fails in its own
direction:

- **Assuming leaves when you have chains.** Every enclosing frame is silently discarded. A construct
  that emits no instruction of its own but encloses others — an empty tied identity is exactly this
  — then reads as **zero records**, and you conclude the instrument cannot see it. It can; you threw
  the evidence away. This is a failure that has shipped in real plotting code more than once.
- **Assuming chains when you have leaves.** Harmless for counting, but a first-colon split on a
  single-frame record yields a fragment of the path rather than a line, which a permissive integer
  extractor will turn into a plausible wrong number.

With the depth known, the rule is unambiguous:

> Tokenize on whitespace, then split each token on its **last** colon.
> **The last frame is what the instruction *is*** — use it, and only it, for "which line emitted
> this instruction."
> **Membership in the whole chain is what the instruction is *inside*** — use it for "is this site
> present in this region."
> A site can be real and load-bearing with **zero** leaf records.

Where chains do exist they carry a free by-product worth knowing: an arm containing such a construct
has multi-frame records where the arm without it has none at all, so **the presence of a chain is
itself evidence the construct is there** — and its absence in the comparison arm is a categorical
difference, not a noisy one.

### Capture it twice. Unconditionally.

Two captures of the **same** arm are not the same numbers, and the split is clean:

> **Everything structural is reproducible** — record count, addresses, the instruction and source
> maps, and every hitcount. **All the variance lives in the stall, latency and idle columns.**

That much is portable, and it is what makes hitcount-based and count-based claims the load-bearing
ones. **How the stall variance distributes across instruction classes is not portable.** Two
independent corpora on this instrument disagree not only on magnitude but on *which class is the
stable one* — a class that was the steadiest reference channel on one was among the movers on the
other. **There is therefore no ordering to inherit and no threshold table worth printing**, here or
anywhere else; a number you did not measure on your own trace is not a floor, it is someone else's
kernel.

> **The control is one extra capture, and it is not optional.** Trace arm A twice. Use the A-to-A
> difference **in the same instruction class you are about to make a claim about** as the error bar
> for the A-to-B difference. An A/B delta smaller than the A/A delta in that class **is not a
> result**. Do this per class, not once for the trace.

**The failure this prevents is a sign flip, not an imprecision.** Two captures of one arm, compared
against the same second arm, have yielded **opposite conclusions** about whether a change raised or
lowered wait stall. Which conclusion you reach was decided by which capture you kept — so an
uncontrolled single capture is not a weak result, it is an **unsigned** one.

**And do not normalize stall to a share of the total to escape this.** It is the natural move and it
makes things worse: stall-per-hit divides by a hitcount, which does not drift, whereas a share
divides by total stall — precisely the quantity that carries the variance. Share-normalizing takes
whatever your most stable class happens to be and imports the whole trace's noise into it, so a
class whose per-hit rate is steady between captures can still have a visibly moving share. **The
per-hit rate is the capture-robust quantity; the share is the contaminated one.**

### Know what the capture does not cover

Capture count and class stability are two axes; **coverage is a third, and it is the one most often
left unstated.** A trace is taken at one shape, on one wave, on one compute unit. A conclusion drawn
from it is scoped to that shape — and if a shape bucket carrying a material share of real calls has
no trace at all, the per-class reading says nothing about it.

**State the covered shape next to every per-instruction claim**, and state which buckets are
untraced. An untraced bucket is a coverage gap, which is a different and more honest finding than a
null.

## Two instrument traps that produce numbers rather than errors

- **A timing read from a counter-collection run is not a timing.** Under hardware-counter collection
  a harness can report a kernel time more than an order of magnitude above the dispatch duration in
  that same run's own trace timestamps. Collect counters and collect time in **separate runs**.
- **Derived percentages are normalised over a window wider than the kernel**, so they under-report
  systematically, and the shorter the kernel the worse it gets.

## Zero-instruction sites are invisible to every PC-attributing instrument

An empty asm emits nothing, so there is no address to sample. In one annotated kernel, most of the
empty-tie sites had **no direct disassembly record at all**. Annotating them honestly means
anchoring to a proxy — the producer of the pinned value, or the next use of the tied register — and
saying in the figure which anchors are direct and which are proxies.

> **Absence of a hotspot at a site is not evidence that the site does nothing.** It is the expected
> appearance of the commonest shape on these pages.

## The hottest line is not the site

In a worked reverse-derivation, the hottest lines by stall were a matrix instruction and a store;
the actual site ranked well down the list and sat over a dozen lines away. The instrument points at
the **consumer** of the pressure. The site sits on the producer side of the edge.

`deciding.md ## What this page can localize, and where it stops` is why this is structural rather
than a tooling gap.
