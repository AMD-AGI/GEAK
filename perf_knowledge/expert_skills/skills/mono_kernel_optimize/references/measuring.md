# Measuring a whole-layer kernel: timeline, floors, which number to trust

A kernel trace shows one kernel (or two) per layer, so it cannot say which stage is slow. The tool is a
per-stage timeline taken inside the kernel, read against each stage's byte floor. Without it the lever
tables are guesses; with too much of it the round produces no candidate.

## Time box and order

- Measurement gets at most about a quarter of the round (a lane round is one engineer with roughly one
  to two hours).
- Cheapest first: (1) the per-case table of the lane's baseline — already there; (2) an existing
  timeline switch — cheap; (3) byte floors — arithmetic; (4) instrumentation of your own — only when
  no switch exists *and* more rounds remain after this one.
- Stop when you can name the stage, or the wait, that holds the largest share of the heaviest-weighted
  case. One more decimal does not change the plan.
- Time box expired without a timeline: plan from the per-case table, the floors and the structure class,
  and write in your notes that the stage split is unmeasured.

## Step 0: is a timeline already built in?

Deployed kernels often carry one. Look before building:

| language | what it looks like |
|---|---|
| HIP C++ | an optional trace / stamp buffer pointer in the kernel's argument struct (null in production) and a helper or macro that stores the realtime counter at numbered points for thread 0 of each workgroup. Passing a buffer of workgroups × points 64-bit words turns it on |
| FlyDSL | a timeline build: numbered stage marks through a stamp helper, written to a scratch region that only the timeline build has (public example: `stamp(c, point)` in sgl-amd-mono `sgl_amd_mono/common/stamps.py` — "a build without the region records nothing"); or a `timeline` / profiling flag in the kernel builder that yields a separately named kernel with one extra buffer argument |

Search the kernel sources for `trace`, `timeline`, `stamp`, `prof`, `realtime`. Check the wrapper and
test helpers for the code that allocates and decodes the buffer (it names the stage behind each index).
Use what exists as it is; do not renumber its points. In a dual-launch kernel one row of marks may hold
separate index ranges for the front and the tail launch (the public example does).

Public FlyDSL (at the commit in the skill's Sources) also ships an experimental in-kernel tracer,
`flydsl.expr.ktrace`: named ranges per wave, one traced device per process, a timeline export, gfx942
supported. *Read from the public source only*, not exercised here on a multi-rank layer kernel: an
option when the installed FlyDSL has it and the kernel has no marks of its own.

## Building one (only when justified by the time box)

1. A separate profiling variant with its own kernel name (so a build or JIT cache never mixes it with
   the real one) and one extra argument: a device buffer of workgroups × events 64-bit words.
2. Thread 0 of every workgroup stores the realtime counter (`s_memrealtime`, 100 MHz, the same clock on
   every CU of one device) at fixed events: kernel entry, start and end of each stage it works on,
   before and after each wait (tile wait, device-wide barrier, cross-rank wait). Plain stores; the
   buffer is read after the kernel finishes.
3. Run it from a small driver of your own, one process per rank, on the task's captured inputs, inside
   the same multi-GPU lease the oracle needs. Replay about 20 times and keep the median per event.
4. Never add the buffer argument to the installed kernel "temporarily": acceptance applies to what is
   installed, and every stamp costs a little.
5. Keep the driver and the variant outside the kernel source tree when the task's static rules apply
   to every file in it.

## Reading a timeline

- Per stage: span = max(end) − min(start) over its workgroups; wait = time spent inside waits.
- Per launch: entry skew (first against last workgroup entry) and the tail after the last stage.
- Compare ticks only within one launch on one device; the counter is not synchronized across GPUs.

| observation | conclusion | open |
|---|---|---|
| a stage far above its floor | that stage's own kernel | [stage_levers.md](stage_levers.md) |
| stages near their floors, layer still slow | structure: imbalance, waits, barriers, the all-reduce | [structure_levers.md](structure_levers.md), [handoff_levers.md](handoff_levers.md) |
| a wait dominates a stage's span | the producer's tail or the hand-off granularity, not the consumer's math | [handoff_levers.md](handoff_levers.md) |
| large entry skew | too many workgroups, or heavy per-workgroup setup before the first stage | [structure_levers.md](structure_levers.md) |
| a gap between layers | a per-layer completion counter | [handoff_levers.md](handoff_levers.md) |

## Floors: a ramp plus bytes at the marginal stream rate

floor(stage) = ramp + bytes the stage must read on this rank ÷ marginal stream rate. Bytes = this rank's
share of the stage's weights (shapes and dtypes, block scales included; only the experts a row routes
to; padded rows are not read) plus, for attention, the KV bytes at the case's context (only the window
for sliding-window layers). If the oracle flushes L2 and the Infinity Cache before every sample (read
its docstring), every byte comes from HBM.

**Do not use "bytes ÷ the peak", nor "bytes ÷ about 85% of peak", as the floor of a decode-sized
stage.** The public hardware card gives 5.325 TB/s peak for MI300X
([arch.md](../../../../hardware/cdna3_mi300/arch.md)). Measured on an idle MI300X, the *effective*
fraction of that peak (bytes ÷ total time) falls sharply as the transfer gets smaller:

| transfer | effective fraction of peak | how measured |
|---|---|---|
| 64 MB read, tuned stream on all CUs inside a kernel | about 58% | microbench inside one reference kernel — the only in-kernel figure |
| 1 GiB read | about 75% | generic framework reduction over the buffer, median of 20, cold and warm alike |
| 1 GiB copy | about 71% (bytes moved) | generic framework copy |
| 256 MB read | about 60% | same reduction |
| 64 MB read | about 31% | same reduction |
| 32 MB read | 17–20% | same reduction |
| 16 MB read | 8–10% | same reduction |

How to read it:

- The generic rows are one framework call each and include that call's fixed latency (launch,
  reduction, result): they are not what a stream inside a persistent kernel gets. They show the shape.
  A straight-line fit of the five points gives a fixed part plus a marginal rate of about 83–85% of
  peak (*derived*); the fixed part is 60–80% of the time at 16–64 MB and about 10% at 1 GiB.
- Inside a kernel the fixed part is smaller but not zero: at that marginal rate the tuned 64 MB stream
  spent about 30% of its time in the ramp (*derived*). A decode layer moves tens of MB to roughly
  100 MB per rank, split over stages, so its stages sit where the ramp matters most. The lane that
  optimized the FlyDSL reference kernel used 60% of peak for a whole layer's bytes in its estimate.
- **Measure your own ceiling** before calling a stage "above its floor": an all-CU stream read (wide
  coalesced loads, one workgroup per CU) after the same cache flush the oracle uses, median of at least
  20 repeats, at two sizes — the stage's bytes and several times that. The slope is your marginal rate,
  the intercept your ramp. Until you have them, use 58% of peak for a stage of about 64 MB and less for
  a smaller one, and say that the floor is an estimate.
- Average-bandwidth slack is not usable slack (*measured*): a stage that seemed to have spare bandwidth
  slowed down when a neighbour's bytes were added to its stream. Probe by injecting the bytes before
  predicting a gain from overlap.
- A floor estimate for a whole layer = byte floor + hand-off latency. In the measured FlyDSL reference
  the byte floor was about half of the achieved layer time and hand-offs another 15–22%, which put the
  achieved time at roughly 1.35–1.5× the estimated floor.

## Which number to trust

| question | number to use | why not the oracle's speedup |
|---|---|---|
| is the candidate accepted | the oracle's own speedup, baseline and candidate timed in the same run, re-measured by the lane's verify step | it is the only number that counts |
| how long does the kernel itself take | the layer kernel's duration in a kernel trace, per process | the oracle times one graph replay per sample after a cache flush: graph launch, the flush kernel and any bookkeeping are in **both** legs. That fixed part was of the order of a fifth to a quarter of a tuned layer's kernel time, so the oracle's ratio understates a kernel-only gain. For kernels shorter than about one graph launch, a single run of the ratio spread from 0.73× to 1.75× |
| which stage is slow | the in-kernel timeline | a trace sees one kernel per launch |
| how far from the hardware | ramp + bytes ÷ your measured marginal rate | bytes ÷ peak leaves out the ramp, which dominates decode sizes |
| what a restructured layer will cost | fixed per-launch part + kernel time summed along the critical path (every stage when all workgroups run every stage; the longest hand-off chain otherwise) | stage speedups do not add up: a fused layer pays the launch part once, not once per stage |

In a trace of a two-leg test, kernels present in both legs are harness work; the layer's kernels are
the ones present in one leg only. Under a profiler a kernel that waits on peers absorbs the ranks' skew:
take the minimum over ranks per sample, then the median.

Noise differs from oracle to oracle: about 2% run to run for a whole-layer case in one harness, and XCD
clocks alone differ by 3–10% ([xcd_chiplet.md](../../../../hardware/cdna3_mi300/xcd_chiplet.md)). Measure
it in your task: run the unmodified kernel as the candidate and read the A/A band. A gain under about
twice that band is not yet evidence — repeat it. Saving a patch follows the role file (as soon as it
beats the incumbent); the verify re-run decides.

## Cost of hand-offs (relative to a tuned layer)

- One device-wide barrier: about 4–5% of the layer's time (*measured*). Ten stage-wide barriers would
  then be 40–50% (extrapolated).
- Layer-to-layer gap (*measured*): about 8% of the layer with a per-layer completion counter, about 3%
  with one epoch per step.
- Why: a global atomic crosses cores and often XCDs (its latency is in
  [xcd_chiplet.md](../../../../hardware/cdna3_mi300/xcd_chiplet.md)), and a barrier makes about 300
  workgroups wait for the slowest.

## Hygiene

- Nothing else on the measured GPUs. Every GPU of the rank group is held in one lease for the whole
  command — the oracle and your own probes alike. The COMMANDMENT's command takes that lease itself:
  never wrap it in a second one.
- Baseline and candidate in the same run. If you time them yourself, alternate them.
- Kill only process ids you started; never kill by name pattern.
- A kernel-lane gain is not a serving gain: steps the kernel does not take (other widths, mixed
  prefill) can dominate a serving metric. Report the lane's number as the lane's number.
