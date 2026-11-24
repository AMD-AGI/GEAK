# Readings that point the wrong way

Part of `../inline-asm-reference.md`. Read when a measurement disagrees with expectation, before acting
on an uncalibrated counter. GPU-measured observations on one build and box unless marked inferred;
any threshold is per box — re-derive on yours.

## "Fewer instructions" was not faster in the cases measured — register class and residency

- **Packed and fewer, yet slower:** replacing scalar scale math with packed 2-wide ops halved that
  region's VALU count (same MFMA count, no spills) and ran materially slower: without the tie, dozens of
  accumulator values moved to AGPRs, the unified footprint crossed an allocation boundary and **waves/SIMD
  halved**. Cutting the ops further algebraically stayed slower — the axis was never op count.
- **Leaner on every count, still slower:** fewer lines/waits/registers/VALU, identical MFMA/memory/LDS
  counts, occupancy and spills — slower. Proves a **timing, not capacity** constraint; the signal was
  dependency-wait cycles rising with active cycles flat (corroborated by sampled stall reasons).
  **Resource counters answer "does it fit", never "does it overlap".**
- **"Which resource is over" is not "which site to remove"** (`classifying.md ## BUDGET-GROUP`).

## Counters whose direction inverts

- **LDS command-queue "full" rate** read near-saturated on the *fast* arm: it counts commands in flight
  (pipeline depth), not back-pressure. For any "full" counter, establish queue-occupancy vs producer-refusal.
- **Occupancy is not goodness:** summed wave-cycles grow with residency (compare kernel time, not summed
  wave-cycles); occupancy rose while the kernel got several times slower (waves stalled on scratch).
- **Contention metrics falling** (arbitration loss ÷10, "no instruction available" halved) accompanied
  regressions — the pipeline had emptied. Check total issue volume first.

## Where register, spill and accumulation-register verdicts come from

The static descriptor and the runtime profiler disagree systematically — **trust the disassembly** (no GPU needed):

| quantity | take it from | why not the profiler |
| --- | --- | --- |
| VGPR demand | metadata VGPR count (cross-check the compiler's total-registers comment), only when spills = 0 | profiler register columns that do not match the descriptor (no fixed ratio on gfx950 / rocprofv3 1.3.5) ⇒ wrong occupancy |
| AGPRs | static AGPR count | runtime `Accum_VGPR_Count` column does not track `.agpr_count` (it was non-zero on kernels with no AGPRs on gfx950 / rocprofv3 1.3.5; earlier read zero on kernels that carry dozens) |
| spills | static spill slots, private-segment size, `scratch_*` census | occupancy-limiter %, scratch-stall rate and spill instruction count all observed silent through a severe spill |

- **Allocation field vs demand field (inferred, not verified):** the allocated VGPR field
  appears to be `max(actual demand, floor implied by a waves-per-EU hint)` — with a hint it climbs while
  demand stays flat. Use demand for "under pressure?", allocation for "how many waves fit"
  (`../../hardware/planning-constants.md ## VGPR / occupancy thresholds`).
- **Spill count nonzero ⇒ the register count is the cap, not the demand**; real demand is visible only
  with the hint loosened/removed.
- **The AGPR base offset is not a register count** — it is aligned padding (can exceed total usage);
  `demand − AGPR = offset` holds by construction when AGPRs > 0 and confirms nothing.

## Reading a per-instruction trace: three things the tool does not enforce

- **Source field = space-separated `path:line` chain, outermost first, leaf last — if your build emits
  inlining debug info** (otherwise depth 1). Histogram the token count first. Tokenize on whitespace, split
  each token on its **last** colon; the **leaf** is the emitting line, **chain membership** is "inside this
  site". Leaf-only parsing makes an empty tied identity read as zero records; a chain present in one arm
  and absent in the other is itself evidence the construct is there.
- **Capture every arm twice.** Structure (record count, addresses, maps, hitcounts) reproduces; **stall /
  latency / idle columns do not**, and which instruction class is stable is not portable. Use the A/A
  delta *in the class you claim about* as the error bar; an A/B delta below it is not a result (it has
  flipped sign). Stall-per-hit is the steadier figure; stall share divides by the noisy total.
- **State the covered shape**; untraced shape buckets are coverage gaps, not nulls.

## Two instrument traps that produce numbers rather than errors

- Timing from a counter-collection run can exceed the trace's own dispatch duration by >10× — collect
  counters and time in **separate runs**.
- Derived percentages are normalized over a window wider than the kernel — they under-report, worse for short kernels.

## Zero-instruction sites are invisible to every PC-attributing instrument

Empty-asm sites have no address; annotate via a proxy (producer of the pinned value, next use of the tied
register) and say so. The hottest lines are usually the **consumer** (an MFMA, a store); the site sits on
the producer side of the edge (`deciding.md ## What this page can localize, and where it stops`).
