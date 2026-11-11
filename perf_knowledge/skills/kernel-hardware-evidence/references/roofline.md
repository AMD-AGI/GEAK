# Roofline and traffic

For matched work `W`, bytes `Q` and elapsed seconds `T`:

```text
intensity = W / Q                 # operations/byte
rate = W / T                      # operations/second
model rate = min(P, B * intensity)
model time = max(W / P, Q / B)     # seconds, allows overlap
model fraction = model time / T
```

`P` and `B` must match the work type and memory interface. State source, case,
device, interval, operands and units. A model percentage is conditional on those
assumptions; it is neither measured utilization nor guaranteed removable time.
Adding the two service times instead requires a justified serialization model.

## Work and bytes

Keep useful operation work separate from executed work (padding, masked lanes,
recomputation and helper stages). State the FMA convention, usually two FLOPs,
operand/accumulator dtype and matrix/vector/integer execution unit. Mixed
instruction types need separate work/capacities, not one universal FLOP roof. Static ISA
sites do not establish loop trips or active-lane work. Lower latency with fewer
executed FLOP/s can be a successful reformulation.

Distinguish these byte meanings:

- **Logical footprint:** required inputs and outputs with stated reuse assumptions;
  include metadata and list workspace separately. It does not measure HBM traffic.
- **Interface requests:** documented counter counts and their conversion at a named
  cache/memory interface, with read/write/atomic treatment and instance reduction.
- **Physical transfers:** use this label only when the counter mapping supports it.

Use the memory level described by `B`. Cache hits, repeated loads, spills, write
policies and request granularity separate these quantities. Tensor size below
cache capacity does not establish residency; support it with matched cache/interface
counters or a controlled warm/cold reuse comparison. Name the byte method and its
evidence. Where reuse is uncertain, show a range from the one-transfer footprint
to the modeled repeated passes, with each assumption explicit. These are logical
byte scenarios, not measured HBM bounds. Similar logical and request totals
neither identify addresses nor prove there is no redundant traffic to remove.

For the reviewed gfx950/ROCm 7.2 SDK definitions, these are DRAM-directed **request
bytes**, with atomics separate:

```text
read   = 32 * TCC_EA0_RDREQ_DRAM_32B_sum
write  = 32 * TCC_EA0_WRREQ_WRITE_DRAM_32B_sum
atomic = 32 * TCC_EA0_WRREQ_ATOMIC_DRAM_32B_sum
```

Verify definitions in the actual SDK before reuse. Do not apply that multiplier
to similarly named events or wildcard sums. A documented request-based comparison
is useful without establishing literal HBM wire traffic.

## Timing and capacities

Match invocation counts and phases: one dispatch, the whole callable, or a workload.
Separate warmup, calibration, copies and unrelated rows. Use sums of work divided
by matching elapsed time; averaging individual rates answers a different question.
Summed dispatch durations double-count overlap. Multi-pass counters come from
replays; do not treat them as simultaneous or clamp negative replay differences.
Profiled timings explain their capture; validate gains with accepted unprofiled
full-operation measurements and current repeat variability.
Keep isolated-kernel variability separate from workload variability: a repeatable
kernel gain may be smaller than the uncertainty of the end-to-end measurement.

Label a capacity as a specification, model or measured comparator. Check the actual
product, instruction mode, dtype, clock/power conditions and dense/sparse convention.
For memory also check access pattern, read/write mix, working set and cache state.
A copy benchmark or tuned library result demonstrates its own conditions, not a
universal ceiling. Separate calibration peaks may not be attainable together.

Arithmetic intensity selects the limiting roof in this model, not the measured
bottleneck. Low rates can reflect dependencies, launch cost, insufficient parallelism
or other resources. A ratio above a comparator calls for checking scope, units,
traffic assumptions and calibration; keep the raw ratio rather than claiming
saturation. Show a conditional range or leave unsupported parts unknown. Choose
a targeted change or measurement to test the explanation; high utilization never
proves all current work is necessary.
