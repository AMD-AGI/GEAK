# Measurement, timing, and counter alignment

Timing, provenance, and cache guidance applies to **all three modes** while
preserving the selected engine's native timing semantics. The counter section at the
end belongs to `rocprof` and `combination`.

## Implement a small workload adapter

Use the existing unit test to enumerate realistic cases and construct inputs.
Keep the baseline and candidate inputs, routing, precision, and correctness
reference equivalent. Export the case signature and input metadata with results.

If import-time overlays, `sitecustomize`, or environment variables select the
implementation, run each implementation in a fresh process. Verify the callable's
source/build provenance and dispatched device symbol. A wrapper filename alone
does not establish which body ran; follow decorators and relevant bindings.

Warm up compilation, autotuning, allocations, and lazy initialization before timing.
Validate the candidate with the unit test's established tolerances. Preserve
validation failures and do not present their speedups as successful optimization.

## Timing scope

Define trace ranges around complete calls and associate device dispatches with
them. For serial retained kernels:

```text
device_us_per_call = sum(retained device durations over measured calls) / call_count
```

The divisor is measured **operator calls**, not necessarily a symbol's dispatch
count. One call may launch the same symbol multiple times. Preserve dispatch
multiplicity, count completeness, and exact symbol/range membership.

When slicing a PyTorch Chrome trace for TL, retain both endpoints of each selected
`ac2g` flow as well as the linked runtime and GPU events. Keeping only events on
the marked host thread drops GPU-side flow endpoints and can make TL's native
operator table empty despite valid kernel durations. Count host `user_annotation`
ranges as calls; matching `gpu_user_annotation` ranges may repeat the same names
and are not additional calls. Preserve the unsliced trace for alignment checks.

Summed durations are GPU work time. For overlapping streams, also report the
relevant elapsed interval or critical path; their sum is not elapsed latency.
For changing decompositions, retain all device work required by the fixed operator
boundary. Do not filter based on an old list of names that a candidate has changed.

Trace activity timing avoids interpreting a host launch floor as device execution.
GPU events are useful for an operator/replay interval, but an event window around
eager host launches can contain device idle gaps. Graph replay changes dispatch
and possibly reuse conditions. Report that timing basis explicitly and cross-check
short-kernel claims against device activities. Do not apply a universal 5/10-us
cutoff to declare all small kernels launch-bound.

## Cache policy and exclusions

Choose a policy matching the question: cold operands, warm repeated operands, or
a defined pool/reuse distance. A repeatedly reused UT buffer may fit in cache
even when deployment rotates weights/KV across layers.

For a cold test, perform the documented cache conditioning before **each measured
call**, outside its measured scope. Record the method, buffer footprint relative
to relevant caches, read/write behavior, and evidence it creates the intended
state. A write-only fill can add writeback traffic or interact with later passes;
do not assume every function named `flush_cache` establishes identical conditions.
Do not flush between internal stages unless the workload contract requires it.

Report excluded setup/flush events separately. Keep required zero initialization,
masking, conversion, or reduction inside a full-operator measurement. Exclusion
by broad symbol substrings can silently remove real work.

Do not divide cache-resident traffic by an HBM reference and call the result an
HBM utilization. Match the modeled memory level, traffic assumptions, and peak.

## Repetition and environment health

Run baseline and candidate on the same GPU configuration under comparable load.
Record raw repeated samples and order, summary statistic, variability, and useful
clock/power/contention observations. Use compatible health probes before/after
long sweeps when shared-node interference is a concern. Probes are evidence about
measurement conditions, not new peaks for every row.

A median is a reasonable ordinary summary. Minimum-of-repetitions can estimate
least-contaminated latency when demonstrated interference is one-sided, but label
it and retain the spread; it does not certify a quiet run. Rerun unstable cases
when they materially affect a conclusion. Avoid universal pass thresholds tied
to one historical GEMM calibration.

## rocprofv3 counter diagnostics (optional)

This section belongs to `mode=rocprof` and `mode=combination`. It covers **raw
`rocprofv3`**, where counters are collected and the roofline is derived by hand
afterwards. That is a *different tool* from `rocprof-compute`, which
[rocprof.md](rocprof.md) and `scripts/run_roofline.py` drive and which computes the
roofline itself. Do not mix their outputs or their units: `rocprofv3`'s
`FETCH_SIZE` / `WRITE_SIZE` are in **KiB**. Inspect the installed tool's help and supported counters
for the selected GPU. Save exact commands, counter definitions, units, and tool
versions rather than assuming names and factors are portable.

Collect incompatible counters in separate passes when hardware slots require it.
Keep input seed/routing, implementation, dispatch order, cache conditioning, and
selection ranges consistent. Record each pass's timing and sample count. Align
by case, kernel/range, and invocation ordinal; dispatch IDs need not persist
across processes. Exclude warmups in every pass and validate dispatch completeness.

The September 2026 gfx950 comparison used this **version-specific convention**:

```text
F_counter = 512 * sum(SQ_INSTS_VALU_MFMA_MOPS_{F8,F6F4,BF16,F16})
          + 64 * (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32
                  + 2*SQ_INSTS_VALU_FMA_F32 + SQ_INSTS_VALU_TRANS_F32)

B_read  = 32*TCC_EA0_RDREQ_32B_sum + 64*TCC_EA0_RDREQ_64B_sum
          + 128*TCC_EA0_RDREQ_128B_sum
B_write = 1024*WRITE_SIZE
B_external_proxy = B_read + B_write
```

Here SIZE values were interpreted as KiB. `1024*FETCH_SIZE` was an independent
read cross-check, not an additional read term to add again. Validate these
conventions against the installed definitions before reuse. The MFMA/VALU formula
covers selected instruction types, not all operations or only useful active lanes;
TRANS counts one operation/lane by convention, not equivalent throughput to FMA.
The TCC-derived total is an external-traffic proxy, not a count of all LDS/cache
loads and stores. Preserve discrepancies between alternative read estimates.

Normalize totals over the retained measured calls. Do not divide whole-process
counts, including warmups and setup, by a nominal repeat count. Missing/unsupported
counters are unavailable, not zero. Report collected categories and components.

Counter throughput uses the selected counter-run device time. If FLOP, read, and
write passes were separate, explain the normalization and non-simultaneous nature
of the aggregate. Do not silently borrow a faster TL timing. To isolate the work
model difference, an additional calculation with one common time can be useful,
provided it is explicitly labeled as recomputed.

Useful expansion diagnostics are `F_counter/F_semantic` and
`B_external_proxy/B_ideal`, subject to matching scope and coverage. Removing
redundancy can lower both counts and runtime; counter-based efficiency may then
fall while the implementation improves. Judge the improvement by correctness,
latency, and the fixed reference.

Use a compatible peak for each stage's precision/resource when interpreting
device-level counters. Do not inherit the parent's matrix peak for every helper.
If only an operator-level TL model exists, leave device-level TL FLOPs/bytes
unavailable instead of allocating them in proportion to timing.
