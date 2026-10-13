# Semantic references and model corrections

**Orthogonal to the engine.** This is not a mode. It is the set of practices for
building an explicit, implementation-independent FLOP/byte reference and for
interpreting the scorecard you compute from it. All three modes may use it; it was
previously bolted into the predecessor skill's `extended` mode, which also bundled
unrelated counter work.

Use it when the question is "did this optimization actually remove work?" rather
than "how fast is this binary?". For such a comparison, **freeze** the workload's
semantic FLOPs, ideal external bytes, and hardware references; measure latency per
implementation; and use counters only to *explain* padding, rereads, and
intermediate traffic, keeping those implementation-dependent counts separate from
the frozen score.

## 1. Define the modeling and measurement contract

Inspect the source, unit test, shape records, and available traces. Resolve missing
information from them before asking the user. Record:

- **Operator boundary:** one device kernel, a named kernel set, or the full operator.
- **Exact case:** dimensions, strides/layout, dtypes, quantization scales,
  batch/context, sparse selection, routing, and prefill/decode phase as applicable.
- **Numerical contract:** accepted precision, approximation, correctness tolerance.
- **Memory level and cache policy:** cold external memory, warm cache, or a
  documented deployment reuse pattern. State which operands may already be resident.
- **Model version**, included/omitted work, and traffic reuse assumptions.
- **Provenance:** TraceLens commit and local changes; GPU identity/partition;
  runtime, compiler, framework, container digest; implementation source and build.

An implementation-independent reference normally belongs to a **logical operator**.
If an optimization changes split-K, fusion, or kernel count, compare the *complete*
implementations of that same operator. A single split stage needs its own stage
contract. Never divide a whole operator's FLOPs/bytes by only its fastest or largest
kernel's time.

## 2. Freeze the hardware reference

Use an explicitly selected architecture JSON with `mem_bw_gbps` and
`max_achievable_tflops`. Record its contents, hash, `source`, and the compute key
actually resolved. Classify the values as **specification peaks**, **calibrated
empirical references**, or **unverified**. Do not infer provenance from the
filename — see the four MI355X variants in
[combination.md](combination.md#7-the-arch-json-is-not-a-constant).

If calibration is needed, run the microbenchmark in an isolated environment,
validate its compute path and bandwidth regime, then **freeze the result across both
implementations and every comparison run**. Current-run health probes detect
contention; they must never silently replace the denominator. (The `rocprof` mode's
`ROOFLINE_MIBENCH_CACHE` is the mechanical form of this rule.)

Choose the compute reference by **execution precision and instruction family**, not
by storage dtype. A BF16 output does not imply BF16 matrix computation. Mixed
matrix/vector work may need separate resource terms; do not assign a matrix peak to
a reduction or scalar helper. If no compatible reference exists, report the missing
reference rather than inventing a precise efficiency.

Do not apply another unexplained efficiency discount to an already-empirical peak —
that is the double-derate failure. Historical MI355X numbers in old reports are
examples, not portable defaults.

## 3. Build an explicit FLOP/byte model

Read [modeling.md](modeling.md) for TraceLens API usage and the GEMM,
attention/MLA, indexer, absorb-BMM, and MoE decisions that matter here.

Start from TL's model, inspect the implementation, and export **named terms and
assumptions**. Distinguish four bases and never let them merge:

| Basis | Purpose | Changes with implementation? |
| --- | --- | --- |
| `semantic_reference` | Agreed useful FLOPs and ideal bytes | No, while the contract is unchanged |
| `legacy_tl` | Reproduce an existing TL/wrapper result without silent corrections | Possibly; inspect the wrapper |
| `implementation_model` | Explain today's padding, split stages, materialization | Yes |
| Hardware counters | Measured execution under stated counter coverage | Yes |

For the frozen reference, **exclude** removable padding, rereads, split-K partial
materialization, and anything else introduced only by today's decomposition;
**include** required inputs/outputs and semantic work inside the agreed boundary.
Count shared storage once under the stated reuse assumption. A corrected model gets
a **new version** and is applied to baseline *and* candidate.

Call the result an **ideal reference under stated assumptions**. Independent FLOP
and byte minima need not be jointly attainable: standard GEMM `2MNK` assumes
conventional arithmetic, it does not prove a minimum over all algorithms.

## 4. Compute and interpret the scorecard

For one matched case, with `F` FLOPs/call, `B` bytes/call, `t` measured s/call,
`P` FLOP/s, `BW` byte/s:

```text
AI                      = F / B
ridge                   = P / BW
t_compute               = F / P
t_memory                = B / BW
t_reference             = max(t_compute, t_memory)
model_roof_axis         = resource with the larger time term
effective_TFLOPs        = F / t / 1e12
effective_GBs           = B / t / 1e9
reference_efficiency    = t_reference / t
distance_to_reference   = t / t_reference
```

Use decimal GB/MB and explicit time units. Store efficiency as a fraction; multiply
by 100 only for display. With `F`, `B`, `P`, `BW` fixed, lower latency always means
higher reference efficiency — which is why the fixed-reference column is a *test*
and a per-leg-recomputed column is not.

`model_roof_axis` classifies the model's resource terms; it is **not proof of the
actual bottleneck**. Low efficiency alone does not establish bandwidth, occupancy, or
launch limitation. Use traffic, launch geometry, stalls, dependencies, and timing
evidence to reach a separate diagnostic conclusion. Very short device kernels are
not automatically host-launch-bound — compare device and host timelines first.

Efficiency above 1 is a reference/model/timing inconsistency to investigate, **not a
value to clamp**. An empirical reference can genuinely be exceeded by a better
execution path; that still means the denominator needs re-deriving before the row is
quotable.

`distance_to_reference` is an ideal ratio, not a promised attainable speedup. High
counter-based efficiency does not rule out *removing* redundant computation or
traffic — a kernel can be at 95% of roof while doing twice the necessary work.

For sequential stages with independently justified stage models, also examine
`Σ max(F_i/P_i, B_i/BW_i)`. It generally differs from an aggregate max of totals.
Do not put today's removable stage boundaries into the frozen operator reference.

## 5. Export reviewable results

```bash
python3 scripts/roofline_report.py \
  --mode extended --input cases.json --arch frozen_arch.json \
  --output results/<mode>/
```

`--mode extended` here is an **input-format label consumed by the calculator**, not a
mode of this skill. Read [report-input.md](report-input.md) for the input contract
and a CPU-only example. The helper computes a single compute-resource /
memory-resource roof; it does not import TraceLens, measure kernels, verify the
physical model, or handle mixed-resource stage schedules.

Each report carries per-case work/traffic, latency and its timing basis, achieved
rates, both roof terms, AI, model axis, efficiency, correctness, provenance, and
unresolved limitations. Label counter-derived rates separately with their own
timing/pass provenance. Keep raw TL values alongside corrected models.

Compare actual baseline/candidate latency **first**. Aggregate only with justified
deployment call counts, keeping phase and coverage explicit:

```text
aggregate_reference_efficiency = Σ calls_i·t_reference_i / Σ calls_i·t_i
workload_speedup               = Σ calls_i·t_base_i / Σ calls_i·t_candidate_i
```

Benchmark repetitions are not deployment weights, and per-case speedups are not
averaged. Do not collapse differing roof axes into one physical utilization figure.
Any end-to-end estimate needs a justified application-time fraction and an overlap
model.

Finish with the observed improvement, model coverage, the evidence pointing at the
next optimization, and what remains unmeasured. A missing counter pass must not
become a measured zero, and a shape-only run must not become a measured performance
claim.
