# SKILL — Measurement-first distributed MoE bottleneck analysis

Framework status: **READY**. A workload analysis remains `awaiting_measurement` until its evidence
contract is complete.

## Doctrine

This Skill converts measured multi-rank evidence into:

1. explicit missing measurements;
2. bounded bottleneck hypotheses;
3. a measured path toward communication/computation overlap and fusion.

It never:

- overwrites the generic `bottleneck`, `top_kernels`, or `top_opportunities`;
- treats profiler category sum as wall-clock critical path;
- treats relative category growth as share of total delta;
- claims dispatch cost from a fused dispatch+GEMM category;
- infers route-to-rank causality without per-rank expert-load and wait evidence;
- hardcodes queue type, CU split, tile order, publication granularity, or fusion schedule;
- prunes another candidate;
- turns a public NVIDIA speedup into an AMD expected speedup.

Every analysis output contains:

- the observation and exact metric;
- constraints implied by that observation;
- bounded hypotheses and what they do not prove;
- unranked public reference patterns with applicability and caveats;
- the next collection experiment that distinguishes hypotheses.

The Skill never ranks or selects implementation candidates. Kernel Workflow's Step-3 TechLead uses
the Step-2 findings, hypotheses, constraints, bounds, unknowns, and references to generate and rank
`directions[]`.

## Required deterministic entrypoint

Run the checked-in analyzer instead of reproducing arithmetic manually:

```bash
python3 <ANALYSIS_SKILL_DIR>/analyze.py \
  --report <multi-rank-analysis.json> \
  --output <EVAL_DIR>/profile_moe_analysis.json \
  --markdown-output <EVAL_DIR>/profile_moe_analysis.md
python3 <ANALYSIS_SKILL_DIR>/validate_output.py \
  --analysis <EVAL_DIR>/profile_moe_analysis.json
```

The runner accepts both the frozen `geak-megamoe-analysis-v1` artifact shape and future generic
reports carrying equivalent `cases[]` / `route_comparisons[]` fields. A runner failure degrades to
the generic profile result; it must never fail Kernel Workflow.

## Inputs

Required:

- generic profile engineer output (`bottleneck`, `top_kernels`, `top_opportunities`);
- versioned multi-rank report with explicit case IDs and rank-max latency;
- route comparisons only when both cases use the same workload contract;
- validated `geak-hardware-context-v2` target context.

## Profile-to-analysis artifact contract

Profile packages raw inputs as `geak-analysis-bundle-v2`. Each case carries embedded rank records
plus optional route summary, Chrome traces, ATT CSVs, and software counters. Every non-UT artifact
must carry `geak-collection-provenance-v1`; a measurement track cannot be `complete` without
non-empty evidence.

```bash
python3 <WORKFLOW_DIR>/scripts/multi_rank_analysis/build_bundle.py \
  --rank-report <PROFILE_DIR>/rank_report.json \
  --metric timing_ms.e2e \
  --expected-world-size 8 \
  --case-artifacts <PROFILE_DIR>/case_artifacts.json \
  --route-comparisons <PROFILE_DIR>/route_comparisons.json \
  --hardware-context <PROFILE_DIR>/hardware_context.json \
  --measurement-tracks <PROFILE_DIR>/measurement_tracks.json \
  --experiment-manifest <PROFILE_DIR>/experiment_manifest.json \
  --output <EVAL_DIR>/analysis_bundle.json

python3 <WORKFLOW_DIR>/scripts/multi_rank_analysis/runner.py \
  --analysis-bundle <EVAL_DIR>/analysis_bundle.json \
  --primary-metric "candidate E2E rank-max latency" \
  --primary-metric-path timing_ms.e2e \
  --category-map <ANALYSIS_SKILL_DIR>/default_moe_category_map.json \
  --output <EVAL_DIR>/multi_rank_analysis.json
```

The generic runner also directly normalizes existing AITER `cases[].ranks[]` or legacy
`records[]` reports. That direct path is sufficient for rank metrics; comprehensive analysis uses
the bundle to join traces, ATT, counters, controlled experiments, and provenance.

## Hardware context is analysis input, not an implementation answer

Record before interpreting measurements:

```text
vendor / model / architecture
device count and topology
execution units per device
wave/warp width
local memory and device memory
runtime/driver/toolchain versions
available communication/synchronization/transfer capabilities
theoretical ceilings with sources
measured pairwise/all-to-all interconnect, HBM, and launch ceilings
```

Every required static field and every non-null measured field carries collector, ISO-8601
timestamp, confidence, and raw-artifact provenance.

Use it to:

- normalize XGMI/HBM utilization against measured target ceilings;
- determine whether low bandwidth indicates transfer inefficiency or unavoidable byte volume;
- evaluate whether candidate workgroup/LDS/VGPR unions can remain resident;
- identify which mechanisms are available for experiments (for example SDMA or device-side signals);
- reject hardware-incompatible public references.

Do not turn hardware capabilities into a prescribed tile, CU split, queue, or scheduler. They constrain
and interpret evidence; Step-3 planning later chooses among candidates. Missing target context or
measured ceilings keeps the analysis `awaiting_measurement`.

**Record the cache/coherence domain count, not just the cache sizes.** On a multi-die part the
last-level cache is per-die, which changes what an in-kernel readiness edge costs and whether it is
even correct. Step-3 feedback (MegaMoE EP8, gfx950 / 8 XCDs, 2026-08), three facts that any
one-kernel fusion analysis on such a part needs as input:

- a poll loop built on a relaxed load sees only its own die's last-level cache, so a counter it waits
  on must be published with system-scope atomics; an agent-scope increment lands in the adder's cache
  and the waiter can hang forever;
- a release fence lowers to a full last-level-cache writeback per issuing block, measured at +0.411 ms
  when every producer block issued one — so per-block release is not a rounding error at this scale;
- marking the payload stores write-through instead (system-scope, non-temporal) removes the fence
  entirely and leaves a store-completion wait as the whole release, improving all four route guards.

Report the die/cache-domain count and the measured cost of a per-block release alongside the
interconnect ceilings; without them a fusion ceiling for an in-kernel readiness edge is unbounded.

## AMD Thread Trace instruction evidence

When stage or fused-kernel time is high but the instruction-pipeline cause is unknown, use the
`rocprofv3_att` collector in `kernel_workflow/knowledge/collectors/`.

ATT can provide per-instruction:

```text
Hitcount
Latency cycles
Stall cycles
Idle cycles
Source/disassembly mapping
VMEM/LDS instruction-level perfcounters
wave states and occupancy context
```

Use it to distinguish MFMA starvation, barrier idle, waitcnt dependencies, LDS pressure, VMEM/global
latency, VALU promote/scale/convert overhead, and producer/consumer wave-role imbalance.

ATT samples selected GPU/CU/SIMD/waves. It does not measure whole-device E2E time, XGMI bytes,
cross-rank waiting, or critical-path contribution. Warm caches outside the trace, capture one
steady-state dispatch with a kernel regex, record target CU/SIMDs, and repeat representative roles.
ROCm defines ATT `Latency` as Stall + Issue/Execute; use `Latency + Idle` for accounted cycles and
never add Stall a second time.

For comprehensive/root-cause or fusion conclusions, the report must also contain:

```text
measurement_tracks.ep_baseline_decomposition
measurement_tracks.communication_bytes
measurement_tracks.wait_padding
measurement_tracks.publication_granularity
measurement_tracks.fusion_dag
measurement_tracks.resource_residency
```

Each track is `{"status":"complete", ...evidence...}` only after its experiment contract and
correctness gate pass. Missing tracks are not silently inferred.

## Metric semantics

For category `c` comparing a skew/candidate case against a uniform/baseline case:

```text
absolute_delta_ms(c) = rank_max_ms_new(c) - rank_max_ms_base(c)
relative_growth_pct(c) = absolute_delta_ms(c) / rank_max_ms_base(c) * 100
positive_absolute_delta_share_pct(c)
  = max(absolute_delta_ms(c), 0) / sum_j max(absolute_delta_ms(j), 0) * 100
```

These answer different questions:

- relative growth: how much that category changed relative to itself;
- positive absolute-delta share: how much of the summed category increase it represents;
- neither is automatically critical-path contribution when kernels overlap.

Worked frozen evidence:

```text
8192 uniform→rank-mixed-skew
combine: +0.3431 ms, +97.68% relative growth, 35.1% positive absolute-delta share
stage2:  +0.3558 ms, +17.13% relative growth, 36.4% positive absolute-delta share
stage1:  +0.2786 ms,  +9.65% relative growth, 28.5% positive absolute-delta share
```

Correct conclusion: combine grows most relative to its own baseline, while Stage2 has the largest
positive absolute category delta. This does not prove combine alone is the root cause.

**Small margins require interleaved measurement, not batch-to-batch comparison.** Step-3 feedback
(MegaMoE EP8, gfx950, 2026-08): the 512-token buckets drift by up to 4% between batches, so a
candidate measured in one batch against a baseline measured in another read as a 1% *regression* on
one route while alternating the two arms inside a single script showed the candidate winning 3 of 4
paired reps. Any margin under roughly 2% must be measured with the arms interleaved in one batch and
reported per-pair, and a frozen single-sample baseline is not precise enough to adjudicate it. State
the method and the rep count next to any delta smaller than the observed drift.

## Category naming

Until separately instrumented, use:

```text
stage1_dispatch_gemm1
stage2
combine
quantize
```

The existing Stage1 kernel fuses planning/counting, dispatch/P2P payload publication, synchronization,
and GEMM1. A Stage1 timing delta cannot be attributed to any one of these.

Category maps are caller-supplied regexes. Record:

- matched kernel names;
- event counts;
- per-call or per-replay normalization;
- unclassified time;
- malformed/missing rank traces;
- physical interval overlap, not only summed duration.

## Six mandatory measurement tracks

### A. Frozen scattered EP baseline decomposition

Use the existing AITER MegaMoEV2 EP8 UT and benchmark directly. Freeze identical inputs, routing,
quantization, graph mode, and iteration count.

Decompose:

- BF16 quantization;
- planning/counting and dispatch/P2P input publication;
- GEMM1 and activation/quantization;
- GEMM2, route weighting, and P2P output publication;
- combine;
- host launch gaps and intermediate materialization;
- rank wait and physical overlap.

The performance denominator is the frozen scattered AITER implementation. TPDP is optional
model-level context and is not a required kernel-analysis track.

### B. Communication bytes and causal controls

Collect software counters per source→destination:

- payload rows and bytes;
- scale/metadata bytes;
- local and remote token copies;
- signal/fence/atomic counts;
- send/receive completion and wait cycles.

Pair with repeated rocprof-sys XGMI sampling and TransferBench topology ceilings.

Run controlled variants:

```text
full
metadata+sync/no-payload
local-loopback payload
compute-only
communication-only (when semantically possible)
```

Variant differences are controlled bounds, not additive components when overlap exists.

### C. First-arrival compute, straggler wait, and padding

Require:

- per-rank/per-expert load matrix;
- per-expert/per-tile readiness;
- local wait cycles or durations;
- useful/padded tile counts;
- epoch-qualified readiness and stale-read/deadlock tests.

Only then may a Step-2 hypothesis be marked route-data-dependent or causally tied to a slow rank;
Step-3 TechLead consumes that evidence label.

**Skew is an intra-rank property; a per-rank load test can miss it entirely.** Step-3 feedback
(MegaMoE EP8, gfx950, 2026-08): on a `rank-mixed-skew` route every rank received approximately the
balanced row count, so a detector comparing this rank's received rows against `cur_tok * topk` never
fired even at threshold 1.0 — while the route was in fact the most skewed one measured. The imbalance
lived *inside* the rank: one local expert owned a disproportionate share of tiles and its tail was
what remote peers waited on. The discriminating statistic is per-expert tile count against the
per-expert mean (`max_expert_tiles` vs `num_valid / tile_m / experts_per_rank`), not any per-rank
aggregate. Report both; a per-rank load matrix alone does not satisfy this track.

### D. Publication granularity

Sweep:

```text
token
remote-owner-aligned row band
GEMM M tile
64/128/256/384-row chunk
```

Report downstream first-start time, atomics/fences, XGMI effective bandwidth, producer/consumer
occupancy, and pipeline bubbles. Never assume token granularity is best.

### E. One-kernel dependency DAG and ceiling

Build measured nodes/edges for:

```text
planning/dispatch
GEMM1
activation/quantization
GEMM2/publication
combine
shared expert
launch/materialization
```

Compute current critical path and Amdahl ceiling for:

1. Stage2 producer + combine consumer readiness;
2. rank-wide persistent expert GEMM scheduling;
3. dispatch receive + GEMM1 readiness;
4. quantization ingress;
5. one complete persistent kernel per EP rank.

The Step-2 record does not emit a fusion direction. It records the measured constraints that a later
Step-3 direction must account for: CU-role resources, buffer lifetime, synchronization requirements,
extra memory, unsupported shapes/routes, and correctness/deadlock gates.

**The ceiling must be charged for the kernel boundary the fusion removes.** An Amdahl bound computed
as "serialized stage time that fusion could hide" silently assumes the boundary itself costs nothing
to replace. It is not free: a kernel boundary ends in a hardware-managed, load-balanced join over
many small workgroups, whereas an in-kernel phase replacing it is typically a software static
partition whose consumers pay `max` over all participating blocks. Step-3 feedback (MegaMoE EP8,
gfx950, 2026-08): folding an ingress pass that costs ~1.4 us of bandwidth into the persistent kernel
made end-to-end *worse* by 0.15%–1.2% across four route guards, entirely from this effect, and a
work-stealing claim queue was worse still at ~89 ns of end-to-end cost per claimed chunk. Record the
fusion ceiling as a *range* whose pessimistic end includes a re-implemented join, and note that the
optimistic end is only reachable if consumers can wait on their own rows rather than on a whole-phase
barrier.

**Straggler-gating has a variance signature, and it is the cheap way to detect it.** The same
feedback: the separate-launch arm reproduced to a 1.7 us spread over three reps while the fused arm
spread 25.6 us at a similar mean. A fused/in-kernel arm that is both slightly slower and markedly
noisier is gated by its slowest participant, not throughput- or latency-bound. Report per-rep spread
alongside the median for every fusion candidate; a mean-only comparison cannot distinguish these.

### F. Resource residency and scheduler liveness

Measure and validate:

- one common workgroup shape for all roles;
- LDS/VGPR/instruction-footprint union;
- actual resident CTA capacity;
- resource bounds across caller-supplied static/dynamic role-allocation probes;
- producer and consumer progress guarantees;
- scheduler-state, stealing, or arithmetic-schedule pressure and termination;
- no oversubscribed spinning consumer can block a required producer.

The final target is one operator launch and one complete persistent kernel per rank. The Skill
reports measured resource constraints and unranked reference patterns; it does not prescribe queues,
role counts, granularity, scheduler, or an intermediate implementation.

**Report which resource binds, not just the LDS figure.** Residency is `min` over every per-CU
resource, and an analysis that reports LDS occupancy alone will propose an unreachable target.
Step-3 feedback (MegaMoE EP8, gfx950, 2026-08): a GEMM1 block at 97.5% LDS looked like an obvious
LDS-reduction candidate, but at 256 VGPR/thread a 512-thread block is pinned to 1 WG/CU by the
register file *regardless of LDS*, so 2 WG/CU requires a 256-thread block first. The one geometry
that did reach 2 WG/CU cost +2.33 ms end to end, because quartering the tile quartered the MACs while
only halving the bytes read. State the binding resource, the block shape it implies, and the
arithmetic-intensity cost of the shape change — an occupancy target without those three is not
actionable.

**A statically partitioned in-kernel phase has a residency precondition that is a deadlock, not a
slowdown, when violated.** If the kernel runs at N WG/CU and work is assigned by arrival ticket, only
the first `num_cu` tickets are resident; assigning a share to a higher ticket that resident blocks
then wait on hangs the grid. Any Step-3 direction proposing static in-kernel partitioning must carry
the measured resident-ticket window and show every participant index inside it.

## Optional model-level extension

Shared/routed expert overlap and DPEP/TPDP can be measured later for model E2E context. They do not
block the MegaMoEV2 fused-kernel work because they include operations outside this EP operator.

## Step-3 handoff acceptance constraints (reported, not decided here)

Correctness:

- pass the existing distributed PyTorch oracle;
- reject non-finite output and protocol errors;
- preserve deterministic top-k reduction order/tolerance;
- cover fixed/compact token buckets, uniform/skew/hot/empty experts, graph replay, and repeated epochs;
- pass deadlock, stale-read, invalid-route, overflow, and non-default-stream gates.

Functionality:

- one complete kernel per EP rank performs quantization, dispatch/planning, GEMM1, activation,
  GEMM2/P2P publication, and combine;
- no whole-stage CPU synchronization or intermediate kernel launch remains;
- the old scattered implementation stays available as fallback/reference.

Performance:

- compare against both the frozen public scattered baseline and the current best scattered incumbent;
- use E2E rank-max latency over at least three repetitions;
- exceed the configured minimum improvement after noise, with no required-case regression;
- report launch count, exposed communication, wait/padding, and overlap—not only total latency.

## Current-implementation safety gates

Before accepting a persistent/fused candidate, audit and test:

- completion/readiness arrays are sized for every index used by each execution mode;
- every rank uses one protocol hash covering world size, capacities, dimensions, transport dtype,
  publication width, and chunk rows;
- unequal rank token buckets cannot select incompatible exact-count protocols;
- invalid/dropped routes cannot leave stale direct-combine slots;
- data and scale stores both complete before system-visible publication;
- every relaxed remote wait is followed by an acquire fence;
- every exact-epoch wait has a bounded diagnostic/timeout path in debug validation;
- non-default stream ownership is explicit across quantize, dispatch, both GEMMs, and combine;
- class-global gates cannot change another instance's IPC contract;
- every waiting consumer has a guaranteed resident producer and vice versa;
- 32-bit source/slot encodings and buffer-resource offsets remain within bounds.

Do not infer liveness from a successful allocator-granularity accident or from one fixed-route UT.

## Public design priors

Use as design references:

- DeepGEMM Mega MoE: symmetric workspace, expert waves, full fusion;
- DeepEP SBO: `block_m` producer signals and Combine Send polling;
- Comet/Flux: shared-tensor decomposition and adaptive block specialization;
- tile-level producer/consumer work: rank-wide persistent GEMM plus a small communication partition;
- MORI/SGLang: dual HIP streams, async/SDMA paths, shared-expert overlap;
- Tutel/FasterMoE/Lina/AEP: adaptive pipelining, dynamic scheduling, and straggler models.

Hardware-specific primitives and performance numbers do not transfer without local reproduction.

## Output contract

`profile_moe_analysis.json` uses `geak-moe-bottleneck-analysis-v4` and contains:

```text
framework_status: ready
analysis_status: awaiting_measurement | evidence_complete
analysis_boundary
  workflow_step: 2
  decision_fields_emitted: false
  decision_owner: Step-3 TechLead
source_schema
source_report_status
source_provenance
analysis_inputs  # resolved path + SHA-256 + byte count
hardware_guidance
measurement_coverage
route_comparisons
findings[]
hypotheses[]
constraints[]
bounds[]
unknowns[]
reference_patterns[]  # unranked
evidence_catalog_summary
degraded[]
claims
  root_cause_proven
  dispatch_independently_measured
  dispatch_metric_refs
```

When any required measurement track is missing:

- analysis status is `awaiting_measurement`;
- root-cause and fusion verdicts are prohibited;
- missing-track collection questions are emitted under `unknowns[]`;
- no Step-3 implementation direction is generated.

A track labelled `complete` without resolved evidence is normalized to `invalid`; labels alone
cannot satisfy the contract. Complete evidence contains non-empty `artifact_refs`, `metrics`, and
`provenance_refs` that resolve to complete typed catalog entries. Track-specific evidence kinds are
required; `fusion_dag` additionally contains finite, internally consistent measured `bounds`.

## Degradation ladder

1. Missing optional route data: omit route causality and continue.
2. Missing/malformed rank trace: record rank/error and lower confidence.
3. Missing category map: use the shipped map, label it default, lower confidence.
4. Missing any comprehensive track: analysis remains `awaiting_measurement`; emit unknowns.
5. Dedicated analysis call failure: return `analysis_result.status=degraded` with an empty
   `analysis_json`; the already-complete generic Profile result is preserved.

## Research dossier

Read the checked-in source and architecture summary:

```text
<ANALYSIS_SKILL_DIR>/RESEARCH.md
```

The originating workspace may also contain a longer experiment dossier under
`artifacts/research/MEGAMOE_FUSION_RESEARCH.md`, but the Skill must not depend on that external path.
