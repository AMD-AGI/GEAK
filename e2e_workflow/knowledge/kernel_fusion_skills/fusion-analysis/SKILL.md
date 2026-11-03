---
name: fusion-analysis
description: Choose which kernel-fusion patterns to implement for one serving workload, from its execution traces, workload weights, model configuration, deployment and framework source. Produces a ranked, quantified candidate list (fusion levels L1-L4) with a non-conflicting recommended set and a seam for each candidate. Covers analysis only; writing the fused kernels and measuring them are separate steps.
---

# Kernel-fusion analysis

The task is to select the fusion patterns that best serve the given inference workload, not to list
every kernel pair that could be fused. Base each decision on:

1. workload characteristics;
2. the execution trace;
3. operator and tensor dependencies;
4. kernel durations and launch overhead;
5. memory traffic and intermediate tensors;
6. resource limits (VGPR, LDS, occupancy);
7. concurrency, batch size, token count and parallelism.

**Core principle: use the smallest fusion scope that removes the dominant bottleneck, and confirm the
choice with measured performance.** A larger fusion scope is not better by itself.

This skill is the analysis half of a search loop: trace → candidate generation → dependency validation
→ kernel implementation → microbenchmark → e2e benchmark → feedback → revised fusion pattern. Its
output feeds the implementation and measurement steps; measured results, not this analysis, decide
what is kept.

## Inputs

The caller supplies these; the file names and paths are given in its instructions.

| Input | Use |
|---|---|
| Trace digest (`kf_trace_digest.py` output) | Per step kind: steps, busy/span/idle, kernels per step, short-kernel share, top kernels, the repeating per-layer block, the full ordered kernel sequence of the median step with gaps and launching frames; the **Live model code** table |
| Raw traces | Any detail the digest does not carry |
| Workload summary | Per-point weights: the share of serving time of each phase and shape |
| Model configuration | Structural parameters |
| Server flags and environment | Parallelism, speculative decoding, graph mode, operator-library switches |
| Framework and operator-library source | Kernel ownership and seams |
| Target metric | The metric the ranking optimizes |
| Implementable levels, unsupported patterns | The fusion levels the implementation step supports, and the patterns inside them it does not support yet |

## Requirements

A candidate that does not meet all of the following must not be recommended.

1. **Evidence.** Every claim about kernel ownership, seam location, infeasibility or "already fused
   to the limit" cites checkable evidence: a trace call stack, a source location that was read, or a
   command run in this environment with its output. A claim without evidence is not a reason to
   recommend or to reject.
2. **The deployed configuration decides.** Weights, shapes (the distribution of M), platform branches
   and the kernel actually called are taken from what this deployment runs, never from defaults,
   another platform or the function definition alone. A model may ship several implementations of
   the same layer; only the implementation listed in the Live model code table is live.
3. **The target metric decides.** Rank by expected gain on the target metric. Candidates that do not
   affect it rank last.
4. **Implementation scope.** Rank every level as found. Only candidates at an implementable level and not
   matching an unsupported pattern may be recommended; mark the others "not implementable in this stage" and
   name the reason (level or pattern). An L3 stage-wise fusion that is not a MegaKernel (see
   [fusion levels](references/fusion-levels.md)) is in scope when L3 is implementable. The recommended set must be
   mutually compatible: its candidates take effect together.
5. **Complete coverage (boundary ledger).** For each phase that drives the target metric, use that
   phase's own trace sequence (never another phase's; list every path when layers differ). List the
   kernel sequence repeated per layer and the sequences run once at the head and tail of each step.
   For every adjacent kernel boundary in those sequences, record either the candidate that covers it
   or the reason it is not fused. A boundary absent from the ledger has not been analysed; complete
   the ledger before recommending.

## Procedure

### Step 0 — Model and deployment

Establish the model structure before judging fusibility; record it as model notes.

- From the model configuration, record: layer count, hidden size, attention type, MoE configuration
  (expert count, top-k, shared experts), quantization, draft or MTP layers, and any other special
  structure.
- In the framework source, locate the model implementation from `architectures` and the Live model
  code table. Trace one layer's forward path: which functions and operators implement attention,
  MLP/MoE, normalization, residuals and model-specific modules. Record any structural difference
  between layers.
- From the server flags and environment, determine:
  - parallelism and communication points (where all-reduce / all-gather run and which implementation);
  - speculative decoding: which step kinds it produces and how rows per step relate to batch size;
  - prefill chunking and whether decode runs under CUDA/HIP graphs;
  - which operators each operator-library switch replaces, and with what.
- Map model structure to the kernels in the trace: a table of module/function → kernels. Every later
  step uses this table.

### Step 1 — Workload characterization

Characterize the workload: concurrency, batch size, token count, prefill/decode split, TP/DP/EP/CP,
average kernel duration, kernel launches per layer, launch gaps, synchronization gaps, HBM traffic,
communication cost, compute utilization and occupancy.

Describe it with the distributions in the workload summary: batch-size and token-count distributions,
the device-time share of each phase, and per step kind the kernel count and duration. Separate the
part repeated per layer from the part executed once per step.

Classify the workload as low concurrency (latency-sensitive), medium concurrency, or high concurrency
(throughput-oriented). Concurrency alone does not determine the fusion level.

### Step 2 — Execution regions

Group kernels into logical regions using the trace and the source, for example: normalization,
attention, KV cache, router, dispatch, MoE, quantization, communication, residual update, speculative
decoding (draft/MTP).

Adjacency in the trace does not imply fusibility. For each region confirm:

- tensor dependencies, multiple consumers and side effects;
- KV-cache mutation and collective communication;
- shape compatibility and precision requirements;
- dispatch conditions: size guards, branches taken for different M, and whether the code runs inside
  graph capture.

Regions also exist outside a single layer; recover these as well:

- cross-layer boundaries (the tail of one layer and the head of the next);
- per-step code outside the layers (metadata, indexing and copy kernels);
- the speculative-decoding draft path.

### Step 3 — Candidate generation

Generate candidates at every applicable level for each hot region. For a layer A → B → C → D → E, for
example: L1 B+C; L2 A+B+C; L3 A+B+C+D; L4 the whole layer. Compare scopes; feasibility of a larger
scope is not a reason to choose it.

"Hot" is judged by weighted share, not by the longest kernels alone. Inspect each of the following
explicitly, since individually short but frequent kernels add up:

- runs of short kernels;
- elementwise, normalization, quantization or cast kernels adjacent to a GEMM or attention kernel;
- several GEMMs that read the same input;
- communication and the computation on either side of it;
- cross-layer boundaries;
- per-step glue;
- the draft path;
- redundant copy, fill and type-conversion kernels.

Enumerate before selecting: list both fusion directions of every adjacent boundary as candidates and
compare them. Give every recommended candidate a smaller-scope fallback.

### Step 4 — Candidate evaluation

Record for each candidate:

- level, the kernels and operators it covers, and why fusion is valid;
- the bottleneck it addresses, the expected gain and the main risks;
- VGPR, LDS and occupancy impact, and any tile or wave-layout compromise;
- producer-consumer and communication overlap opportunities;
- why this scope beats a smaller one, and why it should not be larger;
- occurrences (per layer, per step, which layers);
- applicable shapes and M range, and every phase in which the live server calls the seam (prefill / extend
  chunks, decode, verify, draft) -- not only the phase the region was found in, since the fused kernel either
  serves those calls too or must be guarded off them;
- launch count before and after;
- the source location as `module.function`, and the seam: the entry point where the original call is
  replaced by the fused kernel, specific enough to write a monkeypatch overlay against;
- quantified gain (see Estimation): µs saved per step and weighted e2e percentage, as an upper bound
  and an expected value;
- implementation difficulty (1–5);
- whether the result can be bit-identical to the original;
- relation to other candidates: exclusive (covers the same kernels) or stackable.

When an obstacle appears ("needs a global synchronization", "already a fused kernel", "spans several
modules"), look for a workaround first: redundant computation, a common enclosing seam, or absorption
into an existing fused kernel's prologue or epilogue. Declare a candidate infeasible only when none
applies, and state why.

### Step 5 — Ranking and recommendation

Rank candidates by expected gain on the target metric; list at least the top 20. Resolve exclusivity
and stacking, then state the total expected gain of the recommended set without double counting.

Deliver:

1. the recommended fusion patterns (mutually compatible, implementable scope only, each with a seam);
2. fallback patterns;
3. regions where fusion is not advised, with reasons;
4. information still missing for a final decision.

## Estimation

Gain and cost are estimated at the shapes actually served.

- **Gain** comes from fewer launches and fewer HBM round trips. Graph replay removes only the CPU-side
  launch cost; each kernel's fixed GPU-side cost and the memory round trip between kernels remain, so
  reducing the kernel count also pays inside graph-replayed regions.
- **Cost** comes from lost parallelism (for example split-K or split-KV), replacing heavy kernels
  (GEMMs, library operators) and cross-module integration.
- **Quantify from the trace:** measured duration of the kernels the fusion removes (including each
  kernel's fixed cost) × occurrences per step × the phase weight. Per-layer paths multiply by the layer
  count and usually carry the largest gains; find all of them and rank them first.
- Convert to a share of total device time using the workload-summary weights. Report an upper bound and
  an expected value, and state the reason for the discount between them.

As a summary of the trade-off:

```
net gain = launch saved + sync saved + HBM traffic saved + overlap gain + locality gain
         − resource cost − occupancy loss − tile compromise − load imbalance
```

Prefer the pattern with the largest expected net gain.

## References

- [Fusion levels](references/fusion-levels.md) — definitions of L1–L4 and when each applies.
- [Heuristics](references/heuristics.md) — priors by workload regime; priors only, never rules.
- [Report contents](references/report.md) — the sections the analysis report contains.

The output files, their locations and the machine-readable candidate format are defined by the caller,
not by this skill.
