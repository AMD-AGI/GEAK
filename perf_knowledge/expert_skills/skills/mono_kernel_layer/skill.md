---
id: mono_kernel_layer
title: "Mono kernel: author, fuse and optimize one persistent FlyDSL kernel per decoder layer (decode, TP, in-kernel all-reduce) on gfx942"
kind: expert_skill
authors: [GEAK Team]
scope: kernel            # kernel: validated by kernel_workflow (isolated A/B vs the immutable oracle)
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  # Not operator-gated. What decides whether this skill applies is the SHAPE OF THE TASK -- a whole
  # decoder layer fused into one persistent kernel whose ranks exchange data inside the kernel -- not
  # what a single operator computes, and the capability index has no operator for that. Same reasoning
  # as gluon_authoring: an operator filter would select the wrong thing.
  operator: '*'
  arch_class: ['*']
  gens: [gfx942]                      # every hardware figure in the references is MI300X (304 CUs, 8 XCDs, 64 KiB LDS)
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode]
  from_backend: ""                    # authored from the framework's native layer, or optimized in place
  to_backend: flydsl
# ---- expected effect: the validation gate's pass criteria ----
# isolated_speedup_min is the repo-conventional sanity bar for a kernel-scope skill, NOT a measured
# claim: nothing in this skill has been validated on a box yet (status: draft).
expects:
  isolated_speedup_min: 1.05
  parity: required
# ---- validation: AUTO-FILLED by validate_skill.py — do NOT hand-edit ----
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use

The kernel under work is a **mono kernel**: one persistent FlyDSL kernel per rank that computes a whole
decoder layer for one decode step, including the TP all-reduce, so that one layer is one launch. It is
built from stages that are authored one by one and then inlined. It applies on gfx942 (MI300X), to
tensor-parallel decode at the kernel's widths; the levers below cover block-scaled fp8 and MXFP4
weights.

Two entries:

- **Author** (no seed yet). Write each stage so that it can be inlined into the layer kernel
  ([stage_authoring.md](references/stage_authoring.md)), choose a layout
  ([layer_structure.md](references/layer_structure.md#layouts), `staged_all_ctas` by default), and
  synchronize through the runtime primitives ([runtime_primitives.md](references/runtime_primitives.md)).
- **Optimize** (a seed passes the oracle and launches once per layer, but may be slower than the native
  layer). Measure, pick a lever by symptom, gate every change.

Not for single-operator kernels, for prefill or mixed batches, or for a generation other than gfx942
without re-measuring: the byte floors, CU counts, cache sizes and register budgets in the references are
MI300X figures.

**Terms.**
- *Stage*: a slice of one decoder layer (e.g. QKV projection + RoPE, o_proj + TP all-reduce, router top-k).
- *Native layer*: the serving framework's own implementation of the layer; it is the baseline.
- *Oracle*: the task's pinned `unittest.py` with its cases and `meta.json`; it checks the candidate against
  the native layer (a stage against a float64 reference) and times baseline and candidate.
- *Seed*: the first fused kernel that already passes the oracle's correctness.
- `WIDTHS`: the decode row counts the kernel is built for.
- `PHYSICAL_CU_COUNT`: the number of resident workgroups, at most one persistent workgroup per CU.

## Mechanism

- At decode width 1–8 every weight byte is read once per step and used for a handful of FLOPs, so a
  layer's GEMVs are HBM-bound; attention is latency-bound, and every stage depends on the previous
  stage's output.
- A rocprofv3 trace shows one kernel per layer, so the structure inside it has to be measured with
  in-kernel timestamps against per-stage byte floors ([measuring.md](references/measuring.md)).
- A grid barrier costs several µs of pure latency plus the skew of the slowest workgroup (~300 workgroups,
  global atomics at ~116–202 ns), and a layer has many. The persistent grid (one workgroup per CU, all
  resident) therefore replaces stage-wide barriers with per-tile tagged hand-offs, and overlaps weight
  streaming with the dependency chain
  ([handoffs_and_sync.md](references/handoffs_and_sync.md), [layer_structure.md](references/layer_structure.md)).
- The TP all-reduce is part of the kernel: ranks exchange partials through IPC-shared, uncached peer
  buffers and sum them in fp32 in rank order, so every rank gets bit-identical output.
- Every wait is bounded by a deadline and reports into one sticky error word shared by all ranks, and
  every tag derives from a device-side launch epoch, which keeps a failure a clean stop and CUDA-graph
  replay correct ([runtime_primitives.md](references/runtime_primitives.md)).
- The speedup is realized inside the captured decode graphs; there is no separate eager number to
  optimize ([acceptance_gates.md](references/acceptance_gates.md#what-the-numbers-mean)).

## Procedure

0. **Author mode only.** Write the stages ([stage_authoring.md](references/stage_authoring.md)), inline
   them into one persistent kernel under a layout, with hand-offs and the all-reduce built on the runtime
   primitives. Then continue at step 1 as a seed.
1. **Run the oracle** from the task workspace under a whole-group lease of the TP group, and record the
   weighted speedup (the primary target when a workload is set), the per-width times and the per-case times
   ([acceptance_gates.md](references/acceptance_gates.md#running-the-oracle)).
2. **Measure where the time goes**: build the per-stage timeline and write each stage's byte floor next to
   its span ([measuring.md](references/measuring.md)). Without it the other levers are guesses.
3. **Pick the lever by symptom** (next section). A stage far above its floor points at that stage's
   kernel; stages near their floor with a slow layer point at structure: imbalance, waits and barriers,
   the all-reduce. Order of attack that usually pays (measure to confirm): rebalance → hand-offs →
   all-reduce → per-stage kernels (weights, attention) → placement. Structural changes (which steps the
   kernel covers, how stages overlap) tend to pay more than per-stage polish. When the workload has
   concurrent decode steps, widths > 1 can set the tail latency: cover every width in `WIDTHS` before
   polishing width 1.
4. **Change one thing at a time** and re-measure the timeline; keep the stage → block plan in one table.
5. **Gate every change.** Correctness on every rank and the per-width speedup from the oracle, then the
   launch census, the hollowed-kernel check and the unmodified-runtime check (the CPU-only static pass is
   the cheap pre-check; [acceptance_gates.md](references/acceptance_gates.md#hard-gates)). A candidate
   that violates a gate is failed even when the speedup improved. The levers are directions; the oracle
   and the gates decide.
6. **Record what you learned** (below), only for a layer that was delivered and beat native.

## Choosing a lever by symptom

Start with the stage timeline: without per-stage times, the other levers are guesses.

**Structure of the layer**

| lever | symptom that points to it | typical lever |
|---|---|---|
| [Stage timeline](references/measuring.md#per-stage-timeline) | you do not know where the layer's time goes | in-kernel timestamps per stage, compared with each stage's byte floor |
| [CU rebalance](references/layer_structure.md#rebalance-the-cu-plan) | one stage's span sets the layer time while other partitions idle | CU shares proportional to bytes moved; small stages share CUs |
| [Task tables](references/layer_structure.md#task_tables) | stages that could overlap run on the same CUs; a prefetch starts late | per-stage task tables with staggered base blocks |
| [Wave specialization](references/layer_structure.md#wave_split) | stages never overlap inside a CU | two wave groups per CU, each owning a chain of stages |
| [Multi-row decode](references/layer_structure.md#multi-row-decode) | widths > 1 slower than native | rows as MFMA columns, or block groups per row; attention blocks by context length |
| [XCD placement](references/layer_structure.md#xcd-placement) | hand-off latency high even with balanced stages | producer and consumer workgroups on the same XCD (shared L2) |

**Hand-offs and synchronization**

| lever | symptom that points to it | typical lever |
|---|---|---|
| [Per-tile hand-offs](references/handoffs_and_sync.md#per-tile-hand-offs-instead-of-stage-wide-barriers) | gaps between stages; grid barriers dominate the timeline | per-tile mailboxes instead of stage-wide barriers; fewer barriers per launch |
| [Tagged pairs](references/handoffs_and_sync.md#tagged-value-tag-hand-offs) | layer-to-layer gaps; flags cleared every step | 64-bit (value, tag) stores, one epoch per step, batched polling, bounded waits |
| [Epoch flags](references/handoffs_and_sync.md#epoch-flag-hand-offs-in-flydsl-inside-a-gpu-and-across-gpus) | hand-off cost; peer accesses slow | cache-policy stores + drain + flag = epoch, (value, epoch) pairs, batched polls, SGPR peer bases |
| [All-reduce fusion](references/handoffs_and_sync.md#fuse-the-tp-all-reduce-into-its-neighbours) | the all-reduce span is large or serial after its GEMM | partials written from the GEMM epilogue; residual + next RMSNorm in the reduce epilogue |
| [Early-issue prefetch](references/handoffs_and_sync.md#issue-the-next-stages-loads-before-its-inputs-arrive) | a stage starts with a load ramp | next GEMV's weights and the first K/V tiles issued before their inputs arrive |

**Per-stage kernels**

| lever | symptom that points to it | typical lever |
|---|---|---|
| [Weight streaming](references/stage_kernels.md#stream-the-weights-at-hbm-speed) | GEMV stages far above their HBM floor | wide loads, fp8 dequant in registers, split-K, prefetch by idle CUs |
| [Split-KV attention](references/stage_kernels.md#split-kv-decode-attention) | attention dominates at long context | split-KV (flash-decoding) across workgroups with an in-kernel combine |
| [Wave-local attention](references/stage_kernels.md#wave-local-decode-attention) | attention span far above its KV bytes | per-wave online softmax on MFMA, K from the cache into MFMA, swizzled V^T |
| [Decode MoE](references/stage_kernels.md#decode-moe-router-top-k-fp4-experts) | router / top-k / experts several µs each | wave-level top-k, FP4 decode in registers, scales in read order |
| [Numerics](references/numerics.md) | small per-layer error but FP8 code flips / early greedy forks | native reduction order and rounding points |

**Writing it in FlyDSL and shipping it**

| lever | symptom that points to it | typical lever |
|---|---|---|
| [FlyDSL idioms](references/flydsl_idioms_and_pitfalls.md#flydsl-idioms) | how to express a lever in FlyDSL | traced helpers, cache-policy buffer ops, sched_barrier, DPP, timeline build |
| [Kernel pitfalls](references/flydsl_idioms_and_pitfalls.md#kernel-pitfalls), [serving pitfalls](../apply_mono_kernel_to_framework/skill.md#knobs--pitfalls) | before integrating or measuring | LDS races, spills, KV re-allocation, graph-capture modes, measurement hygiene |

## Reference map

| file | what is in it |
|---|---|
| [references/runtime_primitives.md](references/runtime_primitives.md) | the synchronization / memory primitives: launch epoch, bounded waits, error word, mailboxes, barriers, in-kernel all-reduce, `PeerGroup`, launcher pattern, runtime pitfalls |
| [references/stage_authoring.md](references/stage_authoring.md) | how to write one stage so it can be inlined: oracle, the seven shape rules, read the native kernels first |
| [references/acceptance_gates.md](references/acceptance_gates.md) | running the oracle, what the numbers mean, the seven hard gates, checking them outside the unittest |
| [references/measuring.md](references/measuring.md) | the per-stage timeline and the byte floors |
| [references/layer_structure.md](references/layer_structure.md) | the three layouts and the structure levers (rebalance, task tables, wave split, multi-row, XCD) |
| [references/handoffs_and_sync.md](references/handoffs_and_sync.md) | the three hand-off variants, all-reduce fusion, early-issue prefetch |
| [references/stage_kernels.md](references/stage_kernels.md) | weight streaming, split-KV and wave-local attention, decode MoE |
| [references/numerics.md](references/numerics.md) | matching native reduction order and rounding points |
| [references/native_reference_notes.md](references/native_reference_notes.md) | how vLLM on ROCm computes dense fp8 linears, paged KV, sliding window and sinks, MXFP4 MoE, DeepSeek-V4.1 router / mHC / compressor, all-reduce: facts a float64 reference needs |
| [references/flydsl_idioms_and_pitfalls.md](references/flydsl_idioms_and_pitfalls.md) | FlyDSL idioms, gfx942 resource facts, kernel pitfalls |

Complements: the generic FlyDSL cards in [`languages/flydsl/`](../../../languages/flydsl/) (GEMM and
attention levers, patterns) and the hardware cards in [`hardware/`](../../../hardware/). The serving-engine
side of a mono kernel (vLLM `specialized_decode`, SGLang plugin, smoke tests) is the e2e skill
[`apply_mono_kernel_to_framework`](../apply_mono_kernel_to_framework/skill.md).

## Knobs & pitfalls

- The knobs are per layout (thread count, `n_blocks`, hand-off unit, VGPR / LDS budget, prefetch, XCD
  order, stage → group, task counts and bases): the tables in
  [layer_structure.md](references/layer_structure.md#layouts). Change one at a time and re-measure.
- Kernel and runtime pitfalls: [flydsl_idioms_and_pitfalls.md](references/flydsl_idioms_and_pitfalls.md#kernel-pitfalls)
  (LDS races, spills, alignment, uncached peer accesses, tag wrap) and
  [runtime_primitives.md](references/runtime_primitives.md#pitfalls-seen-so-far) (RCCL under graph capture,
  divergent fallbacks, the FlyDSL compile cache, loop-carried values).
- Serving-engine pitfalls (KV re-allocation, graph-capture modes, watchdog thread, leaked runners) and
  serving-benchmark hygiene: [apply_mono_kernel_to_framework](../apply_mono_kernel_to_framework/skill.md#knobs--pitfalls).

## Recording what you learn

This repository has no separate store for mono-kernel cards: a lesson about this kernel class is recorded
as an experience card in the kernel workflow's store (the `LEARNED_DIR` your orchestrator hands you). The
card format, the discovery header, the content rules (relative figures only: speedup ratios, percent
deltas, fractions of achievable peak, occupancy and hit percentages; never absolute times or rates) and
the ADD-only rule are the kernel workflow's
([`kernel_workflow/knowledge/learned/README.md`](../../../../kernel_workflow/knowledge/learned/README.md),
with `kernel_workflow/scripts/kb.py` as the single writer). What differs for mono kernels:

- **Evidence.** The frozen native layer measured against the mono kernel in the layer task's two-leg A/B:
  device time under graph replay, slowest rank, verified by the Director, plus layer oracle parity and a
  passing hardening step (fault injection: a stalled rank must end in the deadline error). A mono kernel
  that was not delivered earns no card.
- **A card says** that a structure or lever makes a persistent layer kernel faster on this (layer kind,
  gfx, TP, width): a stage CU split, the hand-off granularity, the all-reduce placement. Or it records a
  pitfall seen while writing or fusing stages.
- **Written** by `update_experience` (`SCOPE=lane`) at the end of a mono-kernel run, and for a mono kernel
  only when the best delivered layer beat the native layer.
- **Never claims e2e.** Serving-smoke and GSM8K results are correctness gates here, not speed evidence.

## Do-no-harm notes

- **Advisory only.** This skill seeds the direction; the oracle and the gates decide. When it conflicts
  with the live measurement, the measurement wins.
- **Status: draft.** The levers are model-agnostic: they are derived from the runtime's design, the
  hardware facts cited in each file and the general decode-kernel literature (flash-decoding, the serving
  frameworks' own decode kernels). None of them is a measured result on your model. Treat each as a
  direction to measure, never as a reason to skip the oracle or the gates; every accepted change still
  passes the hard gates.
- **Read the framework sources, not another mono kernel.** For how the native path computes a stage, read
  the framework sources in the stage's native-source list.
- Keep it OFF for: single-operator kernels; prefill, mixed batches and speculative verify (they stay on
  the native path); TP sizes the kernel was not built for; other generations without re-measuring; any
  width not measured against native at short and long context (measure every width before routing it to
  the kernel).
- Never loosen the oracle's tolerance; never patch the runtime inside the candidate (a runtime bug is
  reported); never continue with stale data after a deadline expires; never emit the token of a failed
  step; claim bitwise equality only if the oracle shows max_abs = 0 for every output.

## Sources

In-repository sources: [`kernel_workflow/knowledge/learned/README.md`](../../../../kernel_workflow/knowledge/learned/README.md),
[`languages/flydsl/`](../../../languages/flydsl/), [`hardware/cdna3_mi300/`](../../../hardware/cdna3_mi300/).
