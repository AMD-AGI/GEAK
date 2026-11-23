---
id: mono_kernel_authoring
title: "Write a new whole-decoder-layer persistent kernel (mono kernel) from scratch in FlyDSL: one launch per rank with in-kernel TP all-reduce and KV-cache write, MI300X"
kind: expert_skill
authors: [GEAK Team]
scope: kernel
match:
  operator: '*'
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode, graph]     # real task metadata often says regime: graph (CUDA-graph replay of a decode step)
  from_backend: ""
  to_backend: flydsl
  config: {kernel_class: decoder_layer_monokernel, ranks: multi, entry: author}
expects:
  isolated_speedup_min: 1.0    # nothing in this skill is validated in this repository; a correct seed below 1.0 is still valid
  parity: required
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

**Applicability test** (all three must be true):

1. You are **writing a new kernel**: the lane mode is `author`, and `kernel_src/` holds no device kernel for the layer yet (at most a stub adapter that returns the native `forward`).
2. The target is a **whole decoder layer** for one decode step and a **layer-level oracle** exists: the unit test starts one process per tensor-parallel rank and compares with the native layer on **every rank**.
3. A **synchronization runtime with bounded waits** is vendored in the task, or you will write and self-test one first.

If any line is false, this skill does not apply — stop reading. Do not decide from `op_kind` or `regime` in the task
metadata: they are free text (`decoder_layer`, `fused_layer`, missing; `graph`, `decode`, a dict). Look at the files
and at what the unit test starts.

- An existing whole-layer kernel to make faster (deployed HIP or FlyDSL, or a fresh fused seed): `mono_kernel_optimize`.
- One operator: `flydsl_decode_op_authoring`. Delivery into vLLM / SGLang: `apply_mono_kernel_to_framework`.
- Numerics facts of the native backends: `native_decode_numerics`. Read it with this skill; a layer contains those ops.
- An EP MoE layer (dispatch / GEMM / combine across ranks): the [`mega_fusion` expert workflow](../../expert_workflows/mega_fusion/README.md), not this skill.

**This is a knowledge-only playbook.** No workflow in the public GEAK repository drives the whole-decoder-layer (TP)
authoring flow of this skill. The `mega_fusion` expert workflow is the repository's existing route for multi-rank
fused persistent kernels and is validated for EP MoE layers only. The runtime library, the stage templates and the
gate scripts the original flow used are not in the repository. The skill carries the method, the rules and the review
criteria, so that whoever does this work (the kernel lane in `mode=author` on a decoder-layer task, or an external
workflow) does not have to rediscover them.

## Mechanism

- **What is built.** One persistent FlyDSL kernel per rank, one launch per decode step, at most one workgroup
  resident per CU (a "block" below; the layout names say CTA). Stages hand off inside it through tagged per-tile mailboxes; the TP all-reduce and the KV write
  happen in the kernel. An adapter subclasses the layer class and sends ineligible calls to the native path.
- **Where the gain is.** Decode already runs in a captured graph, so host dispatch is gone. The kernel wins on the
  device timeline: gaps between kernels (a native layer ran 18 to 91 kernels per step), synchronization, memory.
- **Why these rules.** A layer oracle reports one number while the bugs sit in the hand-offs between stages, so
  every stage gets a float64 reference and a derived bound. A stub that returns the native `forward` passes the
  oracle at about 1.0×, so acceptance empties the kernel and counts launches. A graph replay re-runs with frozen
  arguments, so the launch epoch comes from the device.

| part of the original flow (an internal mono-kernel generation workflow, not public) | status |
|---|---|
| layer-oracle freeze with independent verification and negative controls; contract; layout check with copy bodies; direct route with a probe; stages route and fusion; acceptance criteria | implemented and exercised (four ranks, decode width 1; the direct route on two models, the stages route and fusion on MiMo-V2.6-Flash only) |
| layout `wave_split`; partial-layer and dual-launch structures; structure revision during optimization | designed only |
| redesigned robustness suite; framework delivery checks | designed only (a first, self-reported version is implemented; no run on a real kernel is recorded) |

## Procedure

**Planner, first ten minutes** (an external workflow or a person; a lane's author mode has no planner before authoring).
1. Check the three lines above against the task directory. Read the unit test's docstring: ranks, widths, tolerance, which checks run, how long one oracle pass takes.
2. Pick the route (table below); inside a lane pick direct. Take structure `full_one_launch` and layout `staged_all_ctas` unless step 3 of the Steps (structure and layout) says otherwise.
3. Outside a lane, plan several rounds: contract → "correct" → "resources" plus review → hand over. Internal direct authoring took about 1.5 to 3.3 hours per layer kind *after* the contract existed. A task without a synchronization runtime spends its first round writing and self-testing one.
4. Inside a lane's author mode there is exactly one author call and no authoring rounds: contract, "correct" and "resources" must fit that call, and an incorrect seed aborts the lane. A task without a vendored synchronization runtime is out of scope for a lane. If the layer does not fit, return `authored: false` with the reason.

**Engineer, first ten minutes.**
1. Get a first oracle run, holding every GPU of the rank group in one lease for the whole command. Outside a lane: run it on the untouched stub, as the COMMANDMENT file and the unit test's docstring say; a stub that returns the native `forward` passes at about 1.0×. In a lane's author mode `kernel_src/` arrives empty and no COMMANDMENT file exists yet: copy the stub adapter and the vendored runtime, if the task has them, from `TASK_DIR/kernel_src/`, and run the oracle once a candidate exists, with the command form of the lane's author role file (read it for the exact form).
2. Read the native `forward`: list the op sequence, the collectives, and what the layer writes (KV cache, state).
3. Write the stage table with port names before any kernel code.
4. Write the adapter's eligibility test (shape, dtype, configuration only) and its native fallback.

| route | what you do | measured (not controlled comparisons; contract and runtime were already in place) | lean to it when (design memo, not validated) |
|---|---|---|---|
| **direct** | one author writes the whole layer from the per-stage designs, one device function per stage; a probe build localizes failures | layer kernels about 1.9× to 3.1× faster than the native layer on three layer kinds of one model, while a fourth kind failed (2 of 767 sequence checks one byte off in exact KV writes); on an MoE layer of MiMo-V2.6-Flash, with every stage started from tested op code, a seed 1.08× native in under two hours | most stages are standard ops; few stages; tight budget |
| **stages** | each stage is authored and tested alone, then fused (chained, then persistent). The original flow's default, not to be changed without a separate evaluation | on that MoE layer a seed about 80× slower than native in about six hours; an earlier attempt failed at fusion | a new op without precedent; state across steps without precedent; a direct attempt fell back for most stages |
| **direct, then stages** | direct first; only the stages the probe names are re-authored alone. The original flow's recommendation | exercised once: six stages reused, two re-authored, accepted in about three hours | direct fails on a few stages |

**Steps.** Every bold rule below decides a pass, a fail or a number.
1. **Confirm the oracle.** It must judge every rank, run multi-step sequences (also from an empty cache) and graph
   replay, and record the native noise and negative controls. Without that, stop: authoring cannot be judged.
2. **Write the contract** ([`references/layer_contract.md`](references/layer_contract.md)): stages, then a float64 reference and a derived bound per stage.
   - **Cut:** one GEMM-like operation, one reduction or one hand-off per stage. State a stage writes (the KV write) is an output port of exactly one stage.
   - **Ports:** an input reads the nearest earlier output of the same name. A second reader of a layer input that an earlier stage updated reads the nearest earlier output whose name contains it (`residual` reads `residual1`, not the layer input). An input that no output names is a layer input. Example: `input_norm` reads the layer inputs `hidden` and `residual` and outputs `normed` and `residual1`; a later `post_norm` that lists the input `residual` reads `input_norm`'s `residual1`, not the layer input.
   - **`tp`:** write 1 for every stage, also for one that runs on every rank on its own weight shard; write the TP size only for a stage that itself contains a collective (the all-reduce).
   - **Split:** when a stage cannot be authored, cut it at an internal boundary and mark the old stage as replaced by the new parts (they take its place in layer order and keep its external port names). Then build and test only the new parts; every other stage stays as it is. A port is either handed off or a layer output, never both, so never split a stage at a layer output that a later part also reads.
   - **Tolerance:** `layer_tol = min(derived, max(4 × measured, 4 × noise))`: take 4 × the larger of measured and noise, compare it with the derived bound, and the smaller of the two wins. Every negative control of the task must still fail at `3 × layer_tol`. KV writes are byte-exact. In a lane the oracle's tolerance and controls are frozen: compute `layer_tol` as a self-check only, and if it differs from the oracle's tolerance or a control does not fail at 3 ×, report it; do not re-run or strengthen controls.
     Example (invented numbers): derived 0.47, measured 0.016, noise 0.023: 4 × the larger = 4 × 0.023 = 0.092, below 0.47, so `layer_tol` = 0.092. Derived 0.23, measured 0.071, noise 0.009: 4 × 0.071 = 0.284, above 0.23, so the derived bound caps it and `layer_tol` = 0.23. The controls must then fail at 0.276 and at 0.69.
   - **First-draft CU plan:** one persistent workgroup per CU. For stages that run at the same time: CUs of a stage = total CUs × its bytes moved ÷ the sum of bytes moved by those stages (bytes: the weights it reads on this rank, plus KV pages for attention). Not proportional to its operation count or to its share of the native layer's time, and not an even split. Under the default layout stages run one after another and every stage uses every CU. Drop a width that does not fit LDS or registers.
     Example of the proportional split (invented numbers; three stages that run at the same time on 304 CUs): `qkv_proj` moves 175 MiB, `attn` 65 MiB, `experts_ar` 520 MiB, sum 760 MiB, so they get 304 × 175 ÷ 760 = 70, 304 × 65 ÷ 760 = 26 and 304 × 520 ÷ 760 = 208 CUs (70 + 26 + 208 = 304). Ignore every other column: by operation count (1, 3, 2) the split would be 51 / 152 / 101, by share of native time (20%, 45%, 35%) 61 / 137 / 106, evenly 101 / 101 / 102, all of them wrong for this split.
3. **Choose structure and layout** ([`references/structures_and_layouts.md`](references/structures_and_layouts.md)) from structural facts only, never from a model name.
   **Largest stage task count ≥ 8× the smallest: `task_tables`; otherwise `staged_all_ctas`** (a draft rule; its threshold was not validated by measurement). `wave_split` was never run.
4. **Check or write the synchronization runtime** ([`references/sync_patterns.md`](references/sync_patterns.md)): launch epoch read on the device, one deadline per launch (well above the slowest rank's layer time), a sticky error word written to every rank, per-tile mailboxes, one deadline site per hand-off port, all-reduce summed in fp32 in rank order.
   - **All-reduce partial buffer:** slot capacity = (largest decode width sharing the buffer) × hidden elements, and every width passes that same capacity; buffer bytes = 2 × capacity × bytes per element (two halves, chosen by the parity of the launch epoch). Example: widths 1 and 4, hidden 4096, bf16 (2 bytes): capacity 4 × 4096 = 16384 elements for both widths; buffer 2 × 16384 × 2 = 65536 bytes.
5. **Write the kernel** ([`references/fusion_playbook.md`](references/fusion_playbook.md)). Direct route, each milestone committed: **correct** (one kernel, any simple schedule; oracle and probe pass on every rank) → **resources** (ISA scratch 0, VGPR ≤ 512, LDS ≤ 64 KiB, every wait bounded) → **seed** (largest stage first, until the budget ends; inside a lane stop at resources: the seed milestone belongs to the optimize rounds, skill `mono_kernel_optimize`). Stages route: **chained** (one launch per stage) → **persistent**.
6. **Localize failures with a probe build** that exports every port of every stage. Judge each stage on the inputs the kernel actually gave it; run 1, 7 and all blocks, about 3 repeats each, and about 4 consecutive steps from an empty and from a captured cache. Then name the failure class (table under Knobs & pitfalls).
7. **Review** ([`references/review_criteria.md`](references/review_criteria.md)): the criteria below, then robustness: **at least 32 consecutive decode steps** with the KV cache byte-identical after each; kernel ↔ native switching; every width × build flag; fault injection; **10 repeated oracle runs**; memory; **1000 graph replays** of one captured step, the last compared with native.
8. **Hand over.** A correct seed, even slower than native, goes to `mono_kernel_optimize`.

Skip in a short round: separate per-stage lanes; the copy-body layout run (default layout, grid barrier between
stages); all speed work before "resources" passes. Never skip criteria 3 to 8 below. Stop when the budget is used
(return the last commit that passed "correct" and "resources") or the failure is error accumulation or a hollow pass: report these.

## Review criteria (check by hand)

| # (as in the reference) | must hold | how |
|---|---|---|
| 1 | nothing outside `kernel_src/` changed | compare with the task byte for byte |
| 3 | every wait is bounded; the launch epoch comes from the device | read the sources: no bare spin loop, no per-step value from the host |
| 4 | the full layer oracle passes on every rank | re-run in a fresh copy with a fresh compile cache |
| 5 | with **every kernel body emptied the oracle fails** | if it still passes, the outputs are not the kernel's: report, never edit a test |
| 6 | **exactly one launch per layer per rank** on the eligible path, none of the native kernels | count kernels per graph replay with a profiler; count totals, not correlation ids. This holds for a kernel authored to the one-launch structure; a deployed kernel may legitimately use two launches |
| 7 | per-stage evidence | direct route: the probe passes (step 6); stages route: every stage test passes on the fused sources |
| 8 | **resources are pass/fail on their own: ISA scratch (register spill) = 0, VGPR ≤ 512 (≤ 256 in a 512-thread layout), LDS ≤ 64 KiB** | read them from the ISA of the oracle run. Scratch above 0 fails the kernel even when the oracle passes and it is faster than native |
| 9 | the kernel-or-native decision is identical on every rank | it reads shape, dtype and configuration only |
| 10 | no per-call allocation except the outputs | weight repacks are one-time and their bytes reported |

## Knobs & pitfalls

Name the failure class first, then act on it. If several apply, report the first of: oracle or build, the probe's class, error accumulation or race (failing oracle only), resources, launch count, hollow.

| class or symptom | recognize it by | verdict | first thing to do | detail (status) |
|---|---|---|---|---|
| oracle / build | a pinned file changed; a new Python file outside `kernel_src/`; a build or import error | fail | fix this first; restore the pinned files | fusion_playbook §5 |
| resources | ISA scratch above 0, VGPR above 512 or LDS above 64 KiB | **fail on its own**, whatever the oracle and the speed say | the stage with the largest use: smaller tiles | review_criteria §2 |
| hang | a deadline expired or the outer wall clock ran out | fail | all waits share one deadline, so the error word may name a downstream waiter. **Look first at the producer stage of the most upstream port whose tiles do not carry this launch's tag**, not at the stage that reads it: does it publish under every block count and for every tile? | sync_patterns §6 (fault injected) |
| stage numerics | a stage is wrong on its own inputs at every block count | fail | only that stage: its numerics rules against the native source | fusion_playbook §5 |
| hand-off / race | right with 1 block and wrong with more; repeats differ and are wrong; the oracle fails while the probe passes | fail (repeats that differ inside the stage's bound: warning) | the stage's input waits and the upstream publish order | fusion_playbook §3.5 (fault injected) |
| state write | single steps pass, sequences fail; the KV write is missing, misplaced or wrong | fail | check slot, format and scale of the write, on every step | fusion_playbook §4.3 (observed) |
| error accumulation | every stage within its bound on its own inputs **and** the chained error grows over the steps | **a failure only when the layer oracle fails; while the oracle still passes it is a warning to report, not this class** | move the stage's numerics to the deployed path's rounding points; a fix round alone cannot close it: report it, loosen nothing | fusion_playbook §5 (observed, 1.2× the tolerance) |
| launch count | not exactly one kernel per replay, or a native kernel in the trace | fail | remove the extra launch or the native call | review_criteria §2 |
| hollow | the oracle still passes with every kernel body emptied | fail | report it; never change a test | review_criteria §2 |
| stale data after a wait | hand-off data read with plain loads | pitfall | read and write hand-off data with coherent accesses | sync_patterns §8 (measured) |
| an all-reduce wrong without any error | a peer address that is not opened base + offset, or a slot capacity that differs between widths | pitfall | fix the address; one capacity for all widths (step 4) | sync_patterns §10 (observed) |
| an edit to a helper has no effect | the compile cache ignores helper modules | pitfall | rename the launcher or disable the cache | sync_patterns §12 (observed) |

## Do-no-harm notes

- Never edit the oracle, loosen a tolerance, change the multipliers 4 and 3, or change a test to make a check pass.
- Prefill, mixed batches, widths outside the feasible set and refused configurations (expert parallelism, LoRA,
  speculative decoding, other quantization formats) always take the native path, decided identically on every rank.
- A failed wait stops every rank and returns no output. **A stalled rank must end in the deadline error, not a hang.**
- No host sync inside the step. Never read the error word per step: read it in tests, at teardown, or from a watchdog thread in relaxed capture mode.
- Do not quantize more coarsely than native. Report extra weight copies in bytes; create them on the first eligible call in the lane adapter. Delivery moves that build to load time, before KV-cache sizing (skill `apply_mono_kernel_to_framework`).
- A seed slower than native is a valid seed and not a delivery. Do not use this skill to optimize an existing kernel.

## Sources

- An internal mono-kernel generation workflow (not public): its role descriptions, stage design documents, layout
  templates and runtime notes. Measured figures are from its run reports on MiMo-V2.6-Flash and one further model.
- Public cards: [MI300X arch](../../../hardware/cdna3_mi300/arch.md), [occupancy](../../../hardware/cdna3_mi300/occupancy.md), FlyDSL [debugging](../../../languages/flydsl/debugging.md), [authoring optimization](../../../languages/flydsl/authoring_optimization.md).
- Kernels of this class are public in sgl-amd-mono and vLLM; their structures were read from an internal index only.
