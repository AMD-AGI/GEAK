---
name: kernel-ut
description: Build and qualify standalone GPU kernel UTs from a kernel-profile handoff -- declare task boundaries, extract actual kernel source, author seeded-input UTs and qualify their correctness, then deliver them for optimization. Correctness only; timing belongs to optimization. Never profiles or captures numerical oracles.
---

# Kernel UT generation

The input is `profile_handoff.json` from [kernel-profile](../kernel-profile/SKILL.md) and
the artifacts it hashes. This skill declares task boundaries, extracts source and writes
and qualifies UTs. It does not launch serving, profile, rediscover shapes or select
kernels; missing evidence is a task blocker reported with what would unblock it (a
bounded profile capture, not a capture inside the UT).

Set `SKILL_ROOT` to this directory, `SHARED_TOOLS` to `e2e_workflow/scripts` and
`EVAL_DIR` to the profile's output directory. Read [setup](references/setup.md) first.
When invoked by GEAK, also read [workflow handoff](references/workflow-handoff.md).

```bash
python3 "$SKILL_ROOT/../kernel-profile/scripts/profile_handoff.py" verify \
  --handoff "$EVAL_DIR/profile_handoff.json"
```

A failing `verify` means the profile evidence changed after export: re-export from
profile, never edit the handoff.

## 1. Declare complete task boundaries

Read [delivery contract](references/delivery-contract.md) before authoring.
Save `task_plan.json` with one owner per complete math boundary. For each task,
save the original declaration as `run/requested_kernel.json` (called `KERNEL`
in the delivery contract), including exact selected identities, callable
candidates, phases and `source_delivery_required=true`. Every handoff kernel needs a
task or an explicit blocker.

Keep required companions even when below the selection threshold. Use
`joint_ut.phases` for mandatory chains, including split-K plus reduce; use
`dispatch_ut.phases` for mutually exclusive selected branches of one whole op.
Never demand mutually exclusive kernels in every invocation. Whole fused MoE
tasks preserve routing and the complete operation, not a synthetic dense GEMM.

For selected indexer score/top-k, deliver one unified task covering both served
prefill and decode branches. Read [joint indexer UT](references/joint-indexer-ut.md).
For collectives read [multi-rank UT](references/collective-ut.md). Consult the
[GLM example](references/glm-agentx-example.md) only for matching features;
it supplies no default shapes, layer multipliers or kernel identities.

Verify the live callable with `$SHARED_TOOLS/seam_trace.py` and
`python3 "$SHARED_TOOLS/kernel_selection.py" --strict-launch` (`--help` for its inputs)
on the profile's diagnostic traces; preserve complete raw verdicts and deeper-candidate
evidence. Names alone cannot prove a binding or that the requested whole chain executes.
If the declared boundary is wrong, repair the saved task plan with evidence before
extracting; do not edit a failed verdict into a pass or rerun an unchanged UT to repair
JSON fields.

## 2. Extract source and build the standalone UT

Read [source extraction](references/source-extraction.md) and
[task authoring](references/task-authoring.md). Generate inputs from fixed seeds at the
shapes/dtypes/layouts the handoff records. Run one such baseline case per phase, then
expand cases and editable leaves. The UT reads its frozen case manifest; it never
launches serving, profiles, or captures shapes. Each `tasks/<name>/` contains its own
`kernel_src/`, `cases.py`, `meta.json`, `unittest.py`, baseline manifest/overlay and
`run/` evidence. Copy the shared harness/leg/overlay helpers into the task; freeze them
once authored.

Deliver actual native/JIT/generated computation source for every claimed leaf,
with build resources and version/hash provenance. A copied dispatcher that
imports installed computation is not extraction of those leaves. Prove actual
leaf invocation through the candidate overlay, independently of correct output
or import location. Preserve deployed signatures, layouts, quantization,
routing semantics, cache/page state and complete outputs.

Run every GPU command from its task directory under the lease tools in
[setup](references/setup.md) — `gpu_lock.sh` for one device, `gpu_lock_set.py` for the
whole allocated set. Never run a GPU command outside a lease.

Correctness references come from the frozen baseline, run in a separate process on the
same seeds; never use the candidate to generate its own expected output, and never use a
serving-captured output. Nothing persists a golden file: the baseline's outputs are a
temporary per-run artifact.

Randomize ordinary numerical payloads, not the workload definition. Keep observed
shape/layout/quantization, phase and call weights. Routing, indices, page tables and
sequence metadata must retain valid semantics and the observed work/access pattern:
reuse or reconstruct them from handoff metadata, or report the gap. Synthetic
structural cases may extend correctness coverage, but do not claim they reproduce serving
performance without evidence.

## 3. Qualify correctness

`unittest.py` takes `--validation-mode correctness|performance`; the bare entry is
`performance`. UT qualification runs `--validation-mode correctness` and must cover
baseline/candidate binding, seeded random-value correctness at served shapes against the
frozen baseline, phase-specific graph/compile behavior and input refresh, each editable
leaf's numerical negative control and restored-source integrity. Compilation failure is
not a successful numerical negative control. It does not measure timing: record
`performance_measured: false`, keep every weighted signature and its `timing_specs`, and
report performance as unmeasured. Report missing cases and mass; do not renormalize
survivors into full coverage.

Before reporting a verdict, verify deterministic input construction, baseline
self-noise (repeated baseline outputs on one fixed set of inputs) and that numerical
bad-source controls actually fail by a wide margin. See
[task authoring](references/task-authoring.md).

Optimization later runs `unittest.py --validation-mode performance`, which adds
`h.measure_legs()` and `h.serving_weighted_speedup()` from the shared harness. Set
`workload.kind="agentx_trace_replay"`, `weighting="observed"`, and
`serving_weight_model.source="observed_workload"` with the handoff as evidence path.
Each case has a unique `sig`, observed `regime` and finite nonnegative
`effective_calls` on one consistent basis. Aggregate as
`sum(baseline_ms * calls) / sum(candidate_ms * calls)`; no fixed-load fallback
or weighted geometric mean.

## 4. Validate and deliver

Read [source records](references/source-delivery-records.md), freeze the task
snapshot before GPU validation and generate records from actual outcomes.
After real GPU tests, preflight the exact return with the shared delivery gate:

```bash
node "$SHARED_TOOLS/kernel_extraction_delivery.mjs" \
  --kernel "$TASK_DIR/run/requested_kernel.json" \
  --result "$TASK_DIR/run/draft_return.json" --phase extract \
  --out "$TASK_DIR/run/RETURN.json"
```

Use `extract_op` for the whole-op return form defined in the contract. Both
forms require actual source delivery in this skill. The gate reads workflow
schemas and hashed evidence; it does not execute or independently certify GPU
tests. Keep failures visible and continue independent tasks when possible.

Write the UT section of `REPORT.md` and `results.json` with all handoff identities,
task/source locations, actual commands and devices, per-gate status, declared and achieved
coverage, uncovered mass and limitations. Separate inventory completion, task completion,
GPU UT status and delivery consistency. A CPU-only check or a subset of passing tasks is
not full acceptance. Do not report E2E improvement without a separately authorized and
measured E2E result.
