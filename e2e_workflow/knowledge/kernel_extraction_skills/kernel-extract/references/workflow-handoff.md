# GEAK workflow handoff

This reference adapts the standalone extraction procedure to GEAK. The skill's
SKILL.md, references and shared and skill-specific tools own discovery, sampling, source extraction,
UT authoring and validation. The workflow supplies inputs and consumes finished
UTs. Do not also execute the legacy `roles/kernel_extractor.md` procedure.

## Inputs and invocation

| Workflow input | Use in the skill |
|---|---|
| `SKILL_ROOT` | Skill instructions and the additional extraction tools |
| `SHARED_TOOLS` | Shared capture, trace, harness and delivery helpers; vendor task helpers from here |
| `WORKFLOW_DIR` | GEAK e2e directory; used only for the final handoff adapter |
| `EVAL_DIR`, `EVIDENCE_DIR` | Allowed output root and this round's evidence directory |
| `BASELINE_LOG`, `BASELINE_BENCH_DIR`, `BASELINE_SUMMARY` | Completed measured baseline evidence; exclude warmup/discarded intervals |
| `MODEL_PATH`, `CURRENT_FLAGS`, `CURRENT_ENV`, `CURRENT_OVERLAY` | Effective installed stack and accepted baseline |
| `SERVING_TP`, `SERVING_GPU`, `GPU_ID` | Serving layout and allocated test devices; reconcile physical ownership before leasing |
| `TASK`, `WORKLOAD` | User scope and replay specification |
| `PREVIOUS_WORKLOAD_JSON`, `PREVIOUS_RANKING_JSON`, `PREVIOUS_UT_DELIVERY` | Reuse compatible evidence and passing tasks on resume/refresh |
| `BUDGET`, `HEAD_BUDGET`, `REMAINING_MIN` | Scheduling limits; retain the complete selected inventory and explain deferred tasks |
| `CAPTURE_PERSIST_POLICY` | Export into capture processes; defaults to `structure_only` for AgentX |

`PHASE=prepare` executes all five skill stages in one invocation. Prefer seeded
UT input factories from fixed workload metadata; do not collect numerical inputs
or rerun shape capture inside the UT. `CAPTURE_PERSIST_POLICY` controls only the
fallback captures that remain necessary. Write baseline
and sampling evidence under `EVIDENCE_DIR` and tasks anywhere under `EVAL_DIR`.
Preserve supplied UT references and each requested source boundary. Verify actual
installed/container source and build resources before reporting source unavailable.
Do not launch another full replay to obtain discovery evidence. Keep probes bounded
and reuse the supplied serving environment, TP layout and accepted overlays.

`PHASE=repair_delivery` repairs only packaging from saved evidence. Do not restart
selection or rerun unchanged passing GPU tests.

AgentX runs baseline → this complete preparation → head optimization → kernel
milestones → E2E validation. It skips the outer Profile/Strategize, ConfigSweep and
Tune stages; bounded trace sampling inside this skill still runs when needed.
Downstream stages consume the prepared queues in saved order, subject to execution
budgets. They do not nominate new kernels, apply another percentage threshold or
author UTs. Cases, weights and source binding remain fixed; optimization runs the
existing UT against changed candidate code and gates each integration with E2E A/B.
Head wins and completed milestones do not refresh extraction. Saved cost weights
remain those measured during preparation, not measurements of the optimized stack.
Queue exhaustion ends dispatch; undispatched tasks remain in carried state. Fast
and deep modes retain their head-only scheduling. Missing or stale UT evidence
fails that task rather than invoking another extractor from an optimization phase.

Declare `required_gpu_count` for multi-rank tasks in `run/requested_kernel.json`.
Current head/milestone lanes allocate one GPU per task. Completed multi-GPU UTs
remain in the delivery and are listed as deferred for optimization, with evidence
paths; they are not retried on one card. A collective without an explicit rank
requirement is deferred as well. Its extraction coverage remains recorded.

## Package the completed tasks

Run the skill's own per-task preflight first, preserving `run/requested_kernel.json`
and `run/RETURN.json`. The workflow adapter reads these existing files; it does not
repeat kernel selection or GPU validation. Write a `workflow_tasks.json` containing:

```json
{
  "workload_json": "kernel_extraction/round_baseline/kernel_workload.json",
  "ranking_json": "kernel_extraction/round_baseline/kernel_ranking.json",
  "tasks": [{"task_dir": "tasks/example", "phase": "extract_op"}],
  "failed_tasks": [],
  "deferred_tasks": [],
  "coverage_summary": {},
  "notes": "Actual tested scope and limitations"
}
```

All paths are absolute or relative to `EVAL_DIR`, and remain inside that root.
Every selected identity must have one complete owning task, or a failed/deferred
entry with its identities, reason and evidence paths. Retain required companion
leaves, per-phase coverage and uncovered mass. Keep classification and attributed
cost in the saved declaration/ranking; packaging never recomputes them.

```bash
node "$WORKFLOW_DIR/scripts/agentx_kernel_preparation.mjs" \
  --eval-dir "$EVAL_DIR" --manifest "$EVIDENCE_DIR/workflow_tasks.json" \
  --out "$EVIDENCE_DIR/ut_delivery.json"
```

The adapter checks the frozen source/UT evidence, then produces the workflow envelope.
It refuses an invalid purportedly completed task; preserve that task as failed until
its actual defect is fixed. A packaging error alone does not invalidate GPU evidence.
Return only `{"delivery_path":"/absolute/eval/path/ut_delivery.json"}` after all
scheduled work finishes. Keep the skill's full `REPORT.md` and `results.json` as evidence.

The generated envelope replaces each manifest task with its original `kernel`
declaration, `phase` and saved `result_path`. The reader derives missing anchor
cost from the saved ranking, and routes `extract_op` GEMM/attention/MoE tasks to
the head queue and other tasks to the kernel queue. It does not reapply the local
selection preflight. Source execution, numerical negative controls, the frozen
baseline and complete UT evidence must still match the declared boundary.
Descriptive notes retain their original structure.
