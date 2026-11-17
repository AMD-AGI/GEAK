---
name: kernel-brain
description: Control a qualified kernel task through bounded module calls, adapting priorities and parallel work from reports while preserving the parent contract.
---

# Brain

Remain the controller: **Reasoning → Plan → Module → Report → Reasoning**.
Use the parent's task, current steering, selected mode and knowledge restrictions.
Modules may propose directions; the next call and stopping decision remain here.

Reconcile the actual task source and worker receipts with prior `RUN_DIR/HANDOFF.md`.
A rejected or pending handoff is not the worker's accepted source. Keep
`RUN_DIR/STATE.md` as a compact index of source versions, the verified inner incumbent,
open questions, call paths, running handles and next inspection/report times; native
receipts own results and the worker owns accounting. Reuse qualified inputs.

## Decide and act

1. Read new and partial reports alongside running work and current steering. Rank useful
   questions by plausible effect on the native objective, observed bottlenecks, remaining
   headroom and cost through verification and delivery. Use the workload's
   [experience guidance](../../../../brain/experience.md) within this task's scope;
   a missing historical match does not exclude an opportunity.
2. Choose a module and then its implementation under the
   [call contract](../stages/CONTRACT.md). Write a bounded `PLAN.md` stating what evidence
   would change the next decision. Reuse observations or request a discriminating probe;
   do not turn a missing measurement into evidence of no optimization space.
3. Dispatch ready independent work on separate source copies within assigned resources.
   Act when a report arrives without waiting for unrelated calls. Inspect real progress
   and measurements; hold only affected work when a call fails. A frozen-task or binding
   defect returns to the parent for requalification, not an edit to the test.
4. Read the native verdict, reconcile source identities and update the index. Advance the
   inner incumbent only under the shared contract; a merge is a new candidate for Verify.
   Reorder remaining work and budgets, then reason again, including after a rejection.

## Starting route and continuation

When no stronger plan exists, establish or reuse Setup / Benchmark evidence, Profile the
current cost, Optimize a hypothesis and Verify it. Return to Reasoning after each call.
Research, Integrate and Learn are available when they answer the current question.
Every module, including Profile, may repeat; a pass through the catalogue or one accepted
candidate does not finish the search.

Use the parent's actual attempt boundary and observed full-call costs; resuming or
starting a module does not restart the clock. Budget search and delivery together under
the shared contract. Replan when a path cannot fit; if no useful admissible call remains,
return the actual limits and continuation to the parent.

Checkpoint after meaningful reports and send concise Plan / Finish updates to the parent
at its cadence. Preserve unfinished calls, evidence and recovery actions before stopping.
Stop on task completion, exhausted assigned limits, a parent stop or an established
resource/prerequisite limit; a rejected candidate alone is not such a limit.
Use [Deliver](../stages/deliver/README.md) before returning. The parent owns final KEEP,
cold checks and serving acceptance; this loop reports kernel-level evidence only.
