---
name: workload-reasoning
description: Control a workload campaign as its persistent main agent, choosing bounded module calls from evidence and responding to asynchronous progress or failures.
---

# Reasoning

Remain the campaign's main agent: **Reasoning → Plan → Module → Report → Reasoning**.
This is a continuing responsibility, not a new agent per round. Implementations may
delegate work; their reports return here for decisions about the whole campaign.

Keep `BRIEF.md` in the run directory for objectives, quality/comparison criteria,
allowed changes, resource permissions, budgets and the reference to preserve.
Keep `STATE.md` as a compact index of decisions, candidate/input versions, invocation
paths, running handles and next inspection/report times. Native receipts own measured
results and credited time; resume from these records after interruption or migration.

## Decide and act

1. Read new reports alongside running work and accepted/rejected evidence. Identify the
   bottleneck or uncertainty that matters next; consult [experience](experience.md)
   when it can inform that decision.
2. Write a bounded `PLAN.md` using the [module contract](../stages/CONTRACT.md).
   Explain why this call is useful, which module/implementation it uses, and what
   evidence will decide its outcome. Existing qualified inputs need no separate
   readiness stage; investigate only missing prerequisites or observed defects.
3. Invoke the implementation and retain its handle when it runs asynchronously.
   Independent work uses separate candidate/output directories and disjoint measurement
   resources. Inspect it for progress, valid measurements and remaining budget; a
   heartbeat alone is insufficient.
4. Read its native receipts and `REPORT.md`, update `STATE.md`, then adopt, reject,
   repair, investigate or wait. A report may arrive before completion. Reconcile it
   with the current input/source version before acting; stale evidence cannot qualify
   a replacement. Preserve failed and superseded invocations.

A new report or inspection can trigger another decision while other calls are running.
Route a defect to its owning module and hold only affected dependent work. A changed
test boundary returns to UT; after qualification, reason again and resume Optimize with
a separate work root and matching evidence. Healthy branches continue. Keep earlier
costs visible, and credit only compatible work under the declared budget contract.
Join branches only when their inputs or resources require it.

Before replacing a worker, stop its new dispatch, let active measurements finish and
retain their results. Record resource changes and the active location in `STATE.md`.
Check deadlines against remaining search and validation costs; scheduler limits do not
by themselves complete a checkpoint.

## Starting route and adoption

When evidence does not support a better plan, use
**Setup → Baseline → Profile → UT → Roofline → Optimize → Apply back → Evaluate**.
Reuse compatible completed work and combine calls where an implementation covers both.
The [sequential driver](../engine/core/README.md) still supports the configured route.

Preserve its task coverage, parallelism, selected method, credited search budget, cold
verification and E2E reservation. Keep a declared performance-floor candidate and its
recovery artifacts. Extra exploration needs an explicit budget; compare policies from
the same starting state and total resource allowance.
Only the module's native acceptance evidence and all required objectives can admit a
replacement. A kernel win or a completed call does not establish serving improvement.

## Monitor and communicate

At the user's cadence, otherwise `report.interval_sec` from the scenario/defaults,
send **Plan / Finish**: next/running work and blockers; verified completions since the
last update, with evidence links. Say when nothing new has finished. Record the next
inspection and reporting times; background logs do not deliver a conversation update.
Verify wake-up and delivery support before promising unattended reporting; if unavailable,
state the limit and keep the controlling conversation active while monitoring is required.

Stop on demonstrated goal attainment, user stop, exhausted budget or unavailable
prerequisites, and distinguish those outcomes. Preserve continuations for unfinished work.
At handoff or a requested route review, invoke [Report](../stages/report/README.md)
to explain the actual route, including repairs and parallel branches.
