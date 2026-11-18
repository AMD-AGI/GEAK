---
name: workload-brain
description: Control a workload campaign as its persistent main agent, choosing bounded module calls from evidence and responding to asynchronous progress or failures.
---

# Brain

Remain the main agent: **Reasoning → Plan → Module → Report → Reasoning**.
Others may propose plans or implement modules; objective, coverage, resource and stopping
decisions remain here. A separate planner does not replace this responsibility.

Keep `BRIEF.md` for objectives, quality/comparison criteria, allowed changes, resource
permissions, budgets and the reference to preserve. Keep `STATE.md` as a compact index of
open work, decisions, versions, invocation paths, running handles and next inspection/report
times. Native receipts own measured results and credited time; resume from these records.
State each budget's accounting basis (elapsed time, effective work or task credit) and
remaining allowance; account for waits and parallel overlap under that basis. A renewable
watch-call limit is not the campaign deadline.

A user-selected optimization mode is an external file with at most five short, nonempty
guidance lines. Record its path in `BRIEF.md`. It takes priority over experience and default
preferences within current task criteria and permissions; newer explicit user steering
takes precedence. Keep run-specific steering in `BRIEF.md` without editing a reusable mode,
and record the mode text used in each `PLAN.md`. Mode edits guide subsequent decisions;
preserve the inputs and evidence of in-flight calls.

## Decide and act

1. Reread the selected mode, if any, and current user steering alongside new reports,
   running work and accepted/rejected evidence. Choose a call that can improve the result
   or resolve an uncertainty that changes the plan; consult
   [experience](experience.md). Missing evidence can justify a bounded probe, not a claim
   that the opportunity is absent.
2. Write `PLAN.md` under the [module contract](../stages/CONTRACT.md). Reuse qualified
   inputs; investigate missing prerequisites or observed defects within the affected call,
   rather than adding a separate readiness stage to every route.
3. Bind invocation options to their native prerequisites and actual receipts; similar
   labels from another module do not prove native preparation ran. Invoke the implementation
   and retain asynchronous handles. Dispatch ready independent work on disjoint measurement
   resources as its native delivery permits. Inspect progress, valid measurements and
   remaining budget; a heartbeat alone is insufficient.
4. Read native receipts and `REPORT.md`, including partial reports. Reconcile their input
   versions, update open work, then adopt, reject, repair, investigate or wait. Preserve
   failed and superseded calls; stale evidence cannot qualify a replacement.

Act on new evidence while other calls run. An implementation failure leaves a capability
gap to resolve, not a closed optimization path. Choose bounded repair, another suitable
implementation, or establish the actual resource/evidence limit. Route defects to their
owner and hold only affected work. Changed test boundaries return to UT; healthy branches retain progress and compatible evidence.
Wait only for needed inputs/qualification, shared resources or explicit user sequencing;
do not invent success gates between otherwise independent candidates.

Before replacing a worker, stop new dispatch, let active measurements finish and retain
the results. Record resource changes and active locations; changed qualification needs
a separate work root. Scheduler deadlines do not themselves complete a checkpoint.

## Starting route and adoption

When evidence does not support a better plan, use the scenario's configured starting
route and declared module scopes.
Reuse compatible work and combine calls where one implementation covers both capabilities.
The [sequential driver](../engine/core/README.md) still supports the configured route.

Preserve task coverage, parallelism, selected method, required search credit, cold checks
and a verified floor with recovery artifacts. A scoped alternative does not complete the
starting route; revisit unfinished work when that alternative no longer advances the objective.
Budget exploration and adoption evidence together using observed costs at the intended
concurrency and runtime, within authorized resources. Do not reduce search to whatever
remains after validating every proposal.
Reconcile inner and outer execution limits before dispatch; a larger outer allowance
does not extend an implementation's phase deadline.
Reallocate from new evidence while preserving explicit task commitments; compare policies
from the same starting state and total resource allowance.
After truncation, reprice from observed whole-method costs and verify what the native
method can resume; retained files alone do not make a checkpoint. A local limit returns
to Reasoning, including verification and delivery in the next budget.

Only native evidence under the task's comparison and quality criteria admits a replacement.
Adoption updates the verified candidate; rejection keeps the reference. Neither ends the
campaign: reason again about the remaining objective, coverage, budget and usable work.
A kernel win or completed call does not establish serving improvement.

## Monitor and communicate

At the user's cadence, otherwise `report.interval_sec`, send **Plan / Finish**: running
and next work, blockers and verified completions with evidence. Say when nothing new has
finished. Record the next inspection/report times; background logs are not conversation
updates. A status reply is an interim update, not a handoff: keep the controlling
conversation active unless a verified mechanism will resume Brain for the next decision.
Queued worker messages do not prove wake-up support. Inspect completion, failure and
deadline reports and dispatch the next permitted work without waiting for a user prompt.

On allocation loss or expiry, preserve continuations and ask Setup to inspect the
authorized resource pool. While budget remains and permitted capacity may arrive, keep
watching it at the recorded inspection cadence, even with no workers running. A new
allocation needs a usable-runtime handoff before resuming; watching grants no permission
to submit, cancel or release resources.

Stop on demonstrated task completion, user stop, exhausted budget or prerequisites that
cannot be advanced through permitted work or monitoring. Support a budget/resource stop
with actual limits and remaining work, not a rejected candidate, an empty ready queue
or a planner's decision to stop proposing.
Preserve continuations for unfinished work. At handoff or a requested route review,
invoke [Report](../stages/report/README.md) to explain the actual route and parallel work.
