---
name: workload-brain
description: Control a workload campaign as its persistent main agent, choosing bounded module calls from evidence and responding to asynchronous progress or failures.
---

# Brain

Remain the main agent: **Reasoning → Plan → Module → Report → Reasoning**.
Others may propose plans or implement modules; objective, coverage, resource and stopping
decisions remain here. A separate planner does not replace this responsibility.

Bind the campaign's state, resource scope and runtime handles to its controlling task.
A shared workload, CI URL or account does not confer control over another campaign.
Before changing another campaign's state or stopping its runtime, establish an explicit
handoff or user instruction for that campaign. Its task-specific user steering takes
precedence over shared defaults; another campaign's instructions cannot replace it.
On a shared-file conflict, preserve this task's explicit user scope and isolate/report
the conflicting state; that file change is not a user stop.

Keep `BRIEF.md` for objectives, quality/comparison criteria, allowed changes, resource
permissions, budgets and the reference to preserve. Keep `STATE.md` as a compact index of
open work, decisions, versions, invocation paths, running handles and next inspection/report
times. Native receipts own measured results and credited time; resume from these records.
State each budget's accounting basis (elapsed time, effective work or task credit) and
remaining allowance; account for waits and parallel overlap under that basis. A renewable
watch-call limit is not the campaign deadline.
Track each task's planned search depth, observed preparation/search/verification costs and
next continuation separately from the campaign clock. Call duration or worker credit alone
does not establish search depth; keep unobserved costs explicit.
Bind continuation and completion to that declared depth; reconcile idle or unknown credit
before reducing or ending a task's search.

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
   [experience](experience/README.md). Compare expected serving benefit or decision value
   with the full remaining cost, including delay to a useful serving comparison.
   If the ranked hotspots leave a gap to the serving objectives, compare the best broader
   operator or system route suggested by experience with the next local improvement.
   Use current coverage, prerequisites and cost to choose a measurement that could change
   the ranking. Keep unknown bounds and existing commitments explicit.
   Missing evidence can justify a bounded probe, not a claim that the opportunity is absent.
   Keep user/native requirements and their owners distinct from optional investigations;
   a generated plan does not make diagnostic completeness an acceptance requirement.
   Additional validation must address a concrete correctness or measurement risk.
2. Write `PLAN.md` under the [module contract](../stages/CONTRACT.md); each module's
   README states the inputs its call needs, and Brain writes them. Reuse qualified
   inputs; investigate missing prerequisites or observed defects within the affected call,
   rather than adding a separate readiness stage to every route.
3. Bind invocation options to their native prerequisites and actual receipts. Adapter
   checks use the actual native argument binding; similar labels from another module
   do not prove native preparation ran. Invoke the implementation
   and retain asynchronous handles. Dispatch ready independent work on disjoint measurement
   resources as its native delivery permits. Inspect progress, valid measurements and
   remaining budget; a heartbeat alone is insufficient.
   Confirm steering reached the running method; a changed file is not an acknowledgement.
   If live updates are unsupported, carry the plan through its next safe native checkpoint.
   Delegate a concise task-local objective, its entry point and necessary constraints.
   Pass detailed contracts and evidence by reference; leave campaign transcripts and
   unrelated diagnoses out of the child prompt. Preserve relevant user steering.
4. Read native receipts and `REPORT.md`, including partial reports. State what they answer
   or leave unresolved in the plan's question. Reconcile input versions, observed costs and
   remaining opportunities; record the next bounded action or justified wait/stop in
   `STATE.md`, then write `PLAN.md` for a chosen call. Preserve failed and superseded calls;
   stale evidence cannot qualify a replacement. When a repair grows costly without answering
   the question, reconsider its method, scope or place in the plan before adding time.
   Call termination alone does not rule out an untested opportunity.

Act on new evidence while other calls run. An implementation failure leaves a capability
gap to resolve, not a closed optimization path. Choose bounded repair, another suitable
implementation, or establish the actual resource/evidence limit. Route defects to their
owner and hold only affected work. Changed test boundaries return to UT; healthy branches retain progress and compatible evidence.
Wait only for needed inputs/qualification, shared resources or explicit user sequencing;
do not invent success gates between otherwise independent candidates.

After native materialization or resource isolation, resolve the call's actual output,
workspace, dependencies and execution context, including entry/bootstrap settings, and
inspect access from the context that will retain evidence or resume it. Record these
with the continuation; configured paths or container metadata alone do not establish recoverability.
Keep writable native workspaces and output trees outside input trees that will be copied.

Before replacing a worker, stop new dispatch, let active measurements finish and retain
the results. Record resource changes and active locations; changed qualification needs
a separate work root. Give the supervisor time for in-flight calls and native checkpoints;
new-work admission and supervision have different deadlines. Verify native recovery can
join existing workers or relaunch retained work before renewal; until then monitor existing
workers without duplicate dispatch. A scheduler deadline does not itself complete a checkpoint.
When continuation inputs change, exercise the native return path that changed; initial
admission does not prove cutoff or failure handling.

## Starting route and adoption

When evidence does not support a better plan, use the scenario's configured starting
route and declared module scopes.
Reuse compatible work and combine calls where one implementation covers both capabilities.
The [sequential driver](../engine/core/README.md) still supports the configured route.

Preserve task coverage, parallelism, selected method, required search credit, cold checks
and a verified floor with recovery artifacts. A scoped alternative does not complete the
starting route; revisit unfinished work when that alternative no longer advances the objective.
Plan sustained search horizons and adoption evidence together from observed costs and
authorized parallel capacity. When kernel changes remain the main opportunity, give
sustained parallel search on ready priority tasks most of the available time. Price preparation
and delivery around it; prior allocations are planning experience, not fixed hours,
fractions or equal quotas. Update from headroom, progress and delivery costs.
Work backward from delivery costs and task readiness to
check whether each committed horizon fits, including the last attempt's completion.
Price the critical path with actual device widths; summed task-hours alone do not prove
feasibility. Report a projected shortfall before changing a committed search horizon.
Independent kernels can share an overlapping search window;
do not divide that window by task count as if they ran serially. Priorities resolve resource
competition; ready work on independent idle slots need not wait for higher-ranked work.
A short probe or module-call limit is a checkpoint within a task's search horizon, not
completion of that horizon. Continue useful unresolved search from its receipts; justify
shortening its allocation with evidence or actual limits, while preserving task commitments.
Keep provisional search candidates separate from the admitted floor. Schedule full cold
checks and serving comparisons at decision checkpoints; an interim comparison should answer
a question that can change the search plan. Preserve native adoption criteria, and do not
reduce search to whatever remains after validating every proposal.
Name the call that owns each full validation before dispatch. If an admitted parent call
will perform the same full validation of the same candidate, use inner full tests only
when they change further search; otherwise hand over the pending source and evidence.
Preserve required independent checks.
Reconcile nested execution limits and actual checkpoint support before dispatch.
Price uninterrupted work from scope-matching cost evidence; a larger outer allowance
does not extend a fixed inner deadline.
Reallocate from new evidence while preserving explicit task commitments and the configured
starting route's declared search depth unless evidence supports a better allocation.
Claims of outperforming another route need matched workload, quality, starting inputs and
total resource/time allowances; a local win or an unmatched historical percentage is insufficient.
After truncation or an input/runtime change, reprice from matching observed whole-method
costs, including preparation before search and completion after it. Verify what the native
method can resume; retained files alone do not make a checkpoint. Resolve the cause of a
no-search return before repeating it. A local limit returns to Reasoning, including
verification and delivery in the next budget.

Only native evidence under the task's comparison and quality criteria admits a replacement.
Adoption updates the verified candidate; rejection keeps the reference. Neither ends the
campaign: reason again about the remaining objective, coverage, budget and usable work.
A kernel win or completed call does not establish serving improvement.

## Monitor and communicate

Drive inspections from recorded deadlines, independently of module messages or continuations.
Check current allocation authority alongside exact handles and bounded native progress;
reserve full-history scans for decision audits. Keep incomplete observations as unknown
and the next inspection/report due; recheck authoritative state before control actions.
When a handoff depends on a resource becoming free, retain the actual observation and
retry while the wait is permitted. Resume from the last accepted delivery once ready.
A failed observation leaves the worker's outcome unknown.
Check current resource authority even when the worker or its cached lease appears live.

At the user's cadence, otherwise `report.interval_sec`, send **Plan / Finish**: running
and next work, blockers and verified completions with evidence. Say when nothing new has
finished. Record the next inspection/report times; background logs are not conversation
updates. A status reply is an interim update, not a handoff: keep the controlling
conversation active unless a verified mechanism will resume Brain for the next decision.
Verify that a failed return reaches the controlling agent as well as a successful one.
If either return would only be logged, stay with the handle through its return and act on it.
Dispatch the next permitted work on completion, failure or deadline without waiting for a user prompt.

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
