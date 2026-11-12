---
name: kernel-dynamic-loop
description: Optimize a qualified kernel task through an evidence-driven reasoning loop with small, interchangeable modules, within workload's frozen-task, resource and budget contracts.
---

# Kernel dynamic loop

An Optimize implementation: **Reasoning → Plan → Module → Report → Reasoning**.
The coding agent controls this loop using its available tools. Modules are capabilities,
not a prescribed sequence or a set of mandatory agents. This method is independent of
the Kernel Workflow runtime and roles.

Select `stages.optimize.skill=kernel-dynamic-loop` in the scenario or with `run.py --set`.
Read the [module catalogue](README.md#modules), [call contract](stages/CONTRACT.md) and
[design principles](PRINCIPLES.md). Module selection and implementation selection are
separate decisions; the catalogue supplies one initial implementation per module.

## Loop

Read the task contract, current source, worker status and prior `HANDOFF.md` first.
Reuse qualified inputs; call a module because evidence supports its question, not
because it has not run yet. A required native baseline refresh still applies.

Repeat this cycle for the same kernel while a useful next call fits the remaining
budgets. Every module, including Profile, may be revisited across many cycles. A completed
pass or a successful candidate alone is not a stopping condition.

1. **Reason.** Choose a useful question from accepted and rejected evidence, uncertainty,
   running work and remaining budget.
2. **Plan.** Select the module, read its `README.md` contract, then choose an implementation
   `SKILL.md` or tool satisfying it. Record the question and bounded call in `PLAN.md`.
3. **Act.** Follow the selected implementation. Modules may repeat, skip, combine or run
   concurrently within the assigned resources. Retain handles and inspect running work;
   report observed blockers promptly and hold affected dependent calls.
4. **Report.** Read native results and the call's `REPORT.md`, including failures and
   uncertainty. Feed the findings back to Reason to continue, change direction, combine,
   repair or hand off.
   Keep local notes as evidence pointers, not another copy of worker counters.

Use [Deliver](stages/deliver/README.md) before returning, including when blocked or
interrupted. An unfinished call keeps its evidence path and next recovery action.
The parent verifies and accepts candidates; this method makes kernel-level claims only.
