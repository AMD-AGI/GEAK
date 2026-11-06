---
name: workload
description: Improve a serving workload through an evidence-driven reasoning loop. Choose modules and interchangeable implementations for diagnosis, system or kernel changes, and measured comparison; retain the verified reference and the configured compute budget.
---

# Workload optimization

The agent is the controller: **Reasoning → Plan → Module → Report → Reasoning**.
Read the task and scenario config. In its run directory, initialize `BRIEF.md` with the
objective, evaluation/quality contract, allowed changes, resource permissions, budgets
and reference artifacts. `STATE.md` indexes accepted candidates, open questions and
running work; native receipts remain authoritative for measurements and credited time.

Repeat:
1. **Reason.** Read accepted and rejected evidence, uncertainties and running work.
   Choose a useful observation or testable change; consult relevant skills on demand.
2. **Plan.** Write the question, module and implementation, input identities, scope,
   output directory, resource/time budget and decisive check in the invocation's
   `PLAN.md`. Use the [module catalogue](README.md#modules) and
   [invocation contract](stages/CONTRACT.md#direct-module-calls).
3. **Act.** Call the selected skill or tool. Modules may repeat, skip or combine.
   Parallel work gets separate candidate copies and outputs; measurements use isolated
   resources. Retain handles and evidence for work that reports later.
4. **Report.** Read the module's native results and `REPORT.md`, including failures.
   Record what changed, what was established and what remains uncertain. Adopt, reject,
   investigate further or wait; then return to reasoning.

Changing an implementation does not change its module's acceptance contract. Kernel
Workflow and native tools (OOB) share the same frozen-task and cold-verification rules.
System changes use their declared quality and serving checks; they need no artificial UT.
New capabilities enter through a skill path or command, without changing the controller.

The [standard recipe](recipes/standard.md) preserves the existing fixed route and its
configured task coverage, inner methods, compute and validation budgets. Start there
when no better-supported plan is available. Keep a declared performance-floor candidate
and its recovery artifacts; a worse or inconclusive result cannot replace it. Additional
exploration has an explicit budget and must not consume protected reference work.
Compare search policies from the same starting state and total budget.

A completed call is not an optimization win. Adoption requires integration/effect and
quality evidence plus every required objective; a throughput verdict alone is insufficient.
An E2E claim requires the declared serving comparison. Stop on demonstrated goal attainment,
user stop, exhausted budget or unavailable prerequisites; preserve progress and distinguish these
outcomes. Meeting a floor does not prove a higher goal. Report at the task's requested
cadence; the standard recipe's automatic reporting remains available.
