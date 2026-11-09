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
   resources. Retain handles and evidence for work that reports later; the controller
   owns the next inspection and recovery, including after resource or storage migration.
4. **Report.** Read the module's native results and `REPORT.md`, including failures.
   Record what changed, what was established and what remains uncertain. Adopt, reject,
   investigate further or wait; then return to reasoning.

Inspect ongoing work at the requested cadence: progress, native validity, resources and
remaining scope. A heartbeat alone proves none of these. Investigate suspect results
against evidence from the same invocation, source and harness; a large score alone is
not a failed test. Route a confirmed defect to its owner now, hold affected dependent
work and resume after verified repair. Do not wait for final evaluation or blindly retry
`blocked` work. Keep the active location in `STATE.md` and leave a pointer after migration.

At the user's reporting cadence (otherwise `report.interval_sec` or its existing default),
send a short **Plan / Finish** update to the user: **Plan** — next actions, running work
and blockers; **Finish** — verified completions since the last update, linked to evidence.
Say when nothing new has finished. Record the update and next reporting time in `STATE.md`
and continue monitoring between reports. Background logs do not replace this user-facing
update. Before promising unattended updates, verify both scheduled wake-up and delivery
to this conversation. If either is unavailable, state that limit and keep the primary
conversation active when ongoing reporting is required.

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
outcomes. Meeting a floor does not prove a higher goal.

## Multi-config scenarios

A multi-config scenario (`adopt-ci-artifacts` baseline, `multi_config: true`) optimizes every
CI point of one EP group in one campaign. The split path is
`baseline → profile → ut → roofline → optimize → applyback`: `profile: kernel-profile` locates
runtime dispatch and shapes and freezes a `profile_handoff.json`; `ut: kernel-ut` consumes it
and generates/qualifies the fixed-template UTs, one UT per task pooling every config
(`kernel-extract/scripts/merge_policy.py` decides what pools). Set `task_source_stage: "ut"` so
consumers use the separate task output, and run/retry `ut` without repeating captures. See
[the handoff contract](stages/profile/HANDOFF.md) and [the UT stage](stages/ut/README.md).

Applyback measures every CI point a UT covered (`stages/applyback/points.py`), each in that
point's own CI configuration, and judges it against that point's official CI number
(`reference: ci` in `stages/evaluate/finalize.py`); a covered point left unmeasured makes the
verdict incomplete.

