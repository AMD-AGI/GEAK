---
name: system-oob
description: Use native agent tools to optimize a declared serving configuration or implementation in an isolated candidate, with quality and end-to-end evidence supplied by the plan.
---

# System optimization with native tools

Read the [invocation contract](../../CONTRACT.md#direct-module-calls). This is a direct
Optimize implementation; the standard kernel recipe uses `native-frozen` for OOB tasks.

Read the plan's hypothesis, parent system, allowed changes, evaluation contract and
budget. Work in a separate candidate copy. Inspect available implementations and native
controls, then make a bounded change that tests the hypothesis. Related changes may be
combined when their interaction is the question; record the bundle so its result is not
attributed to an individual edit without evidence.

Use targeted checks to screen the candidate. Changes to computation may need a new UT
boundary and qualification; a supplied frozen harness stays frozen. Scheduling, cache,
layout, backend and runtime settings are possible search dimensions, not a checklist.
Do not change the workload or quality criterion to make a candidate look faster.

Return `REPORT.md` with the changed source/configuration/build identity, checks, costs
and unresolved assumptions. Hand the candidate to [system integration](../../applyback/system-runtime/SKILL.md)
and [Evaluate](../../evaluate/native-e2e/SKILL.md). A screen is provisional; only the
required comparison can justify replacing the accepted system. Keep losing candidates
and their reports so the next plan can use what was learned.
