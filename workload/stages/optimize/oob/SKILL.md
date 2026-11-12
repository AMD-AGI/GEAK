---
name: oob
description: Produce an isolated kernel or serving-system candidate with native coding tools, preserving the plan's scope and native validation contract.
---

# Optimize with native tools

Read the [invocation contract](../../CONTRACT.md#direct-module-calls) and the plan's
hypothesis, inputs, allowed changes, resources, budget and decisive check. The agent
chooses its coding approach. Scope comes from the plan; a supplied frozen kernel task
keeps its frozen contract throughout.

## Qualified kernel task

Select `stages.optimize.skill=oob`. Use [Optimize's kernel recipe](../README.md#kernel-recipe)
and this directory's worker hooks. Dispatch, accounting, KEEP and cold verification
remain with the shared scheduler/worker and native measurement reader.

Reuse the qualified harness and edit only declared computation sources with native
coding tools. Preserve every output and the complete case sweep, source/build bindings,
inputs and assigned device/rank set. Independent coding can use isolated copies;
measurements retain the task's resource isolation, including for collectives.

Use the supplied native task entry for complete checks; retain diagnostic evidence
separately. Leave the selected source in `kernel_src/` and a concise `HANDOFF.md` for
the worker. Its subsequent receipt owns acceptance; a local score is not an E2E claim.

## Serving-system experiment

Use a direct module call with a separate candidate copy. Inspect available implementations
and native controls, then make a bounded source/configuration change testing the plan.
Related changes may be combined when their interaction is the question; retain the bundle
identity. A computation change may require UT qualification; a configuration-only call
uses its declared system checks and does not go through the kernel worker hooks.

Screen the candidate without changing the workload or quality criterion. Return `REPORT.md`
with source/configuration/build identity, checks, costs and unresolved assumptions.
Hand it to [system integration](../../applyback/system-runtime/SKILL.md) and
[Evaluate](../../evaluate/native-e2e/SKILL.md). Preserve losing candidates and evidence;
only the required comparison can justify replacing the accepted system.
