---
name: kernel-dynamic-loop
description: Optimize a qualified kernel task through an evidence-driven reasoning loop with small, interchangeable modules, within workload's frozen-task, resource and budget contracts.
---

# Kernel dynamic loop

An Optimize implementation: **Reasoning → Plan → Module → Report → Reasoning**.
Read the supplied task, [module catalogue](README.md#modules) and
[principles](PRINCIPLES.md), then follow [Brain](brain/SKILL.md) as the same controlling
agent throughout the attempt. Modules are capabilities with replaceable implementations;
they are not mandatory agents or a fixed execution order.

Select `stages.optimize.skill=kernel-dynamic-loop` in the scenario or with `run.py --set`.
The [call contract](stages/CONTRACT.md) preserves the parent's task, resources, budgets
and acceptance criteria. This method uses the agent's tools directly and is independent
of the Kernel Workflow runtime and roles.
