---
name: kernel-hardware-evidence
description: Use AMD profiling, roofline, compiler IR and assembly evidence to choose kernel experiments or review hardware claims. Applies when an optimization decision depends on execution details; ordinary edits do not require profiling.
---

# Kernel hardware evidence

Use the task's available tools and the existing job owner's execution setup.
This prospective skill source does not install tools, change active investigations
or grant GPU access.

Start from the complete callable and its numerical/state contract. Separate
required work from the baseline's tiles, splits and temporary buffers. Keep the
best correct measured source while developing useful alternatives, including
correct slower implementations whose remaining cost is understood.

For a decision that depends on hardware evidence:

- Check the cited artifact and field against the actual source/configuration,
  case (shape, dtype, strides), runtime, physical device and measured phase.
  Correctness and timing must cover the same source and selected implementation;
  a passing fallback does not qualify a different timed path.
- State the observation separately from its explanation. Choose an edit or
  diagnostic that distinguishes plausible causes, then check that compilation
  and dispatch realized the intended change. An unrealized change leaves the
  idea untested; faster timing alone does not prove the proposed mechanism.
- Preserve the full-operation correctness and unprofiled timing boundary. Include
  all replaced stages and required state updates; do not move cost outside it.
  Reuse applicable evidence and refresh only what the change or question affects.

Read only the reference needed for the current question:

| Question | Reference |
|---|---|
| Which diagnostic, command or failed collection needs attention? | [Collection](references/collection.md) |
| Is the byte/FLOP model, percentage or headroom claim defensible? | [Roofline](references/roofline.md) |
| Did a rewrite change ownership, movement, instructions or resources? | [IR, ISA and dataflow](references/ir-isa-dataflow.md) |

Keep a brief rationale with raw links in the existing experiment notes: the claim,
its operands/units and scope, unresolved alternatives, and what the result changes.
No additional report form or complete diagnostic suite is required. Missing evidence
stays unknown; useful conditional models and untested ideas can still guide work.
Correct inherited claims against their actual sources, retain failed attempts,
and carry supported observations forward without turning a local outcome into a
universal hardware rule.
