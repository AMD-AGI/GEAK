---
name: task-inspect
description: Inspect a supplied kernel task and its native receipts to establish measurement readiness without rebuilding qualified inputs.
---

# Task inspection

Read the [Setup / Benchmark contract](../README.md).
Inspect the declared computation, editable leaves, full case sweep and native entry.
Check missing or suspect prerequisites and reuse valid qualification. Keep the original
baseline distinct from the currently accepted source. Obtain a required fresh baseline
through the unchanged native measurement path in the parent-provided environment.

Return the supporting receipts and observed test cost. Report a defect requiring changed
frozen inputs, bindings or device width to the parent instead of redefining the task.
