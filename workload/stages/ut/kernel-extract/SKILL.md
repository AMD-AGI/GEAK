---
name: workload-kernel-ut
description: Build or repair qualified kernel tasks from workload evidence using the existing kernel-extract tools and native delivery gates.
---

# Kernel task extraction

Read the [UT contract](../README.md). Use the declaration, task construction and delivery
parts of [kernel-extract](../../profile/kernel-extract/SKILL.md).
Set `SKILL_ROOT` to that vendored directory, `SHARED_TOOLS` to its `shared/`, and
`EVAL_DIR` to this invocation's output. Reuse compatible Profile evidence; request
missing observations through Brain instead of silently launching another full replay.
The plan sets scope; GPU capacity limits concurrent execution, not extraction coverage.

Finish callable bindings, case geometry/weights, editable leaves and device requirements
before freezing and running the full native UT and delivery gate. Use
`workload/engine/campaign/run_task_entry.py <task-dir>` in the pinned runtime so the
task's `unittest.py` cannot shadow Python's standard library.
Qualify that a semantics-preserving edit to a declared leaf remains runnable while native
negative controls still detect incorrect outputs. The oracle and harness remain frozen.

For a repair, retain the original failure and task version, change only the affected
delivery in a new directory and rerun its native qualification. Report new task identities,
receipts and remaining blockers promptly to Brain for rescheduling.
The legacy combined Profile/UT delivery uses [Profile's artifact mapping](../../profile/README.md#driver-delivery).
