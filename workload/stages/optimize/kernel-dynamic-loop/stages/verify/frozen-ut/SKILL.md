---
name: frozen-ut
description: Verify an exact kernel candidate in a fresh process through the unchanged native task entry, retaining its source-bound evidence.
---

# Frozen native UT

Read the [Verify contract](../README.md).
Use the supplied source tools to check the edit boundary and snapshot the canonical
computation tree. Exclusively materialize the candidate there to exercise existing
bindings. For `source_tree.py check`, exit 1 means permitted changes; exit 2 rejects them.

Run the supplied `run_task_entry.py` in a fresh process with the full assigned device
set and a timeout covering recorded test cost. Preserve stdout/stderr, exit status,
source/input identities and generated receipts before another run; use existing evidence
tools as required by the task. Keep pending evidence on failure. Wait for measurement
completion, then restore the prior canonical tree even after an invalid result.

Return the native verdict and full-task score without introducing another scoring formula.
