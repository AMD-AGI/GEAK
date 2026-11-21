---
name: frozen-ut
description: Verify an exact kernel candidate in a fresh process through the unchanged native task entry, retaining its source-bound evidence.
---

# Frozen native UT

Read the [Verify contract](../README.md).
Use the supplied source tools to check the edit boundary.
Use an independent complete task copy where native bindings support relocation;
retain all declared inputs and workspace metadata, including hidden entries, and
prove the candidate's actual binding through the native checks. Otherwise snapshot the canonical
computation tree and materialize the candidate there exclusively.
For `source_tree.py check`, exit 1 means permitted changes; exit 2 rejects them.

Run the supplied `run_task_entry.py` in a fresh process with the full assigned device
set and a timeout covering recorded test cost. Preserve stdout/stderr, exit status,
source/input identities and generated receipts before another run; use existing evidence
tools as required by the task. Keep pending evidence on failure. Wait for measurement
completion; if the canonical task was used, restore its prior tree even after an invalid result.

Return the native verdict and full-task score without introducing another scoring formula.
