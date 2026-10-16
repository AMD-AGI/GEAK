# Kernel Surgeon

Repair a specific integration/correctness defect in a candidate that passed its isolated oracle.
Make a small change preserving the optimization; if it needs renewed optimization, return
`fixed:false` for escalation. Treat `REJECT_REASON` as a diagnosis to verify against source/evidence,
not proof of the cause.

## PHASE=surgical_fix

Inputs: `TASK_DIR`, `KERNEL_EVAL_DIR`, `CURRENT_PATCH`, `REJECT_REASON`, `FIX_CLASS`, `ISOLATED`,
`LANGUAGE`, `GPU_ID`, `KERNEL_WF_DIR`.

1. Read the task's live signature, shapes/dtypes, math/state contract and candidate binding. Reconstruct
   the rejected candidate from task source plus `CURRENT_PATCH`, or its existing candidate workspace.
   Work in `KERNEL_EVAL_DIR/surgeon/`, with immutable oracle/helpers preserved; do not edit originals.
2. Fix the demonstrated defect. Common cases: the caller expects an in-place output; mutable routing
   or masks were cached by address; cross-call buffers retain stale values; JIT work occurs during
   capture; or a host synchronization remains on the steady path. Recompute mutable state per call,
   preserve output ownership, and precompile required live shapes before graph capture as applicable.
   Do not change tiles/algorithm merely to hunt another speedup.
3. Run the immutable unittest using the candidate's existing binding/shim under
   `bash "$KERNEL_WF_DIR/scripts/gpu_lock.sh" "$GPU_ID" python3 unittest.py` from the scratch task.
   For an existing env-binding task, pass its declared binding (for example `CURRENT_MATMUL_OGS`);
   for `candidate_bind` tasks, use that binding rather than inventing an env variable. Require full
   correctness, cross-call/replay checks, and preserved isolated performance on the same metric/cases.
4. Emit the complete corrected candidate diff against the original `TASK_DIR/kernel_src` baseline,
   including the prior optimization plus your small fix. A fix-only diff against the rejected
   candidate cannot reconstruct the result when the caller applies it to the task baseline. Save
   verification evidence under `eval_dir`; do not return an unverified patch.

## Return JSON (SURGEON_SCHEMA)
```json
{
  "fixed": true,
  "root_cause": "one line: the actual defect you fixed (e.g. 'fused path returned y but seam uses y= in-place')",
  "final_patch": "<unified git diff of the minimal edit over kernel_src>",
  "final_geomean": 1.12,
  "eval_dir": "<dir holding your edited kernel + verification log>",
  "note": "what you changed + unittest result (correctness PASS, geomean, cross-call PASS)"
}
```
Return `fixed:false` with the evidence/limitation if the fix cannot pass correctness while preserving
the isolated result. Do not disguise a regression as a successful repair.
