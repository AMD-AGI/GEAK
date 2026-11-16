# Merging UT tasks across configurations

After reviewing the draft declarations and before authoring UTs, write a provisional
plan without templates. `merge_policy.py` reads `task_plan.json`, not the declarations;
running it before this plan exists cannot decide the task boundaries.

```bash
python3 "$SKILL_ROOT/scripts/build_task_plan.py" --eval-dir "$OUT" --check
python3 "$SKILL_ROOT/scripts/build_task_plan.py" --eval-dir "$OUT" --no-scaffold
python3 "$SKILL_ROOT/scripts/merge_policy.py" --group-dir "$OUT" \
    --baseline "$RUN_ROOT/baseline/out/baseline.json" \
    --source-root <path-in-evidence>=<same-pinned-source-tree> --out "$OUT/merge_policy.json"
```

Use the matching baseline's `launch_diff`; for standalone UT generation, supply its
path explicitly. Preserve pinned config launch settings when deciding splits. Missing
source or launch-setting evidence is a blocker, not evidence that settings are equal.
The policy reads the handed-off `configs/*/evidence` without collecting new profiles.

Apply the result; record each application in `decisions.json`. **An operator dispatched by
batch size is one UT. Whether it is one is decided strictly by the policy; once it is, it
is merged -- the harness being hard to write is never a reason to keep it apart.**
1. Same launch group, same entry: one UT whose cases pool every config the task runs in.
2. `per_task.<t>.decision == split_by_config_value`: the task's own source reads a setting
   that differs between configs (`launch_diff`); one UT per value. Nothing else splits.
3. `branch_groups[*]` (`merged: true`): different entries chosen above them by batch size.
   Declare ONE task for the group; `how` says how:
   - `dispatcher` -- `target_callable` = `seam`; its threshold is part of the task.
   - `rebuilt_instance` -- `target_callable` = `seam` (a method); `cases.py` builds the
     instance from `attribute_manifest` (every `self.<attr>` the seam and its helpers read)
     and fails on an attribute outside it.
   - `scope_equalized` -- the seam is not callable from a UT (it takes a ForwardBatch...): one
     entry per member (`member_entries`), each case calls its member's; a member whose path
     stops earlier also runs the `scope_region` calls the other's path runs, so every case
     compares the same output. The threshold is not part of the task.
   Copy the group into `meta.json` as `merge` (members, how, seam, call_lines, conditions,
   observed_m, boundaries, threshold_movable, expected_kernels, and the manifest or
   region). Every case carries `branch` (its member, from `case_sources`), its
   `expected_kernels`, any non-size `context` it fixes (an enabled-kernel set, an optional
   FP8 input, a layer type), and `boundary: true` with weight 0 for cases placed at each
   `boundaries` edge. Each branch keeps its own tolerance; during optimization the timing gate reports
   `per_branch` and `worst_branch`, and a kept result needs every branch above the noise
   band, not only the pooled figure.
4. `refused_pairs` say why two tasks are NOT one (launched together, interleaving sizes, a
   call not under a conditional): keep them separate.
5. Different launch groups (EP, ...): same entry and same env, one UT with cases tagged by
   group; otherwise separate (`cross_group`).

Apply the decisions to `task_decls.json`, then rerun `build_task_plan.py --check` and
`build_task_plan.py` to generate the final scaffolds. Move obsolete provisional task
directories outside `tasks/` before starting a wave, so only final owners get lanes.
