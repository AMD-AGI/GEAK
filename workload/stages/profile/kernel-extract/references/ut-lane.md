# UT lane: write, validate and fix one task's UT

`scripts/ut_wave.py` runs one of these per task, in parallel, each on its own GPU(s). You
are one lane. Everything shared -- the profile, the census, the selection, the task
boundary, the case plan, the server evidence -- is already done and frozen; you turn one
scaffolded task directory into a UT whose every gate passes on the installed baseline.

## Rules

- **Write only inside `TASK_DIR`.** Other lanes are editing their own tasks right now.
  Read anything else; never write `task_plan.json`, `task_decls.json`, the evidence or a
  sibling task.
- **Start no server, take no profile, no capture.** The evidence is what it is. If the
  task cannot be built from it, that is a blocker (below), not a reason to collect more.
- **Only your GPUs.** `GPU_IDS` lists them (0..k-1 inside the lane). `unittest.py` spreads
  its legs over all of them by itself; do not pin a GPU by hand and do not take a
  `gpu_lock` on a device outside `GPU_IDS`.
- **Do not edit the boundary.** `run/requested_kernel.json` fixes the kernels, the entry
  (`target_callable`), the phases and the family. If it is wrong, block with the evidence.
- The frozen helpers copied into the task (`ut_driver.py`, `ut_contract.py`,
  `harness_lib.py`, `leg_runner.py`, `compile_watch.py`, `reference_f64.py`) are not yours
  to change. A gate that fails is fixed in the task's own files: `cases.py`, `meta.json`,
  `run/ut_cases.json`, `kernel_src/`.

## Procedure

1. **Read the request.** `run/requested_kernel.json` (kernels, entry, phases, family,
   `server_evidence`, `server_preimports`), `run/ut_case_plan.json`, the scaffold the plan
   left (`meta.json`, `cases.py`, `run/ut_cases.json`, `run/ut_contract.json`) and, when
   present, `run/server_calls.json` -- the real calls the server made into the entry,
   warmup included. Then [source extraction](source-extraction.md),
   [task authoring](task-authoring.md) and [UT contracts](../ut_contracts/README.md), and
   the family's `pitfalls` in `ut_contracts/families/<family>.json`.
2. **Extract the source** (`kernel_src/`, `meta.ut_contract.editable_leaves`,
   `meta.candidate_bind`): the installed files the entry's kernels compile from, resolved
   inside this container (the serving image), with their hashes.
3. **Fill the scaffold.** `cases.py` `call()` exactly as the deployed caller calls the
   entry -- optional arguments, dtypes and None-ness as `run/server_calls.json` shows them,
   not as defaults suggest -- the family's hooks, `replay_bundle()` for graph phases; the
   case axes in `run/ut_cases.json` if the plan left any open; one numerical negative
   control per editable leaf; `meta.workload.serving_weight_model.evidence_path`. Then
   `python3 $SKILL_ROOT/ut_contracts/scaffold_ut.py sync-meta --task $TASK_DIR`.
4. **CPU lint until clean:** `python3 $TASK_DIR/ut_contract.py $TASK_DIR`.
5. **Iterate on the boundary cases:** `cd $TASK_DIR && python3 unittest.py --quick`. It runs
   every gate except timing on the smallest, middle and largest case of each phase and a
   few server calls, one draw, and writes `run/ut_report.quick.json`. It never passes;
   it is how you find a wrong `call()`, a hook, a bound or a control in minutes instead of
   an hour. Fix and rerun until every gate it runs passes.
6. **Full run:** `python3 unittest.py --skip timing` when the prompt says TIMING is done by
   the wave (the default), otherwise `python3 unittest.py`. Fix what fails and rerun; add
   `--reuse` to keep the gates that already passed when nothing they read changed.

The lane ends by itself once the receipt (`run/ut_result.json`, written by the UT, newer
than every file it reads) shows every required gate passing. Do not write the receipt, do
not edit reports, do not loosen a bound or a gate to get there: a departure from the family
is an `--override` with a stated reason, recorded by the lint.

## Reading the gates

- `launches`: the baseline did not launch a kernel the server launched in that phase
  (`missing`, with `seen` listing what it did launch). The task owns the whole operator --
  every stage, every dispatch branch, both phases when the server ran both -- so add the
  cases or the call path that reach it. Waive one in `meta.ut_contract.launch_waivers`
  `[{kernel, phase, why}]` only when it cannot run from this entry (say why).

- `correctness` failing on an **unmodified** extraction means `call()`, a hook or a layout
  fact differs from the deployed call -- the candidate is the installed code, so the two
  legs should agree to the family bound.
- `server_calls` / `compile_variants`: the server compiled a Triton variant your cases do
  not reach, or ran a call your `call()` cannot express. `compile_variants.missing[].differs`
  names the field (`signature[M]: constexpr vs i32` is an int argument equal to 1; a
  `constexprs[...]` difference is a dispatch branch). Add a case that reaches it; waive it in
  `meta.ut_contract.compile_variant_waivers` only with the reason it cannot occur in
  deployment. `env_differs` means a module is missing from `meta.server_preimports`.
- `negative_controls`: a mutation must fail the compare by >= the margin and must run; a
  mutation that breaks compilation is not a control.
- `replay`, exit 3 / `UT_HARNESS_INCOMPLETE`: a graph phase needs `replay_bundle()`.

## Collectives

A task with `required_gpu_count` > 1 in `run/requested_kernel.json` is a collective on the
deployment's world size; the wave gives the lane that many GPUs. The generic driver runs
one process, so build the multi-rank harness of `references/collective-ut.md` inside the
task (every rank's output checked, the deployed graph mode, a numerical negative control)
and keep `unittest.py` as its entry. If it cannot be built here, that is a blocker naming
what is missing -- the kernel stays a selected target.

## Blockers

If the task cannot pass on this evidence -- the baseline itself fails its own family bound
for a reason you have pinned down, the entry in the request does not launch the kernels,
a structural fixture the family needs was never recorded -- stop and record it:

```bash
python3 $SKILL_ROOT/scripts/ut_wave.py block --task-dir $TASK_DIR \
    --reason "<what is missing or wrong, and the command whose output shows it>"
```

A vague reason ("doesn't work") is the one unacceptable outcome; nobody can act on it.
