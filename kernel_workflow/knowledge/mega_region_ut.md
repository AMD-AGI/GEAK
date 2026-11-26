# mode=mega on a region UT (args.region_ut)

A caller (e.g. the e2e kernel track, or a user calling kernel_workflow directly) can hand mode=mega an
operator as a UT task dir and the GPU set. The UT may be laid out in any way: mode=mega does not require a
fixed format. Before any other role, the region UT adapter reads the UT and writes ONE unified contract,
`<region_ut>.mega_ut.json` (beside the UT dir, never inside it). Every role then works from that contract;
this file changes only what the UT fixes, everything else in your role and its mode=mega fragment holds.

## Adapter (label region_ut, before Setup)

Read the UT however it is laid out: `meta.json` (or whatever declares the cases), `cases.py`, `unittest.py`
and the harness it calls, its README/notes, and its own evidence under `run/` (a passing result, a
self-noise record, the declared GPU count). Do not run it on a GPU; CPU-only probes (`--help`, imports of
pure metadata, a dry parse) are fine. `python3 <WORKFLOW_DIR>/tools/region_ut_args.py <region_ut>` reads the
layouts GEAK already produces; when it prints a contract, start from it and correct whatever the UT says
otherwise; when it prints `{"error"}`, derive every field yourself.

Write the contract and return the same JSON. Its top level is the mega args (they fill only what the caller
left unset):

| field | meaning |
|---|---|
| `benchmark_harness` | the file that runs the UT (its `unittest.py`, or your wrapper) |
| `candidate_import_modules` | the module(s) a candidate replaces (what the UT binds into the candidate leg) |
| `target_guards` | `[sig]`: the case with the largest served weight (call mass); one target |
| `regression_guards` | every other timed case sig |
| `regression_floor_by_guard` | `{sig: floor}`: 0.98, lowered to `1 - 2·abs(1 - ratio)` where the UT's own baseline-vs-baseline record shows that case noisier |
| `required_accuracy_cases` | every case sig the UT checks |
| `accuracy_metric`, `accuracy_threshold` | the UT's own error measure and its bound, in the UT's units |
| `required_replays` | how many captured-graph replays per graph case one passing UT run performs (its replay or timing iterations); omit when the UT replays no graph |
| `require_graph_capture` | `"false"` when the UT captures and replays the seam itself |
| `positive_control` | `{name, guard: <target>, how: "a calibrated device-side delay inside a copy of the seam in the editable files, measured with the UT", expected_pct_lo: -9, expected_pct_hi: -4, magnitude: "constructed", abort_on_fail: false, required_control_pairs: 5, required_null_pairs: 8}` |

and `ut`, how every role runs and reads this UT:

| field | meaning |
|---|---|
| `run` | `{command, cwd, gpus, timeout_s}`: the exact command (under `gpu_lock_set.py <ids> --`) and the number of GPUs one run needs |
| `editable` | the files a candidate may change (e.g. `kernel_src/`); everything else is the acceptance test |
| `verdict` | where the run says pass/fail on correctness, per rank when multi-rank (a line, an exit code, a JSON field) |
| `per_case` | where each case's baseline and candidate time come from (a line pattern or a JSON path), and that the slowest rank counts |
| `headline` | where the weighted speedup comes from |
| `replay` | where the replay evidence is and how it was counted |
| `null_arm` | how to measure the baseline against itself (a flag, or two baseline runs) |
| `notes` | what a candidate must keep (e.g. a frozen routing it must consume), anything unusual |

Adapt, do not refuse: if the UT lacks something the contract needs (per-case times only in a JSON file, a
command that needs a flag for timing), say in `ut` how to get it; if it lacks it entirely (no case split, no
baseline leg in the same run), write a thin wrapper beside the UT (`<region_ut>.mega_ut/`, never inside the
UT) that runs the UT's own code and produces it, and point `benchmark_harness` and `ut.run` at it. Never
weaken the UT's own correctness check or tolerance. Return `{"error": "<why>"}` only when the UT cannot give
a correctness verdict and a time for both legs at all.

## Roles

- **benchmark_engineer** (setup): the COMMANDMENT is `ut.run.command` from the candidate tree under one lease
  of `ut.run.gpus` GPUs. First prove the contract on that first run, on the unchanged editable files: the
  verdict reads PASS (on every rank), every case in `required_accuracy_cases` gets a baseline and a candidate
  time from `ut.per_case`, the headline reads where `ut.headline` says, and the replay evidence is there with
  at least `required_replays` replays. If any of these does not hold, return no baseline and say which
  (the run stops in Setup: a search on a wrong contract measures nothing); a contract field you corrected
  from the run goes into your report. Then the baseline per case is the UT's baseline time on the unchanged
  editable files; TARGET_GUARDS are the contract's case sigs. The null arm is `ut.null_arm`. The positive
  control is the contract's `positive_control`, measured with the same command.
- **engineer**: the candidate tree is the task dir; change only `ut.editable`, and keep what `ut.notes` says a
  candidate must keep. Before you return, also write `_handback/support.json` inside the editable dir
  (`{"phases": [...], "widths" or "m_range": ..., "dtypes": [...], "refusals": [...]}`: where the
  implementation engages, and every case where it keeps the stock path) and `_handback/notes.md` (the
  design, per-case times, known limits). If the implementation needs an enable switch, default it on inside
  the editable files; the server will not set it.
- **verify_engineer**: score with `ut.run.command` exactly; the per-case times (`ut.per_case`) are the paired
  readings, the verdict (`ut.verdict`) is the accuracy verdict against ACCURACY_THRESHOLD in the UT's units.
  Liveness is the UT's own graph replay (`ut.replay`): when the UT passes, report `liveness:"pass"` and
  `replay_count` = the contract's `required_replays`. The UT captures the seam itself, so `graph_safe` is
  never "n/a" here, not even for a config-only change: `"pass"` when the UT passes, else `"fail"`.
- **director** (validate): when `benchmark_harness` is a wrapper the adapter wrote, also run the UT's own
  command (its `unittest.py` or equivalent, without the wrapper) on the final patch; its verdict and
  headline are the ones that count, and a disagreement with the wrapper rejects the patch.
- **director / tech_lead**: the final patch is the diff of `ut.editable` (plus `_handback/`). That is the whole
  result: the caller applies it to the UT, re-runs the same command, and binds the editable files into the
  server the way the UT binds its candidate leg.
