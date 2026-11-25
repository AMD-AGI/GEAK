# mode=mega on a region UT (args.region_ut)

A pipeline (e.g. KernelFusion for an L3 MegaKernel region, or the e2e kernel track) can hand mode=mega a
multi-rank operator as a region UT: a GEAK task dir whose `unittest.py` runs every rank. The caller passes
only `region_ut` (= KERNEL_PATH) and the GPU set; `tools/region_ut_args.py` turns the UT into the mega args
the caller left unset (benchmark, guards, accuracy, bound modules, positive control). This file changes only
what the UT fixes; everything else in your role and its mode=mega fragment holds.

## What the task dir fixes

- `meta.json`: the seam (`target_callable`), the modules a candidate may replace (`candidate_bind`, one entry
  per module, files under `kernel_src/`), `world_size` (ranks per run), `cases` (sig, step type, width,
  capacity, tolerance, served weight), `correct_metric`, `acceptance.min_speedup`, `deploy` (the live stack's
  PYTHONPATH and environment).
- `unittest.py` (with `cases.py`, `harness_lib.py`, `leg_runner.py`, `overlay_setup.py`): the benchmark. Two
  legs that differ only in the overlay: the baseline leg is the live stack (`baseline_overlay/`), the
  candidate leg is the same stack with every `candidate_bind` module taken from `kernel_src/`. It checks
  every rank's outputs against the baseline leg (eager and under one captured graph per case), times every
  case on every rank (the slowest rank counts), and prints `CORRECTNESS: PASS|FAIL`, `GRAPH_REPLAYS: <n>`
  (`meta.graph_replays`, the replay stress every rank passed), one line per case
  (`case <sig>: baseline_ms=... candidate_ms=... speedup=...`), `GEAK_WEIGHTED_SPEEDUP=` and
  `GEAK_GEOMEAN_SPEEDUP=`. Exit 0 = every rank correct and every case timed.
- Only `kernel_src/` may change. `unittest.py`, `cases.py`, the harness files, `meta.json`,
  `baseline_overlay/`, `baseline_ref/` and `controls/` are the acceptance test: never edit them.

## Roles

- **benchmark_engineer** (setup): the COMMANDMENT runs `python3 unittest.py` from the candidate tree under
  one lease of the whole GPU set (`python3 <GEAK>/e2e_workflow/scripts/gpu_lock_set.py <ids> python3
  unittest.py`). The baseline per case is the UT's baseline leg on the unchanged `kernel_src/`
  (`baseline_ms`); TARGET_GUARDS are the UT's case sigs. The positive control is a calibrated device-side
  delay added inside a `kernel_src/` copy of the seam, measured with the same `unittest.py`.
- **engineer**: the candidate tree is the task dir; change only files under `kernel_src/`. Before you return,
  also write `kernel_src/_handback/support.json` (`{"phases": [...], "widths" or "m_range": ...,
  "dtypes": [...], "refusals": [...]}`: where the implementation engages, and every case where it keeps the
  stock path, e.g. a capacity it falls back on) and `kernel_src/_handback/notes.md` (the design, per-case
  times, known limits). If the implementation needs an enable switch, default it on inside `kernel_src/`
  (`os.environ.setdefault(...)` in the seam's module); the server will not set it.
- **verify_engineer**: score with `unittest.py` exactly as the COMMANDMENT says; the per-case speedups it
  prints are the paired readings (rank-max), its CORRECTNESS line is the accuracy verdict per rank. Its
  `GRAPH_REPLAYS: <n>` line is the liveness evidence: every rank replayed each case's captured graph n
  times with refreshed inputs and every replay matched the baseline, without a hang. Report
  `liveness:"pass"` and `replay_count:<n>` when CORRECTNESS is PASS; never re-derive a replay count of your
  own. The UT captures the seam itself, so `graph_safe` is never "n/a" here, not even for a config-only
  change: `graph_safe:"pass"` when CORRECTNESS is PASS, else `"fail"`.
- **director / tech_lead**: the final patch is the `kernel_src/` diff (plus `kernel_src/_handback/`). That is
  the whole result: the caller applies it to the UT's `kernel_src/`, re-runs the same `unittest.py`, and
  binds `kernel_src/` into the server by replaying `candidate_bind` (e2e integrate).
