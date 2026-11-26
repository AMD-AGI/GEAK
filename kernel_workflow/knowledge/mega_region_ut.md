# mode=mega on a region UT (args.region_ut)

A pipeline (the e2e kernel track, or KernelFusion for an L3 MegaKernel region) can hand mode=mega an
operator as a GEAK UT task dir. The caller passes only `region_ut` (= KERNEL_PATH) and the GPU set;
`tools/region_ut_args.py` turns the UT into the mega args the caller left unset (benchmark, guards and their
noise floors, accuracy, replay count, bound module, positive control). This file changes only what the UT
fixes; everything else in your role and its mode=mega fragment holds.

## The two UT formats

Both are one task dir: `meta.json`, `cases.py`, `unittest.py`, the vendored harness files
(`harness_lib.py`, `leg_runner.py`, `overlay_setup.py`; a multi-rank UT adds its own rank launcher, e.g.
`mp_harness.py`), `kernel_src/`, `baseline_overlay/`, `baseline_ref/`, `run/`. The baseline leg is the
installed stack; the candidate leg binds the seam (`meta.target_callable`) to `kernel_src/` through
`meta.candidate_bind`. Tell them apart by `meta.cases` + `meta.acceptance`:

- **kernel_ut** (e2e kernel extraction, the kernel-ut skill): cases in `meta.workload.cases` (`sig`, `regime`,
  `effective_calls`); tolerance `meta.tol`; graph regimes in `meta.phase_regimes`. `unittest.py` runs the
  gates (identity, correctness with a refreshed-input graph replay per graph case, negative controls), then
  times both legs; it prints `GEAK_WEIGHTED_SPEEDUP=` and a final `{"UT": "pass"|"fail", ...}` line, and
  writes the per-case times to `run/ut_result.json` (`gates.timing.per_case`: `sig`, `baseline_ms`,
  `optimized_ms`). If its argparse has `--validation-mode`, run it with `--validation-mode performance`.
- **fusion_contract** (KernelFusion, `fusion_ut_contract.md`): cases in `meta.cases` (`sig`, `step_type`,
  `width`, served `weight`, `regime` eager|graph); `meta.acceptance` (`tol` with `tol_units`, `noise_bound`,
  `min_speedup`). `unittest.py` runs every rank and prints `IDENTITY_RESOLVE`, one gate line per mode
  (`ORACLE`/`RANDOM`/`SERVED`, `SELF` with `--self-noise`: `<MODE>: PASS|FAIL max_err=... tol=...`), one line
  per case (`case <sig>: <baseline> ms / <candidate> ms / <speedup>x`, slowest rank),
  `CORRECTNESS: PASS|FAIL max_err=... tol=... per_rank=[...]` and `GEAK_WEIGHTED_SPEEDUP=`; per case also in
  `run/ut_result.json`. Each graph case is captured once per leg, checked after an in-place input refresh,
  and replayed `--iters` times per leg while timed.

Ranks per run: `python3 <GEAK>/kernel_workflow/scripts/detect_gpu_requirement.py --declaration <task>`
(`required_gpu_count` from `run/RETURN.json` / `run/requested_kernel.json`, or
`meta.capabilities.multi_rank.world_size`). Only `kernel_src/` may change; everything else in the task dir
is the acceptance test: never edit it.

## Roles

- **benchmark_engineer** (setup): the COMMANDMENT runs `unittest.py` (with the flag above) from the candidate
  tree under one lease of the UT's GPU count (`python3 <GEAK>/e2e_workflow/scripts/gpu_lock_set.py <ids> --
  python3 unittest.py ...`). The baseline per case is the UT's baseline time on the unchanged `kernel_src/`;
  TARGET_GUARDS are the UT's case sigs. The null arm is the baseline against itself (`--self-noise` where the
  UT has it, else `run/self_noise.json`). The positive control is a calibrated device-side delay added inside
  a `kernel_src/` copy of the seam, measured with the same `unittest.py`.
- **engineer**: the candidate tree is the task dir; change only files under `kernel_src/`, and keep consuming
  what the UT's notes say the seam must consume (e.g. a frozen routing). Before you return, also write
  `kernel_src/_handback/support.json` (`{"phases": [...], "widths" or "m_range": ..., "dtypes": [...],
  "refusals": [...]}`: where the implementation engages, and every case where it keeps the stock path) and
  `kernel_src/_handback/notes.md` (the design, per-case times, known limits). If the implementation needs an
  enable switch, default it on inside `kernel_src/`; the server will not set it.
- **verify_engineer**: score with `unittest.py` exactly as the COMMANDMENT says; its per-case times are the
  paired readings, its correctness verdict (`CORRECTNESS` line, or the `{"UT": ...}` line and exit code) is
  the accuracy verdict against ACCURACY_THRESHOLD in the UT's own error units. Liveness is the UT's own graph
  replay: when the UT passes, report `liveness:"pass"` and `replay_count` = the UT's `--iters` when it has that
  option (default in its argparse), else 1 (one refreshed-input replay per graph case). The UT captures the seam itself, so
  `graph_safe` is never "n/a" here, not even for a config-only change: `"pass"` when the UT passes, else
  `"fail"`.
- **director / tech_lead**: the final patch is the `kernel_src/` diff (plus `kernel_src/_handback/`). That is
  the whole result: the caller applies it to the UT's `kernel_src/`, re-runs the same `unittest.py`, and binds
  `kernel_src/` into the server through `candidate_bind`.
