# mode=mega on a region UT (args.region_ut)

A pipeline (e.g. KernelFusion for an L3 MegaKernel region) can hand mode=mega a multi-rank operator as a
region UT: the GEAK task dir KernelFusion builds for a fusion region (e2e_workflow
`roles/_fragments/fusion_ut_contract.md`). The caller passes only `region_ut` (= KERNEL_PATH) and the GPU
set; `tools/region_ut_args.py` turns the UT into the mega args the caller left unset (benchmark, guards and
their noise floors, accuracy, replay count, bound module, positive control). This file changes only what the
UT fixes; everything else in your role and its mode=mega fragment holds.

## What the task dir fixes

- `meta.json`: the seam (`target_callable`), the module a candidate replaces (`candidate_bind`, its file
  under `kernel_src/`), `required_gpu_count` (ranks per run; also in `run/RETURN.json`), `cases` (sig,
  step_type, width, context, served `weight`, `effective_calls`, `regime` eager|graph), `acceptance`
  (`tol` and `tol_units`: the error measure and its bound; `noise_bound`; `min_speedup`).
- `unittest.py` (with `cases.py`, `mp_harness.py`, `harness_lib.py`, `leg_runner.py`, `overlay_setup.py`): the
  benchmark. Two legs that differ only in the overlay: the baseline leg is the installed stack, the
  candidate leg rebinds the seam to `kernel_src/`. It runs every rank (one process per rank, mp_harness)
  and prints `IDENTITY_RESOLVE`, then one gate line per mode (`ORACLE`, `RANDOM`, `SERVED`, and `SELF` with
  `--self-noise`: `<MODE>: PASS|FAIL max_err=... tol=...`), one line per served case
  (`case <sig>: <baseline> ms / <candidate> ms / <speedup>x`, the slowest rank's time per call),
  `CORRECTNESS: PASS|FAIL max_err=... tol=... per_rank=[...]`, `GEAK_GEOMEAN_SPEEDUP=` and
  `GEAK_WEIGHTED_SPEEDUP=` (Σ calls·baseline / Σ calls·candidate). `run/ut_result.json` holds the same per
  case. Exit 0 = identity, every gate on every rank and every case timed.
- Graph-regime cases (target verify, draft) are captured once per leg with different values in the static
  input, checked after an in-place input refresh, and replayed `--iters` times per leg while timed.
- Only `kernel_src/` may change. `unittest.py`, `cases.py`, the harness files, `meta.json`,
  `baseline_overlay/`, `baseline_ref/`, `reference_io/`, `weights/` and `controls/` are the acceptance
  test: never edit them.

## Roles

- **benchmark_engineer** (setup): the COMMANDMENT runs `python3 unittest.py` from the candidate tree under one
  lease of `required_gpu_count` GPUs (`python3 <GEAK>/e2e_workflow/scripts/gpu_lock_set.py <ids> -- python3
  unittest.py`). The baseline per case is the UT's `<baseline> ms` on the unchanged `kernel_src/`;
  TARGET_GUARDS are the UT's case sigs. The null arm is `--self-noise` (baseline against itself,
  `run/self_noise.json`). The positive control is a calibrated device-side delay added inside a
  `kernel_src/` copy of the seam, measured with the same `unittest.py`.
- **engineer**: the candidate tree is the task dir; change only files under `kernel_src/`. Keep consuming
  what the UT's `acceptance` notes say the seam must consume (e.g. a frozen routing). Before you return,
  also write `kernel_src/_handback/support.json` (`{"phases": [...], "widths" or "m_range": ...,
  "dtypes": [...], "refusals": [...]}`: where the implementation engages, and every case where it keeps the
  stock path) and `kernel_src/_handback/notes.md` (the design, per-case times, known limits). If the
  implementation needs an enable switch, default it on inside `kernel_src/`; the server will not set it.
- **verify_engineer**: score with `unittest.py` exactly as the COMMANDMENT says; its per-case speedups are
  the paired readings (slowest rank), its CORRECTNESS line is the accuracy verdict per rank (`max_err`
  against `tol`, in `acceptance.tol_units`). Liveness: when SERVED passes, every rank replayed each graph
  case's captured graph `--iters` times per leg without a hang, after a refreshed-input replay that matched
  the baseline; report `liveness:"pass"` and `replay_count:<iters>` (the UT's `--iters`, default in its
  argparse). The UT captures the seam itself, so `graph_safe` is never "n/a" here, not even for a
  config-only change: `graph_safe:"pass"` when CORRECTNESS is PASS, else `"fail"`.
- **director / tech_lead**: the final patch is the `kernel_src/` diff (plus `kernel_src/_handback/`). That is
  the whole result: the caller applies it to the UT's `kernel_src/`, re-runs the same `unittest.py`, checks
  `acceptance.min_speedup`, and binds `kernel_src/` into the server through `candidate_bind`.
