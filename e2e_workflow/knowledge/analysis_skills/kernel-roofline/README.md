# kernel-roofline — install and what is in the box

A Claude Code skill for GPU kernel/operator roofline analysis in two explicit
modes. `SKILL.md` is the entry point and the mode-selection table; everything else
hangs off it.

In GEAK this directory is already bundled and selected by default through
`analysis_skill: "kernel-roofline"` (legacy `"roofline"` is an alias). See
[the GEAK integration contract](references/geak-integration.md) for the Profiler
and Architect handoff. The installation below is for standalone use.

## Install

```bash
tar xzf kernel-roofline.tar.gz -C /some/shared/place
mkdir -p ~/.claude/skills
ln -s /some/shared/place/kernel-roofline ~/.claude/skills/kernel-roofline
```

Start a new Claude Code session; `kernel-roofline` should appear in the skill list.
A symlink is preferred over a copy so that everyone picks up the same package, but a
copy works — nothing in `scripts/` or the docs depends on an absolute path.

## Prerequisites, by mode

| Mode | Needs |
| --- | --- |
| `tl-only` | Latest upstream TraceLens `main`, resolved and frozen per analysis. Existing-trace/model-only analysis needs no GPU; new capture does. |
| `rocprof` | `rocprof-compute` + a GPU + a runnable per-case command. |

`scripts/roofline_report.py` and its synthetic example need only the standard
library — that is the cheapest way to confirm the package unpacked intact:

```bash
python3 scripts/roofline_report.py --mode extended \
  --input assets/example_cases.json --arch assets/example_arch.json --output /tmp/krf
# -> Wrote 2 rows to /tmp/krf
```

See `references/installation.md` for the real installs and
`references/runner.md` for the implemented measurement/counter/TL workflow.
Use `case_runner_template.py --capture` followed by `tl_only.py` for TL, or the
same runner under `run_roofline.py` for counters. Old campaign timing/model drivers
are not used. Run `python3 -m unittest discover -s tests -v` for CPU regressions;
GPU validation is separately available through `tests/gpu_smoke.py`.
For TL review, generate with `--include_call_stack` and pass the complete selected
`unified_perf_summary.csv` row. Its `name` selects the operation model; device
names live in `kernel_details_summary`. See `references/unified-summary.md` for
`perf_params`, per-kernel stacks, version-dependent schemas, and the row extractor.

## Two gates you will hit on gfx95x, read before your first run

1. **rocprof-compute version.** The microbench in builds below 3.6.0 issues CDNA3
   MFMA on CDNA4, so bf16/fp16/int8 empirical peaks come out ~4x low. If you cannot
   upgrade the binary, you are **not** blocked: its *counters* are fine, only its
   *microbench* is wrong. Seed a `roofline.csv` of independently measured ceilings —
   `references/rocprof.md` §1–2 has the procedure and the two cheap checks that a CSV
   really carries corrected ceilings. Record which route your ceilings took; "3.4.0"
   alone is not provenance.
2. **gfx950 L2→HBM read bytes are halved** by three independent copies of a formula
   that infers 128B requests from `TCC_BUBBLE`, which is identically 0 on this part.
   `scripts/rocprof_hbm_patch.py` carries the backport. `references/rocprof.md` §3.

`references/pitfalls.md` is the one-page index of every trap in this package. It is
worth ten minutes before the first campaign and is the right place to look when a
number is surprising.

## Layout

```
SKILL.md              mode selection, the cross-mode rules, provenance requirements
references/           tl-only.md  rocprof.md          (one per mode)
                      folding.md                 per-shape -> per-kernel, all modes
                      modeling.md  corrections.md  measurement.md  report-input.md
                      installation.md  pitfalls.md  unified-summary.md
scripts/              fold.py                    THE shared fold: per-shape -> per-kernel
                      extract_tl_row.py          native row + per-kernel stack context
                      run_roofline.py            patched rocprof-compute driver
                      rocprof_leg_template.sh    batch driver — copy and edit CONFIG
                      case_runner_template.py    shared one-case measurement runner
                      measurement_runtime.py     owned cache/NUMA/graph lifecycle
                      trace_scope.py rocprof_window.py   measured-window selection
                      tl_only.py                 latest-main native report driver
                      collect_results.py         preserve requested case coverage
                      dispatch_probe.py          diagnostic only, never a bound
                      rerender.py repeak.py      offline re-score, no GPU
                      new_mibench.py             standalone microbench -> roofline.csv
                      rocprof_hbm_patch.py       gfx950 HBM-read backport
                      roofline_report.py         standalone calculator
assets/               arch JSON variants + md5/source README, synthetic example
examples/krf_0927/    independent tl-only and rocprof examples (see below)
```

## The worked example

`examples/krf_0927/` contains a real MI355X campaign with baseline and
candidate results, kept separately for each mode:

- `tl-only/COMPARE.md` and `tl-only/rows_*.json`: historical campaign TL-based results and per-case data, not current-main native coverage evidence.
- `rocprof/leg/*_roofline.txt`: independent per-kernel counter reports.

The TL results cover all modeled cases; the rocprof reports cover the
dominant case per seam. Preserve those separate coverage scopes. To reproduce
the standalone TL rollup without a GPU:

```bash
python3 scripts/fold.py --rows examples/krf_0927/tl-only/rows_candidate.json --label TL
```

Not included, to keep the archive small: the `*_roofline_raw.txt` files (which is
what `rerender.py` / `repeak.py` consume), the per-run logs, and the case runners.
`tl_leg_*.py` / `tl_compare.py` were campaign-specific seam drivers and are not part
of this package — the per-seam driving is yours to write, the folding rules are not.

The numbers in these reports are internal kernel measurements from 2026-09-27/28.
Treat the archive accordingly.
