# kernel-roofline — install and what is in the box

A Claude Code skill for GPU kernel/operator roofline analysis in three explicit
modes. `SKILL.md` is the entry point and the mode-selection table; everything else
hangs off it.

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
| `tl-only` | A working TraceLens install. **No GPU.** |
| `rocprof` | `rocprof-compute` + a GPU + a runnable per-case command. |
| `combination` | Both. |

`scripts/roofline_report.py` and its synthetic example need only the standard
library — that is the cheapest way to confirm the package unpacked intact:

```bash
python3 scripts/roofline_report.py --mode extended \
  --input assets/example_cases.json --arch assets/example_arch.json --output /tmp/krf
# -> Wrote 2 rows to /tmp/krf
```

See `references/installation.md` for the real installs.

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
references/           tl-only.md  rocprof.md  combination.md  (one per mode)
                      folding.md                 per-shape -> per-kernel, all modes
                      modeling.md  corrections.md  measurement.md  report-input.md
                      installation.md  pitfalls.md
scripts/              fold.py                    THE shared fold: per-shape -> per-kernel
                      run_roofline.py            patched rocprof-compute driver
                      rocprof_leg_template.sh    batch driver — copy and edit CONFIG
                      case_runner_template.py    one-shape-per-command runner + arm guard
                      join_legs.py               fold rocprof ratios onto TL's time
                      compare_legs.py            AI / bound axis side by side
                      rerender.py repeak.py      offline re-score, no GPU
                      new_mibench.py             standalone microbench -> roofline.csv
                      rocprof_hbm_patch.py       gfx950 HBM-read backport
                      roofline_report.py         standalone calculator
assets/               arch JSON variants + md5/source README, synthetic example
examples/krf_0927/    a finished two-engine two-arm campaign (see below)
```

## The worked example

`examples/krf_0927/` is a real campaign over nine seams of a production model, run
on MI355X, both arms, both engines. Start at `combination/REPORT.md`.

The `rocprof/leg/*_roofline.txt` files are included, so the join is reproducible
offline with no GPU:

```bash
python3 scripts/join_legs.py --rocprof-dir examples/krf_0927/rocprof/leg --arm candidate \
    --tl-rows examples/krf_0927/tl-only/rows_candidate.json --out /tmp/join.json
```

That must reproduce `combination/join_candidate.txt` line for line, ending with
`[48 of 56 TL rows had no rocprof file -- this table covers 8 case(s), NOT the whole
leg]`. The TL leg covered every case; the rocprof leg deliberately covered the
dominant case per seam, and the coverage line is how you see that rather than
assuming the table is complete. It is also a working regression test for the join.

Not included, to keep the archive small: the `*_roofline_raw.txt` files (which is
what `rerender.py` / `repeak.py` consume), the per-run logs, and the case runners.
`tl_leg_*.py` / `tl_compare.py` were campaign-specific seam drivers and are not part
of this package — the per-seam driving is yours to write, the folding rules are not.

The numbers in that report are internal kernel measurements from 2026-09-27/28.
Treat the archive accordingly.
