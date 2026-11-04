---
name: stub-roofline
description: Copy configured CPU fixture evidence into the roofline slot to exercise its artifact contract without an analysis tool or agent session.
---

# stub-roofline

Performs no analysis. Copy `rows.json` and `folded.json` from `params.fixture_dir` to
`$OUT`; the slot's `finalize.py` produces the ranking and verdict. Paths resolve against
the GEAK repository root. The fixture must describe the tasks, case signatures and
selection shares in the upstream profile delivery; see `../README.md` for the interface.

It is the **executable proof** of §5 of `stages/CONTRACT.md`: a new implementation is a
directory, a `SKILL.md` (plus `run.sh`), and config — with no registry to
touch. Swap it for `kernel-roofline/` and the other four stages need no edit at all.

```bash
python3 workload/run.py --config <cpu-fixture-config> \
    --set stages.roofline.skill=stub-roofline --set stages.roofline.mode=script \
    --set stages.roofline.params.fixture_dir=<fixture-dir> run roofline
```

Use a separate CPU smoke run with `target.enforce=false` and a written fixture-only
reason. Keep synthetic fixtures separate from measured evidence and never report their
ranking as a performance result. This implementation ships no `scripts/fold.py`: its
ability to pass the same finalizer demonstrates that the contract is the artifacts.
