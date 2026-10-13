---
name: stub-roofline
description: A placeholder roofline implementation returning a fixed ranking. It exists to demonstrate that replacing an implementation is a one-line config change.
---

# stub-roofline

Performs no analysis; emits a fixed `roofline.json`.

It is the **executable proof** of §5 of `stages/CONTRACT.md`: a new implementation is a
directory, a `SKILL.md` (plus `run.sh`), and one line of config — with no registry to
touch. Swap it for `kernel-roofline/` and the other four stages need no edit at all.

```bash
python3 workload/scripts/stage.py --config <cfg> --set stages.roofline.skill=stub-roofline run roofline
```

Its artifact conventions match `kernel-roofline` (a `kernels` list in `roofline.json`,
descending by `expected_e2e_gain_pct`), so the downstream optimize stage consumes it as
an optimization order exactly as usual — except that this particular order rests on
nothing, which `caveats` states outright.
