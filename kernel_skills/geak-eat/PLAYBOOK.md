# GEAK playbook for the EAT supervisor

This is about the harness, the measurements and team structures, not about kernel techniques. Finding
the optimizations is the team's job. Everything here is advice: the score and your own diagnosis win.

## What the eval gives you
- `bash eat/eval.sh [--ws <dir>]` = correctness + paired timing against the frozen baseline in a fresh
  copy, under a GPU lease. It is parallel-safe: unlike the default EAT setup, several GPU-using agents
  may work in the same phase, each in its own workspace. The number of GPUs in `$EAT_GPUS` caps how
  many evals run at once; extra ones wait in the lease queue.
- The noise band is in `eat/COMMANDMENT.md`. Treat a score inside it as a tie. Re-run before believing
  a small win, and do not let the team build on one.
- `solutions/results.tsv` + `solutions/<id>/` form a candidate tree (each row has a parent). The shipped
  kernel is just the current pick. A new line of work may start from any candidate.

## Measurement traps seen on these AMD tasks (have a role check them when a result looks odd)
- A win in an agent's own harness that does not show up in `eat/eval.sh` usually means a different
  regime: cache warm vs flushed, eager vs graph replay, autotune keyed without the shape, power or clock
  state. Ask for the regime difference, not a re-run.
- Two candidates scoring exactly equal, or a change with no effect at all, usually means the edited
  code is not the code that runs (a different dispatch path, a stale build, a second copy of a module).
  Check before calling it a dead end.
- Under CUDA/HIP graph replay, host-side savings are worth nothing. When the task deploys under graphs,
  judge host work by the replay timing.
- Exit 86 / `GEAK_SOURCE_INVALID` is a broken measurement, never a negative result.
- Profiles go stale after a structural change; re-profile before planning on them.

## Team shapes that have been useful (seeds to adapt, not a fixed menu)
- **Solo optimizer** with the whole task. A good first iteration and a good control.
- **Parallel branches**: 2–4 implementers on distinct ideas, each in its own workspace from a named
  parent candidate, then one **integrator/shipper** in a later phase that merges, evals and ships.
- **Analyst before implementers**: a profiler / roofline / workload analyst writes a short gap analysis
  (where the time goes vs what the hardware allows) that the implementers must read.
- **Auditor / challenger**: re-derives the best candidate's claim independently. Checks the measurement
  traps above, generalization beyond the eval shapes, and whether a declared ceiling is real.
- **Dead-end revisitor / fresh eyes**: takes a "closed" idea from `results.tsv` and asks what assumption
  closed it. Useful once the team starts declaring ceilings.
- **Knowledge synthesizer**: keeps `LEARNINGS.md` (facts with evidence) and `SYNTHESIS.md` (the current
  plan joining everyone's findings), and makes sure every next prompt points at them.
Patterns proven in earlier runs (evidence in `@SK@/v2/EVIDENCE.md`): measure where the time goes before
optimizing; challenge every declared ceiling with an independent probe; one fresh-architecture role at a design
bottleneck (plateau + one design family + headroom to the roof); cut work by regime / size band; validate the steering tool against an
official; ship first, every iteration; adversarial audit every iteration; map the deployed path early.
The v2 role skills under `@SK@/v2/` encode these and can be copied into role prompts.
Starter role prompts from the earlier fixed design (`geak-engineer`, `geak-integrate-verify` under
`@SK@`) may be copied into role prompts when they fit.

## Memory every team must keep
- `LEARNINGS.md`: what was measured, under which regime, with the eval line or log path.
- `SYNTHESIS.md`: the current best understanding and plan, rewritten (not appended) each iteration.
- `solutions/results.tsv`: every experiment, including failures, with the parent.
Each role prompt should say which of these it reads and which it writes.

## Lessons from earlier GEAK × EAT runs
@LESSONS@
