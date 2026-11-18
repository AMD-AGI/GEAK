---
name: geak-lead
description: Lead a GEAK GPU kernel optimization run (v4) — run the search shell: each round observe the archive, run the active strategy's policy hook, reason with its STRATEGY.md, write and check a decision (mutate, fresh, combine, measure, investigate, ship, custom, stop), dispatch it to roles, and record outcomes. Strategies are pluggable packages under v4/search. Use when asked to optimize a kernel or task dir end to end.
---

# Lead (v4)

You run a search. The **state** is the archive of measured nodes plus the roofline cap. The **policy**
is a search template you read each round. Your **reasoning** turns state + policy into actions that roles
execute. Read `$SK/v4/COMMON.md` first. Evidence: `../EVIDENCE.md`.

## Paths and inputs
- `SK` = `kernel_skills/`; `KW` = `$SK/../kernel_workflow`. Inputs: `INPUT`, `GPUS`, optional `TASK`,
  `DEADLINE`, `TARGET`, `APPLY=1`, `SEARCH` (strategy package under `$SK/v4/search/`: `evo` default,
  `hillclimb` = v3 behaviour, `avo`; or your own).
- `RUN` holds: `ledger.md`, `SYNTHESIS.md`, `LEARNINGS.md`, `inbox.md`, `cands/`, `roles/`, `archive.json`,
  `decisions/round_<n>.json`.

## Roles (subagents: "Read `$SK/v4/<role>/SKILL.md` (or `RUN/roles/<role>/SKILL.md`) and follow it")
| role | in the search |
|---|---|
| geak-benchmark | fitness: official + validated fast probe, per-case rows with weights |
| geak-roofline | the cap per case; distance to it = where headroom is |
| geak-investigator | facts that change the state: budget, ceiling, instrument, deployment, prior-art |
| geak-engineer | `mutate` on one parent and one cut; `fresh` via `MODE=fresh-arch` |
| geak-ship | `combine` (merge or per-regime dispatch), attack, `ship` gate, final |
You may add roles or edit prompts (`RUN/roles/<name>/SKILL.md`, ≤ 100 lines; log under `## Harness changes`).

## Round 0 (in parallel)
geak-benchmark `MODE=setup`; geak-roofline; investigator `budget` and `deployment`. Then the round loop.

## Each round (the search shell, `$SK/v4/search/SHELL.md`)
`S=$SK/v4/search/$SEARCH` (or `RUN/search/<name>` for a strategy you wrote from `$SK/v4/search/_template`).
1. **Observe.** `python3 $SK/bin/archive.py build --run $RUN && python3 $SK/bin/archive.py views
   $RUN/archive.json --noise <NOISE> --cap $RUN/roofline/cap.json --md $RUN/archive_views.md`. Also read
   `SYNTHESIS.md`, `LEARNINGS.md`, `inbox.md`, the roofline report, and the past decisions with outcomes.
2. **Policy hook.** If `$S/policy.py` exists, run it:
   `python3 $S/policy.py --run $RUN --archive $RUN/archive.json --state $RUN/search/$SEARCH/state.json`.
   Its proposals are advice.
3. **Reason.** Read `$S/STRATEGY.md`. Diagnose the last round's failure mode, as in v3: false ceiling →
   investigate `ceiling`; flipping verdicts → `instrument`; ideas only from inside → `prior-art`; engineers
   quitting early → deliverables; verified work unshipped → ship first. Then write
   `decisions/round_<n>.json` (`$SK/v4/search/decision.schema.json`) with `strategy`, the proposal indices you
   used, and the actions. Every `why` cites the state it rests on; deviations from the strategy say why.
4. **Check.** `python3 $SK/bin/check_decision.py --run $RUN --decision $RUN/decisions/round_<n>.json
   --strategy $S --gpus <n>`. Fix every error before dispatching.
5. **Dispatch** (the shell's fixed mapping):
   | op | executor |
   |---|---|
   | `mutate` | geak-engineer: `PARENT`, `CUT`, `DIRECTION`, `REFS`, `RETAIN`, `TAGS` |
   | `fresh` | geak-engineer `MODE=fresh-arch` |
   | `combine` | geak-ship: `kind=merge|dispatch` |
   | `measure` | geak-benchmark `MODE=measure` (or `xover`) |
   | `investigate` | geak-investigator |
   | `ship` | geak-ship |
   | `custom` | `$S/roles/<role>.md` |
   | `stop` | the end rule below |
   Engineers get deliverables: a bound for their cut, ≥ 2 designs, and a result with a mechanism for each.
   New nodes carry the strategy's tags.
6. **Record.** Fill each action's `outcome` (new node ids, status, score delta, joined front, shipped).
   Rewrite `SYNTHESIS.md`, including what the strategy's choices paid.

## Ship rule (pre-registered in SYNTHESIS.md before measuring)
- Speed: beat the shipped candidate by more than the decision instrument's noise, and do not lose in the
  official eval. State the rule per regime when regimes differ in deployment weight.
- Speed-neutral fixes: non-inferiority plus a clean audit.
- Deployment vs score: losing up to `NOISE` on the score for a documented deployed-path win is allowed.
  More needs the user.
- A per-regime dispatch `combine` ships only if each regime's branch passed its own audit and the
  dispatch condition is a regime or size band, never one benchmark shape.

## Restart
Read `ledger.md`, `decisions/`, each role's `progress.md`, and detached GPU jobs. Resume roles; never redo
finished actions.

## Ending the run
End early only if all hold, with the evidence in `SYNTHESIS.md`:
1. A fresh geak-roofline rerun shows gap closed ≥ 80%, or an independent `investigator MODE=ceiling`
   explains each top-gap case with a mechanism measured on this hardware.
2. The archive's front has no node whose regime beats the shipped one beyond noise (the dispatch composite
   adds < NOISE), and every family worth a fresh attempt has had one.
3. geak-ship's audit passed, and the deployment path runs the shipped code.
Otherwise run to `DEADLINE` / `TARGET`. Then geak-ship `MODE=final` and write `RUN/final/report.md`:
headline, cap and gap closed, per-regime table, the archive front, the search trace (actions per round
and which ones paid off), and `## Harness notes`. Write `ENDED <time>` to `ledger.md`; reopen only for a
changed precondition, or for an extended deadline with gap < 80%.
