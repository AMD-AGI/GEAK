---
name: geak-lead
description: Lead a GEAK GPU kernel optimization run (v4) — treat optimization as a search over an archive of measured candidates; each round, read the archive and a search template (hillclimb, evo, avo), decide actions (mutate any node, crossover, fresh architecture, investigate, ship, stop) with reasons, and dispatch roles to execute them. Use when asked to optimize a kernel or task dir end to end.
---

# Lead (v4)

You run a search. The **state** is the archive of measured nodes plus the roofline cap. The **policy**
is a search template you read each round. Your **reasoning** turns state + policy into actions that roles
execute. Read `$SK/v4/COMMON.md` first. Evidence: `../EVIDENCE.md`.

## Paths and inputs
- `SK` = `kernel_skills/`; `KW` = `$SK/../kernel_workflow`. Inputs: `INPUT`, `GPUS`, optional `TASK`,
  `DEADLINE`, `TARGET`, `APPLY=1`, `SEARCH` (template name, default `evo`; `hillclimb` is the v3 behaviour).
- `RUN` holds: `ledger.md`, `SYNTHESIS.md`, `LEARNINGS.md`, `inbox.md`, `cands/`, `roles/`, `archive.json`,
  `decisions/round_<n>.json`.

## Roles (subagents: "Read `$SK/v4/<role>/SKILL.md` (or `RUN/roles/<role>/SKILL.md`) and follow it")
| role | in the search |
|---|---|
| geak-benchmark | fitness: official + validated fast probe, per-case rows with weights |
| geak-roofline | the cap per case; distance to it = where headroom is |
| geak-investigator | facts that change the state: budget, ceiling, instrument, deployment, prior-art |
| geak-engineer | mutation operator on one parent and one cut; `MODE=fresh-arch` = new island |
| geak-ship | crossover (code merge or per-regime dispatch), attack, selection gate, final |
You may add roles or edit prompts (`RUN/roles/<name>/SKILL.md`, ≤ 100 lines; log under `## Harness changes`).

## Round 0 (in parallel)
geak-benchmark `MODE=setup`; geak-roofline; investigator `budget` and `deployment`. Then read the template.

## Each round
1. **State.** `python3 $SK/bin/archive.py build --run $RUN && python3 $SK/bin/archive.py views
   $RUN/archive.json --noise <NOISE> --cap $RUN/roofline/cap.json --md $RUN/archive_views.md`. Also read
   `SYNTHESIS.md`, `LEARNINGS.md`, `inbox.md`, the roofline report, and the last round's decisions with
   their outcomes.
2. **Policy.** Read `$SK/v4/search/<SEARCH>.md` (or `RUN/search/<name>.md` if you wrote one). It says how to
   pick parents, how to split budget across actions, and when to restart, merge or stop. You may deviate;
   write why.
3. **Diagnose** the last round's failure mode, as in v3: false ceiling → investigator `ceiling`; flipping
   verdicts → `instrument`; ideas only from inside → `prior-art`; engineers quitting early → deliverables;
   verified work unshipped → ship first.
4. **Decide.** Write `decisions/round_<n>.json`: a list of actions, each one of
   - `{"op": "mutate", "parent": id, "cut": ..., "direction": ..., "why": ...}` → geak-engineer;
   - `{"op": "fresh", "cut": ..., "why": ...}` → geak-engineer `MODE=fresh-arch` (seeds a new family);
   - `{"op": "crossover", "parents": [ids], "kind": "merge|dispatch", "why": ...}` → geak-ship;
   - `{"op": "investigate", "mode": ..., "question": ..., "why": ...}` → geak-investigator;
   - `{"op": "ship", "id": ..., "why": ...}` → geak-ship;
   - `{"op": "stop", "why": ...}` (only under the end rule).
   Every `why` cites the archive view or measurement it rests on. Size the round to `GPUS` and the eval
   cost: with one GPU, at most two GPU-heavy actions at once.
5. **Dispatch** the actions. Engineers get deliverables (a bound for their cut, ≥ 2 designs, a result with
   a mechanism each). Children inherit or set `family`.
6. **Close the round.** geak-ship applies the ship rule. Add each action's outcome to its decision file
   (`"outcome": {...}`). Rewrite `SYNTHESIS.md`, including which policy choices paid off.

## Ship rule (pre-registered in SYNTHESIS.md before measuring)
- Speed: beat the shipped candidate by more than the decision instrument's noise, and do not lose in the
  official eval. State the rule per regime when regimes differ in deployment weight.
- Speed-neutral fixes: non-inferiority plus a clean audit.
- Deployment vs score: losing up to `NOISE` on the score for a documented deployed-path win is allowed.
  More needs the user.
- A per-regime dispatch crossover ships only if each regime's branch passed its own audit and the
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
