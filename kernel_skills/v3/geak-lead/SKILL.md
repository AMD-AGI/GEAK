---
name: geak-lead
description: Lead a GEAK GPU kernel optimization run (v3) — compute the design-independent cap first, compose each round's team from six roles, branch from any candidate, challenge every declared ceiling, ship continuously, survive restarts, and end only when the gap to the cap is closed or independently explained. Use when asked to optimize a kernel or task dir end to end.
---

# Lead (v3)

You own strategy and the run, not the code. Roles work and leave results in files; you decide the next
round and when the run is over. Read `$SK/v3/COMMON.md` first. Evidence for each rule: `../EVIDENCE.md`.

## Paths and inputs
- `SK` = `kernel_skills/` (two levels above this file); `KW` = `$SK/../kernel_workflow`.
- `INPUT` (never edited), `GPUS`, optional `TASK`, `MAX_ROUNDS`, `DEADLINE`, `TARGET`, `APPLY=1`.
- `RUN` = the user's choice, else `$SK/../exp/<basename>_<timestamp>`. Files: `ledger.md` (yours),
  `SYNTHESIS.md` (rewritten by you each round), `LEARNINGS.md` and `inbox.md` (everyone), `cands/`, `roles/`.

## Roles: subagents told "Read `$SK/v3/<role>/SKILL.md` (or `RUN/roles/<role>/SKILL.md`) and follow it"
| role | job |
|---|---|
| geak-benchmark | instrument sized to the eval's cost; official result; validated fast probe; cross-overs |
| geak-roofline | design-independent cap on the score; gap closed |
| geak-investigator | `budget`, `ceiling`, `instrument`, `deployment` (+ UT blind spots), `prior-art` |
| geak-engineer | owns one cut from one parent; `MODE=fresh-arch` designs blind to the shipped code |
| geak-ship | merge, attack, apply the ship rule, final verdict |
You may add roles or edit any prompt. Save it as `RUN/roles/<name>/SKILL.md` (≤ 100 lines) and log it
under `## Harness changes` in `ledger.md`, with what, why, and later whether it paid off.

## Start (round 0, in parallel)
geak-benchmark `MODE=setup`; geak-roofline (cap with the baseline); investigator `MODE=budget` and
`MODE=deployment`. These need only `RUN/orig` and the harness. Do not wait for setup to finish. Then
engineers on the cuts that hold the largest weighted gap to the cap.

## Scale to resources
Read `GPUS` and the eval cost that benchmark measured. With one GPU, run at most two GPU-heavy roles
at once. When one official run takes > ~10 minutes, decisions use the validated fast probe, and official
runs only confirm (geak-benchmark "slow-eval mode"). Give each role a time box that ends before `DEADLINE`.

## Each round
1. Read `SYNTHESIS.md`, `LEARNINGS.md`, `inbox.md`, the candidate table, the roofline report and every
   role's notes. Diagnose the last round's failure mode:
   - **a declared ceiling** measured with the team's own probe or design → `investigator MODE=ceiling`;
   - **one design family, ≥ 5% weighted gap to the cap** → `engineer MODE=fresh-arch`. Dispatch it after a
     2-round plateau, or at once if less than half the time is left;
   - **verdicts near noise or flipping** → `investigator MODE=instrument`;
   - **ideas only from inside the team** → `investigator MODE=prior-art`, then an engineer who ports one
     technique;
   - **engineers stop early with "tie"** → deliverables, not stop rules;
   - **verified work not shipped** → ship first.
2. Cut work by ownership: regime, size band, measured cost component, or the ISA instruction list each
   engineer owns. A role whose biggest measured cost lies in another cut tells the owner via `inbox.md`.
3. Choose parents on purpose. Engineers may also deliver small "option" patches on top of the shipped node.
4. Every engineer delivers: a measured bound for its cut, ≥ 2 designs, and a result with its mechanism
   for each.
5. geak-ship every round. In the last third of the run, run a checkpoint `MODE=final` as soon as
   something has shipped.
6. Write the decision, roles, parents and reasoning to `ledger.md` before dispatching.

## Ship rule (pre-registered in SYNTHESIS.md before measuring)
- Speed: beat the shipped candidate by more than the decision instrument's noise, and do not lose in the
  official eval. State the rule per regime when regimes differ in deployment weight.
- Speed-neutral fixes: non-inferiority plus a clean audit.
- Deployment vs score: a change that wins on the deployed path but loses up to `NOISE` on the score
  ships only with the deployment evidence written down. A larger score loss needs the user's decision.
- If a rule blocks a real, audited gain twice, the rule is wrong: fix it and log why.

## Restart (you were resumed, or roles vanished)
Read `ledger.md`, each role's `progress.md`, and the last file write in each role's dir. Check for
detached GPU jobs (pid / log). Resume a role via its transcript, or start a fresh one from its
`progress.md`. Do not redo finished work.

## Ending the run
End early only if **all** hold, with the evidence in `SYNTHESIS.md`:
1. A **fresh geak-roofline rerun** shows **gap closed ≥ 80%**. Otherwise, for the remaining gap: an
   `investigator MODE=ceiling` from a role that did not build the shipped design has measured the
   specific mechanism that blocks each top-gap case on this hardware. A bound built from the shipped
   design's own costs does not count.
2. Every lever inside the editable code is closed with a measured mechanism. A fresh-arch design landed in
   the same family or lost.
3. geak-ship's audit passed, and the deployment path runs the shipped code.
Otherwise run to `DEADLINE` / `MAX_ROUNDS` / `TARGET`. Then geak-ship `MODE=final` and write
`RUN/final/report.md`: headline from `final/verify.json`, the cap and gap closed, per-case and per-regime
tables, the candidate tree, closed levers, deployment status, and `## Harness notes` (roles per round,
changes you made, what to change next). Add reusable lessons per `$KW/knowledge/learned/README.md`.
After that, write `ENDED <time>` to `ledger.md`. If resumed after `ENDED`, reopen only for closures
whose precondition has changed, or if the deadline was extended and the gap is < 80%.
