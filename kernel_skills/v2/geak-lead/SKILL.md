---
name: geak-lead
description: Lead a GEAK GPU kernel optimization run (v2) — measure first, compose each round's team from five roles, branch from any candidate, challenge every declared ceiling, ship continuously, and end the run on independent evidence. Use when asked to optimize a kernel or task dir end to end.
---

# Lead (v2)

You own strategy and the run, not the code. Roles do the work and leave results in files; you
read the files, decide the next round, and decide when the run is over. No fixed round count.
v2 encodes what worked across 4 kernels and 35 evolved team iterations (see `../EVIDENCE.md`).

## Paths and inputs
- `SK` = `kernel_skills/` (two levels above this file). `KW` = `$SK/../kernel_workflow`.
- `INPUT` (never edited), `GPUS`, optional `TASK`, `MAX_ROUNDS`, deadline, `TARGET`, `APPLY=1`.
- `RUN` = the user's choice, else `$SK/../exp/<basename>_<timestamp>`. If `RUN/ledger.md` exists, resume.
- Layout as in v1 (`$SK/geak-techlead/SKILL.md` "Run layout"), plus `RUN/LEARNINGS.md` (facts with
  evidence, appended live by every role) and `RUN/SYNTHESIS.md` (current understanding + plan,
  rewritten by you each round). Every role prompt names which of them it reads and writes.

## Roles (run each as a subagent: "Read `$SK/v2/<role>/SKILL.md` and follow it", then KEY=value lines)
| role | job |
|---|---|
| geak-benchmark | setup the instrument; official result per candidate; cross-over pairs |
| geak-investigator | `MODE=budget` (where the time goes), `ceiling`, `instrument`, `deployment`, `prior-art` |
| geak-engineer | one cut from one parent; `MODE=fresh-arch` designs a new architecture without reading the current best |
| geak-ship | integrate candidates, adversarial audit, pre-registered ship decision, final verdict |
Trust only files. Start independent roles in one message. Without subagents, run the skills yourself.

## Start
1. Materialize `RUN/orig`, record `RUN/input_path.txt`, run geak-benchmark `MODE=setup`.
2. Round 1 is always: geak-investigator `MODE=budget` and `MODE=deployment` in parallel with one
   engineer on the most obvious lever. No optimization plan before the time budget exists.

## Each round
1. Read `SYNTHESIS.md`, `LEARNINGS.md`, the candidate table, and every role's notes. Diagnose the
   failure mode of the last round before choosing roles. The ones seen most often:
   - **a declared ceiling** ("at the HBM roof", "fixed X µs", "lens exhausted") measured with the team's
     own probe → `investigator MODE=ceiling`: independent roof (best raw probe, vendor kernels, ISA);
   - **a design bottleneck** → one `engineer MODE=fresh-arch` (rules below);
   - **gains near the noise band, or verdicts that flip** → `investigator MODE=instrument`;
   - **the search stays inside the team's ideas** → `investigator MODE=prior-art`;
   - **engineers stop early with "tie"** → give deliverables, not stop rules (see below);
   - **verified candidates not shipped** (time, reclaim, rule too strict) → ship first next round.
2. **Cut the work by ownership, not by technique**: regime (decode / prefill), size band (small / large
   n or M), or measured cost component (loop, prologue, tail, stores, host). Each engineer owns one cut.
3. **Parents on purpose.** Default is the shipped best. Branch from an older node when it wins the
   dominant cases or the best line closed a door; merge branches that win different cuts.
4. **Every engineer gets deliverables**: a measured lower bound for its cut, ≥ 2 designs tried, and a
   result for each (kept / closed with mechanism). A "tie" is only accepted with its mechanism.
5. Every round ends with geak-ship (it may also run mid-round). Ship the first verified piece early;
   node reclaims and timeouts lose unshipped work.
6. Append the decision, roles, parents and reasoning to `ledger.md` before dispatching.

## Fresh architecture: only at a design bottleneck
Dispatch `MODE=fresh-arch` only when all three hold (otherwise it mostly yields an expensive "no"):
- **plateau**: ≥ 2 rounds without a shipped gain beyond noise;
- **one design family**: the recent candidates are perturbations of the shipped design (same work
  decomposition, same data flow), and their ties/closures share a cause rooted in that design;
- **headroom**: `investigator MODE=budget`/`ceiling` shows a weighted gap to the independent roof of
  ≥ ~5%, so the limit is the design, not the hardware.
With a small gap (< ~2%) use it at most once, time-boxed to ~1/4 of a round, as a convergence check.
Judge it over two rounds: its design is a new parent that the next round's engineers optimize.

## Ship rule (pre-register it in SYNTHESIS.md before measuring; geak-ship applies it)
- Speed: the candidate must beat the shipped one by more than the measured noise on the decision
  instrument (cross-over pairs or a validated deployment-like probe), and must not lose in the
  official eval. Do not ask an instrument to resolve an effect below its noise.
- Speed-neutral fixes (robustness, deployment path): ship on non-inferiority plus a clean audit.
- If a rule blocks a real, audited gain twice, the rule is wrong: fix it and record why.

## Ending the run (you must end it; an idle loop costs money)
End when all hold, and write the evidence into `SYNTHESIS.md`:
- every lever inside the editable code is closed with a measured mechanism;
- the roofs were measured independently (raw probe / vendor kernel / ISA), not with the team's kernel;
- a fresh-arch design landed in the same family or lost to the shipped one;
- geak-ship's audit passed, and the deployment path runs the shipped code.
Or end at `TARGET`, `MAX_ROUNDS` or the deadline. Then geak-ship `MODE=final`, write
`RUN/final/report.md` (headline from `final/verify.json`, per-case and per-regime table, candidate tree,
closed levers with mechanisms, deployment status), and add reusable lessons per
`$KW/knowledge/learned/README.md`. Tell the user: speedup, verdict, why the run ended, cost.
