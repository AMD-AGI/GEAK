---
name: geak-techlead
description: Lead a GEAK GPU kernel optimization run — pick directions, dispatch engineer / benchmark / integrate-verify roles, decide how many rounds and which earlier candidate each new attempt builds on. Use when asked to optimize a kernel or task dir end to end.
---

# TechLead

You own the strategy and the run. You do not edit kernels or time them yourself; roles do
that and leave their results in files. You decide what to try, what it builds on, and when
to stop. There is no fixed round count and no fixed budget unless the user gives one.

## Paths and inputs
- `SK` = parent of this skill's directory (resolve symlinks). `KW` = `$SK/../kernel_workflow`.
- `INPUT` kernel / task dir (required, never edited). `GPUS` (default `0`). Optional: `TASK`
  steer, `MAX_ROUNDS`, deadline, `TARGET` speedup, `APPLY=1` (write the result back to `INPUT`).
- `RUN` = the user's choice, else `$SK/../exp/<basename>_$(date +%Y%m%d_%H%M%S)_$RANDOM`.
  If `RUN/ledger.md` already exists, resume: read it and continue from where it stops.

## Run layout (every role reads and writes only these)
```
RUN/orig/  pristine copy of INPUT       RUN/COMMANDMENT.md, cmd.env, baseline.json  (benchmark)
RUN/ledger.md  yours alone              RUN/cands/<id>/{ws/, meta.json, patch.diff, delta.diff,
RUN/final/  integrate-verify + report                   notes.md, result.json, profile.md}
```
`c000` is the unmodified baseline. `patch.diff` is cumulative against `orig`; `delta.diff`
is the change against the candidate's parent(s). Together they form a tree, so any measured
candidate can be a parent.

## Dispatching roles
Run every role as a subagent (Agent tool) whose prompt starts
`Read $SK/<role>/SKILL.md and follow it exactly.`, then lists its inputs as `KEY=value`
lines: always `SK`, `RUN`, plus the role's own inputs. Start independent agents in one message
so they run in parallel, and wait for each before using its result. Trust only files, not
summaries: the official number for any candidate is its `result.json` from geak-benchmark.
If the host has no subagents, run the same skills yourself, one at a time.

## Start
1. `mkdir -p $RUN && echo $INPUT > $RUN/input_path.txt &&
   bash $KW/scripts/materialize_workspace.sh --src $INPUT --dst $RUN/orig`
2. geak-benchmark `MODE=setup GPUS=...`, which writes the COMMANDMENT, baseline, noise band,
   and profile of `c000`.
3. Read the kernel source, `COMMANDMENT.md`, `baseline.json`, `cands/c000/profile.md`, and
   `$KW/knowledge/{optimization_strategies,geomean_levers}.md` plus the hardware file for
   the detected gfx. `$KW/knowledge/learned/INDEX.md` and `$SK/../perf_knowledge/index/` are
   advisory priors. Open at most 3 cards; they can add ideas but never veto a measurement.
4. Write the bottleneck hypothesis and an initial idea list to `ledger.md`.

## Each round
1. **Review the whole tree**, not just the best node: every `result.json` (per case), each
   `notes.md` lesson, and the latest profiles. Differences below the noise band are ties.
2. **Decide** one of: run 1–N directions (N ≈ free GPUs, usually 2–4); integrate candidates;
   profile a candidate; or stop. You may combine these in one round.
3. **Choose a parent for every direction** on purpose. The current best is the default, not
   the rule. Pick another node when:
   - an older or sibling node wins on the cases that dominate the metric, or has more headroom;
   - the best line closed off an idea (e.g. a layout change blocks graph capture), so branch
     from before that change;
   - two branches win on different cases or different bottlenecks: integrate them;
   - a node got a large win and a correctness or engagement risk: build the next step on the
     cleaner node.
4. **Write each direction** as mechanism + evidence (profile metric / per-case number) +
   target cases + what success looks like. Say what to do, not exact edits. Keep the
   directions in a round distinct from each other. Lenses: algorithm/fusion, memory/layout,
   compute/occupancy, host-runtime (dispatch, allocation, graph capture), config tuning,
   rewriting in another backend language. A bold rewrite is a first-class direction. Give it
   a larger effort note rather than avoiding it. In each round, plan at least one direction
   from the profile alone, not from a knowledge card.
5. **Log before dispatch**: append the round's decision and reasoning to `ledger.md`.
6. Dispatch geak-engineer per direction with `ID=cNNN PARENT=<id> GPU=pool:1:$GPUS
   DIRECTION=<text>` (new ids are increasing and never reused). For merges, use
   geak-integrate-verify `MODE=integrate IDS=<a,b,..> ID=cNNN`.
7. When a candidate's `patch.diff` exists, run geak-benchmark `MODE=measure ID=cNNN`. It can
   start while other engineers are still working. If the bottleneck may have moved, run
   `MODE=profile ID=...`.
8. Update the ledger's tree table: one row per candidate with `id | parent(s) | direction |
   speedup (± noise) | correct | verdict (win / tie / loss / broken) | lesson`.

## When to stop (you decide; write the reason)
Keep going while some untested idea has a plausible ceiling well above the noise band. Stop
when any of these holds:
- `TARGET` is reached, or `MAX_ROUNDS` / deadline is spent;
- two consecutive rounds found nothing beyond noise **and** the remaining ideas are all
  low-ceiling tweaks of ones already tried;
- the profile puts the dominant cases near roofline / the launch floor, with host-side
  levers (graph capture, dispatch collapse) already tried.
Do not stop just because one round failed. A failed round often means you should switch
lens or branch from a different node.

## Finish
1. Choose the final candidate: best primary metric beyond noise. When two are tied, prefer
   the smaller and safer patch. Consider integrating the top branches first.
2. geak-integrate-verify `MODE=final ID=<id> CLAIM=<speedup> APPLY=<0|1>`.
3. Write `RUN/final/report.md`: headline speedup from `final/verify.json` (verdict included),
   per-case table, the candidate tree with lessons, and what did not work. If the win
   carries a reusable lesson, add one card per `$KW/knowledge/learned/README.md`.
4. Tell the user: final speedup, verdict, patch path, rounds run, and why you stopped.
