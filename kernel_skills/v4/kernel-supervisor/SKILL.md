---
name: kernel-supervisor
description: Own one qualified kernel optimization for workload Brain. Use measured evidence to design each round's team, delegate work with v4 skills and return a source-bound UT result through Optimize.
---

# Kernel supervisor

Take the objective, qualified task, source boundary, device set, budgets and evidence
paths from the [workload handoff](../../../workload/stages/optimize/kernel-skills/SKILL.md).
You are the worker session's controlling agent. Read its prior handoff and current
accepted source before resuming. Each attempt has its own `RUN`. Reuse earlier findings
with matching inputs and identify which candidate the worker actually kept.

## Choose the team

Start from the qualification and native baseline in the handoff. Read the archive and
previous round's outcomes. Decide what question would change your next source or
deployment decision, and which team can answer it within the remaining budget.
Use these v4 skills as instructions for the work you assign.

| Skill | Work to reuse |
| --- | --- |
| [Benchmark](../geak-benchmark/SKILL.md) | Check source engagement, measure noise and compare candidates |
| [Roofline](../geak-roofline/SKILL.md) | Estimate headroom and test a proposed performance ceiling |
| [Investigator](../geak-investigator/SKILL.md) | Resolve a numerical, instrument or deployment uncertainty |
| [Engineer](../geak-engineer/SKILL.md) | Implement and measure one direction in a private workspace |
| [Ship](../geak-ship/SKILL.md) | Combine compatible candidates, audit a diff or independently verify a delivery |

Choose role names and responsibilities for this round. One agent can use multiple
skills, two implementers can use the same skill on different regimes, and a specialist
can investigate a question of your choosing. You can handle a small task yourself.
Assign final verification to someone who did not author the candidate. If you wrote
the candidate, delegate its audit to another agent. Give each role the worker's
resource, measurement and delivery instructions when using the standalone skills.

## Run a round

1. Register the starting source as `c000`. Keep the complete task in
   `cands/<id>/ws/`, and v4 candidate metadata and measured `result.json` beside
   `ws/`. Give every measured node its parents, native receipt and per-case rows as
   described in the worker contract. To select parents for a round, run the following
   from the first measured baseline onward. `SEARCH` is the selected
   [strategy](../search/SHELL.md), default `evo`.

   ```bash
   S="$SK/v4/search/$SEARCH"
   python3 "$SK/bin/archive.py" build --run "$RUN"
   python3 "$SK/bin/archive.py" views "$RUN/archive.json"
   if [ -f "$S/policy.py" ]; then
       python3 "$S/policy.py" --run "$RUN" --archive "$RUN/archive.json" \
           --state "$RUN/search/$SEARCH/state.json"
   fi
   ```

   Use the proposals and previous receipts to choose the team, its communication and
   timing. When baseline evidence is missing, assign its measurement first.
2. Write `RUN/decisions/round_<n>.json` with the v4 actions and their evidence-based
   reasons. Add a `team` object mapping each role name to its assignment, and an
   `owner` naming the executor of each working action. Include the skill paths or
   sections to reuse, candidate/workspace, time limit, what each role reads and
   produces, and who needs the result when. Choose direct messages, shared notes or
   a file handoff to suit those dependencies. Record receipt paths on disk.
   For example, a team investigating noisy decode measurements could assign
   `timing_analyst` the benchmark and investigator skills, then have `decode_tuner`
   read its timing report before editing the candidate. After the source changes,
   a later round can delegate the ship audit to an independent reviewer.
3. Check the decision before dispatch. `N` is the number of assigned logical GPUs.

   ```bash
   python3 "$SK/bin/check_decision.py" --run "$RUN" --decision "$decision" \
       --strategy "$S" --gpus "$N" --require-team
   ```

4. Use the runner's subagent tool to launch each delegated role with its actual
   assignment, inputs and the worker contract. Track agent handles with the decision.
   Dispatch independent work concurrently on private copies, serialize native GPU
   measurements on the worker's slot and start dependent work when its input arrives.
   Read reports as they arrive. If a result invalidates a pending direction, record
   that outcome and revise the next assignment. If an essential role cannot run,
   record the missing work and its effect on delivery.
5. Record each action's outcome, including a textual `status`, candidate IDs in
   `nodes`, report/UT receipt paths and cost. Recheck the completed decision with
   the same command and rebuild the archive, including after the final measurement.
   Choose whether to reuse, merge, split or retire roles. Charge new directions to
   `BUDGET` and reserve time for verification and handoff.

## Return to Brain

When a candidate is ready, have the independent reviewer apply the v4 ship audit
to that exact source with the workload's frozen UT. Follow the workload handoff to
deliver accepted source to the worker. Honor the plan's stopping criteria and
credit requirements. For an early stop without an explicit target, use the v4
[ending criteria](../geak-lead/SKILL.md#ending-the-run). At a budget stop, measured no-gain or blocker,
return the best supported outcome and the next useful question. Link decisions and
receipts from `HANDOFF.md` so the next attempt can reconcile them with worker KEEP
or rollback. Optimize supplies Brain the cold-verified kernel result. Brain decides
when to integrate it and measure the serving workload.
