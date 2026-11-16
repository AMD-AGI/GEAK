---
name: geak-eat
description: Run EAT (Evolving Agent Teams, arXiv 2610.36848) on a GEAK kernel or extracted task dir — build the frozen GEAK eval, launch the supervisor that designs and evolves agent teams, monitor, verify the result, and harvest lessons that evolve this harness across runs.
---

# GEAK × EAT

EAT supplies the loop: a supervisor that reflects, re-designs the team (roles, phases, prompts) and
relaunches. GEAK supplies what EAT assumes but leaves to the user: a frozen, parallel-safe, noise-aware
eval for AMD kernels, plus lessons carried from run to run.
`SK` = parent of this skill dir; `KW` = `$SK/../kernel_workflow`; `GEAK` = `$SK/..`.
EAT: `EAT_HOME=${EAT_HOME:-$HOME/.cache/eat/EAT}`. If it is missing, `git clone
https://github.com/caoshiyi/EAT $EAT_HOME && git -C $EAT_HOME checkout 6c564358` (the version this
was written against; read its `SKILL.md` once).

Inputs: `INPUT` (kernel or task dir, never edited), `DEST` (new task repo), `GPUS` (comma list given by
the user; never take others), budget $, max iterations, max team size, optional `TASK` steer.

## 1. Build the task repo
```bash
mkdir -p $DEST/.work && cd $DEST
bash $KW/scripts/materialize_workspace.sh --src $INPUT --dst $DEST            # working tree
bash $KW/scripts/materialize_workspace.sh --src $INPUT --dst $DEST/.work/orig  # pristine, eval base
printf '%s\n' .work/ 'eat/log/.supervisor_*' .torch_ext/ __pycache__/ build/ '*.so' > .gitignore
mkdir -p eat solutions && git init -q
```
Find the instrument the way `$SK/geak-benchmark/SKILL.md` setup steps 2–4 do: reuse the harness,
detect GPUs per job, run the engagement probe. Write `eat/cmd.env`:
- extracted task with `unittest.py` printing `GEAK_WEIGHTED_SPEEDUP`: `CORRECTNESS_CMD=""`,
  `BENCH_CMD='python3 unittest.py'`, `PAIRING=in_process`, `EDITABLE="kernel_src"`;
- otherwise: the correctness and benchmark commands, `PAIRING=interleave`, `EDITABLE="<paths>"`, and
  `LAT_REGEX` if the output is not `GEAK_RESULT_LATENCY_MS=<ms> <case>`;
- always `GPU="pool:<n>:$GPUS"`, `EVAL_TIMEOUT=<s>`, optional `HOLDOUT_CMD` (correctness on shapes
  outside the scored set; use one if the oracle can build them).

`eat/env.sh`: export `EAT_ROOT=$DEST`, `EAT_WORK=$DEST/.work`, `EAT_GPUS="${EAT_GPUS:-<GPUS as space
list>}"`, `GEAK_SK=$SK`, `GEAK_KW=$KW`, plus whatever the harness needs (venv, `PYTHONPATH`, caches).
Copy `templates/eval.sh` to `eat/eval.sh`. Fill `templates/task.md` → `eat/task.md` and `PLAYBOOK.md` →
`eat/PLAYBOOK.md` (`@LESSONS@` = the contents of `$SK/geak-eat/lessons.md`). Write
`eat/COMMANDMENT.md`: commands, metric, editable / frozen paths, the engagement probe result.

## 2. Calibrate (before any agent runs)
- Run `bash eat/eval.sh` 3× on the untouched repo. Every score must be ≈1.0. `NOISE` = max(0.01,
  max |1 − score|). Put it in `cmd.env`, `COMMANDMENT.md` and `task.md`. If the spread is >5%, find out
  why first (busy GPU, clocks, autotune).
- Plant a deliberately wrong result in a scratch copy of an editable file and run
  `eval.sh --ws <scratch>`. It must print `SCORE: 0`.
- `git add -A && git commit -m "EAT task setup"`. Show the user `eat/task.md` and the calibration
  numbers, and get approval.

## 3. Launch
Agents run with `--dangerously-skip-permissions`. Run only inside the job container/sandbox, and tell
the user. Confirm budget, iterations and team size, then:
```bash
tmux new-session -d -s eat_<name> "cd $DEST && export EAT_GPUS='<space list>'; bash $EAT_HOME/agent_team/launch_stateless.sh --persistent <iters> <team> <budget> 2>&1 | tee -a .work/eat.log"
```
Gateway auth (`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`) passes through. EAT only unsets
`ANTHROPIC_API_KEY`. Its `$` costs come from the CLI's list-price accounting.

## 4. Monitor
Follow EAT's `SKILL.md` §4: `tail .work/eat.log`, `eat/log/BUDGET.md`, `solutions/results.tsv`,
`eat/log/iter_*/official_eval.txt`. Send information (not optimization hints) through
`eat/log/OPERATOR_NOTES.md`. If a score looks too good, run step 5 at once rather than waiting.

## 5. Verify the result
1. `EVAL_REPS=7 bash eat/eval.sh` on the shipped state. For iteration commits use EAT's
   `ops/verify_iter.py`.
2. Spawn one independent subagent with `$SK/geak-integrate-verify/SKILL.md`'s `MODE=final` audit
   checklist (step 2) on `git diff` of the editable paths against `.work/orig`. It must not see the
   team's notes before forming its own judgment.
3. Report: verified score ± noise, audit result, held-out result, cost, iterations, patch path.

## 6. Harvest (this is how the harness itself evolves)
Read `eat/log/iter_*/team_config.md`, `EVOLUTION_HISTORY.md` (if the supervisor kept one) and
`BUDGET.md`. Append at most 8 lines to `$SK/geak-eat/lessons.md`, each one:
`<task> · <plateau symptom> → <team change> → <score before → after> · <$>`.
Also add eval or harness defects the run exposed, each with its fix. Keep the file under 60 lines:
merge, do not append duplicates. A transferable kernel lesson goes to `$KW/knowledge/learned/` per its
README. Before trusting new lessons, compare a run with them against one without them at matched
budget: KB priors have hurt planning here before (`$KW/roles/tech_lead.md`, LEARNED_KB budget).
