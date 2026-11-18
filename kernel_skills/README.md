# kernel_skills — GEAK kernel optimization as skills

Two ways to run, sharing the same instrument (`bin/`, the frozen eval) and knowledge:
- **`geak-eat`** (recommended for long runs): [EAT](https://arxiv.org/abs/2610.36848)'s supervisor
  designs and re-designs the agent team every iteration. GEAK supplies the frozen, parallel-safe,
  noise-aware eval, a candidate tree, a supervisor playbook, and `lessons.md`, which carries what
  worked from run to run. In the paper, a fixed four-role team scored 0.31× on its MLA task where
  the evolving team reached 0.74×.
- **v4** ([`v4/`](v4/)): v3's roles, plus search as state + policy. `bin/archive.py` keeps every measured node with
  per-regime results (best, noise-aware front, dispatch composites, families). Each round the lead reads a
  pluggable strategy package (`v4/search/<name>/STRATEGY.md` + optional `policy.py`; see `v4/search/SHELL.md`)
  and writes checked actions to `decisions/round_<n>.json`: mutate, fresh, combine, measure, investigate, ship,
  custom, stop. Examples: `hillclimb`, `evo`, `avo`. Evidence and offline replay: `v4/EVIDENCE.md`.
- **Fixed roles v3** ([`v3/`](v3/)): six skills (v2's five + `geak-roofline`) and shared `COMMON.md` rules.
  A run may end only when a design-independent roofline cap shows ≥ 80% of the gap closed, or when an
  independent role explains the rest. It also scales to one GPU and slow evals and survives restarts.
  Changes and evidence: [`v3/EVIDENCE.md`](v3/EVIDENCE.md). Tools: `bin/roofline.py`, `bin/hbm_stream.py`.
- **Fixed roles v2** ([`v2/`](v2/)): five skills (lead, engineer, investigator, benchmark, ship) distilled from
  the evolved EAT teams on four kernels; each rule is traced to its evidence in [`v2/EVIDENCE.md`](v2/EVIDENCE.md).
  Not yet compared head-to-head with EAT at equal budget.
- **Fixed roles v1** (the four skills below): cheap, predictable, and a useful control arm for
  measuring whether `geak-eat` pays off.

A small alternative to `kernel_workflow/` (`kernel_workflow.js` + `kernel_lane.js` + 12 role
prompts). No JS orchestrator: the techlead is an agent following a skill. It dispatches the
other roles as subagents, and every role reads and writes one run directory.

| Skill | Role |
|---|---|
| [geak-techlead](geak-techlead/SKILL.md) | Picks directions and each one's parent candidate, decides when to stop, writes the report |
| [geak-engineer](geak-engineer/SKILL.md) | Implements one direction on top of one parent, in a private workspace |
| [geak-benchmark](geak-benchmark/SKILL.md) | Builds the measurement contract; official paired A/B result and profile per candidate |
| [geak-integrate-verify](geak-integrate-verify/SKILL.md) | Merges candidates across branches; final independent audit and re-measurement |
| [geak-eat](geak-eat/SKILL.md) | Turns a kernel / task dir into an EAT task (`eat/eval.sh`, `task.md`, `PLAYBOOK.md`), launches, verifies, harvests lessons |

How it differs from `kernel_workflow`:
- **No fixed rounds or budget.** The techlead decides each round and records why it stops
  (`MAX_ROUNDS` / `TARGET` are optional caps).
- **A candidate tree, not a single best line.** Every candidate stores `patch.diff`
  (vs the original) and `delta.diff` (vs its parent), so a new direction can start from
  any earlier node, and integrate can merge branches.
- **Paired timing.** `bin/ab.sh` runs baseline and candidate alternately (ABBA) under one GPU
  lease. A setup-time A/A run gives the noise band; differences inside it are ties.

Reused from `kernel_workflow/` unchanged: `scripts/gpu_lock.sh`, `materialize_workspace.sh`,
`profile_kernel.sh`, `detect_gpu_requirement.py`, `workspace_sources.py`, and
`knowledge/` (including `learned/`). The skills locate them at `../kernel_workflow`, so keep
this folder inside the GEAK checkout.

## Use
Symlink the four skill dirs into the host's skill directory (e.g. `~/.claude/skills/`), or
point the agent at `geak-techlead/SKILL.md` directly:
```
/geak-techlead optimize /path/to/task_dir GPUS=0,1,2,3
/geak-techlead optimize /path/to/task_dir GPUS=0,1 MAX_ROUNDS=4 TASK="focus on decode shapes"
```
Headless: `claude -p "..." --allowed-tools Agent,Bash,Read,Write,Edit,Skill`.

## Run directory
```
RUN/orig/                 pristine copy of the input (never edited)
RUN/COMMANDMENT.md        commands, parse rule, metric, noise, EDITABLE / FROZEN paths
RUN/cmd.env               the same commands, shell-sourceable (used by bin/ab.sh)
RUN/baseline.json
RUN/ledger.md             techlead's decisions + candidate tree table
RUN/cands/cNNN/           ws/ meta.json patch.diff delta.diff notes.md result.json profile.md
RUN/final/                verify.json final_patch.diff report.md
```
