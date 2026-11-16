## Hard Requirements (MUST be in every agent's prompt)
1. Task: make the GPU kernel in `@TASK_NAME@` faster on AMD @GPU_NAME@ (@GFX@, ROCm/HIP — not CUDA) while
   staying correct. @TASK_STEER@ Every agent reads `eat/COMMANDMENT.md` and `CLAUDE.md` (if present) first.
2. Score = the last line of `bash eat/eval.sh` (`SCORE: <x>`): @METRIC@ over the frozen baseline, 0 on any
   correctness failure. The A/A noise band is ±@NOISE@: a difference inside it is a tie, not a result.
   Report only scores from `eat/eval.sh`; your own timings are for steering.
3. Editable: @EDITABLE@ — nothing else. Never modify `eat/`, the oracle / harness (`unittest.py`, `meta.json`,
   `baseline_*`, `reference_io.pt`, `cases.py`, `leg_runner.py`), or tolerances. The eval copies only the
   editable paths into a pristine task copy, so edits elsewhere have no effect on the score.
4. Generalization: results must hold for any valid input of the served shapes. No branching on the
   eval's exact shapes, no caching outputs across calls, no skipping work the oracle happens not to check.
   @HOLDOUT_NOTE@
5. GPUs: only `$EAT_GPUS` (@GPUS@). Run every GPU command as
   `bash @KW@/scripts/gpu_lock.sh "$GPU" <cmd>` from your workspace dir (`GPU` is in `eat/cmd.env`); it
   leases a free GPU and isolates the build cache. Never set `HIP_VISIBLE_DEVICES` yourself. Wrap
   anything that may hang in `timeout`. Exit 86 / `GEAK_SOURCE_INVALID` = invalid measurement, not a result.
6. Sources: all code on this machine and the web may be read. Knowledge bases (advisory, measurement
   decides): `@KW@/knowledge/`, `@GEAK@/perf_knowledge/`. Do not push to any git remote.

## Shared Environment
The supervisor reads `eat/PLAYBOOK.md` before designing a team. All agents share this repository.

### Workspaces and candidates (MUST be in every agent's prompt)
- Never edit the shipped kernel in place while experimenting. Make a private workspace with
  `bash @SK@/bin/mkws.sh .work .work/ws_<role>_<n> [solutions/<parent>/patch.diff]` (it starts from the
  pristine task plus the parent's patch) and score it with `bash eat/eval.sh --ws <that dir>`.
  Any number of agents may run evals at the same time.
- A candidate worth keeping becomes `solutions/<id>/` (next free `cNNN`): `patch.diff`
  (`git -C <ws> add -A && git -C <ws> diff --binary --cached base`), `notes.md` (what, why, lesson), and a
  row in `solutions/results.tsv`: `id  parent  score  author_role  status  description`. Log discarded and
  failed experiments too, with the reason. Any candidate may be a parent, not only the best.
- Only the agent the supervisor names as shipper applies a candidate to the editable paths of this
  repository (`git apply solutions/<id>/patch.diff` on a clean tree) when it beats the shipped score
  beyond the noise band.
