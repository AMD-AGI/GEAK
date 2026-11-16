# Profile: several configs (`adopt-ci-artifacts` baseline)

When `baseline.json` has `multi_config: true`, the run covers every config of one group
(default: one EP) of an InferenceX CI run, and each task still gets **one** UT. `$OUT` is
the group's eval dir; each config's observations and deployment live under
`$OUT/configs/<pid>/`. Plan the shared capture union before starting probe servers,
then project costs onto each config and merge. One group per run: a second EP group is a sibling
scenario whose baseline `select` names that EP.

1. **Per config**, for every `points[].point_id` in `baseline.json`
   (`EVAL_DIR=$OUT/configs/<pid>`): prepare observations and deployment before profiling.
   `observed_workload.py` takes its inputs from the point's
   `$RUN_ROOT/baseline/out/points/<pid>/observed_inputs.json` (the CI's own scheduler log,
   decode interval, measured window and completed tokens) -- or reuse the baseline's
   validated `observed_workload.json` -- and `serving.json` is that config's deployment.
   Follow [shared sampling](kernel-extract/references/multi-config-sampling.md):
   `multi_config.py plan` groups compatible deployments and deduplicates their observed
   points, `capture` profiles each pool once, and `project --merge` recomputes each
   config's own weights. Pass the scenario's global `sampling.num_points` budget,
   `context_tolerance`, and `target_pct`. Preserve every config's uncovered mass.
2. **Merge** (already done by `project --merge`; for independently captured legacy inputs):

   ```bash
   python3 "$WORKLOAD_ROOT/stages/profile/multi_config.py" merge --group-dir "$OUT" \
       --config <pid1>="$OUT/configs/<pid1>" --config <pid2>="$OUT/configs/<pid2>" ...
   ```

   This writes `$OUT/sampling_points_final.json` (all points, ids `<pid>__<point>`, per-config
   `phase_summary`), the per-point shape and census files, merged `source_map` /
   `adjacency` / `compile_variants`, and `kernel_ranking.json` in which every row's share is
   the equal-weight mean of its share in each config (absent = a measured 0) -- the common
   denominator shares are reported on, not the selection rule (step 3).
3. **Select per config, then take the union**:

   ```bash
   python3 "$WORKLOAD_ROOT/stages/profile/multi_config.py" select --group-dir "$OUT"
   ```

   An operator is a candidate as soon as it passes the configured rule -- phase Top
   `selection_top_k` AND a share of **that single config's own workload** above
   `selection_overall_pct_gt` (the multi-config scenario: Top 10 and >3%) -- in any one
   config; not on its share averaged over the
   configs, which would bury an operator that dominates some concurrencies and is absent at
   others. Each config is selected by the unchanged `select_targets.py` on its own
   `kernel_ranking.json`, by operator (against the merged source map, so operator keys
   agree across configs), with the configured scope; the union is written to
   `$OUT/selection_targets.json` with each target's `selected_in_configs` and
   `config_pct`. Then continue with §3-§5 on `$OUT` unchanged: `draft_task_decls.py`,
   review, `build_task_plan.py`, the UT wave.

   **Communication kernels are listed, not extracted.** Operators whose kernels are all of
   a kind in `selection_list_only_kinds` (default `["collective"]`: all-reduce, all-gather,
   NCCL/RCCL) are taken out of the targets and recorded under `listed_only` with their
   per-config share; they stay in the denominator and the report and get no task and no UT.
   A standalone collective UT reproduces neither the interconnect state, the rank skew nor
   the overlap with compute that decide its serving time; its real levers (fusing the
   all-reduce with the following norm, quantized or fewer collectives) lie outside one
   kernel's seam; and its multi-rank UT costs the whole node and is the flakiest gate.
   Set the list to `[]` to extract them anyway.
4. Every planned case now carries its `config_id` (sig `<pid>/<PHASE>_<shape>`), each config
   planned on its own bands. The UT pools the configs with
   `serving_weight_model.config_weighting` (`harness_lib.serving_weighted_speedup`, which
   also reports `per_config`):
   - `calls` (default): each config's call distribution is normalized to 1 and the configs
     are averaged, `w(M) = sum_c alpha_c * freq_c(M)`, metric `sum_M w(M) * t(M)`. A kernel
     sees M, not the concurrency, so this pools "a launch at M" across concurrencies; a
     config whose launches cost more (larger M) counts for more GPU time.
   - `time`: each config's baseline-time mass is normalized to `alpha_c`, i.e. the
     alpha-weighted harmonic mean of the per-config speedups.
   `alpha_c` is equal unless `serving_weight_model.config_weights` names it. A task whose
   kernels do not run in a config has no cases there and that config drops out of its pool.

## Merging across concurrency: what decides the task boundary

- **"The gain disappears at another concurrency" is usually a path switch, not a shape
  change.** At a larger M the server routes the same operator to a different kernel. So
  **the unit of merging is the operator entry, above the
  path selection**, never one Triton kernel: put `target_callable` above the M-dependent
  dispatch, so every config's cases enter the same seam, reach their own path, and the
  switch threshold itself is something the optimizer can move.
- **Select by operator, per config.** `multi_config.py select` selects each config by
  operator on its own workload and unions the results, so an operator that matters at some
  concurrencies only is not averaged away with zeros from the others.
- **M-dependent paths merge; launch-dependent paths do not.** A UT runs in one environment.
  Paths chosen by M are reached by the cases themselves (the `launches` gate sees all of
  them across the phase's cases). Paths chosen by a launch setting (attention backend, MoE
  backend, KV dtype, speculative settings, ...) cannot be: the baseline splits such configs
  into separate groups (`adopt-ci-artifacts` `split_on`), and `baseline.json`
  `launch_diff` lists what still differs inside a group (e.g. HiCache sizes) for review.
- **Cover every graph bucket.** Under CUDA graph the served M is the padded batch
  (`M = pad(bs) x tokens_per_req`, e.g. DFLASH 8 x the captured bs list). Add the captured
  buckets no point covers as task `envelope_cases` (tested, weight 0), so a regression at an
  M no sampled point hit is still visible.
- **Who decides:** `kernel-extract/scripts/merge_policy.py`, from the evidence; the rules are
  its docstring, the procedure `kernel-extract/references/multi-config-ut-merge.md`.
- **One owner per file.** When two tasks touch one module, one owns it; a whole-module
  replacement also replaces the other functions in that file: rebind the function instead.
- A merged dispatch file that loads `_*.py` helpers from its own directory must ship them in
  the overlay; a missing helper fails at import and the server reports the model as not
  registered. Check imports on one GPU before starting the 8-GPU server.
