# Kernel extraction (AgentX) — multi-node

Read this only when `SERVING_TOPOLOGY` is in your inputs.

- `CAPTURE_BY_ENGINE` names each engine's capture inputs; `CAPTURE_ENGINE_LOG` is that engine's
  server log.
- The sampling points run in ONE serving leg: `bench_e2e.sh` with the serving env every leg in this
  run uses plus `PROFILE=1 REPEATS=0 SKILL_ROOT="$SKILL_ROOT" PROFILE_PROBE_POINTS=<sampling_points.json>
  PROFILE_PROBE_IDS=<id,id,...>`. It boots the server once and runs the skill's serving_probe.py on
  each point (the single-node rules: drive the shape, confirm it in the decode engine's log, one
  step-bounded window on that point's engine, verify the trace). Each point leaves
  `$PROFILE_DIR/<id>.journal.json` (what coverage_ledger.py reads, as on one node), its traces in
  `$PROFILE_DIR/<id>/` and the rank-set report in `$PROFILE_DIR/<id>/ranks/`. A journal whose status
  is not `ok` is a failed point; `bench_e2e.sh` has already spent the skill's one recapture on it
  (the first try is `<id>.try1.journal.json`), so report the gap and go on, as on one node — give
  coverage_ledger.py only the `ok` points' censuses. Never start a server yourself and never run serving_probe.py
  directly: the driver has no GPU and no route to a server it did not boot.
- `SERVING_TOPOLOGY=pd_disagg`: write `engine` (`prefill` | `decode`) into each task's
  `run/requested_kernel.json` — the engine whose trace (`mn_trace_roles.json`) the kernel was selected
  from. The packer refuses a task that names no engine.
- `SERVING_TOPOLOGY=pd_disagg`: do NOT weigh the two engines yourself. After writing `kernel_ranking.json`
  (`share_of_phase_pct` keyed `PREFILL` = the prefill engine's trace, `DECODE` = the decode engine's,
  each taken verbatim from the ledger's `phase_mass_share_pct`, variants of one kernel summed — never
  over another denominator), run `python3 "$SKILL_DIR/scripts/multi_node/mn.py" pd-mass
  --coverage <coverage.json> [--coverage ...] --roles <mn_trace_roles.json> [--roles ...] --apply
  <kernel_ranking.json>` (one `--roles` per engine: `$PROFILE_DIR/<id>/ranks/mn_trace_roles.json` of a
  DECODE point and of a PREFILL point),
  THEN `select_targets.py`. It rewrites every row's `selection_pct_gpu_time` from both phases' GPU-time
  mass × each engine's ranks and sets `denominators`; the packer refuses a ranking whose `denominators`
  it did not write. Exit 2 is a failed selection.
- Copy the skill's GPU scripts (regen_reference.py, oracle_power.py, kernel_self_noise.py) into the
  task directory, as op_bench.py is, and run each through gpu_lock.sh, which places it on a node.
  gpu_lock_set.py does not run on a multinode run.
