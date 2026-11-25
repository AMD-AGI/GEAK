# Tuning roles — PD disaggregation

Read this only when `SERVING_TOPOLOGY=pd_disagg` is in your inputs: the shared part below, then the
section for your role. Skip the other role's section.

Prefill and decode are separate servers, and each takes its config in two layers:
- `EXTRA_SERVER_ARGS` / `EXTRA_ENV` / `OVERLAY_PYTHONPATH` reach BOTH engines (the shared layer).
- `GEAK_MN_PD_SWEEP_ARGS_<ROLE>` / `GEAK_MN_PD_SWEEP_ENV_<ROLE>` (`<ROLE>` is `PREFILL` or `DECODE`)
  reach that engine alone. Merge a new value into the one already in your inputs, key by key, and
  omit an empty variable. Never put a one-engine flag or env into the shared layer.
- Prove what each engine got: `mn_launch.json` `pd.effective_args_by_role`, and that engine's log
  (the `log` of its entry in `mn_launch.json` `pd.engines`).

## config_tuner

- A direction whose `engine` is `prefill`/`decode` changes that engine only: its value goes into that
  engine's `GEAK_MN_PD_SWEEP_*`, and the other engine's accepted layer passes through unchanged.
  `engine: ""` stays shared. Record `engine` on the trial.
- In the role file's bench command, `EXTRA_SERVER_ARGS` / `EXTRA_ENV` carry the shared layer only
  (current shared config, merged with a shared change), and the per-engine layer is added beside them:
  ```bash
  EXTRA_SERVER_ARGS="<shared flags>" EXTRA_ENV="<shared env>" \
  GEAK_MN_PD_SWEEP_ARGS_PREFILL="<prefill flags>" GEAK_MN_PD_SWEEP_ARGS_DECODE="<decode flags>" \
  GEAK_MN_PD_SWEEP_ENV_PREFILL="<prefill env>" GEAK_MN_PD_SWEEP_ENV_DECODE="<decode env>" \
  ```
- Verify the swap in the named engine's log. If a one-engine change reached both engines, fix the
  strings and re-run the trial.
- Return JSON: each trial also carries `"engine": "prefill|decode|\"\""`; `accepted_flags` /
  `accepted_env` are the shared layer; and add each engine's whole layer after this sweep:
  ```json
  "accepted_flags_by_role": {"prefill": "<prefill flags>", "decode": "<decode flags>"},
  "accepted_env_by_role": {"prefill": "<KEY=VAL ...>", "decode": "<KEY=VAL ...>"},
  ```
  Leave out an engine this sweep did not change (it keeps its layer); `""` clears one.

## tuning_specialist

An op on the merged Top-N carries an `engine` saying which server it ran on. Tune it on THAT engine's
shapes (its own `profile_workload_<role>.json`) and put `engine` on its `ops_tuned` entry. Code and
config deploy differently:
- Code (a routing/dispatch overlay) is one tree on both engines: it goes in `apply_overlay`, and both
  servers import it. If it must act on one engine's shapes only, make it decide by shape inside the
  code, and keep the other engine's shapes on their default path.
- Env / flags that bind one engine's artifact go in `apply_env_by_role.<role>` /
  `apply_flags_by_role.<role>`. A shape-keyed table (e.g. a GEMM config CSV) is safe in the shared
  `apply_env`: the other engine's M does not match its rows and falls back to the default. Say which
  you chose and why in `tuning_report.md`.
- In your A/B pass the per-engine config as `GEAK_MN_PD_SWEEP_*` and the overlay as
  `OVERLAY_PYTHONPATH`, and prove engagement in that engine's log.

Your return JSON (and `tuning_result.json`) carries two more keys beside `apply_overlay` /
`apply_env` / `apply_flags`, each `{"prefill": "...", "decode": "..."}` with `""` for an engine that
needs nothing:

```json
  "apply_env_by_role": {"prefill": "", "decode": "<KEY=VAL for that engine alone>"},
  "apply_flags_by_role": {"prefill": "", "decode": ""},
```
