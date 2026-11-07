# Tuning Specialist — PD disaggregation

Read this only when `SERVING_TOPOLOGY=pd_disagg` is in your inputs.

Prefill and decode are separate servers, and an op on the merged Top-N carries an `engine` saying which
one it ran on. Tune it on THAT engine's shapes (its own `profile_workload_<role>.json`) and put `engine`
on its `ops_tuned` entry. Code and config deploy differently:
- Code (a routing/dispatch overlay) is one tree on both engines: it goes in `apply_overlay`, and both
  servers import it. If it must act on one engine's shapes only, make it decide by shape inside the
  code, and keep the other engine's shapes on their default path.
- Env / flags that bind one engine's artifact go in `apply_env_by_role.<role>` /
  `apply_flags_by_role.<role>`. A shape-keyed table (e.g. a GEMM config CSV) is safe in the shared
  `apply_env`: the other engine's M does not match its rows and falls back to the default. Say which
  you chose and why in `tuning_report.md`.
- In your A/B pass the per-engine config as `GEAK_MN_PD_SWEEP_ENV_<ROLE>` / `GEAK_MN_PD_SWEEP_ARGS_<ROLE>`,
  merged with any value already in your inputs, and the overlay as `OVERLAY_PYTHONPATH`. Prove
  engagement in that engine's log: the `log` of its entry in `mn_launch.json` `pd.engines`.

Your return JSON (and `tuning_result.json`) carries two more keys beside `apply_overlay` /
`apply_env` / `apply_flags`, each `{"prefill": "...", "decode": "..."}` with `""` for an engine that
needs nothing:

```json
  "apply_env_by_role": {"prefill": "", "decode": "<KEY=VAL for that engine alone>"},
  "apply_flags_by_role": {"prefill": "", "decode": ""},
```
