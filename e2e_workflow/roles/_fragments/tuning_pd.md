# Tuning Specialist — PD disaggregation

Read this only when `SERVING_TOPOLOGY=pd_disagg` is in your inputs.

Prefill and decode are separate servers, and an op on the merged Top-N carries an `engine` saying which
one it ran on. Tune it on THAT engine's shapes (its own `profile_workload_<role>.json`), put `engine`
on its `ops_tuned` entry, and deploy it to that engine only:
- A routing/dispatch overlay that only one engine should load goes in `apply_overlay_by_role.<role>`,
  not `apply_overlay` — the shared overlay is imported by both servers, and the other one runs the
  rerouted op at an M this tune never saw.
- Env / flags that bind one engine's artifact go in `apply_env_by_role.<role>` /
  `apply_flags_by_role.<role>`. A shape-keyed table (e.g. a GEMM config CSV) is safe in the shared
  `apply_env`: the other engine's M does not match its rows and falls back to the default. Say which
  you chose and why in `tuning_report.md`.
- In your A/B pass the per-engine part as `GEAK_MN_PD_OVERLAY_<ROLE>` / `GEAK_MN_PD_SWEEP_ENV_<ROLE>` /
  `GEAK_MN_PD_SWEEP_ARGS_<ROLE>`, merged with any value already in your inputs. Prove engagement in
  that engine's log: the `log` of its entry in `mn_launch.json` `pd.engines`.
- A `TUNING_TARGETS` entry with `lever_kind: "code_overlay"` (the Architect's `tuning_directions`)
  is not a kernel. Typically it is the KV-transfer path, when `kv_handoff.verdict.handoff_bound` is
  true. There is no shape to tune and the skillset's op loop does not apply. Build the change as an
  overlay for the named engine: `add-module --patch` on the file that holds `target_callable`,
  seeded from `CURRENT_OVERLAY`. Deploy it via `apply_overlay_by_role.<engine>` and A/B it like any
  other candidate. Its evidence is the `evidence_metric` it names (decode `#transfer-req` mean and
  prefill `#inflight-req` mean over the timed round, pre vs post), plus identical outputs. A
  throughput delta without that log shift is noise, not a win. Log both means in
  `tuning_report.md` even when the lever loses.

Your return JSON (and `tuning_result.json`) carries three more keys beside `apply_overlay` /
`apply_env` / `apply_flags`, each `{"prefill": "...", "decode": "..."}` with `""` for an engine that
needs nothing:

```json
  "apply_overlay_by_role": {"prefill": "<overlay for that engine alone>", "decode": ""},
  "apply_env_by_role": {"prefill": "", "decode": "<KEY=VAL for that engine alone>"},
  "apply_flags_by_role": {"prefill": "", "decode": ""},
```
