# E2E Integrator — multi-node serving

Read this only when `SERVING_TOPOLOGY` is in your inputs.

## PD (`SERVING_TOPOLOGY=pd_disagg`)

**PHASE=integrate (only if `SERVING_ENGINE` is in your inputs): code goes to BOTH engines; only
config can be one engine's.** Both engines run one code tree, and the kernel's cases already span
both engines' shapes. So:
- A code change (patch, rebind, authored kernel) is built from `CURRENT_OVERLAY` exactly as in step 2
  and passed as `OVERLAY_PYTHONPATH` on both legs, so both engines import it. Never give it to one engine.
- An `env` / `flag` winner goes in `<ENGINE_ENV_VAR>` / `<ENGINE_ARGS_VAR>` on the candidate leg, merged
  key by key with the value already in your inputs — not in `EXTRA_ENV` / `EXTRA_SERVER_ARGS`.
- Engagement is proven in `SERVING_ENGINE`'s front-end log: the `log` of its entry in the leg's
  `mn_launch.json` `pd.engines` (`ENGINE_LOG` in your inputs says which). The other engine runs the
  same code; check its log too and say in `note` whether the marker appears there.
- `accepted_overlay` is the candidate dir (`$CAND`), the new overlay for both engines.

**PHASE=finalize step 1.** When your inputs contain `GEAK_MN_PD_SWEEP_ARGS_PREFILL` / `_DECODE` or the
matching `GEAK_MN_PD_SWEEP_ENV_*`, `final_launch.sh` exports those beside `EXTRA_SERVER_ARGS`. They are
config accepted for one engine only. Do not fold them into the shared variables — those are applied to
both engines. Code has no per-engine layer: `OVERLAY_PYTHONPATH` and `final_patch.diff` are the same on both.
