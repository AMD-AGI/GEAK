---
name: adopt-ci-artifacts
description: Take an official InferenceX CI run as the baseline, one baseline per config (search-space point), straight from the run's own artifacts. Nothing is re-measured locally. Each point's bmk_agentic result supplies the deployment identity, the official throughput, the completed output tokens and the measured window; its server_logs supply the scheduler log the profile stage turns into the observed shape distribution. Points are grouped (default by EP) and the lowest- and highest-concurrency point of each group is marked representative for applyback.
---

# Adopt an InferenceX CI run, every config of it

An InferenceX workload is a curve, not a point: the run's search space is several
deployments (TP / EP / speculative decoding / KV offload) each swept over a concurrency
list. This baseline keeps all of them. The rest of the pipeline then extracts kernels from
every config, gives each task ONE unit test that weights every config equally, and
measures end-to-end at every point a UT covered (`applyback/points.py`).

**The reference is the CI's own number.** `throughput` is the run's
`request_metrics.throughput.output.tokens_per_second` for that point; `measured` is true
because the CI measured it, and `provenance.measured_on` is `inferencex_ci`. Applyback
compares its candidate against this number directly, so the node and session difference
between the CI runner and the local machine is part of any delta it reports; the noise band
must cover it.

## Input `$PARAMS`

```json
{
  "ci": {"repo": "SemiAnalysisAI/InferenceX", "run_id": "39659301076",
         "token_env": "GITHUB_TOKEN",
         "artifacts_dirs": ["/path/to/already/downloaded/artifacts"]},
  "select": {"model_prefix": "musespark", "ep": [8], "conc": [1, 4, 8, 16]},
  "group_by": ["ep"],
  "representative": null,
  "noise_band": 1.045,
  "require_power_valid": true,
  "max_decode_disagreement_pct": 5.0,
  "allow_partial": false
}
```

- `ci.run_id`: the Actions run. **Optional.** With no run id and a `ci.resolve` block, the
  baseline adopts what the site publishes *now* instead: the newest AgentX date for the
  declared `hardware`/`framework`/`precision`, the single run its rows name, and — unless
  `select.ep` already says — the EP group owning that curve's Optimal-Only peak tok/s/chip,
  because a campaign covers one EP group and that is the group the site itself puts at the top
  of the curve. The choice is written to `$OUT/resolution.json` and to `baseline.json`
  (`resolution`), so a campaign that selected adaptively still records exactly which run and
  image it ran against; pin `ci.run_id` to freeze it. Resolution needs the network, so it
  happens when the stage runs, never at `--dry-run` time. Declaring `target.image` alongside a
  resolved run makes `target.enforce` refuse any run whose image differs — leave it out to let
  the image follow the resolution, and the model stays the hard guard.
  Listing a run's artifacts needs no credential; **downloading
  them does**, even for a public repository. The token is read from the variable named by
  `ci.token_env` (default `GITHUB_TOKEN`) and is sent to `api.github.com` only, never to the
  storage host the download redirects to. Keep it in `runner_env.env_file`, not in config.
- `ci.artifacts_dirs`: directories that already hold artifacts, one subdirectory per
  artifact name (what a manual download leaves). Local copies win over downloads; with no
  `run_id` only local copies are used.
- `select`: equality filters on the bmk fields (`model_prefix` = `infmax_model_prefix`,
  `tp`, `ep`, `conc`, `kv_offloading`, `spec_decoding`, ...). A list means any of.
- `group_by`: fields that define a group. Kernel selection is per group (the operator's
  share of *its own* group's workload), and the representative points are per group.
- `split_on`: launch settings that select kernel paths (default: attention backend, KV
  dtype, page size, speculative algorithm/draft tokens/draft attention backend, DP
  attention, MoE runner backend, quantization, `SGLANG_USE_AITER*`, `MUSE_MOE_BACKEND`),
  read from each point's CI recipe. Configs that differ in any of them land in different
  groups: one UT runs in one environment and cannot time two configs' cases on two
  launch-selected paths. Whatever else differs inside a group (e.g. HiCache sizes) is
  listed in `baseline.json` `launch_diff` for review.
- `representative`: explicit point ids; default is each group's lowest and highest
  concurrency.
- `allow_partial`: by default one unusable selected point blocks the stage; a curve with a
  silent hole in it is not the curve that was asked for.

## What each point needs, and where it comes from

| Needed by `observed_workload.py` | Source |
| --- | --- |
| scheduler log | `server_logs_*` tarball, `logs/*_agg_w*.out` |
| `decode_log_interval` | the server log's printed server args |
| measured window | bmk `power_audit.window_start_unix/end_unix`; else the client's `Phase profiling (profiling) started/complete` lines in `logs/agentic/benchmark.log`, dated by the server log |
| completed output tokens (decode cross-check) | bmk `server_metrics.tokens.generation_total` |

The skill runs `observed_workload.py` on every point as a check and blocks a point whose
decode expansion disagrees with the completed tokens by more than
`max_decode_disagreement_pct`. Disaggregated serving (separate prefill/decode logs) is
reported as unsupported rather than read as an aggregated log.

## Output

```
$OUT/baseline.json                    first representative point's baseline + points/groups/representative
$OUT/points/<pid>/baseline.json       the single-point schema (stages/baseline/README.md)
$OUT/points/<pid>/observed_inputs.json  log paths, interval, window, completed tokens
$OUT/points/<pid>/observed_workload.json
$OUT/points/<pid>/launch.json         the CI recipe's serving args/env (logs/config.yaml)
$OUT/points/<pid>/serving_env.sh      that launch as an env file (EXTRA_SERVER_ARGS + env); applyback's default
$OUT/points/<pid>/server_logs/        only the members above, extracted with path checks
$OUT/artifacts/                       downloaded artifacts (when any were downloaded)
$OUT/stack_identity.json              when every point names one model and one image
```

`<pid>` is `tp<T>-ep<E>-c<C>-kv<offload>-spec<spec>`. Disaggregated recipes (roles other
than `agg`) and points without a readable recipe are reported, not guessed.

`blocked` when artifacts cannot be listed or downloaded, a selected point lacks a server log,
a throughput, a decode interval or a window, `power_valid` is false (with
`require_power_valid`), or the decode cross-check fails.
