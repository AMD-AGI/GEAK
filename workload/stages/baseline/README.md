# The `baseline` slot — pin down what the number was before any optimization

**Input** none (first stage).
**Output** `$OUT/baseline.json`, the single reference every downstream comparison uses.

## Implementations

| Skill | Shape | Notes |
| --- | --- | --- |
| `adopt/` | script | **default**. Reuse an already-measured serving receipt. No GPUs, no tokens, replayable |
| `measure/` | agent | measure on this machine. Which benchmark runs is config (`bench_e2e.sh` against a running service, or the site's own CI harness in a fresh container), and whether a reproduction target is *proven* is `params.pins` |
| `adopt-ci-artifacts/` | script | take an official InferenceX CI run as the baseline, **every config of it**: one baseline per search-space point straight from the run's `bmk_agentic_*` and `server_logs_*` artifacts, grouped (default by EP) with each group's lowest/highest concurrency marked representative. With no run pinned it adopts the latest published curve for the target hardware/framework/precision. Nothing is re-measured; the reference is the CI's own number. Needs a GitHub token to download artifacts, or pre-downloaded copies |

Choose by what the baseline has to withstand. `adopt` is right when the number is already
trusted and the question is only "what was it". `measure` with `pins` is right when **this
machine's ability to hit the reference point is itself in question** — before a long
campaign, after a node or image change, or whenever a previous baseline and the reference
figure disagree and nobody can say which pin drifted.

`measure` is **agent-shape** (contract §2B): there is no `run.sh`. Starting a container
and running somebody else's harness is a handful of commands; what makes the stage hard is
knowing which pin silently produces a believable number, and that is prose. It used to be
1,520 lines across three directories, two of which differed only in which benchmark
command ran and whether pins were checked.

The default reuses rather than re-measures because the reference run has already been
reproduced in full, to within a fraction of a percent. The run id, the CI figure and the
measured one live in the scenario config and in the run's own receipt, not in this
document. Spending another GPU-hour × 8 reproduces the same number plus noise.

## The shared artifact, `baseline.json`

Both skills emit the same schema; downstream (roofline's expected gain, applyback's A/B)
reads only this file.

```json
{
  "metric": "output_token_throughput",
  "unit": "tok/s",
  "throughput": 446.87889,
  "higher_is_better": true,
  "noise_band": 1.01,
  "config": {"tp": 8, "ep": 1, "gpus": 8, "concurrency": 10},
  "reference": {"name": "ci_output_tokens_per_second", "value": 446.56324, "ratio": 1.00071},
  "measured": true,
  "provenance": {"method": "adopt", "receipt": "...", "receipt_sha256": "...", "node": "..."}
}
```

`noise_band` records the E2E decision threshold from `stages.baseline.params` in
`config/default.json` or the scenario override. It is policy, not a measured noise
estimate or the kernel `keep_threshold`. [Evaluate](../evaluate/README.md) owns its use;
report repeat variability separately.

`measure` fills the same schema and additionally records every pin plus the check-by-check
alignment result under `provenance` (`alignment_verdict`, `alignment_checks`,
`image_digest`, `benchmark_id`, `cpu_binding`, `cold_cache_dirs`, `measurement_mode`).

## Artifacts → `result.json`

`finalize.py` writes it, for either shape. `metrics` carries `throughput`, `noise_band`,
`node`, `drift_pct` and `reference_ratio` when a reference is recorded, and `alignment`
when the baseline was a pinned reproduction.

## What is judged, and where

`finalize.py` is the verdict (`stages/CONTRACT.md` 3.1) and the rules are written only
there: that the number is measured, that the harness's own validity flags hold, that drift
stays inside the configured `max_drift_pct`, that a node name is recorded, and that a
scenario declaring `pins` carries a proven alignment. Read that file rather than looking
for a second copy here — the same four rules were once written out three times, and
`agentx-trace-replay` checked none of them.

Scenario config supplies whatever a particular receipt or harness calls its fields —
`throughput_key`, `reference_key`, `valid_flags`, `pins`, `env_map`, `isolation`,
`harness` — because those are one harness's JSON key names, which are data, not skill.

## Two measurement rules

- **An A/B must be same-node.** On this workload the same configuration varies by
  fractions of a percent across nodes, so a baseline taken on node A against a candidate
  measured on node B reports nothing but the node difference. `finalize.py` requires the
  node name; applyback checks it.
- **`REUSE_SERVER=1` and `REUSE_SERVER=0` are not comparable.** `provenance.measurement_mode`
  records which was used, and applyback must use the same one.
