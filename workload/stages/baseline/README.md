# Stage `baseline` — pin down what the number was before any optimization

**Input** none (first stage).
**Output** `$OUT/baseline.json`, the single reference every downstream comparison uses.

## Implementations

| Directory | Shape | Notes |
| --- | --- | --- |
| `adopt-measured-receipt/` | script | **default**. Reuse an already-measured serving receipt. No GPUs, no tokens, replayable |
| `agentx-trace-replay/` | script | re-measure with a fresh AgentX trace replay. Occupies the whole machine; gate on `agentx_smoke.sh` first |
| `ci-run-replay/` | script | reproduce a *named CI run* from a fresh digest-pinned container and accept the number only if it aligns. Pins image digest / data+seed / `benchmark_id` / cache-bust prefix, binds one rank per physical core, and proves each pin afterwards from the run's own records |

Choose by what the baseline has to withstand. `adopt-measured-receipt` is right when the
number is already trusted and the question is only "what was it". `ci-run-replay` is right
when **this machine's ability to hit the CI point is itself in question** — before a long
campaign, after a node or image change, or whenever a previous baseline and the CI figure
disagree and nobody can say which pin drifted. It is the only one that reports `blocked`
on a run that completed successfully but landed outside the band, which is the failure
mode that otherwise quietly re-labels every downstream delta.

The default reuses rather than re-measures because the CI run named in `instruct.md`
(`34783501425`) has already been reproduced in full: `output_tokens_per_second =
446.87889` against a CI reference of 446.56324 — a **0.07%** difference — with 1455
successes and 0 errors on TP8/EP1, 8 GPUs, global concurrency 10. Spending another
GPU-hour × 8 reproduces the same number plus noise. To re-measure anyway, set
`stages.baseline.skill` to `agentx-trace-replay`.

## The shared artifact, `baseline.json`

Both implementations emit the same schema; downstream (roofline's expected gain,
applyback's A/B) reads only this file.

```json
{
  "metric": "output_token_throughput",
  "unit": "tok/s",
  "throughput": 446.87889,
  "higher_is_better": true,
  "noise_band": 1.045,
  "config": {"tp": 8, "ep": 1, "gpus": 8, "concurrency": 10},
  "reference": {"name": "ci", "value": 446.56324, "ratio": 1.00071},
  "measured": true,
  "provenance": {"method": "adopt-measured-receipt", "receipt": "...", "sha256": "..."}
}
```

- `noise_band` is the **keep threshold**, and it is calibrated rather than quoted: four
  kernels in a previous campaign whose source was md5-identical to the original
  re-measured at 1.0246–1.0453 from cold, so the default is **1.045**, not the commonly
  cited 1.02. Anything below that is noise, not speedup.
- A baseline with `measured=false` must not enter applyback's A/B.

`ci-run-replay` fills the same schema and additionally records every pin plus the
check-by-check alignment verdict under `provenance` (`alignment_verdict`,
`alignment_checks`, `image_digest`, `benchmark_id`, `cpu_binding`, `cold_cache_dirs`), and
sets `reference` to the CI run with the measured `drift_pct`.

## Artifacts → `result.json`

All three implementations are script-shaped; `run.sh` writes `result.json` itself, so
`stage.py run baseline` completes in one step. `metrics` carries `throughput` and
`reference_ratio` (`ci_drift_pct` and `alignment` for `ci-run-replay`).

## Two measurement rules

- **An A/B must be same-node.** On this workload the same configuration varies by
  fractions of a percent across nodes, so a baseline taken on node A against a candidate
  measured on node B reports nothing but the node difference. The node name is recorded
  in `provenance`, and applyback checks it.
- **`submission_valid` / `power_valid` being false in `measured_baseline.json` is a
  `blocked`**, not a warning. A non-compliant submission cannot serve as a reference.
