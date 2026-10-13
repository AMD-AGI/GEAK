---
name: agentx-trace-replay
description: Measure a fresh AgentX trace replay against an already-running service and use it as this pipeline's baseline. Drives GEAK e2e_workflow's bench_e2e.sh with the agentx client adapter, gated on agentx_smoke.sh readiness checks. When the environment is incomplete it reports blocked rather than producing a throughput number that measured something else.
---

# Measure an AgentX trace replay as the baseline

The alternative to `adopt-measured-receipt`. This one really occupies the GPUs and really
measures.

**It exists because reusing a receipt presupposes an unchanged environment.** After a
new image, new weights, a different node, or changed serving parameters, the old receipt
is no longer this machine's baseline, and re-measurement is mandatory.

## Precondition: the readiness check is not optional

Run `e2e_workflow/scripts/agentx_smoke.sh` first. The failures it catches share one
shape: **nothing goes wrong until the client has already started and warmed the
service**, and the error arrives twenty minutes later. The worst of them never raises an
error at all — a stock `aiperf` without `--scenario` support will cheerfully run a
*different* workload and hand back a perfectly plausible throughput number.

Hence `params.smoke_required` defaults to `true`, and any blocker yields `blocked`.

## Input `$PARAMS`

| Field | Required | Meaning |
| --- | --- | --- |
| `base_url` | yes | address of the running service, e.g. `http://127.0.0.1:8000`. This skill does **not** start the service |
| `model` | no | weights path passed to the smoke check for extra validation |
| `inferencex_path` | no | InferenceX checkout; `map_aiperf.py` lives under its `benchmarks/`. It is on neither `$PATH` nor `/opt`, so it must be given explicitly |
| `duration_s` | no | measurement window in seconds; default `3600`, matching the InferenceX reference client's canonical window |
| `repeats` | no | default `1`. The canonical trace-replay procedure is one warm-up round plus one timed round |
| `metric` | no | scoring axis; default `p90_intvty_inferencex` |
| `noise_band` | no | keep threshold; default `1.045` |
| `smoke_required` | no | default `true` |
| `env` | no | extra environment variables, passed through to `bench_e2e.sh` |

## How it runs

```bash
python3 "$WORKLOAD_ROOT/scripts/stage.py" run baseline     # with skill set to agentx-trace-replay
```

Everything `run.sh` does is reproducible from `$OUT/bench.log`:

1. `agentx_smoke.sh` readiness check — stop on any blocker.
2. `BENCH_CLIENT=agentx REUSE_SERVER=1 bash e2e_workflow/scripts/bench_e2e.sh`, against
   the **already-warm** service at `base_url`.
3. Read `throughput_tok_s_median` and `metric_basis` from `bench_summary.json` and
   translate them into the canonical `baseline.json`.

## Output

`$OUT/baseline.json`, `$OUT/bench/` (bench_e2e.sh's full OUT_DIR), `$OUT/bench.log`,
`$OUT/smoke.log`, `$OUT/result.json`.

## Two rules that must not be broken

- **Numbers from `REUSE_SERVER=1` and `REUSE_SERVER=0` are not comparable.** How far a
  round on a warm service differs from one on a cold isolated service, and in which
  direction, cannot be derived from the configuration. `baseline.json` records which was
  used in `provenance.measurement_mode`, and applyback must use the same one.
- **Do not open the profile window just as the load starts.** At that point aiperf is
  still parsing the corpus and staging per-lane warm-up, so the window captures the ramp
  (batch=1, no steady-state MoE/attention mix). The agentx adapter places the window at
  the 75% mark by itself; do not override that with `AGENTX_PROFILE_WARMUP_S` unless you
  know why.
