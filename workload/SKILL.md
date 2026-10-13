---
name: workload
description: End-to-end kernel optimization pipeline for AgentX serving workloads — measure the baseline, profile and extract per-kernel unit tests, rank by roofline headroom, optimize one kernel per GPU in parallel (48 h by default), then apply the winners back and measure real end-to-end throughput. Swapping the profiler, the roofline method, the optimization method or the agent CLI is a one-line config change.
---

# workload — AgentX optimization scenario

Five stages. Each is a **slot** holding one or more interchangeable **skills**:

```
baseline  ──>  profile  ──>  roofline  ──>  optimize  ──>  applyback
measure        extract UTs    rank by        one kernel     wire back,
the target                    headroom       per GPU        measure e2e
```

Run the whole thing:

```bash
cd <GEAK repo root>
CFG=workload/config/agentx_dsv4.json

python3 workload/scripts/stage.py --config $CFG --dry-run   # check resolution first
python3 workload/scripts/stage.py --config $CFG run baseline
python3 workload/scripts/stage.py --config $CFG run profile
python3 workload/scripts/stage.py --config $CFG run roofline
python3 workload/scripts/stage.py --config $CFG run optimize    # 48 h per kernel by default
python3 workload/scripts/stage.py --config $CFG run applyback
```

`next` tells you which stage is due; `status` shows progress. Each stage's result is
recorded in `$RUN_ROOT/STATE.json`; after an interruption, re-issue the same command and
the run resumes where it stopped.

## Stages

| Stage | Default skill | What it does | Key artifact |
| --- | --- | --- | --- |
| `baseline` | `adopt-measured-receipt` | reuse an already-measured receipt (no GPUs); `agentx-trace-replay` re-measures instead | `baseline.json` |
| `profile` | `agentx-kernel-extraction-standalone` | profile, then extract frozen-harness unit tests | `tasks/<k>/`, `kernel_ranking.json` |
| `roofline` | `kernel-roofline` | per-kernel roofline; rank by **expected e2e gain** | `roofline.json` |
| `optimize` | `geak-kernel-workflow` | one kernel per GPU in parallel, 48 h by default; `subagent-raw` optimizes without the GEAK workflow | `optimized.json`, `verified.json` |
| `applyback` | `overlay-e2e-ab` | reversible overlay, fresh server, A/B against the baseline | `ab.json` |

Each slot's I/O, artifact mapping and known pitfalls are in
`stages/<stage>/README.md`; the procedure itself is in `stages/<stage>/<skill>/SKILL.md`.

## Replacing a module

This is the reason the pipeline is shaped this way. **All of it is configuration; none
of it is code.**

```bash
# a different roofline implementation
--set stages.roofline.skill=my-roofline
# optimize without the GEAK workflow
--set stages.optimize.skill=subagent-raw
# a different agent CLI
--set runner=codex
# turn a knob
--set stages.optimize.params.time_budget_sec=7200
```

A new implementation is a new directory under `stages/<stage>/` containing a `SKILL.md`
(plus `run.sh` if it is a script), named in the config. **There is no registry to
update** — follow `stages/CONTRACT.md`.

Agents work the same way; see `runners/CONTRACT.md`. The repository ships `claude.sh`
(validated), `codex.sh` (written to the contract, not yet validated) and `stub.sh`
(calls no agent — it exists to prove the scheduler does not depend on one).

## The two contracts

Modularity rests on two contracts and nothing else:

- **Stage-skill contract** (`stages/CONTRACT.md`): read `$IN`/`$CONFIG`, write artifacts
  to `$OUT`, finish by writing `$OUT/result.json`.
- **Runner contract** (`runners/CONTRACT.md`): `runner.sh <prompt> <workdir> <log>`,
  exit `0` on success, `78` on an unmet precondition (do not retry), anything else is
  retryable.

## Three things to know before reading further

1. **The failure mode of this scenario is an operator win with no system movement.** A
   MoE sorting kernel at 1.80× in microbenchmark moved the full machine by **+0.2356%**;
   MoE split4 at 1.342× moved it by **−0.0872%**. Kernels are therefore ranked by
   `pct_gpu_time × (1 − 1/attainable_speedup)` rather than by operator speedup, and
   **only `applyback` may claim an end-to-end gain**.
2. **The noise band is 1.045, not 1.02.** Measured: four kernels whose source was
   md5-identical to the original re-measured at 1.0246–1.0453 from cold.
3. **No speedup counts until it survives a cold rerun.** `scripts/verify.py` previously
   caught a kernel that passed hot and failed cold (a unit-test leg allocates from a
   virgin allocator, so zeroed pages let a partially-written output compare equal), and
   rescued a genuine 1.51× that a node failure had reverted.

## Self-check

```bash
python3 -m pytest workload/tests -q          # CPU-only: no GPUs, no tokens
```
