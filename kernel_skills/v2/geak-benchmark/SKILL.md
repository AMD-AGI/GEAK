---
name: geak-benchmark
description: GEAK measurement role (v2) — freeze the instrument for a kernel run (reuse the harness, prove it runs the edited code, measure A/A noise per GPU), give each candidate its official result, and compare candidates with cross-over pairs. Invoked by geak-lead and geak-ship.
---

# Benchmark (v2)

You own the instrument; every speedup in the run is a number you wrote. Never edit kernel code to
make a measurement pass. Inputs: `SK`, `RUN`, `MODE` = `setup | measure | xover`, plus
`GPUS` (setup), `ID` (measure) or `IDS=<x>,<y>` (xover). `KW` = `$SK/../kernel_workflow`.
GPU commands only via `cd <ws> && bash $KW/scripts/gpu_lock.sh $GPU <cmd>`. Exit 86 /
`GEAK_SOURCE_INVALID` invalidates everything that invocation printed.

## MODE=setup
1. `bash $SK/bin/mkws.sh $RUN $RUN/cands/c000/ws`.
2. **Reuse the harness, never rewrite it**: an oracle `unittest.py` (if it prints
   `GEAK_WEIGHTED_SPEEDUP` it is also the perf harness), `config.yaml` commands, `task_runner.py`,
   `test_*.py` / `bench*.py`. Only if none exists write `test_harness.py` (`--correctness` against a
   trusted reference; `--benchmark` printing `GEAK_RESULT_LATENCY_MS=<ms> <case>`), timing the regime
   the kernel is deployed in (graph replay if it is captured; the same cache state). The denominator is
   the original implementation, never a naive reimplementation.
3. GPUs per job: `python3 $KW/scripts/detect_gpu_requirement.py <harness>`, confirmed from the code.
   `GPU=pool:<n>:$GPUS`. For a long harness export `GEAK_GPU_RUN_TIMEOUT` above its run time.
4. **Engagement probe**: in a scratch copy, put a deliberate error in the editable source; correctness
   must fail. Also check that the eval's code path is the one the edited file controls.
5. Write `RUN/cmd.env`: `GPU`, `CORRECTNESS_CMD`, `BENCH_CMD`, `PAIRING=interleave|in_process`,
   `LAT_REGEX` if needed.
6. **Noise, per GPU**: three A/A runs (baseline vs baseline copy) on each of the first two GPUs. Report
   per-GPU means and spread; a GPU-to-GPU offset is a bias, not noise. `NOISE` = max(0.01, worst
   |1 − A/A|); if the spread is > 5%, find the cause first (busy GPU, clocks, host socket, autotune).
7. `RUN/COMMANDMENT.md`: commands, `PARSE`, `METRIC` (weighted ratio-of-sums if the harness weights,
   else geomean of base/cand), `NOISE` and any GPU bias, `EDITABLE`, `FROZEN`, the engagement result.
   `RUN/baseline.json` and `cands/c000/result.json` (speedup 1.0).
8. Ask the lead to run `geak-investigator MODE=deployment` in round 1; if the eval does not exercise the
   deployed path, write that in the COMMANDMENT as a known gap.

## MODE=measure (official result for `cands/$ID`)
1. `M=$RUN/cands/$ID/m_$(date +%s)_$$`; `mkws.sh $RUN $M/cand <patch>` and `mkws.sh $RUN $M/base`.
   Patch fails to apply → `apply_failed`. Changes outside `EDITABLE` → `frozen_touched`.
2. Correctness in `$M/cand`; failure → `incorrect` with the log tail.
3. `bash $SK/bin/ab.sh $RUN $M/base $M/cand $M/ab 3` (`-` for base when `PAIRING=in_process`), then
   `python3 $SK/bin/score.py $M/ab` → median of per-rep primary metric and spread.
4. `cands/$ID/result.json`: `{id, status, speedup, metric, noise, spread, reps, gpu, per_case, notes}`.

## MODE=xover (decide between two candidates)
Build fresh `cand` workspaces for both and run
`bash $SK/bin/xover.sh $RUN <base|-> <ws_x> <ws_y> $RUN/xover/<x>_<y>_<ts> 2` (2 pairs; 4 if the lead
asks for a close call). Each pair runs both candidates at the same time on two GPUs, then swapped.
Write `delta_mean` (y − x), the per-pair deltas and both series to `RUN/xover/<x>_<y>_<ts>.json`.
A delta inside `NOISE` is a tie. Never compare two candidates measured at different times on
different GPUs.

Return 3 lines: status, speedup ± noise (or delta), result path.
