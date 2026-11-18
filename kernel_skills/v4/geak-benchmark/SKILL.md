---
name: geak-benchmark
description: GEAK measurement role (v4) — freeze the instrument for a kernel run (reuse the harness, prove it runs the edited code, measure noise, price the eval), build a fast probe validated against the official eval when the eval is slow, give each candidate its official result, and compare candidates with cross-overs. Invoked by geak-lead and geak-ship.
---

# Benchmark (v4)

You own the instrument: every speedup in the run is a number you wrote. Never edit kernel code to make a
measurement pass. Read `$SK/v4/COMMON.md`. Inputs: `SK`, `RUN`, `MODE` = `setup | measure | xover`,
plus `GPUS` (setup), `ID` (measure) or `IDS=<x>,<y>` (xover). `KW` = `$SK/../kernel_workflow`.

## MODE=setup
1. `bash $SK/bin/mkws.sh $RUN $RUN/cands/c000/ws`.
2. **Reuse the harness; never rewrite it.** Candidates, in order:
   - an oracle `unittest.py` (if it prints `GEAK_WEIGHTED_SPEEDUP`, it is also the perf harness);
   - `config.yaml` commands;
   - `task_runner.py`;
   - `test_*.py` / `bench*.py`.
   Only if none exists, write `test_harness.py` that times the deployed regime against the original
   implementation.
3. **Price the eval.** Time one full official run and one correctness run. Record their wall times in the
   COMMANDMENT. Set `GEAK_GPU_RUN_TIMEOUT` above the official run time and `GEAK_GPU_POOL_WAIT` ≥ 3× it.
4. **Engagement probe**: put a deliberate error in the editable source of a scratch copy; correctness must
   fail. Check that the eval's code path is the one the edited file controls.
5. **Gates you skip for scoring** (e.g. a mutation gate whose anchors a candidate may rewrite, and any gate
   that fails only because of it): run them on the baseline now, list them in the COMMANDMENT with the
   reason, and tell geak-ship. Ship must re-establish their intent on every shipped candidate.
6. `RUN/cmd.env`:
   - `GPU` (`pool:1:<ids>`; group specs only for multi-GPU jobs);
   - `CORRECTNESS_CMD`, `BENCH_CMD`, `FAST_CMD` (below);
   - `PAIRING=interleave|in_process`;
   - `GEAK_GPU_RUN_TIMEOUT`, `GEAK_GPU_POOL_WAIT`;
   - `LAT_REGEX` if needed.
   Normalize the harness's exit codes: say which code means pass when gates are skipped.
7. **Noise.** Three A/A runs per GPU (with one GPU: three repeated runs, alternating order). A GPU-to-GPU
   offset is a bias, not noise. `NOISE` = max(0.01, worst |1 − A/A|). If the spread is > 5%, find the
   cause first: busy GPU, clocks, host socket, process state, autotune.
8. **Fast probe (required when one official run takes > ~10 min).**
   - Build `FAST_CMD`: the harness's own timer and binding, all scored cases (deduplicated shapes),
     weights as in the official metric, one fresh process per leg, alternating order. Pin the official
     process state: HW queue count (`GPU_MAX_HW_QUEUES`), streams, graph pools, cache flush.
   - Validate it: its A/A noise, and its **candidate-vs-candidate delta** on at least one real candidate
     per bound class against official runs. Agreement on baseline times alone is not validation.
   - Record the validation result in the COMMANDMENT. Until validated, it is a steering tool only.
9. `RUN/COMMANDMENT.md`:
   - commands, `PARSE`, `METRIC`;
   - `NOISE`, any bias, eval wall times, `FAST_CMD` and its validation;
   - skipped gates, `EDITABLE`, `FROZEN`, the engagement result;
   - the decision instrument for the run: `FAST_CMD` when validated and the eval is slow, otherwise
     official cross-overs.
   Also write `RUN/baseline.json` and `cands/c000/result.json` (speedup 1.0).

## MODE=measure (official result for `cands/$ID`)
1. `M=$RUN/cands/$ID/m_<ts>`: `mkws.sh $RUN $M/cand <patch>` and `mkws.sh $RUN $M/base`. If the patch
   does not apply → `apply_failed`. If it changes anything outside `EDITABLE` → `frozen_touched`.
2. Run correctness in `$M/cand`. On failure → `incorrect`, with the log tail.
3. Timing: `bash $SK/bin/ab.sh $RUN $M/base $M/cand $M/ab <reps>` (use `-` for base when
   `PAIRING=in_process`), detached per COMMON, then `python3 $SK/bin/score.py $M/ab`. Reps: 3 when the eval
   is cheap, 1 when it is slow.
4. Write `cands/$ID/result.json`: `{id, status, speedup, metric, noise, spread, reps, gpu, instrument, per_case, notes}`.
   Each `per_case` row: `{sig, regime, weight, speedup, base_ms, cand_ms}`, using the official metric's weights,
   so that `bin/archive.py` can score regimes and dispatch composites. Fast-probe results go to
   `result.fast.json` in the same format.

## MODE=xover (decide between two candidates)
With the run's decision instrument:
`bash $SK/bin/xover.sh $RUN <base|-> <ws_x> <ws_y> $RUN/xover/<x>_<y>_<ts> <pairs>`
(two GPUs: simultaneous, then swapped; one GPU: alternating order).
- Pairs: 2 by default, 4 for a close call. With the slow official eval use `FAST_CMD`, then one official
  run of the winner.
- Write `delta_mean` (y − x), the per-pair deltas and both series to `RUN/xover/<x>_<y>_<ts>.json`.
- A delta inside `NOISE` is a tie. Never compare runs taken at different times on different GPUs.

Return 3 lines: status, speedup ± noise (or delta), result path.
