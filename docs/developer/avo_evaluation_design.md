# AVO Profiling & Evaluation — Module Design

The scoring function `f` in AVO (Agentic Variation Operators) decides whether a
variation step produced real progress: it must turn a candidate kernel into a
**trustworthy, independently-verified** speedup (and a profiler reading) that the
commit gate and the supervisor can rely on — never the agent's self-report. This
document specifies AVO's profiling & evaluation subsystem as a self-contained
module: the per-step verification flow, how the verified speedup is computed,
the profiling stages and their timeouts, the cost-saving guards, how results
feed the commit gate / supervisor / memory, ESCALATE evaluation, finalize, and
the isolation guarantees.

> Scope: a deep-dive companion to [`avo_design.md`](avo_design.md) §4.2 (commit
> gate), §13 (P0 scoring & verification), §16.3 (delayed profiling), and §17.5/
> §17.7 (cost guards & noise robustness). The AVO-side code lives in
> `src/minisweagent/run/avo/controller.py` (`_apply_verified_score`,
> `_build_verify_ctx`, `_eval_once`, …); the heavy lifting reuses GEAK's
> `run/postprocess/evaluation.py` (`evaluate_round_best`, `run_profile`) without
> modification.

---

## 1. The scoring contract `f`

AVO's `f = correctness + throughput` is realized by GEAK's existing machinery:

- **Correctness + throughput definition** lives in `COMMANDMENT.md` (produced by
  preprocess): the SETUP / CORRECTNESS / FULL_BENCHMARK / PROFILE sections.
- **Per-step measurement** is the agent's `save_and_test` tool (writes
  `patch_*.patch` + `patch_*_test.txt` into the worker dir).
- **Independent verification** is `evaluate_round_best`: it re-applies the best
  patch in a *fresh* worktree and re-runs FULL_BENCHMARK + PROFILE, producing the
  authoritative number.

Cardinal rule: **the agent's self-reported speedup is never trusted for a
commit.** Only the independently-verified value enters the lineage (§5).

---

## 2. Where a step's candidates live

A variation step writes into GEAK's canonical round layout so the evaluator can
consume it unchanged:

```text
<output_dir>/results/round_{step}/avo-worker/
├── patch_*.patch            # save_and_test patches
├── patch_*_test.txt         # per-attempt correctness + speedup logs
└── best_results.json        # the agent's self-selected best (speedup, patch file, per-shape)
```

ESCALATE rescue workers use `results/round_{9000+n}/rescue-worker-{k}/` instead
(via the `_escalate_patch_dir` hint). Verified evaluations are written to
`<output_dir>/round_{step}_evaluation.json`.

---

## 3. The per-step verification flow

After a step runs, the controller calls `_apply_verified_score`, which overwrites
the step's score with a noise-robust **verified** speedup before the commit gate:

```python
def _apply_verified_score(result, verify_ctx, step_index, output_dir, *,
                          repeats=1, min_commit_speedup=1.0, skip_on_no_gain=True,
                          cache=None, cache_path=None) -> None:
    self_sp, cand_patch = _peek_candidate(output_dir, step_index)   # agent's self-report

    # C3: agent itself reports no gain → can't clear the floor → skip the expensive bench
    if skip_on_no_gain and self_sp is not None and self_sp <= min_commit_speedup:
        result.best_speedup = None; result.best_correct = False; result.best_patch_path = None
        return

    # C2: identical patch already verified → reuse cached verified speedup
    fp = _patch_fingerprint(cand_patch)
    if cache is not None and fp and fp in cache:
        result.best_speedup = float(cache[fp]["speedup"]); ...; return

    # B1: run evaluate_round_best `repeats` times, take the median (+ per-shape median)
    samples, per_shapes, best_patch = [], [], None
    for _ in range(max(1, repeats)):
        out = _eval_once(verify_ctx, step_index, output_dir)
        if out is None: continue
        sp, patch, per_shape = out
        samples.append(sp); per_shapes.append(per_shape); best_patch = patch

    if not samples or best_patch is None:
        result.best_speedup = None; result.best_correct = False; result.best_patch_path = None
        return

    verified = statistics.median(samples)
    result.best_speedup = float(verified)
    result.best_correct = True
    result.best_patch_path = Path(best_patch)
    result.per_shape_speedups = _median_per_shape(per_shapes) if repeats > 1 else (per_shapes[0] if per_shapes else {})
    result.profiling = _read_profile_metrics(output_dir, step_index)   # P-mem-3 causal signal
    ...                                                                 # cache + record attempt
```

`_eval_once` is one independent verification:

```python
def _eval_once(verify_ctx, step_index, output_dir):
    from minisweagent.run.postprocess.evaluation import evaluate_round_best
    results_dir = output_dir / "results" / f"round_{step_index}"
    round_eval = evaluate_round_best(dict(verify_ctx), step_index, results_dir)   # fresh worktree + FULL_BENCHMARK
    if round_eval is None or not getattr(round_eval, "best_patch", ""):
        return None
    fb = getattr(round_eval, "full_benchmark", None)
    verified = fb.verified_speedup if fb and fb.verified_speedup is not None else round_eval.benchmark_speedup
    if verified is None:
        return None
    return float(verified), round_eval.best_patch, _read_per_shape_speedups(output_dir, step_index)
```

Because each call builds a fresh worktree and re-runs FULL_BENCHMARK, repeated
calls are **independent measurements of the same candidate** → pure measurement
noise, which the median suppresses.

The verify context is built once per run, pointed at the **isolated work repo**
(§9):

```python
def _build_verify_ctx(repo, output_dir, gpu_ids, task) -> dict:
    return {
        "output_dir": str(output_dir), "preprocess_dir": str(output_dir),
        "repo_root": str(repo),                   # = work_repo (isolated clone)
        "harness_path": _discover_harness_path(output_dir),
        "gpu_ids": list(gpu_ids), "num_parallel": 1, "metric": None,
        "starting_patch": "", "_best_global_speedup": 0, "user_instructions": task,
    }
```

---

## 4. Inside `evaluate_round_best` (the verified score)

File: `src/minisweagent/run/postprocess/evaluation.py`.

### 4.1 Candidate selection

It scans `results/round_{step}/*/best_results.json`, drops non-improving /
patchless candidates, and picks the best by **min kernel time** when all
candidates report it, else by **max self-reported speedup**:

```python
all_have_kernel_time = all(c["kernel_time_ms"] is not None for c in candidates)
best = min(candidates, key=lambda c: c["kernel_time_ms"]) if all_have_kernel_time \
       else max(candidates, key=lambda c: c["speedup"])
```

(For a normal AVO step there is a single `avo-worker` candidate; selection
matters for ESCALATE's N workers.)

### 4.2 Apply → CORRECTNESS → FULL_BENCHMARK → PROFILE

```python
eval_worktree, eval_env = resolve_eval_worktree(repo_root, best_patch_file, harness_path, output_dir, eval_gpu_ids)
try:
    run_correctness_and_benchmark(eval_worktree, eval_env, commandment_path, pp_dir, round_eval, round_num, ...)
    run_profile(eval_worktree, eval_env, commandment_path, pp_dir, round_eval, round_num, results_dir)
finally:
    cleanup_eval_worktree(repo_root, eval_worktree)
```

The eval worktree is a **separate** detached worktree (`_eval_worktree`) created
from `repo_root` (= AVO's work repo) and removed afterward, so verification never
dirties the step's working tree.

### 4.3 Verified speedup — direct latency, then per-shape geomean

```python
candidate_ms = extract_latency_ms(candidate_stdout)
baseline_ms  = extract_latency_ms(baseline_text)
if candidate_ms is None or baseline_ms is None:
    # multi-shape harness: compute per-shape speedups, take the geometric mean
    shape_speedups = compute_shape_speedups(baseline_shapes, candidate_shapes, direction)
    geomean = math.exp(sum(math.log(v) for v in vals) / len(vals))
    round_eval[section_key]["verified_speedup"] = round(geomean, 4)
    round_eval[section_key]["per_shape_speedups"] = shape_speedups
else:
    verified_speedup = compute_speedup(baseline_ms, candidate_ms, direction)
```

Key properties:

- **Metric direction** comes from the baseline metric (`lower_is_better` by
  default), so latency and throughput metrics are handled correctly.
- **Per-shape geomean** is the multi-config score (matches AVO §4.1); per-shape
  values are persisted and used by the commit gate's regression guard (§5).
- **Broken-measurement rejection**: a non-positive candidate latency is
  rejected with a distinct `failure_reason` (not silently treated as "1.0x").

### 4.4 The typed result

`write_eval_results` returns a `RoundEvaluation` whose
`full_benchmark.verified_speedup` is the authoritative number; `speedup_source`
records whether it is a "FULL_BENCHMARK verified result" or only agent-reported
(no verified value available).

---

## 5. From verified score to a commit

The verified value feeds `LineageStore.maybe_commit`, whose gate combines four
checks (`avo_design.md` §4.2 / §16.1 / §17.3 / §17.7):

```python
# 1. correctness + verified candidate exists
if not result.produced_verified_improvement_candidate: return False
# 2. anti-lazy floor: must exceed min_commit_speedup vs baseline (reject ~1.0x trivial rewrites)
if candidate_speedup <= self.min_commit_speedup: return False
# 3. vs current best: stricter of epsilon tolerance and the B1 significance margin
threshold = self.best_speedup * max(1.0 - self.epsilon, 1.0 + self.significance_margin)
if candidate_speedup < threshold: return False
# 4. per-shape regression guard: reject if any shape < min_per_shape_speedup
if self.min_per_shape_speedup > 0 and any(v < floor for v in result.per_shape_speedups.values()): return False
```

Only on passing all four does a new `avo-v{N}` enter the lineage. `committed` is
then the single "progress" signal the StagnationDetector consumes (see
[`avo_supervisor_design.md`](avo_supervisor_design.md)).

---

## 6. Profiling

### 6.1 `run_profile` (verification-time profiling)

```python
profile_script = build_eval_script(str(commandment_path), ["SETUP", "PROFILE"])
profile_result = subprocess.run(["bash", profile_script],
                                capture_output=True, text=True,
                                timeout=1800, cwd=str(eval_worktree), env=eval_env)
# → writes profile.json; mutates round_eval["profile_comparison"]
```

It runs the COMMANDMENT's SETUP + PROFILE on the best kernel; output drives
`profile.json` (consumed downstream).

### 6.2 Two consumers of profiling

- **Supervisor bottleneck** — `supervisor._read_profile_bottleneck` reads
  `profile.json`'s `bottleneck` / `limiter` / `summary` field into the
  stagnation bundle, so re-planning is grounded in the real limiter.
- **Causal memory (P-mem-3)** — `controller._read_profile_metrics` scans
  `round_{N}_evaluation.json` for scalar perf fields (occupancy, bandwidth,
  tflops, latency, register, lds, …) and stores up to 8 on the result; the
  evolution log then shows the bottleneck **delta** across versions (see
  [`avo_memory_design.md`](avo_memory_design.md)).

### 6.3 Delayed profiling (CuTeGen, §16.3)

Profiler feedback is **withheld** early so the agent first gets the structure
right, then switches to profiling-guided micro-tuning:

```python
# controller, per step:
profiling_enabled = step_idx > profiling_after_step or len(lineage.committed) > 1
```

`compose_task` injects a `STRUCTURAL (profiling withheld)` stage note for the
first `profiling_after_step` steps (default 3, unless a real commit already
landed), then a `PROFILING-GUIDED` note. Set `profiling_after_step: 0` for
simple/elementwise kernels to profile early.

---

## 7. Timeouts (important, and easy to confuse)

AVO defines **no** profiling timeout of its own; it inherits GEAK's. There are
two distinct profiling paths with different timeouts:

| Stage / path | Timeout | Source | Configurable |
|--------------|---------|--------|--------------|
| **Verification PROFILE** (`evaluate_round_best` → `run_profile`) | **1800 s (30 min)** | `evaluation.py` `run_profile`, hardcoded | ❌ no env/config knob |
| Verification FULL_BENCHMARK (candidate) | 1800 s | `evaluation.py` | ❌ |
| Verification baseline FULL_BENCHMARK | 1200 s | `evaluation.py` | ❌ |
| Verification CORRECTNESS | 600 s | `evaluation.py` | ❌ |
| Harness `--profile` mode (`run_harness` MODE_TIMEOUTS) | **120 s** | `GEAK_PROFILE_TIMEOUT` | ✅ env |
| Harness correctness / benchmark / full-benchmark | 900 / 600 / 900 s | `GEAK_BENCH_TIMEOUT` / `GEAK_CORRECTNESS_TIMEOUT` | ✅ env |
| Preprocess baseline profiler-mcp | 120 s | `GEAK_PROFILE_TIMEOUT` | ✅ env |
| Agent env command (per variation step) | 3600 s | `avo.env_timeout_s` (geak_avo.yaml) | ✅ config |
| Agent profiling tool (make/analysis) | 6 h each | `profiling_tools.py` | code |

> The common confusion: `run_harness`'s `MODE_TIMEOUTS["profile"]` (120 s, env
> `GEAK_PROFILE_TIMEOUT`) governs the `--profile` *harness* mode (mainly
> preprocess), while AVO's per-step verification profiling goes through
> `run_profile` (a bash SETUP+PROFILE script) at a **hardcoded 1800 s**. To make
> the latter configurable, `evaluation.py`'s `run_profile` timeout must be
> parameterized (not currently done).

---

## 8. Cost & noise guards

A multi-day run cannot afford a full FULL_BENCHMARK every step when it can't
change the outcome. Three guards (config-driven):

| Guard | Knob | Effect |
|-------|------|--------|
| **C3 skip-on-no-gain** | `skip_verify_on_no_gain` (default on) | if the agent self-reports ≤ `min_commit_speedup`, skip verification (can't commit anyway) |
| **C2 dedup cache** | always on (`avo_state/verify_cache.json`) | identical patch (by content hash) reuses its cached verified speedup |
| **B1 noise-robust median** | `verify_repeats` (default 1), `commit_significance_margin` (default 0) | re-measure N× and take the median; require `best·(1+margin)` to count as a new best |
| **whole-step verify toggle** | `verify_each_step` (default true) | set false to fall back to the lightweight log parse when per-step FULL_BENCHMARK is too costly |

These raise the *objectivity* of the progress signal that drives the
deterministic supervisor while keeping GPU cost in check.

---

## 9. Isolation

Two layers of isolation keep verification from touching the user's repo:

1. AVO's whole loop runs on the **isolated work repo** (`avo_repo`, an
   independent clone), so `verify_ctx["repo_root"]` = work repo, never the
   user's repo.
2. `evaluate_round_best` itself builds a **temporary `_eval_worktree`** from
   `repo_root`, applies the patch there, runs the benchmark/profile, and removes
   it in a `finally`. So even the work repo's tree is not mutated by
   verification.

Net: per-step verification is doubly sandboxed; the verified `best → candidate`
patch applies onto the same base the agent edited (the work repo's current best).

---

## 10. ESCALATE evaluation (best-of-N)

The diversified rescue reuses the *same* evaluator to pick and verify the best of
several distinct directions, then folds it into the lineage through the same
commit gate:

```python
rescue_round = 9000 + len(lineage.committed)
for k, strat in enumerate(_diversified_directions(n_workers)):
    run_variation_step(..., avo_config=_with_patch_dir(avo_cfg, rescue_dir / f"rescue-worker-{k}"))
round_eval = evaluate_round_best(verify_ctx, rescue_round, rescue_dir)   # best-of-N, verified
lineage.commit_from_round(round_eval, repo=repo)                          # commit gate (prefers verified)
```

`commit_from_round` prefers `full_benchmark.verified_speedup` over the
agent-reported `benchmark_speedup` and rejects patchless rounds.

---

## 11. Finalize

At run end, `_finalize` reuses GEAK's `auto_finalize` so `final_report.json`
keeps its canonical shape. Two things make finalize reflect the AVO-authoritative
best:

- `LineageStore.build_postprocess_ctx` first **materializes the verified
  committed best** into `results/round_999999/avo-best/best_results.json` (+ the
  patch), so the `auto_finalize` scanner surfaces it.
- `auto_finalize` also runs `select_best_verified_round_evaluation`, which
  prefers FULL_BENCHMARK-verified round evaluations — aligning the final report
  with the lineage's verified best.

`_write_trajectory` additionally emits `trajectory.json` (running-best curve,
commit rate, per-strategy stats) for observability.

---

## 12. Configuration

`avo.*` in `src/minisweagent/config/geak_avo.yaml`:

```yaml
avo:
  verify_each_step: true            # per-step FULL_BENCHMARK + per-shape geomean (accurate; slower)
  verify_repeats: 1                 # B1: re-measure N×, take median (noise-robust); 3 on noisy harnesses
  commit_significance_margin: 0.0   # B1: require best*(1+margin) to commit (noise floor)
  skip_verify_on_no_gain: true      # C3: skip verify when self-report ≤ floor
  min_commit_speedup: 1.0           # anti-lazy floor vs baseline
  min_per_shape_speedup: 0.0        # per-shape regression guard (0 = off; e.g. 0.95)
  commit_epsilon: 0.001             # tolerance vs current best
  profiling_after_step: 3           # delayed profiling: structural-first for N steps (0 = profile early)
  env_timeout_s: 3600               # per-step agent env command timeout
```

Environment overrides for the harness-level timeouts: `GEAK_PROFILE_TIMEOUT`,
`GEAK_BENCH_TIMEOUT`, `GEAK_CORRECTNESS_TIMEOUT`. `GEAK_AGENT_SELECT_PATCH=1`
trusts the agent-selected patch and skips CORRECTNESS/FULL_BENCHMARK/PROFILE
(debug only — do not use for real AVO runs, it defeats verification).

---

## 13. Residual gaps (by design / future)

- **Verification PROFILE timeout is hardcoded (1800 s).** Not exposed via config;
  parameterizing `run_profile` is a small follow-up if needed.
- **Cost of `verify_each_step`.** Per-step FULL_BENCHMARK is the dominant GPU
  cost; the C2/C3 guards mitigate it, and `verify_each_step: false` trades
  accuracy for speed.
- **Single-GPU verification** (`num_parallel: 1` in `verify_ctx`). ESCALATE is
  the only place AVO exploits multiple candidates; a best-of-K *main* loop is
  deferred (`avo_design.md` §17.5 C1).
- **Profiler metric extraction is heuristic** (`_PROFILE_METRIC_HINTS` keyword
  scan), cross-backend best-effort; richer structured profiling could improve
  the causal-memory delta.

---

## 14. References

- [`avo_design.md`](avo_design.md) — commit gate (§4.2), P0 scoring &
  verification (§13), delayed profiling (§16.3), cost/noise guards (§17.5,
  §17.7).
- [`avo_supervisor_design.md`](avo_supervisor_design.md) — consumes the verified
  `committed` signal + `profile_bottleneck`.
- [`avo_memory_design.md`](avo_memory_design.md) — consumes profiler metrics for
  the causal evolution log (P-mem-3).
- `src/minisweagent/run/avo/controller.py` — `_apply_verified_score`,
  `_eval_once`, `_build_verify_ctx`, `_read_profile_metrics`, `_do_escalate`,
  `_finalize`.
- `src/minisweagent/run/postprocess/evaluation.py` — `evaluate_round_best`,
  `_compute_verified_speedup`, `run_profile`, `setup_eval_worktree`,
  `MODE_TIMEOUTS`.
- `src/minisweagent/run/avo/lineage_store.py` — `maybe_commit`,
  `commit_from_round`, `build_postprocess_ctx`.
