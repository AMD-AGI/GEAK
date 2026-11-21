# avo-source — Testing (kernel-level)

Rigorous, layered testing for the AVO kernel engine, **before** any e2e
integration. Three layers, increasing cost: deterministic unit tests (no
GPU/LLM), an integration smoke run on the `knn` example (GPU + LLM), and an
acceptance/quality gate.

> Run the cheap deterministic layer by hand after every change here (it is not
> part of GEAK's CI); run the integration layer on the rig before trusting a real
> evolution run.

**Headless shim:** the versioned entry point is `run_avo.mjs` in this directory; see
the README one level up (*4. Node shim*). Commands below run from this directory
(`avo-source-impl/`).

---

## Layer 1 — Deterministic unit tests (no GPU, no LLM)

These pin the **exactness-critical** logic — the commit gate and the lineage
writer — which must never be left to LLM judgement.

> **Prereq:** the `node` suites import from `lib/`. GEAK's root `.gitignore` has a
> generic `lib/` rule; the `.gitignore` in this directory negates it (`!/lib/`). If a
> fresh clone has no `lib/`, the import (and the workflow) fails — re-check that file.

### 1a. Commit gate (JS) — `tests/commit_gate.test.mjs`

Tests `lib/commit_gate.mjs` (the single source of truth imported by
`avo_workflow.js`): geomean math, anti-lazy floor, B1 significance margin,
correctness/failed handling, custom opts.

```bash
node tests/commit_gate.test.mjs    # exit 0 = all pass
```

(Node is required only on the rig; the gate is pure and has no runtime deps.)

### 1b. Lineage writer (Python) — `tests/test_avo_lineage.py`

Tests `scripts/avo_lineage.py`: geomean, `seed` (+ `baseline.json` for resume),
`commit` (message format incl. **latency + expected**, lineage.json advance,
per-shape, side-effect files), **commit refusal** below best / no positive speedup,
subject truncation, `context` (top-k priors), and **`resume`** (reconstructs
committed/best/baseline/failed/direction from durable files).

```bash
python -m pytest tests/test_avo_lineage.py -q
```

**Invariant under test (defense in depth):** even if the JS gate were wrong,
`avo_lineage.py commit` refuses (non-zero exit) a candidate that does not beat the
current best — so a bad number can never enter `lineage.json`. The roofline tests
also pin the `flops`/`bytes`→regime classification and that it stays **inert** (no
crash, fields omitted) when telemetry is absent.

### 1c. Start-point selection (JS) — `tests/start_select.test.mjs`

Tests `lib/start_select.mjs` (Diverse Fan-out): `best` returns only the global
best; `topk` is best-first by speedup; `diverse` is one-per-strategy-bucket;
`k=1` collapses to best; baseline (v0) excluded; empty lineage → `[]`.
(`elites` policy is covered by `archive.test.mjs`.)

```bash
node tests/start_select.test.mjs    # exit 0 = all pass
```

### 1d. Roofline model (JS) — `tests/roofline.test.mjs`

Tests `lib/roofline.mjs`: ridge point per dtype, regime classification
(memory-bound / compute-bound / basement / unknown), `analyzeCase` fraction +
potential-speedup on synthetic memory/compute cases, `rooflineSummary` worst-case
selection, lever→specialty mapping, and graceful nulls without telemetry.

### 1e. Headroom & stop (JS) — `tests/headroom.test.mjs`

Tests `lib/headroom.mjs`: roofline early-stop (at/above target, min-commits guard,
relative-only no-stop) and Amdahl `allocateAttention` (uniform vs time-weighted
focus; saturated cases filtered).

### 1f. MAP-Elites archive (JS) — `tests/archive.test.mjs`

Tests `lib/archive.mjs`: cell keying (strategy×regime), best-per-cell archive,
`selectElites` (best-first, baseline excluded), `crossoverPairs` (different regimes),
and `selectMergeGroup`/`mergeImproves` (same-base complementary merge gate).

### 1g. Measurement rigor + autotune (JS) — `tests/measure.test.mjs`, `tests/autotune.test.mjs`

`measure.mjs`: median/MAD/cv, MAD outlier drop, `summarizeRepeats`. `autotune.mjs`:
capped+even-subsampled `expandConfigSpace`, `selectBestConfig` (verified then
roofline tie-break), `excludeFailed`. The variance-aware commit gate is covered in
`commit_gate.test.mjs`.

### 1h. rocprof parse (py) — `tests/test_rocprof_probe.py`

`rocprof_probe.py` parses counter CSV into measured bytes, occupancy, and MFMA
metrics using a deterministic fixture. Standalone ASM/IR and IR-feedback tests
are intentionally excluded from this source-only integration.

### 1i. Agent call end (JS) — `tests/agent_contract.test.mjs`, `tests/agent_early_finish.test.mjs`
- `agent_contract`: what counts as an agent's final answer — the whole result text
  is one JSON object (a single code fence is accepted) with every required key;
  progress text, JSON quoted inside prose, arrays/scalars and objects missing a key
  do not.
- `agent_early_finish`: drives the real `run_avo.mjs` against a fake `claude`
  (`tests/fixtures/fake_claude.mjs`) that replays `result` events and then stays
  alive. An agent that exits right after its answer takes the unchanged path; an
  answered agent that stays alive is stopped after the grace; progress text, quoted
  JSON, error results and results missing a key do not end the call; a background
  process that ignores SIGTERM is gone when `agent()` returns; without a final
  answer the call still ends on process exit or on the wall-clock timeout.
- `agent_reaper` (extended): `waitTagGone` returns as soon as no tagged process is
  left, and reports the survivors at its deadline.

### Run the whole deterministic layer

One command (runs pytest and every node suite; a missing pytest or Node.js 18+
is reported and counts as a failure):

```bash
bash run_tests.sh
```

…or manually:

```bash
python -m pytest tests/ -q
for t in tests/*.test.mjs; do node --test "$t"; done
```

---

## Layer 2 — Integration smoke (rig: GPU + LLM)

End-to-end on GEAK's canonical `examples/tasks/knn` task with a **short wall-clock
budget**, to prove the whole chain runs and the contract holds. Run it on a copy
outside the GEAK checkout, so the run writes nothing into the repository.

`handoff.json`:

```json
{
  "kernel_path": "<COPY>/knn",
  "budget_s": 900,
  "gpu_ids": "<idle gpu>",
  "verify_repeats": 3,
  "apply_to_original": "false"
}
```

```bash
python3 run_geak_avo.py handoff.json result.json --dry-run
python3 run_geak_avo.py handoff.json result.json
```

`budget_s: 900` is a 15-minute smoke; raise it once green. Pick the GPU with
`rocm-smi`.

Pre-flight (fair baseline + idle GPU):
```bash
rm -rf <COPY>/knn && cp -r <GEAK>/examples/tasks/knn <COPY>/knn   # <COPY> is outside <GEAK>
rocm-smi   # pick an idle GPU id for gpu_ids
```

### What to assert after the run

1. **Phases ran in order:** Setup → Benchmark → Seed → (Profile) → Evolve → Report
   → Validate (check the workflow log).
2. **Lineage seeded:** `avo_repo` has tag `avo-v0`; `avo_state/lineage.json` has a
   `v0` node.
3. **Contract object returned** with the team_workflow-compatible shape:
   `{eval_dir, final_geomean, final_patch, validation_status, report_path}`.
4. **`director_validation.json` exists** and its
   `director_verified_speedup_geomean` is **independently** measured (not the
   runner's number).
5. **Integrity:** the copied `knn` kernel directory is byte-identical: rerun the
   `find ... sha256sum` command from `roles/director.md` on it and compare with
   `orig_manifest.sha256`.
6. **If any commit happened:** `final_patch.diff` applies cleanly on a fresh copy
   of the original and passes the COMMANDMENT correctness command; each `avo-v{N}`
   commit message carries speedup/strategy/per-shape (rich message).
7. **Lineage report renders:**
   `python scripts/avo_lineage_report.py <eval_dir>/avo_state --format md`.

### Smoke acceptance

- The run **completes without an exception** and `result.json` has `status`
  `ok` or `no_gain` (`validation_status` `accepted` or `no_significant_gain`).
- `final_geomean >= 1.0` and, if `num_commits > 0`, the final patch is correct.
- A 0-commit smoke (budget too short to find a win) is still a PASS for the
  *plumbing* as long as 1–7 hold — it tests the loop, not the optimizer quality.

### 2b. Fan-out variants (operator=team / Diverse Fan-out)

Run the same knn smoke with the heavier operator + diverse starts to exercise the
parallel paths and the **tree** lineage:

```json
{ "operator": "team", "operator_directions": 2, "start_policy": "diverse",
  "start_k": 2, "gpu_ids": "<g0>,<g1>" }
```

Add these keys to the handoff above (one worker per GPU).

Extra assertions on top of 1–7:

8. **Parallel workers ran:** the log shows N `variation_<step>_i@<startRef>` workers
   per step, each on its own GPU; each was independently re-measured by `avo_verify`
   on its **own base ref** (no cross-GPU benchmark interference).
9. **Tree (when start_policy≠best):** at least one committed node's `parent_id` is
   **not** the immediately-preceding best (a branch off an earlier version) —
   inspect `avo_state/lineage.json` / `git -C avo_repo log --graph --oneline`.
10. **Best still advances on global improvement only:** `best_id` always has the
    max `score.speedup` in `lineage.json` (a behind-branch commit never demotes it).
11. **start=best is a pure chain:** re-running with `start_policy:"best"` yields a
    linear `parent_id` chain (regression guard that the default is unchanged).

### 2c. Resume

Interrupt a run (Ctrl-C mid-Evolve), then re-invoke with the **same `eval_dir`**:

```json
{ "eval_dir": "<the interrupted run's eval_dir>" }
```

Add this key to the same handoff.

Assert:
12. Setup logs `(RESUME)` and **skips Benchmark/Seed** (no re-baseline); the loop
    resumes from the prior best (`best=avo-v{N}` with the prior commit count).
13. New step dirs **don't collide** with the prior run's (`variation_*` numbering
    continues past existing ones).
14. Previously-failed directions are still avoided (restored from
    `supervisor_log.jsonl`); final `director_validation.json` reflects the combined
    lineage.

---

## Layer 3 — Quality / acceptance gate (rig, longer)

Run a real budget on `knn` and compare against the existing engines.

| Check | Target |
|-------|--------|
| Correctness of final kernel | 100% (COMMANDMENT correctness passes on the original) |
| Verified geomean (Director) | > 1.0; ideally competitive with `team_workflow` on knn |
| Commit discipline | committed versions ⊂ verified improvements; `attempts.jsonl` ≫ commits |
| Noise robustness | re-running Director validation twice agrees within the variance band (verify_repeats median working) |
| No fake progress | every `avo-v{N}` reproduces its speedup on a clean re-apply |

A/B vs `team_workflow` (depth vs breadth) is **Step 4** in `PLAN.md` — same
kernel, same wall-clock, compare speedup/hour and final geomean.

---

## What is NOT yet covered (follow-ups)

- **Supervisor / ESCALATE paths** are only exercised on a kernel that actually
  stalls; add a deterministic stall fixture (e.g. force `stall_no_improve=1`,
  `supervisor_cycles_to_escalate=1` with a no-op kernel) to smoke the
  REDIRECT/ESCALATE branches cheaply on the rig.
- **Backtrack** (`supervisor.backtrack_to`) needs a multi-commit lineage to test
  the checkout path. Note: `avo_verify` is now base-ref aware, so a backtracked
  start verifies on the correct ref (smoke this once a multi-commit run exists).
- **Kernel roles** (`benchmark_engineer` / `profile_engineer` / `tech_lead`) are
  AVO's own copies in `kernel_base/roles/`; the Layer-2 smoke confirms that the
  inputs AVO passes match what they expect.
  Independent verification uses avo-workflow's **own** `roles/avo_verify.md`
  (base-ref aware), not team's `verify_engineer`.
- **Island model** (persistent per-branch budgets/best) is explicitly out of scope
  for now; `start_policy=diverse` is single-best-advancing tree exploration, not
  islands (see the start-point design doc).
