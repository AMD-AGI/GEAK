# AVO Roofline Design (avo-workflow)

How `avo-workflow` turns AVO's objective from **purely relative** ("beat the last
version") into a **hardware-anchored** one ("approach the MI300X roofline, then
stop") — the change that lets the engine deliberately close the gap to the limit
and out-optimize a human expert instead of plateauing early.

> Companion: [`avo_knowledge_and_profiling.md`](avo_knowledge_and_profiling.md)
> (profiling staging), [`avo_supervisor_design.md`](avo_supervisor_design.md)
> (steering on stall), and the lever map in
> [`../knowledge/roofline_playbook.md`](../knowledge/roofline_playbook.md).

---

## 1. The problem with a relative-only objective

The commit gate scores a candidate against the **running best** (`geomean(baseline/
candidate)` ≥ best, within margin). That answers *"did it improve?"* but never:

1. **How close to the hardware ceiling are we?** A 2.0× kernel may be at 20% of the
   roofline (huge headroom) or 90% (done) — relative scoring can't tell.
2. **Which ceiling binds us?** Memory BW, compute (MFMA), or neither (occupancy /
   launch overhead)? Without this, steering is a guess.
3. **When should we stop?** A wall-clock budget keeps grinding for sub-noise gains
   long after the real limit is reached.

A roofline model answers all three, so the search has a true north star.

## 2. Deterministic vs LLM split (the Workflow doctrine)

| Concern | Owner |
|---------|-------|
| Roofline math (AI, bound, % of peak), regime classification, lever menu | **pure JS** `lib/roofline.mjs` ⇄ **pure Python** `scripts/avo_lineage.py` (kept in sync; unit-tested) |
| Early-stop decision + Amdahl attention allocation | **pure JS** `lib/headroom.mjs` |
| Quality-diversity archive (cells) + crossover pairing | **pure JS** `lib/archive.mjs` |
| Wiring: compute summary after verify, inject steer, stop loop, store regime | **deterministic** `avo_workflow.js` |
| Picking the actual edit for a regime; emitting `flops`/`bytes` telemetry | **LLM** `roles/variation_engineer.md`, `roles/avo_verify.md` |
| Regime-targeted re-plan on stall | **LLM** `roles/supervisor.md` |

The numbers and decisions are **code** (reliable, reproducible); only the *edit* and
the *FLOP/byte derivation* are LLM judgement.

## 3. The model (per shape)

Given per-case telemetry `{latency_ms, flops, bytes, dtype}` and the MI300X peak
table (`kernel_knowledge/hardware/cdna3_mi300/peak_tables.md`, encoded once per
language):

- **Arithmetic intensity** `AI = flops / bytes` (FLOP/byte).
- **Roofline bound** `= min(peak_compute[dtype], AI · HBM_BW)` (attainable FLOP/s).
- **Achieved** `= flops / latency_s`; **achieved BW** `= bytes / latency_s`.
- **Fraction of roofline** `= achieved / bound` ∈ (0,1] — the headline number.
- **Potential speedup** `= bound / achieved` (how much faster is still attainable).
- **Ridge** `AI* = peak_compute / HBM_BW` (fp16 ≈ 245 FLOP/B).

MI300X anchors (gfx942): fp16/bf16 **1307 TF**, fp8/int8 **2615 TF**, fp32 163 TF,
HBM3 **5.325 TB/s**. (`gfx942_mi325` = MI325X, 6.0 TB/s, same compute.)

## 4. Regime = the binding constraint

Let `mem_frac = achieved_BW/peak_BW`, `cmp_frac = achieved/peak_compute`.

| Regime | Condition | Limiter | Lever track |
|--------|-----------|---------|-------------|
| **basement** | `max(mem_frac, cmp_frac) < 0.35` | latency / occupancy / launch | `host_runtime` |
| **memory-bound** | not basement, `AI < ridge` | HBM bandwidth | `memory` |
| **compute-bound** | not basement, `AI ≥ ridge` | MFMA / FLOP ceiling | `compute` |
| **unknown** | no flops/bytes telemetry | — (relative-only) | `algorithm` |

`recommendLevers(regime)` returns the matching specialist track (same taxonomy as
`team_workflow`) + a ranked lever list (see the playbook).

## 5. Gap-driven steering

Each step the loop builds a steer block from the **current best**'s roofline:

```
ROOFLINE = { regime, fraction_of_roofline, worst_case, focus_case, specialty, levers }
```

- `worst_case` / `focus_case` = the shape with the most **potential speedup** (via
  `allocateAttention`, Amdahl-weighted by runtime when shares are known) — attack
  where the opportunity is largest, not an already-saturated shape.
- Injected into `variation_engineer` (single + fan-out), `supervisor`, and the
  `operator=team` planner. The agent picks one lever from the regime's list.

This is "profiling-guided" raised to the roofline level: the regime is derived from
*measured* achieved-vs-peak, not a profiler heuristic.

## 6. Headroom control — stop & allocation (`lib/headroom.mjs`)

- **`shouldStopRoofline(summary, {target, commits, minCommits})`** — stop the Evolve
  loop once `geomean(fraction_of_roofline) ≥ roofline_target` (default 0.85) **and**
  `min_commits` met. Banks the remaining wall-clock budget at the real limit.
- **`allocateAttention(analyses, {weights})`** — rank shapes by
  `(potential_speedup − 1) × weight`; saturated shapes (≈1.0) drop out. `weights`
  defaults to uniform; pass runtime shares for Amdahl prioritization.

## 7. Quality-diversity search (`lib/archive.mjs`, pillars 5+6)

A single best-chain collapses to one local optimum. The MAP-Elites archive keeps
the **best version per behavioral cell** = `(strategy × regime)`:

- **`selectElites(committed, k)`** → diverse fan-out start points (best first, one
  per cell). Exposed as `start_policy="elites"` (richer than `diverse`'s 1-D
  strategy buckets).
- **`crossoverPairs(committed)`** → pairs of elites from **different regimes** (a
  memory-bound winner + a compute-bound winner), ranked by combined speedup — the
  substrate for a future crossover variation primitive.
- **`archiveStats`** → cells/regimes coverage for the report.

## 8. Telemetry contract & storage (pillar 7)

- `roles/avo_verify.md` (and the COMMANDMENT benchmark) emit per-case `flops`,
  `bytes`, `dtype`. These are **problem properties** (identical for baseline and
  candidate) — copied from `BASELINE_PER_CASE` when present, or derived from the op
  shape per the playbook recipes.
- `scripts/avo_lineage.py` (`seed`/`commit`, `--hw`) computes the roofline from
  `baseline_ms` (baseline) / `optimized_ms` (candidate) and stores
  `score.roofline = {geomean_fraction, regime, per_shape}` in `lineage.json` — so the
  whole lineage's progress toward the ceiling is auditable + resumable. A `roofline`
  subcommand prints the summary for debugging.
- `cmd_resume` restores the best's `roofline_fraction`/`regime`, so a resumed run can
  apply the stop rule and steer immediately.

## 9. Graceful degradation (non-negotiable)

If a kernel's COMMANDMENT does not emit `flops`/`bytes`, **every** roofline path is
inert: `rooflineSummary.ok=false` → `regime='unknown'`, no early-stop, steer falls
back to `DIRECTION`/`PROFILE_SUMMARY`. The run behaves exactly as Step 2 (pure
relative mode, wall-clock budget governs). This is enforced by tests
(`summary ok=false when no telemetry`, `no telemetry -> never stops`).

## 10. Tunables (args)

| arg | default | meaning |
|-----|---------|---------|
| `hw` | `gfx942` | peak table: `gfx942` (MI300X) \| `gfx942_mi325` (MI325X) |
| `roofline_target` | `0.85` | geomean fraction-of-roofline at which to stop early |
| `start_policy` | `diverse` | `best` \| `topk` \| `diverse` \| **`elites`** (MAP-Elites) |

(All existing AVO tunables — budget, commit gate, stall — are unchanged.)

## 11. Files & tests

- `lib/roofline.mjs` ← `tests/roofline.test.mjs` (14)
- `lib/headroom.mjs` ← `tests/headroom.test.mjs` (8)
- `lib/archive.mjs` ← `tests/archive.test.mjs` (8); `lib/start_select.mjs` `elites`
- `scripts/avo_lineage.py` roofline ← `tests/test_avo_lineage.py` (+7, total 24)
- knowledge: `knowledge/roofline_playbook.md`

## 12. Acceptance (roofline objective is "working")

- A kernel **with** flops/bytes: `lineage.json` nodes carry `score.roofline`; the log
  shows `…%-roofline <regime>` per commit; the loop stops at `roofline_target` once
  `min_commits` is met; fan-out with `start_policy=elites` starts from distinct cells.
- A kernel **without** flops/bytes: identical behavior to relative-only AVO (no
  crash, no premature stop).
- Unit tests green on a node-equipped machine (`run_tests.sh`).

## 13. Accuracy & rigor add-ons (A2 / C1 / A4 — landed 2026-06-16)

Three additions harden the objective so it can be trusted over a long lineage:

- **A2 — measured bytes over analytic.** `analyzeCase` now prefers
  `measured_bytes` (rocprof HBM traffic) over the analytic `bytes` and tags
  `bytes_source` (`measured`/`analytic`); rocprof `counters` (occupancy/MFMA-active)
  ride along on the case. This fixes the biggest mis-classification risk (a
  cache-friendly kernel looking memory-bound). `scripts/rocprof_probe.py` is the
  deterministic **parse** half (counter CSV → vector; unit-tested on fixtures); the
  `collect` run is rig-only/inert. Mirrored in `avo_lineage.py`.
- **C1 — variance-aware commit gate.** `avo_verify` aggregates repeats robustly
  (outlier drop + median) and reports the measured **cv**; `passesCommitGate` widens
  the required margin to `max(significanceMargin, noiseK·cv)` (`commit_noise_k`), so a
  within-noise "win" can't commit and seed a false best. Robust stats live in
  `lib/measure.mjs`. Backward compatible (no `cv` → legacy fixed margin).
- **A4 — bounded autotune sweep.** `lib/autotune.mjs` (`expandConfigSpace` capped +
  even-subsampled, `selectBestConfig` = best verified then closest-to-roofline) is the
  single source of truth for a best-of-N config sweep the operator can run for
  last-mile co-tuning (`autotune`, `autotune_max_configs`). One sweep = one gated
  candidate.

## 14. Deviations & follow-ups

1. **rocprof `collect` + IR `dump` are rig-only** (toolchain-dependent); the parse
   halves are landed + tested, the actual on-rig capture is wired via the verify role.
2. **No expert-library bar yet.** "Surpass expert" is currently measured vs the
   roofline, not vs aiter/hipBLASLt/CK on the same shapes. *Follow-up (B1):* a
   reference-bar measurement + `surpassed_expert` flag.
3. **Crossover is same-base only** (`selectMergeGroup`); a cross-lineage crossover
   *edit* primitive is a follow-up. **Orchestrator-driven** autotune (the loop fanning
   out over configs, vs the operator running the sweep in-step) is also a follow-up.
4. **No cross-run memory / seeding** by `(op, dtype, arch)` — same gap as IR/memory.
