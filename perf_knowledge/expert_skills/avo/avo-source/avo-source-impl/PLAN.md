# Implementation plan: avo-workflow (4 steps)

## Goal

Bring GEAK's AVO (continuous single-lineage kernel evolution) into PerfSkills as a
**third optimization engine** — a deterministic Workflow that exposes the **same
contract as `team_workflow`** (kernel dir in → verified `final_patch.diff` +
geomean out) and is **pluggable into `workflow_e2e_team`** as an alternative
`kernel_workflow_dir`. Do **not** merge AVO into `team_workflow` (breadth vs depth
are different regimes).

## Output directory

`PerfSkills/avo-workflow/` — fresh module. Reuse (copy) `scripts/` and
`knowledge/` from `workflows/`; do not modify existing PerfSkills files.

## Source of truth (reuse, don't reinvent)

- **GEAK AVO** (`GEAK/src/minisweagent/run/avo/`): `controller.py` (loop, ESCALATE,
  infra-abort), `lineage_store.py` (commit gate, `avo-v{N}` tags, rich commit
  messages), `supervisor.py` (stagnation re-plan), `variation_step.py` (prompt
  assembly, memory injection). These define the proven behavior to port.
- **GEAK docs**: `docs/developer/avo_design.md`, `avo_strategy_dataflow.md`,
  `avo_paper_gap_and_roadmap.md` (esp. Tier-1 measurement rigor).
- **PerfSkills `workflows/`**: `team_workflow.js` orchestration style, `verify_engineer`,
  `gpu_lock.sh`, COMMANDMENT contract, `knowledge/`.
- **GEAK `skills/avo-evolution/`**: variation-step contract + playbook + stagnation
  self-check (becomes this module's `knowledge/`).

---

## Step 1 — Thin wrapper (fastest end-to-end validation)

**Aim:** prove "AVO as a PerfSkills kernel engine" with minimal work, before any
re-implementation.

- `avo_workflow.js` (v0): a thin Workflow that shells out to the existing Python
  `geak-avo` and **adapts its outputs to the `team_workflow` contract**
  (`final_patch.diff`, geomean, `director_validation.json`).
- Setup/validation reuse PerfSkills conventions: isolated `eval_dir`, read-only
  original `kernel_path`, Director independent validation via `gpu_lock.sh`.
- Map args: `kernel_path`, `budget_s`, `gpu_ids`, `task` → `geak-avo` CLI flags.

**Deliverable:** run on the `knn` example end-to-end; produce a valid
`director_validation.json`. Two runtimes (JS + Python) is acceptable here.
**Exit criteria:** AVO completes a short time-boxed run and the Director verifies a
geomean against the TRUE baseline.

## Step 2 — Native `avo_workflow.js` (deterministic shell + roles)

**Aim:** remove the Python dependency; make AVO a first-class PerfSkills workflow,
structurally consistent with `team_workflow`.

- **Deterministic JS** (port of `controller.py`): wall-clock budget loop,
  `reset-to-best`, commit gate (geomean compare + anti-lazy floor), `git commit` +
  `tag avo-v{N}`, stagnation counters, ESCALATE fan-out, infra-failure abort,
  verify scheduling, `min_commits` guard.
- **Roles (agents):**
  - `roles/variation_engineer.md` — one variation step (port the AVO contract +
    memory/lineage/exemplar injection from `variation_step.py`).
  - `roles/supervisor.md` — REDIRECT re-planner (port `supervisor.py`: diagnose,
    mark-failed, propose untried directions, optional backtrack).
  - `roles/avo_verify.md` — own base-ref-aware independent verify (derived from
    `workflows/roles/verify_engineer.md`; needed for Diverse Fan-out / backtrack).
  - `roles/tech_lead.md` (report), `roles/director.md` (setup/validate/cleanup).
- **State files** (port `lineage_store.py`): `lineage.json`, `attempts.jsonl`,
  `evolution_log/`, `supervisor_log.jsonl`, `patches/v{N}.patch`; `avo_lineage.py`
  enforces the commit gate's data ops (geomean, rich messages, atomic writes).
  (`notebook/` / `direction.json` kept in-memory by design — see README *Output*.)
- **Apply Tier-1 measurement rigor by default** (per `avo_paper_gap_and_roadmap.md`):
  `verify_repeats>=3` (median) + non-zero significance margin.

**Deliverable:** `avo-workflow` runs with no Python AVO dependency; same contract.
**Exit criteria:** parity with Step 1 results on `knn` (within noise), single
runtime.

## Step 3 — Integrate into `workflow_e2e_team` (pluggable kernel engine)

**Aim:** let the e2e system layer deep-optimize a dominant hot kernel with AVO.

- Set e2e `args.kernel_workflow_dir = "<...>/avo-workflow"` to route a kernel to
  AVO instead of `team_workflow` (the e2e contract is already pluggable).
- Add a **routing rule** in the e2e System Architect: use AVO only for a kernel
  with **high `pct_gpu_time` AND remaining headroom after `team_workflow`**, and
  pass a **time-boxed `budget_s`** (a slice, not literal days).
- Keep `team_workflow` as the e2e default; AVO is the opt-in "heavy" engine.

**Deliverable:** an e2e run that extracts one hot kernel, optimizes it via AVO,
overlays it back, and gates on the end-to-end throughput delta.
**Exit criteria:** e2e accepts an AVO-produced kernel patch with a measured e2e
throughput gain above the noise band.

## Step 4 — A/B validation (depth vs breadth)

**Aim:** quantify when AVO's depth beats `team_workflow`'s breadth.

- On the same kernel + same total wall-clock: run `team_workflow` (bounded budget)
  vs `avo-workflow` (time-boxed), comparing **speedup/hour** and **final geomean**.
- Record where AVO wins (long-horizon, single high-value kernel) vs where
  `team_workflow` wins (quick broad gains), to drive the e2e routing rule in Step 3.

**Deliverable:** a short results note + a routing recommendation table.
**Exit criteria:** a documented, reproducible comparison on ≥1 kernel (ideally an
attention-class kernel to match the paper's regime).

## Step 5 — Roofline-anchored objective (surpass experts, approach the ceiling)

**Aim:** move AVO's objective from *purely relative* ("beat the last version") to
*hardware-anchored* ("approach the MI300X roofline, then stop"), so the agent
deliberately closes the gap to the limit instead of plateauing early. Built
**design-first / generically** (deterministic, unit-tested; no kernel-type
branching), with full graceful degradation when no FLOP/byte telemetry exists.

- **Pure model — `lib/roofline.mjs`** (mirrored in `scripts/avo_lineage.py`): per
  shape `AI = flops/bytes`, roofline bound `= min(peak_compute[dtype], AI·HBM_BW)`,
  `% of roofline`, and regime classification (`memory-bound` / `compute-bound` /
  `basement`). Peaks from `kernel_knowledge/hardware/cdna3_mi300/peak_tables.md`.
- **Gap-driven steering:** the loop passes a `ROOFLINE` block (regime + ranked
  levers + worst/most-headroom shape) into `variation_engineer` + `supervisor`, so
  the change attacks the *binding constraint* (port of "profiling-guided" but at the
  roofline level). Levers map to the team specialist taxonomy.
- **Headroom control — `lib/headroom.mjs`:** roofline early-stop at
  `roofline_target` (default 0.85, after `min_commits`) + Amdahl `allocateAttention`
  to spend the next step on the shape with the most remaining headroom.
- **Quality-diversity search — `lib/archive.mjs`:** MAP-Elites archive keyed by
  (strategy × regime) → `start_policy="elites"` for diverse fan-out starts, plus
  `crossoverPairs` to combine complementary elites (escape the single-chain optimum).
- **Telemetry contract (pillar 7):** `avo_verify` (and the COMMANDMENT benchmark)
  emit per-case `flops`/`bytes`/`dtype`; `avo_lineage.py` stores the candidate's
  roofline fraction + regime in `lineage.json` (auditable + resumable). Real-machine
  rocprof counters (occupancy, MFMA active%) are a follow-up that slots into the same
  per-case telemetry vector.

**Deliverable:** roofline objective + steering + early-stop + elites archive, all
behind graceful degradation. **Exit criteria:** deterministic unit tests pass
(`lib/{roofline,headroom,archive}.mjs` + `avo_lineage.py` roofline); a kernel
without FLOP/byte counts behaves exactly as Step 2 (pure relative mode).

> **Not done yet (follow-ups):** (a) real rocprof telemetry capture in `avo_verify`
> on the rig; (b) an **expert-library reference bar** (measure aiter/hipBLASLt/CK on
> the same shapes → "surpassed_expert" flag) — deliberately deferred; (c) a
> parametric **autotune-sweep** variation primitive for the last-mile co-tuning.

---

## Contract & budget notes

- **Contract parity** with `team_workflow` is the non-negotiable invariant
  (enables Step 3). The one intentional difference: AVO budget is **wall-clock
  `budget_s`**, not direction count.
- **Reliability** comes from the independent `verify_engineer` + deterministic
  commit gate in JS — never trust an agent's self-reported speedup.
- **Isolation**: the user's `kernel_path` is read-only; all work in `eval_dir/`
  (mirror `team_workflow`/`skills/team` isolation + integrity manifest).

## Risks

- **Two runtimes in Step 1** (JS + Python) — acceptable as scaffolding; removed in
  Step 2.
- **Wall-clock budget in e2e** — must be time-boxed so a multi-day engine doesn't
  stall an e2e run; enforced by the Step 3 routing rule.
- **Measurement noise compounding** over a long lineage — mitigated by Tier-1
  defaults (repeats + median + significance margin) from Step 2 on.
