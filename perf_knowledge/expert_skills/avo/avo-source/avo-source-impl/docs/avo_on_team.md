# AVO on Team — Design

How AVO is (and should be) built on the **Team Workflow** rather than reimplemented
from GEAK. Records the component mapping, what changes, and the `operator=single|team`
design that lets each variation step be a single agent (paper-faithful,
explicit) or a bounded Team round (default).

> Companion: [`README.md`](../README.md) (engine overview), [`PLAN.md`](../PLAN.md)
> (4-step plan), [`TESTING.md`](../TESTING.md),
> [`avo_roofline_design.md`](avo_roofline_design.md) (the hardware-anchored objective
> that now steers the Team operator), [`avo_ir_feedback_design.md`](avo_ir_feedback_design.md)
> (IR evidence per worker). Source bases: `../../workflows/team_workflow.js` (Team
> Workflow) and GEAK `run/avo/`.

> **Status (2026-06-15):** `operator=single|team` is **landed**; the reused Team
> `tech_lead` plans under the **roofline `ROOFLINE` steer** with **regime→specialist
> dispatch** (`assignSpecialists`); fan-out start points include `start_policy=elites`
> (MAP-Elites); and the **Integrator merge of complementary winners** (§7) is now
> **landed** too — both `operator=team` and ESCALATE try a gated merge before
> committing the single best. This doc is the complete design; §6 marks the status.

---

## 1. AVO has two layers — Team supplies the scaffolding for both

| Layer | Essence | GEAK realization | Team realization |
|-------|---------|------------------|------------------|
| **Outer evolution loop** | single lineage, wall-clock budget, commit gate, anti-stall supervisor | `controller.py` | Team Workflow's budget loop + advancing canonical + independent verify (already a proto-lineage) |
| **Variation operator** `Agent(P_t,K,f)` | produce ONE candidate from the lineage | one `OptimizationAgent` | a single specialist Engineer **or** one bounded Team round (TechLead → Engineers → Integrator) |

**Key insight:** Team Workflow already has most of AVO's reliable outer scaffolding —
a budget loop that commits each round's winner into an **advancing canonical**
(`current_best.diff`), an **independent `verify_engineer`** (trust only re-measured
numbers), an **Integrator**, **cross-round memory** (insight blackboard + hypothesis
ledger), and **Director arbitration**. AVO does not need to rebuild this; it only
needs to swap the **policy** (bounded breadth → unbounded single-lineage depth).

> We are already on this path: the current `avo-workflow` reuses Team's
> `benchmark_engineer` / `profile_engineer` roles and Team's commit-gate pattern;
> independent verification uses avo-workflow's **own** base-ref-aware
> `roles/avo_verify.md` (derived from Team's `verify_engineer`, but able to verify a
> Diverse-Fan-out / backtracked candidate on its real base). The default operator is
> a single agent.

---

## 2. Component mapping (team_workflow ↔ avo-workflow)

| Team Workflow | avo-workflow (AVO) | Note |
|---------------|--------------------|------|
| budget loop (`dispatched < BUDGET`) | wall-clock loop (`Date.now() < DEADLINE`) | AVO budget = seconds, not direction count |
| advancing canonical + `current_best.diff` | git repo `avo_repo`, HEAD=best, tags `avo-v{N}` | AVO adds a real, taggable, backtrackable lineage |
| "commit round winner if `> cumulative*1.05`" | deterministic commit gate (`passesCommitGate`): correct ∧ ≥ best·(1+margin) ∧ > anti-lazy floor | same idea, formalized + Tier-1 margin |
| TechLead `plan_round` (orthogonal directions) | supervisor `redirect` (new untried directions on stall) | AVO plans reactively (on stall), Team every round |
| specialist Engineers (algorithm/memory/compute/host_runtime) | variation operator (single agent **or** team round) | see §4 |
| `verify_engineer` (independent re-measure) | own **`avo_verify`** (base-ref aware) | derived from Team's; the shared source of trust |
| Integrator (merge round winners) | **landed** — own base-ref-aware `avo_integrator` (§7); team round / ESCALATE try a gated merge first | derived from Team's `integrator`, base-ref aware like `avo_verify` |
| insight blackboard + hypothesis ledger | AVO memory (`history.insights`/`failed`/`ledger`) | could be replaced by Team's directly |
| Director setup + validation | **reused** (`setup_native` + `validate`) | + integrity manifest |
| (none) | StagnationDetector → REDIRECT/ESCALATE | AVO's anti-stall, not in Team |
| (none) | **roofline regime → `assignSpecialists()`** | **landed** — steers Team `plan_round` + backfills each worker's specialty (§7.1) |
| (none) | **MAP-Elites archive** (`lib/archive.mjs`) + `start_policy=elites` | landed; `selectMergeGroup`/`crossoverPairs` drive the Integrator merge (§7.2) |
| (none) | **IR evidence** per worker (`avo_ir_feedback_design.md`) | sharpens each worker's lever (read-only) |

---

## 3. What changes vs. team_workflow (only 4 things)

To turn Team's bounded breadth into AVO's unbounded single-lineage depth:

1. **Budget** → wall-clock `budget_s`; run to the deadline (drop "stop when budget
   directions exhausted"; keep `min_commits` as a warning).
2. **Lineage** → commit each accepted candidate as a git **`avo-v{N}` tag** (not just
   an advancing `current_best.diff`), enabling `git show`, audit, and **backtrack**.
3. **Anti-stall** → add a deterministic **StagnationDetector** that fires
   `REDIRECT` (LLM supervisor re-plan) and `ESCALATE` (diversified fan-out). Team's
   TechLead re-plan + early-stop is the seed of this; AVO formalizes it.
4. **Operator** → a step may be a single agent (explicit) or one bounded Team round (default)
   (§4). Everything else (verify, integrator, memory, Director) is reused.

---

## 4. The variation operator: `operator = single | team`

The operator is pluggable; the outer AVO loop is identical for both.

### `operator=single` (explicit — paper-faithful)
One `variation_engineer` agent per step: start from best → one focused change →
`save_and_test` → emit incremental patch → independent verify → commit gate.
- **Pros:** matches the paper's `Agent(P_t,K,f)`, cheap per step, a true single
  lineage of single-agent decisions.
- **Use for:** explicitly requested long-horizon, single-chain depth.

### `operator=team` (default — Team round as the operator) — **as landed today**
Each step = **one bounded Team round**, realized in `avo_workflow.js` as:
1. `planTeamDirections(step)` → the **reused Team `tech_lead` `plan_round`** proposes
   `K` orthogonal directions (each with a `specialty` ∈ algorithm/memory/compute/
   host_runtime), **now receiving the `ROOFLINE` steer** (regime + ranked levers +
   worst/most-headroom shape).
2. `pairStartsWithDirections(dirs, START_K)` → zips the K directions with the start
   points chosen by `start_policy` (`best`/`topk`/`diverse`/`elites`).
3. `fanOutEvolve(pairs)` → fan-out **`variation_engineer` workers** (one per
   direction × start point, each on its own `avo_repo@startRef` worktree + GPU),
   each **independently re-measured by `avo_verify` on its own base ref**.
4. The **single best** candidate that passes the **same deterministic commit gate** is
   committed + tagged `avo-v{N}` (parented on its own start ref → tree).

- **Pros:** much stronger per step (breadth inside depth); reuses Team's `tech_lead`
  planning + AVO's parallelism + per-GPU isolation + independent verify; the global
  best still advances only on a verified global improvement.
- **Cons:** ~K× cost per step; deviates from the paper's single-agent operator.
- **Use for:** hard, high-value kernels where one agent stalls.

> **Completion (§7, now landed):** (a) each worker is dispatched a **regime-aligned
> specialty** (`assignSpecialists`) and scopes to that specialist's knowledge; (b) the
> round first attempts an **Integrator merge** of complementary winners (`avo_integrator`),
> committing the merged candidate only if it is independently verified to beat the best
> individual — otherwise it commits the best winner as before.

> **`operator=team` is just ESCALATE promoted to a normal step.** AVO's ESCALATE
> already runs a diversified fan-out (multiple directions → verify → commit best);
> `operator=team` makes that the standard step (with `tech_lead` planning). Both share
> **one** helper, `fanOutEvolve` — so the Integrator/crossover completion in §7 upgrades
> *both* paths at once.

### Decision
Default `team` with `start_policy=diverse`; expose `args.operator`,
`args.operator_directions`, `args.start_policy`, and `args.start_k`. Keep
explicit `operator=single,start_policy=best` as the clean reproduction of the
paper-faithful AVO operator.

---

## 5. Why NOT build on the `team` *skill* (markdown)

The `team` **skill** is LLM-interpreted control flow — the LLM decides when to loop,
how many engineers to spawn, and when to stop (non-deterministic, non-reproducible).
AVO's two hard requirements are exactly the opposite:

- **Deterministic commit gate** (no fake progress) — must be code, not LLM judgement.
- **Controllable budget** over a multi-day run — must be code.

So the **core loop must be a Workflow** (code-driven), as concluded in the
Skill-vs-Workflow analysis. The team skill's `sub_skills/` + `knowledge/` are fine to
**reuse as operator knowledge**, but they must not own the loop.

---

## 6. Status — landed vs designed-pending

**Landed** (in `avo_workflow.js` + roles + `lib/`):
- Reuse of Team `benchmark_engineer` / `profile_engineer` / `tech_lead` (`plan_round`)
  + `gpu_lock.sh` / `profile_kernel.sh`; the commit-gate pattern formalized as the
  deterministic `passesCommitGate`.
- `operator=single|team` switch (`args.operator`, `args.operator_directions`);
  `fanOutEvolve` shared by `operator=team` and ESCALATE.
- Independent verify via AVO's base-ref-aware `avo_verify`.
- **Roofline steer** (`ROOFLINE` block) injected into `variation_engineer` /
  `supervisor` / the Team `plan_round`; roofline early-stop; per-version `regime`.
- **MAP-Elites** archive + `start_policy=elites`; `crossoverPairs`/`selectMergeGroup`.
- **(P1) Regime → specialist dispatch** — `assignSpecialists(regime,k)` orders the
  Team round's specialties (dominant ceiling first) and backfills each worker's track.
- **(P2) Integrator merge** — `avo_integrator` composes complementary same-base
  winners; the merge is independently re-verified and committed **only if** it passes
  the commit gate **and** beats the best individual (`mergeImproves`), else best wins.
  Used by both `operator=team` and ESCALATE (shared `commitRoundResult`).
- Director setup/validate reuse + integrity manifest.

**Designed, pending** (this doc; none change the outer loop's invariants):
- **(P3) Adopt Team's insight blackboard + hypothesis ledger** in place of AVO's
  ad-hoc `history` memory (optional consolidation).
- **(P4) IR evidence per worker** — see [`avo_ir_feedback_design.md`](avo_ir_feedback_design.md).

One engine, two operator strengths: the default faithful to the paper, the heavy mode
reusing Team's hierarchy — now under a hardware-anchored (roofline) objective.

---

## 7. Completing AVO-on-Team — roofline-regime specialist dispatch + Integrator merge (landed)

The two gaps from §4 are where the **remaining** Team advantages get fully absorbed.
Both ride the seams the roofline work already built (`steerFrom`, `fanOutEvolve`,
`commitCandidate`, the archive), so neither touches the outer loop's guarantees
(deterministic gate, independent verify, wall-clock budget, lineage). Both are now
implemented; the deterministic cores (`assignSpecialists`, `selectMergeGroup`,
`mergeImproves`) are unit-tested.

### 7.1 Regime → specialist dispatch (P1)

**Goal:** put the *right specialist* on the *binding ceiling*, not K generic workers.

- `recommendLevers(regime)` maps the regime to Team's taxonomy (`memory-bound→memory`,
  `compute-bound→compute`, `basement→host_runtime`, `unknown→algorithm`).
- **Implemented:** `assignSpecialists(regime, k)` (`lib/roofline.mjs`) returns the K
  specialties **dominant-ceiling first**, then orthogonal cover of the others (the next
  ceiling shifts after each commit). `planTeamDirections` passes this as `SPECIALISTS`
  to the Team `plan_round` **and** backfills any direction the planner left without a
  `specialty`. The worker then honors `DIRECTION.specialty` and loads **only** that
  specialist's knowledge (`roles/variation_engineer.md §4`) — a real knowledge-scoped
  specialist, not an advisory label.
- **Deterministic core (tested):** `assignSpecialists` is pure + unit-tested
  (`tests/roofline.test.mjs`). Graceful: `regime='unknown'` → algorithm-first generic
  order (today's behavior).

### 7.2 Integrator merge of complementary winners (P2)

**Goal:** when two workers win on **different** regimes/shapes, *combine* them instead
of discarding the runner-up — the Team Integrator's core value.

- **Trigger (deterministic):** `commitRoundResult` calls `selectMergeGroup(passers)`
  (`lib/archive.mjs`) — among the round's gated winners it finds those that **share a
  base ref** (so patches can stack) **and** attack **different regimes** (so the combo
  compounds), keeping the best per regime. Cross-base or single-regime → no merge,
  fall back to "commit best". Unit-tested.
- **Merge step (LLM):** `roles/avo_integrator.md` — a base-ref-aware integrator
  (derived from Team's `integrator`, parallel to how `avo_verify` is base-ref-aware):
  it worktrees `BASE_REF`, stacks compatible patches or hand-merges, and emits one
  incremental merged patch.
- **Gate (unchanged):** the merged candidate is independently re-measured by
  `avo_verify` on `BASE_REF`, then committed **only if** `passesCommitGate` (beats the
  global best vs the TRUE baseline) **and** `mergeImproves` (beats the best individual
  winner it combined). Parent = the shared base ref (tree edge).
- **Anti-cheat:** the merge can never bypass the gate; a failed/below-best merge
  simply commits the single best winner, exactly as before. Shared by `operator=team`
  and ESCALATE.

### 7.3 Why this is the right shape

- It keeps the **outer loop deterministic** and the **commit gate authoritative** —
  the merge is "another candidate", not a new trust path.
- It makes `operator=team` strictly stronger: regime-targeted specialists explore the
  binding constraint, and the Integrator compounds complementary wins — directly
  serving "approach the roofline / surpass the expert".
- It reuses Team wholesale (specialist `engineer` knowledge + `integrator`), finishing
  the "AVO on Team" thesis: AVO owns the reliable depth loop; Team supplies the strong
  per-step breadth operator.

### 7.4 Acceptance (the completion is "done")

- `assignSpecialists` unit-tested (regime→specialty ordering; unknown fallback).
- An `operator=team` round on a kernel with a clear dominant regime dispatches the
  matching specialist as the primary worker.
- When two complementary winners exist, the `integrator` produces a merged candidate
  that is committed **iff** it beats both via independent verify; otherwise the best
  single winner is committed (regression guard).
- All paths inert/identical to today when `regime='unknown'` (no telemetry).
