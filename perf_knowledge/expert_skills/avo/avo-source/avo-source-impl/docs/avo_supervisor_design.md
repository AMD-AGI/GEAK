# AVO Supervisor Design (avo-workflow)

How `avo-workflow` realizes the paper's **self-supervision** (anti-stall) mechanism,
its two layers, the exact triggers/levels, and what deviates. Grounded in the paper
(arXiv:2603.24517 §3.3) and the avo-workflow implementation.

> Companion: [`avo_on_team.md`](avo_on_team.md), [`avo_memory_design.md`](avo_memory_design.md),
> and (deeper GEAK background) `GEAK/docs/developer/avo_supervisor_design.md`.

---

## 1. What the paper assumes

> §3.3: "two failure modes can impede progress: the agent may **stall** when it
> exhausts its current line of exploration, or it may enter **unproductive cycles**
> of edits that repeatedly fail to improve scores. To mitigate both, AVO
> incorporates a **self-supervision mechanism that detects these scenarios and
> intervenes**. Once triggered, the mechanism **reviews the overall evolutionary
> trajectory and steers the search toward several candidate optimization
> directions**." … "the supervisor maintained forward progress by intervening
> during periods of stagnation."

So: detect (stall | cycle) → review trajectory → propose several new directions.

## 2. Two-layer realization (detector + re-planner)

Following the Workflow doctrine (deterministic scaffold + LLM judgement):

| Layer | Where | Role |
|-------|-------|------|
| **Deterministic detector** | `avo_workflow.js` (loop counters) | always fires on stall/cycle; owns levels + ESCALATE trigger |
| **LLM re-planner** | `roles/supervisor.md` (`PHASE=redirect`) | reviews lineage/attempts/profile, proposes untried directions, optional backtrack |

The detector is **code** (reliable, reproducible); the *judgement* of "what to try
next" is the LLM. This mirrors how GEAK splits `StagnationDetector` vs the
`avo-supervisor` subagent.

## 3. Detector — triggers & levels

State tracked in the loop (`avo_workflow.js`): `noImprove`, `recentHashes`
(last `STALL_PATCH_REPEAT`), `supervisorCycles`, `infraFailures`.

| Signal | Condition (default) | Maps to paper |
|--------|---------------------|---------------|
| **no-improvement stall** | `noImprove >= stall_no_improve` (5) | "exhausts its current line" |
| **patch cycling** | last `stall_patch_repeat` (3) patch hashes identical | "unproductive cycles of edits" |
| **infra failure** | a step / all team workers return `status=infra_failure`; abort after `max_infra_failures` (2) | (not in paper — engineering guard: broken harness/env) |

**Budget-aware gating.** A REDIRECT is only fired if the remaining wall-clock can fit
the supervisor re-plan itself (`supervisor_overhead_s`, default 150) **plus at least one
redirected step** (estimated from the trailing average of recent step durations).
Otherwise the stall is logged and the budget is banked — the re-plan is skipped so its new
directions are not produced after the deadline and wasted (a failure mode observed in
practice: the supervisor fired at `noImprove=8` with no time left to try its directions,
going straight to Report).

Level escalation (on a stall/cycle trigger):

```
trigger → REDIRECT (LLM supervisor re-plan; reset counters)
        → if supervisorCycles >= supervisor_cycles_to_escalate (3): ESCALATE (diversified fan-out)
```

(There is no separate NUDGE/INTERRUPT level in avo-workflow; a single REDIRECT
covers "steer", and ESCALATE covers "diversify". GEAK's finer NUDGE→INTERRUPT→
REDIRECT→ESCALATE ladder can be added if needed.)

## 4. REDIRECT — the LLM re-planner (`roles/supervisor.md`)

**Inputs (read directly):** `lineage.json` (what worked), `attempts.jsonl` (what
keeps failing), `evolution_log/` (causal trail), `PROFILE_SUMMARY` (bottleneck),
`FAILED_DIRECTIONS`, `CURRENT_DIRECTION`.

**Output (directive JSON):**
- `diagnosis` — evidence-grounded, one/two sentences.
- `mark_failed[]` — dead directions (added to `history.failed`; never re-proposed).
- `new_directions[]` — 2–3 untried, bottleneck-consistent, mutually-orthogonal
  `{name, expected, prompt}`.
- `next_direction` — the single best to try next.
- `backtrack_to` — optional earlier `avo-v{K}` to resume from (local-trap escape).
- `converge` — optional `true` when a hard structural/HW wall is evident and no
  orthogonal lever remains; the loop may exit early (see §5b).

**Self-validation (objectivity):** the orchestrator drops any proposed direction
already in `failed`/current; the supervisor must still return at least one fresh
generic direction so the run always has a next move (mirrors GEAK
`_validate_directive` + fallback taxonomy). When the runtime exposes
**`recordSupervision`**, the orchestrator appends the **normalized** directive
(`lib/supervisor_directive.mjs`: every direction has `name`, `expected`, `prompt`) to
`supervisor_log.jsonl` with a real wall-clock `ts` — the LLM-only append is then
optional (see `roles/supervisor.md`).

**Effect on the loop:** `direction` is set to the first novel `new_directions`
entry (used by the next `operator=single` step, or as a seed for `operator=team`
planning); counters reset; `backtrack_to` moves `bestVersion` so the next step
checks out that tag.

## 5. ESCALATE — diversified fan-out

After `supervisor_cycles_to_escalate` REDIRECTs without a commit, run a **wider
parallel rescue**: `fanOutEvolve(dirs, "escalate_<step>")` runs
`escalate_workers` distinct directions in parallel (one per GPU), independently
verifies each, and commits the single best that passes the **same commit gate**.

> ESCALATE and `operator=team` share **one** helper (`fanOutEvolve`): a team round
> is "fan-out as the normal step"; ESCALATE is "a wider fan-out only when stalled".

## 5b. Root-cause convergence (anti sterile churn)

The per-direction `stall_no_improve` counter only sees one direction at a time, so
the loop can still burn budget across **repeated supervisor cycles** that re-state
the same bottleneck without a commit. The orchestrator adds **deterministic**
root-cause guards (in addition to the LLM-only judgement):

1. **`converge` flag** — In the supervisor JSON, optional `converge: true` asserts
   a hard structural / hardware wall with no novel lever left; the Evolve loop
   exits and banks the remaining wall-clock (same spirit as roofline early-stop).
2. **Barren redirects** — Count successful supervisor interventions since the last
   commit; at `converge_barren_cycles` (default 4), stop (no further REDIRECT /
   ESCALATE).
3. **Repeated diagnosis** — Token–Jaccard similarity between consecutive
   `diagnosis` strings; when similarity ≥ `dx_sim_threshold` (default 0.55) for
   `converge_dx_repeat` consecutive pairs (default 3), treat as “same root cause”
   and stop.

Disable any guard by setting its integer knob to `0`. The detector still owns
triggers; these rules only decide **early exit** so parallel rescue does not run
after the evidence already says “stuck”.

## 6. Backtrack (revisit earlier `P_t`)

> Paper §3.3: the agent "autonomously decided … when to **revisit earlier
> approaches in `P_t`**".

avo-workflow realizes revisit **externally**: the supervisor's `backtrack_to` sets
`bestVersion` to an earlier tag, and the next variation step checks it out as its
base. This is a deliberate deviation (the paper's agent self-decides because it
holds all of `P_t` in context; our fresh-agent steps need an explicit pointer).

## 7. Deviations vs the paper (tracked)

1. **External vs agent-internal steering/revisit.** Paper: one agent self-steers
   and self-revisits. avo-workflow: a deterministic detector + an LLM re-planner +
   an explicit best pointer. *Rationale:* reliability (deterministic anti-stall +
   no fake progress) over fidelity; this is the Workflow trade-off.
2. **Coarser level ladder.** REDIRECT+ESCALATE vs GEAK's 4-level ladder. Add NUDGE/
   INTERRUPT if runs need gentler nudges before a full re-plan.
3. **Infra-failure abort** is an addition (not in the paper) — a long autonomous
   run must not burn budget on a broken harness/env.

## 8. Tunables (args)

| arg | default | meaning |
|-----|---------|---------|
| `stall_no_improve` | 5 | no-improvement steps before REDIRECT |
| `stall_patch_repeat` | 3 | identical patch hashes ⇒ cycling ⇒ REDIRECT |
| `supervisor_cycles_to_escalate` | 3 | REDIRECTs without commit before ESCALATE |
| `supervisor_overhead_s` | 150 | est. seconds for a REDIRECT; with the trailing step-time estimate, gates out re-plans that cannot run a redirected step before the deadline |
| `escalate_workers` | max(2, #GPU) | parallel rescue directions |
| `max_infra_failures` | 2 | consecutive infra failures before abort (0 disables) |
| `converge_barren_cycles` | 4 | supervisor REDIRECTs with no commit before deterministic stop (0 disables) |
| `converge_dx_repeat` | 3 | consecutive near-duplicate diagnoses ⇒ stop (0 disables) |
| `dx_sim_threshold` | 0.55 | token–Jaccard similarity that counts two diagnoses as “same root cause” |

## 9. Acceptance (supervisor is "working")

- A deliberately-stalling kernel (`stall_no_improve=1`) triggers a REDIRECT whose
  `new_directions` are not in `failed`.
- `supervisor_cycles_to_escalate=1` then triggers an ESCALATE fan-out.
- `supervisor_log`/`history.insights` records each intervention; a `backtrack_to`
  directive makes the next step start from the named earlier tag.
- A broken harness (`infra_failure`) aborts after `max_infra_failures` instead of
  spinning to the deadline.
- With defaults, four fruitless supervisor rounds since the last commit exit Evolve
  (`converge_barren_cycles`); or three consecutive near-duplicate diagnoses do the
  same (`converge_dx_repeat` + `dx_sim_threshold`).
