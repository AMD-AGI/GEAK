# avo-workflow product roadmap and scheduling (pending approval)

| Field | Value |
|-------|-------|
| **Document status** | `Pending approval` — not a committed schedule until approved; planning draft only |
| **Version** | 0.2 |
| **Date** | 2026-06-18 |
| **Scope** | `PerfSkills/avo-workflow` (orchestration, `lib/`, `roles/`, `docs/`, contract with `avo_lineage.py`) |

---

## 1. Approval (fill in and return)

- [ ] **Reviewed**: priorities and phasing are reasonable  
- [ ] **Approved**: may execute per this document (or note Phase 1 / 1+2 / … only)  
- **Approver**: ________________  
- **Date**: ________________  
- **Notes (scope cuts, deadlines, resources)**:  

---

## 2. Priority overview (execution order)

| Order | Priority | Topic | Suggested effort* | Dependencies |
|-------|----------|-------|-------------------|--------------|
| **P1** | **P0** | Supervisor output validation, audit chain, schema alignment | 1–2 weeks | None → **Landed** (`normalizeSupervisorDirective` + `recordSupervision` audit writes) |
| **P1** | **P0** | Multi-case objectives and commit gate (design freeze + optional implementation) | 2–3 weeks | Can parallel previous item (design first) |
| **P2** | **P1** | Policy gate when roofline telemetry is missing (optional args) | 1 week | Better after P0 supervisor stabilizes |
| **P3** | **P1** | Progress metadata consistency (heartbeat / step single source of truth) | 3–5 person-days | None → **Landed** (`syncAvoHeartbeat` + `SYNC_HEARTBEAT` skips role heartbeat) |
| **P4** | **P2** | Stall gradient (lightweight NUDGE-style prompts vs full REDIRECT) | 1–2 weeks | Needs runtime prompt cost estimate |
| **P5** | **P2** | MAP-Elites **crossover variation step** (cross-regime parents → committable child) | 2–4 weeks | `archive.mjs` + Integrator groundwork exists |
| **P6** | **P2** | Near-threshold **adaptive verify** (repeat runs / FULL_BENCHMARK triggers) | 1–2 weeks | Tie to gate parameter docs |
| **P7** | **P3** | Large `kernel_knowledge` **optional retrieval** (MCP/RAG; measurement still authoritative) | 2–4 weeks | Infra selection |
| **P8** | **P3** | **Short-budget E2E smoke** (fixture or mock agent) | 1–2 weeks | CI capacity |
| **P9** | **P3** | JS / `avo_lineage.py` **dual-implementation sync** checklist and CI parity | ongoing / ~0.5 week per milestone | Strongly tied to every gate change |

\*Effort is **one engineer full-time equivalent** familiar with this repo; calendar time shrinks with parallelism.

---

## 3. Phased detail (deliverables and acceptance)

### Phase P1 — P0: Supervisor contract and audit chain

**Context:** LLM-produced directives sometimes miss fields vs `SUPERVISOR_SCHEMA` / `supervisor.md`, breaking the next step and `supervisor_log.jsonl` trust.

**Deliverables**

1. Orchestration (`avo_workflow.js` or runtime shim) **validates and normalizes** supervisor returns (when `prompt`/`expected` missing: fill defaults or one narrow repair call — pick one in design).  
2. Optional: `appendSupervisorLog`-style primitive for **deterministic append** of `{"ts": epoch, "directive": ...}`.  
3. Update `docs/avo_supervisor_design.md` §4 (self-check), `roles/supervisor.md` (match implementation).

**Acceptance**

- Evolve **does not crash** on malicious truncation / empty `new_directions`; logs or insights locate the failure.  
- New or updated **unit tests** (pure validators) or minimal integration case.

---

### Phase P1 — P0: Multi-case objectives and commit gate

**Context:** Plain geomean can hide per-shape regressions; long no-commit stretches plus variance gate can block candidates slightly better on geomean.

**Deliverables**

1. **Design doc** (new `docs/avo_commit_objectives.md` or extend existing): optional modes — per-case veto, weighted geomean, Pareto archive linked to `elites`; default unchanged.  
2. Implement **at least one** optional arg (e.g. `commit_per_case_min_speedup` or equivalent), default off, with `CHANGELOG` + README args table.

**Acceptance**

- Default path compatible with existing `tests/commit_gate.test.mjs` (or explicit versioning).  
- Doc states **relationship to roofline `worst_case`** to avoid contradictory double gates.

---

### Phase P2 — P1: Roofline “unknown” gate

**Deliverables:** Optional arg, e.g. `escalate_requires_roofline` (default `false`): when `bestRoofline.ok === false`, skip or downgrade ESCALATE / or require supervisor to mark `converge` conditions; exact semantics in a new `avo_roofline_design.md` section.

**Acceptance:** With switch on and no telemetry, ESCALATE behavior and logs match design; off → identical to today.

---

### Phase P3 — P1: heartbeat and step single source of truth

**Deliverables:** Define **sole writer** or field semantics (e.g. `heartbeat` = last step that finished a commit; `attempts` = max attempt step); update `docs/avo_memory_design.md` + shim write path if any.

**Acceptance:** Long-run samples: no unexplained conflict between `heartbeat.step` and `attempts.jsonl` max(step).

---

### Phase P4 — P2: Stall gradient (NUDGE class)

**Deliverables:** Before `stall_no_improve` is exhausted, insert **low-cost** re-prompts (insights / compressed checklist only), no full supervisor; knobs and trigger counts in `avo_supervisor_design.md`.

**Acceptance:** Same budget, measurable drop in **supervisor invocations** or token cost vs baseline; no increase in false-positive commits (sampled).

---

### Phase P5 — P2: Crossover variation step

**Deliverables:** Use `crossoverPairs` parent refs; define one **crossover variation** step (new role or `variation_engineer` phase extension), same verify/commit gate; update “not yet landed” wording in `docs/avo_start_points_design.md`.

**Acceptance:** At least one E2E or integration path for `operator=single` or `team`; failure falls back to current path.

---

### Phase P6 — P2: Adaptive verify

**Deliverables:** When candidate geomean falls in narrow band `(best, best+δ]`, auto-increase repeat count or force FULL_BENCHMARK; `δ` and caps controlled by args.

**Acceptance:** Aligned with `avo_verify` / COMMANDMENT docs; does not slow every step by default.

---

### Phase P7 — P3: Knowledge retrieval boost

**Deliverables:** Optional MCP or vector retrieval; default off. Update `avo_knowledge_and_profiling.md` Part A.

**Acceptance:** Off → zero behavior change; on → doc lists keys and quotas.

---

### Phase P8 — P3: E2E smoke

**Deliverables:** New section in `TESTING.md`; script or CI job with tiny `budget_s`, offline or mock.

**Acceptance:** Main-branch CI green (or nightly).

---

### Phase P9 — P3: Dual-implementation sync

**Deliverables:** In `docs/`, maintain **JS vs Python** responsibility matrix (gate, roofline summary, resume fields); PR checklist item.

**Acceptance:** Every change to `lib/commit_gate.mjs` or gate-related constants in `avo_workflow.js` forces checklist parity or documented waiver.

---

## 4. Suggested timeline (Gantt-style intent, not mandatory calendar weeks)

```
Weeks:  |--W1--|--W2--|--W3--|--W4--|--W5--|--W6--|--W7--|--W8--|
P1 P0 supervisor     [====]
P1 P0 multi-case     [===....====]
P2 roofline gate         [==]
P3 heartbeat                 [=]
P4 nudge                           [====]
P5 crossover                             [====....]
P6 adaptive verify                         [====]
P7 RAG (opt)                                   [..........]
P8 E2E smoke                                       [====]
P9 sync matrix         (throughout; close each milestone)
```

Note: **Both P1 P0 tracks can run in parallel** (different PRs); P2–P3 can start ~W3; P5 and P6 may overlap but **gate docs should be frozen** before adaptive verify.

---

## 5. Explicitly out of scope for this document

- Merging AVO **into** `team_workflow.js` (architecture: keep dual engines).  
- A **separate workflow** that replaces Triton with CK/asm (`asm-workflow`, etc.) — link only via `kernel_knowledge` and task copy.  
- Changing per-customer `COMMANDMENT.md` measurement contracts (except backward-compatible verify/roofline field extensions).

---

## 6. Revision history

| Version | Date | Notes |
|---------|------|-------|
| 0.2 | 2026-06-18 | P1-P0 supervisor normalization + P3 heartbeat: `recordSupervision`, `syncAvoHeartbeat`, `SYNC_HEARTBEAT`. |
| 0.1 | 2026-06-18 | First draft, pending approval |
