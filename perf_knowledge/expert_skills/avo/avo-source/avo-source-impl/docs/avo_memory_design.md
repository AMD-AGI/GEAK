# AVO Memory Design (avo-workflow)

How `avo-workflow` realizes the **memory** that the NVIDIA AVO paper assumes, what
deviates, and exactly which artifacts are written/read. Grounded in the paper
(arXiv:2603.24517 §3.2/§3.3/§4.1) and the avo-workflow implementation.

> Companion: [`avo_on_team.md`](avo_on_team.md), [`avo_supervisor_design.md`](avo_supervisor_design.md),
> and (deeper GEAK background) `GEAK/docs/developer/avo_memory_design.md`.

---

## 1. What the paper assumes

> §4.1: "The agent … maintains **persistent memory through its conversation
> history**, which accumulates the **full context of prior edits, compiler
> outputs, profiling results, and reasoning** across the evolutionary process."
> §3.2: the agent "frequently **examines multiple prior implementations in `P_t`**
> within a single variation step, comparing their profiling characteristics."

So the paper's memory is **one long-running agent's raw, continuous conversation
history** + on-demand access to all prior committed solutions `P_t`.

## 2. Why avo-workflow cannot keep that verbatim

avo-workflow (like GEAK-AVO) runs a **fresh agent per variation step** (Workflow
model: each `agent(...)` call is a bounded sub-agent). This is what keeps a
multi-day run's context **bounded** and the commit gate **deterministic**. The
cost: there is no single growing raw conversation. avo-workflow therefore
**reconstructs** memory from durable artifacts and injects a bounded summary each
step. This is the **single biggest, deliberate deviation** from the paper.

## 3. The three reconstructed layers

| Layer | Artifact | Role (paper analogue) |
|-------|----------|------------------------|
| **L1 — Lineage `P_t`** | `avo_state/lineage.json` + git `avo_repo` (commits/tags `avo-v{N}`, rich messages) | the committed solution set + scores the agent "examines" (§3.2) |
| **L2 — Attempt history** | `avo_state/attempts.jsonl` (every attempt incl. failures) | the internal search trajectory (§3.2: "unsuccessful attempts remain part of the trajectory") |
| **L3 — Causal evolution log** | `avo_state/evolution_log/step_*.json` (strategy, committed?, verified_speedup, per_shape, rationale) | the "reasoning across the process" (§4.1), compacted |

Plus an **in-loop working set** the orchestrator threads between steps
(`history` in `avo_workflow.js`): `insights[]`, `failed[]`, `ledger[]`.

## 4. Write path (who writes what, when)

Per step, after the commit gate decision (deterministic JS in `avo_workflow.js`):

- **Every attempt** (operator=single AND each operator=team worker, on every exit
  path): the variation engineer appends one line to **attempts.jsonl**
  (`committed:false`) via `avo_lineage.py attempt` — strategy, status, self-speedup,
  patch hash, rationale. This is the deterministic L2 write and is symmetric to how
  the commit is written (a script invoked by a role, not hand-edited JSON). The loop
  ALSO mirrors it in-memory (`history.ledger`) for same-run steering.
- `lineage_engineer commit` → `avo_lineage.py commit` writes the new node to
  **lineage.json**, appends **attempts.jsonl** (`committed:true`) and an
  **evolution_log** entry; the role then `git commit`+`tag` with the rich message.
- The supervisor's diagnosis/dead-directions are mirrored to `history.failed` /
  `history.insights`. **`supervisor_log.jsonl`** is appended by the orchestrator
  when the runtime provides **`recordSupervision`** (canonical `ts` + normalized
  directive); otherwise the supervisor role appends manually (see `supervisor.md`).

### `heartbeat.json` vs `attempts.jsonl` (single source of truth)

- **`attempts.jsonl`** remains the authoritative **search trajectory** (every
  attempt line, including `step` / `step_index` and `committed`).
- **`heartbeat.json`** is a **live progress snapshot** for humans / dashboards:
  - When the runtime provides **`syncAvoHeartbeat`** (e.g. `run_avo.mjs`), **`avo_workflow.js`**
    merges `phase`, `step` (same counter as the Evolve loop), `commits`,
    `best_version`, `best_speedup`, and optional `best_latency_ms` **after each loop
    iteration** and on supervisor / escalate — this **`step` always matches** the
    orchestrator's completed Evolve index (fixes drift vs mid-step role heartbeat).
  - `avo_lineage.py commit` / `seed` still merge into `heartbeat.json` (best speedup
    after a commit); those fields align with the orchestrator snapshot on the next pulse.
- **`variation_engineer.md` §1**: when `SYNC_HEARTBEAT: true` in Inputs, **skip** the
  optional `avo_lineage.py heartbeat` bash stanza so a worker does not overwrite
  `step` mid-iteration with a parsed dirname.

```
variation/team worker → (always) avo_lineage.py attempt: attempts.jsonl += {step, strategy, status, self_speedup, committed:false}
                      → verify → commit gate
   ├─ pass → avo_lineage.py commit: lineage.json + attempts.jsonl (committed:true) + evolution_log/ + git tag (rich msg)
   └─ fail → history.ledger += {step, strategy, verified, committed:false}   (durable copy already in attempts.jsonl)
supervisor (on stall) → (recordSupervision) supervisor_log.jsonl += {ts, directive} ; history.failed += mark_failed ; history.insights += diagnosis
orchestrator (shim)   → syncAvoHeartbeat merges heartbeat.json (step/phase/best_*)
```

> **Durability across resume:** because every attempt is in attempts.jsonl,
> `cmd_resume` reconstructs failure memory from BOTH `supervisor_log.jsonl` (named
> dead directions) AND `attempts.jsonl` (strategies that recur with `committed:false`
> — surfaced as insights). So a run that committed steadily and never triggered a
> supervisor still does not re-tread dead strategies after a resume.

## 5. Read / inject path (into each step)

The orchestrator injects a **bounded** summary into every variation step and the
supervisor; the agent additionally **pulls** raw detail on demand (the paper's
"examines multiple prior implementations" → `git show`):

- **Injected to `variation_engineer`** (push, bounded): `BEST_VERSION`,
  `CUMULATIVE_SPEEDUP`, `DIRECTION`, `INSIGHTS` (`history.insights`),
  `FAILED_DIRECTIONS` (`history.failed`), `PROFILE_SUMMARY`.
- **Pulled by the agent** (on demand, per `knowledge/lineage_usage.md`):
  `git -C avo_repo log --oneline` / `git show avo-v{N}`, `lineage.json`,
  `evolution_log/`.
- **Injected to `supervisor`**: it reads `lineage.json`, `attempts.jsonl`,
  `evolution_log/`, `profile` directly (see supervisor design).

**Boundedness (the key property):** what is *pushed* is hard-capped (best summary
+ insights list + failed list), independent of step count, so the prompt never
grows with run length. Unbounded raw history lives on disk and is only sampled via
`git show` when the agent chooses.

## 6. `operator=team` memory

When a step is a bounded Team round, two memory systems coexist:
- **AVO cross-step memory** (L1–L3 above) — spans steps/rounds, owns the lineage.
- **Team intra-round memory** — the reused `tech_lead` carries Team's *insight
  blackboard + hypothesis ledger* within the round (`history` is passed in as
  `HISTORY`). The round's outcome folds back into L1–L3 via the commit gate.

So Team's richer per-round memory is **subordinate** to AVO's lineage memory: it
informs direction planning, but only committed winners enter `P_t`.

## 7. Remaining gaps vs the paper (tracked)

1. **Continuous raw context (P-mem-3 / option B).** We carry a compacted summary +
   per-step rationale, not the full raw transcript (compiler/profiler logs +
   cross-version reasoning chain). *Mitigation:* a rolling compressed agent-context
   window across steps (future); a literal persistent agent conflicts with the
   per-step worktree reset and is out of scope.
2. ~~Multi-prior-implementation injection (§3.2).~~ **CLOSED.** Each step now
   examines the top-k prior implementations via
   `avo_lineage.py context --state <AVO_STATE> --k 3` (bounded: per-version
   speedup + per-shape + truncated patch, excluding v0 and the current best), plus
   `git show` on demand. See `roles/variation_engineer.md` step 2.
3. **Cross-run memory.** Each run cold-starts; persisting committed patches keyed
   by `(kernel, dtype, arch)` for cross-run seeding is future work (FACT-style).

## 8. Acceptance (memory is "working")

- Step N's prompt contains the current best + insights from steps < N (push),
  bounded regardless of N.
- A failed direction at step i is in `history.failed` and is NOT re-proposed at
  step j>i (supervisor self-validation).
- `lineage.json`, `attempts.jsonl`, `evolution_log/` are all non-empty and
  consistent after a multi-commit run; `avo_lineage_report.py` renders the chain.
