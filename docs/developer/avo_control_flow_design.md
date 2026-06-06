# AVO Execution Model & Control Flow — Module Design

This document specifies **how an AVO run is structured at runtime**: the two-level
loop model (the budget-driven outer loop of variation steps, and the bounded
inner agent loop), the per-step lifecycle, what `submit` means, every exit
condition at each level, budgeting/termination, and resume. It is the
"control-flow" companion to the subsystem docs
([supervisor](avo_supervisor_design.md), [memory](avo_memory_design.md),
[evaluation](avo_evaluation_design.md)) and to the top-level
[`avo_design.md`](avo_design.md).

> Code: `src/minisweagent/run/avo/controller.py` (`run_avo`, the outer loop) +
> `variation_step.py` (`run_variation_step`, one step) + `lineage_store.py`
> (commit gate) + `stagnation.py` (intervention levels). AVO modifies no GEAK
> core scheduler; it composes `OptimizationAgent`, `RunBudget`, and
> `evaluate_round_best`.

---

## 1. Two levels of looping

AVO has **two** nested loops. Keeping them distinct removes most confusion about
"rounds", "iterations", and "exits".

| Level | What one iteration is | Bound | Index / dir |
|-------|-----------------------|-------|-------------|
| **Outer** (controller) | one **variation step** = one `OptimizationAgent.run()` | wall-clock **budget** (no fixed count) | `step_idx` → `results/round_{step_idx}/` |
| **Inner** (the agent) | one **agent iteration** = one LLM decision + tool execution | `variation_step_limit` (default **200**) | the agent's internal step counter |

There is **no `max_rounds`** in AVO (unlike standard GEAK's `max_rounds=2/5`).
The number of variation steps is whatever fits in the budget.

---

## 2. The flow

```text
┌──────────────────────────────────────────────────────────────────────┐
│ AVO outer loop (budget-driven; no fixed round count)                   │
│  while not budget.soft_stop.is_set():        ← the only outer exit      │
│                                              (wall-clock budget spent)  │
│   step_idx += 1                                                         │
│   (1) reset_worktree_to_best(avo_repo)   # avo_repo back to clean best  │
│   (2) agent_repo = fresh ephemeral worktree (variation_N/repo)         │
│   (3) ┌────────────────────────────────────────────────────────┐       │
│       │  one variation step = one OptimizationAgent.run()         │       │
│       │  inner agent loop (iterations):                           │       │
│       │    for it in 1..variation_step_limit (default 200):       │       │
│       │       LLM decision -> tool (edit / save_and_test / ...)   │       │
│       │       ├─ calls submit ──────────► exit: Submitted (typical)│      │
│       │       ├─ it == 200 ─────────────► exit: LimitsExceeded     │      │
│       │       ├─ hits deadline ─────────► exit (deadline)          │      │
│       │       └─ exception ─────────────► exit (exception name)    │      │
│       └────────────────────────────────────────────────────────┘       │
│   (4) remove ephemeral worktree                                        │
│   (5) verify: _apply_verified_score (re-apply patch on clean best →    │
│       FULL_BENCHMARK + per-shape geomean verified speedup)             │
│   (6) maybe_commit (commit gate: correct & verified ≥ best·(1-ε) &     │
│       > min_commit_speedup & per-shape guard)                          │
│   (7) detector.evaluate -> intervention level:                        │
│        NONE → continue   NUDGE → hint next step                       │
│        REDIRECT → supervisor re-plan (new direction)                  │
│        ESCALATE → diversified rescue round                            │
│       (none of these exit the outer loop — they only change direction)│
└───────────────────────────────────┬────────────────────────────────────┘
                                     │ budget.soft_stop fires
                                     ▼
                          finalize (final_report.json, trajectory.json)
```

---

## 3. Outer loop (variation steps)

`controller.run_avo`:

```python
while not budget.soft_stop.is_set():
    step_idx += 1
    lineage.reset_worktree_to_best(work_repo)        # (1) clean best base
    agent_repo = _make_step_worktree(work_repo, step_dir / "repo")  # (2)
    try:
        result = run_variation_step(repo=agent_repo, ...)           # (3)
    finally:
        _remove_step_worktree(work_repo, agent_repo)                # (4)
    lineage.record_attempts(result)
    if verify_each_step:
        _apply_verified_score(result, verify_ctx, step_idx, ...)    # (5)
    committed = lineage.maybe_commit(result, repo=work_repo)        # (6)
    signal = detector.evaluate(result, committed)                   # (7)
    if   signal.level == NUDGE:    pending_nudge = ...
    elif signal.level == REDIRECT: _do_redirect(...)
    elif signal.level == ESCALATE: _do_escalate(...)
```

**Outer exit condition — exactly one:** `budget.soft_stop` is set. A watchdog
(`schedule_optimization_watchdog`) sets it ~`finalize_grace_s` before the
optimization deadline (`deadline = total_s − preprocess − grace`). Nothing else
ends the outer loop:

- Stagnation never ends the run — it routes to NUDGE / REDIRECT / ESCALATE.
- `min_commits_before_stop` is only a **warning** if the budget elapses with too
  few commits; it does not extend or cut the loop.
- `KeyboardInterrupt` / kill exit the process (state is already persisted; see
  [resume](#7-resume)).

---

## 4. Inner loop (one variation step) and `submit`

One variation step is a single `OptimizationAgent.run(task)`. Internally the
agent iterates up to `variation_step_limit` (default 200; config
`avo.variation_step_limit`), each iteration being one LLM decision + tool call
(`bash`, `str_replace_editor`, `save_and_test`, `strategy_manager`,
`profile_kernel`, RAG, …).

### What `submit` is

`submit` is the agent's **terminal tool** (`minisweagent.tools.submit`). Calling
it raises `Submitted` (a `TerminatingException`), which ends the agent's inner
loop with `exit_status="Submitted"`. In AVO, `submit` means:

- ✅ "**this** variation step is done" — and nothing more.
- ❌ NOT "the overall optimization is complete" — the AVO step contract
  explicitly says *"DO NOT declare the overall optimization complete — only THIS
  step ends."*
- ❌ NOT "commit to the lineage" — whether the step's result enters `P_t` is
  decided **independently** by the controller's commit gate (§6), after
  re-verifying. `submit` and "committed" are unrelated.

The step contract instructs the agent to `submit` only when it has a **verified
improvement** OR it has **exhausted the current direction and documented why**.

### Inner exit conditions (does a step exit early?)

**Yes — a step almost always exits before the 200-iteration cap**, because the
agent submits. The 200 limit is a fallback, not the norm.

| Exit | Trigger | `exit_status` | Early? |
|------|---------|---------------|--------|
| `submit` | agent has a verified improvement, or exhausted+documented the direction | `Submitted` | ✅ typical |
| step limit | inner iterations reach `variation_step_limit` (200) | `LimitsExceeded` | ❌ fallback cap |
| cost limit | `variation_cost_limit` reached (default 0 = off) | `LimitsExceeded` | ✅ if enabled |
| deadline | the step hits `budget.optimization_deadline()` | (deadline) | ✅ |
| exception | compile/run crash, etc. | exception class name | ✅ |

The controller treats **every** termination as "this step is done" and proceeds
— the core anti-"lying-flat" property: *an agent stopping ≠ the run stopping.*

---

## 5. Per-step lifecycle (the 7 stages)

1. **Reset** — `reset_worktree_to_best(avo_repo)`: `avo_repo` returns to the
   clean best (`git checkout -f avo-v{best}` + `git clean -fd`), serving as the
   verification base.
2. **Ephemeral worktree** — `_make_step_worktree`: the agent runs in a throwaway
   `git worktree` at `variation_NNNN/repo` (GEAK-aligned), so it never dirties
   `avo_repo`. Removed in step 4. (See [`avo_design.md`](avo_design.md) §1.)
3. **Agent run** — `run_variation_step` builds + runs one `OptimizationAgent`
   (skills injected, contract + memory + lineage context in the task body); the
   inner loop of §4 happens here. Patches saved to `results/round_{step}/avo-worker/`.
4. **Discard worktree** — `_remove_step_worktree`.
5. **Verify** — `_apply_verified_score` reuses `evaluate_round_best`: re-apply the
   step's best patch on a clean base, run FULL_BENCHMARK (+ optional repeats →
   median), compute the **independently verified per-shape geomean** speedup.
   (See [`avo_evaluation_design.md`](avo_evaluation_design.md).)
6. **Commit gate** — `lineage.maybe_commit`: a version enters `P_t` iff correct
   **and** `verified ≥ best·(1−ε)` (or `best·(1+margin)`) **and**
   `> min_commit_speedup` **and** no per-shape regression below the floor. This,
   not `submit`, decides `avo-v{N}`.
7. **Stagnation** — `detector.evaluate(result, committed)` returns a level; the
   controller acts (NUDGE / REDIRECT / ESCALATE). `committed` is the single
   "progress" signal. (See [`avo_supervisor_design.md`](avo_supervisor_design.md).)

---

## 6. The commit gate vs `submit` (why they differ)

A common point of confusion:

- `submit` ends the **agent's** loop (the agent's own judgment that the step is
  done). It is self-reported and not trusted for correctness.
- The **commit gate** is the controller's independent, verified decision about
  whether the step's best candidate becomes a new lineage version. It re-runs
  FULL_BENCHMARK on a clean base and applies the §6 thresholds.

So a step can `submit` "with an improvement" yet **not** be committed (if the
verified speedup doesn't clear the gate), and conversely a step that hit
`LimitsExceeded` can still be committed if it left a verified-better patch.

---

## 7. ESCALATE — the rescue "round"

When the supervisor has intervened `supervisor_cycles_without_commit` times with
no commit, `_do_escalate` runs a **diversified rescue round**
(`results/round_{9000+n}/`): `rescue_workers` (default 4) variation steps under
*distinct* generic directions, each in its own ephemeral worktree, evaluated
together with `evaluate_round_best`, best folded back via `commit_from_round`.
This is the only place AVO runs more than one step "per round id"; it does not
change the outer loop's exit logic.

---

## 8. Budget & termination

| Knob (`geak_avo.yaml` → `run.budgets.{mode}`) | Role |
|-----------------------------------------------|------|
| `total_s` | absolute wall-clock cap for the whole run |
| `preprocess_soft_cap_s` / `preprocess_hard_cap_fraction` | preprocess time bounds |
| `finalize_grace_s` | reserved at the end so finalize can run cleanly |
| `kill_buffer_s` | hard `os._exit` margin after the deadline |

`soft_stop` is the cooperative signal the outer loop polls at each step boundary.
On exhaustion the loop ends and `_finalize` writes `final_report.json` +
`trajectory.json`. Per-step `variation_step_limit` / `variation_cost_limit`
bound the inner loop independently of the wall clock.

---

## 9. Resume

State is durable: `avo_state/lineage.json` (committed `P_t` + tags in
`avo_repo/.git`), `attempts.jsonl`, `heartbeat.json`, the strategy file, the
working notebook, and the evolution log. Re-running `geak-avo` with the same
`-o` directory resumes: `_prepare_work_repo` reuses the existing `avo_repo`
(preserving `avo-v{N}` tags), `LineageStore` loads `lineage.json`, and the loop
continues producing variation steps from the current best. Preprocess is skipped
when `COMMANDMENT.md` already exists. (The in-memory `StagnationDetector` counters
restart at zero — a conservative choice.)

---

## 10. Quick answers

- **How many iterations per round?** Up to `variation_step_limit` (default 200)
  agent iterations per variation step; usually fewer because the agent `submit`s.
- **When does a round exit?** On `submit` (typical), the 200-iteration cap, a
  cost/deadline limit, or an exception — whichever comes first.
- **When does the whole run exit?** Only when the wall-clock budget's
  `soft_stop` fires (then finalize). Stagnation never exits; it redirects.
- **Is `submit` a commit?** No — `submit` ends the agent's step; the commit gate
  independently decides whether the verified result enters the lineage.

---

## 11. References

- [`avo_design.md`](avo_design.md) — top-level design (controller §5, variation
  step §6, repo isolation §1).
- [`avo_supervisor_design.md`](avo_supervisor_design.md) — stagnation levels +
  REDIRECT/ESCALATE.
- [`avo_evaluation_design.md`](avo_evaluation_design.md) — verified scoring +
  the commit gate's inputs.
- [`avo_memory_design.md`](avo_memory_design.md) — what each step injects/records.
- `src/minisweagent/run/avo/controller.py`, `variation_step.py`,
  `lineage_store.py`, `stagnation.py`; `minisweagent/tools/submit.py`,
  `agents/optimization_agent.py` (`Submitted` / `LimitsExceeded`).
