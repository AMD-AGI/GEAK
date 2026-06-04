# AVO Supervisor — Module Design

The **Supervisor** is the subsystem that keeps a multi-day AVO
(Agentic Variation Operators) run from *stalling* or *"lying flat"* (摆烂). This
document specifies it as a self-contained module: its responsibilities, the
two-layer architecture, every data structure and function, the control flow,
the anti-stall guarantees, configuration, and extension points.

> Scope: this is a deep-dive companion to [`avo_design.md`](avo_design.md) §8.
> The Supervisor lives in `src/minisweagent/run/avo/stagnation.py` +
> `supervisor.py`, is dispatched by `controller.py`, and is configured by
> `subagents/avo-supervisor/`. It modifies no GEAK core module.

---

## 1. Problem statement

A single-lineage AVO run executes one `OptimizationAgent` "variation step" at a
time for hours or days. Three failure modes threaten such a run:

1. **Stalling** — the agent grinds on a dead direction (no committable
   improvement) and never self-corrects.
2. **Cycling** — the agent re-emits the same patch, or oscillates between two
   equivalent attempts.
3. **LLM co-stall** — a purely LLM-based "manager" can get stuck in the same
   blind spot as the worker it supervises.

The Supervisor's single responsibility: **detect non-progress and force a
productive change of direction**, with a guarantee that intervention happens
*even if every LLM in the system misbehaves*.

---

## 2. Design principle: two layers

```text
                 one variation step completes
                            │
                            ▼
        ┌───────────────────────────────────────────┐
  L1 →  │ StagnationDetector  (deterministic, no LLM) │
        │   counters → StagnationLevel  (0..4)        │
        └───────────────────────┬─────────────────────┘
                         signal.level
                            │  controller dispatch
              ┌─────────────┼───────────────┬───────────────┐
            NONE          NUDGE           REDIRECT         ESCALATE
           continue   next-prompt hint   ┌──────────┐   diversified
                                          │   L2     │     rescue
                                          │ LLM      │
                                          │ Supervisor│
                                          └──────────┘
```

- **L1 — `StagnationDetector` (deterministic).** A pure-counter "alarm clock".
  It owns the decision of *when* to intervene and *how hard*. No model call, so
  it always fires.
- **L2 — LLM Supervisor (`avo-supervisor`).** Invoked only at `REDIRECT`. It
  owns the decision of *what new direction to try*, grounded in evidence. It is
  read-only: it proposes; the controller executes.

**Why split.** A single LLM supervisor can stall together with the worker. The
deterministic outer alarm (L1) guarantees an intervention regardless of model
behavior; the LLM (L2) supplies intelligence on top, and even L2 has a
deterministic fallback (§5.4).

---

## 3. Layer 1 — `StagnationDetector`

File: `src/minisweagent/run/avo/stagnation.py`.

### 3.1 Intervention levels

```python
class StagnationLevel(enum.IntEnum):
    NONE = 0       # continue current direction
    NUDGE = 1      # inject a reminder into the next step's prompt
    INTERRUPT = 2  # force-end the current step early (handled by step_limit)
    REDIRECT = 3   # call avo-supervisor, switch direction, reset worktree
    ESCALATE = 4   # run a diversified parallel rescue round
```

The controller acts on `>= REDIRECT` by calling L2; `NUDGE` is a soft, no-LLM
hint; `INTERRUPT` is realized by the per-step `step_limit` rather than a
mid-step kill.

### 3.2 Tracked counters and default thresholds

| Counter | Default | Triggers |
|---------|---------|----------|
| `steps_without_commit` | 80 | wasted effort with nothing committed |
| `wall_time_without_commit_s` | 2700 | 45 min with no commit |
| `consecutive_correctness_failures` | 5 | repeated compile/correctness fails |
| `consecutive_no_improvement` | 8 | speedup ≤ 1.001 repeatedly |
| `patch_hash_repeat` | 3 | identical diff → cycle |
| `supervisor_cycles_without_commit` | 3 | L2 tried repeatedly, still nothing → ESCALATE |
| `trend_window` | 3 | window for "still climbing" rescue; 0 = off |

Defaults live in `_DEFAULTS` and are overridden by `avo.stagnation` from the
merged config.

### 3.3 The single "progress" signal

The only definition of progress is **`committed`** — i.e. whether
`LineageStore.maybe_commit` accepted a new *independently-verified* version this
step. The detector never trusts the agent's self-reported speedup.

```python
def evaluate(self, result: VariationResult, committed: bool) -> StagnationSignal:
    if committed:
        self._reset_progress_counters()         # commit ⇒ everything clears
        return StagnationSignal(StagnationLevel.NONE, "committed new best", self._snapshot())

    self.steps_without_commit += 1
    self.wall_time_without_commit_s += float(result.wall_time_s or 0.0)

    any_correct = any(a.correctness_passed for a in result.attempts)
    if result.attempts and not any_correct:
        self.consecutive_correctness_failures += 1
    else:
        self.consecutive_correctness_failures = 0

    improved = result.best_speedup is not None and result.best_speedup > 1.001
    climbing = self._is_climbing(result.best_speedup)       # trend awareness
    self.consecutive_no_improvement = 0 if (improved or climbing) else self.consecutive_no_improvement + 1

    repeat_hits = self._update_patch_cycle(result)
    return self._classify(repeat_hits)
```

### 3.4 Trend awareness (anti-false-stall)

A non-committing step whose verified speedup sets a **new high vs the recent
window** is treated as "still climbing" and does *not* accrue the
no-improvement stall — rescuing slow-but-real directions from a premature
redirect:

```python
def _is_climbing(self, speedup: float | None) -> bool:
    window = int(self._threshold("trend_window"))
    if window <= 0 or speedup is None:
        if speedup is not None:
            self._recent_speedups.append(speedup)
        return False
    prior = list(self._recent_speedups)
    self._recent_speedups.append(speedup)
    recent = prior[-window:]
    return len(recent) >= 2 and speedup > max(recent) * 1.001
```

### 3.5 Cycle detection

A bounded `deque(maxlen=16)` of recent `patch_hash` values; the max repeat count
of any hash this step is compared to `patch_hash_repeat`:

```python
def _update_patch_cycle(self, result: VariationResult) -> int:
    hits = 0
    for attempt in result.attempts:
        if not attempt.patch_hash:
            continue
        self._recent_hashes.append(attempt.patch_hash)
        count = sum(1 for h in self._recent_hashes if h == attempt.patch_hash)
        hits = max(hits, count)
    return hits
```

### 3.6 Classification (priority order)

```python
def _classify(self, repeat_hits: int) -> StagnationSignal:
    # ESCALATE first: L2 already intervened repeatedly with no commit.
    if self.supervisor_cycles_without_commit >= self._threshold("supervisor_cycles_without_commit"):
        return StagnationSignal(StagnationLevel.ESCALATE, ...)

    redirect_reasons = [...]   # any threshold hit → REDIRECT (with a human reason)
    if redirect_reasons:
        return StagnationSignal(StagnationLevel.REDIRECT, "; ".join(redirect_reasons), ...)

    # halfway to a redirect threshold → soft NUDGE
    if (self.consecutive_no_improvement >= max(1, thr_noimp // 2)
            or self.consecutive_correctness_failures >= max(1, thr_fail // 2)):
        return StagnationSignal(StagnationLevel.NUDGE, ...)

    return StagnationSignal(StagnationLevel.NONE, "progressing", ...)
```

Priority: **ESCALATE > REDIRECT > NUDGE > NONE**.

### 3.7 Counter reset semantics

```python
def reset(self, *, partial: bool = False) -> None:
    self.steps_without_commit = 0
    self.wall_time_without_commit_s = 0.0
    self.consecutive_correctness_failures = 0
    self.consecutive_no_improvement = 0
    self._recent_hashes.clear()
    self._recent_speedups.clear()       # new direction ⇒ fresh trend window
    if not partial:
        self.supervisor_cycles_without_commit = 0
```

- `partial=True` (after a REDIRECT): clear the per-direction stall counters that
  the redirect is expected to fix, but **keep** `supervisor_cycles_without_commit`
  so repeated fruitless redirects eventually ESCALATE.
- `partial=False` (after a commit or an ESCALATE): clear everything.

### 3.8 Data structures

```python
@dataclass(frozen=True)
class StagnationSignal:
    level: StagnationLevel
    reason: str        # human-readable, surfaced in logs + the L2 bundle
    counters: dict     # snapshot for the bundle
```

`note_supervisor_cycle()` increments the supervisor-cycle counter; it is called
by the controller every time it dispatches a REDIRECT (§6).

---

## 4. Layer 2 — the LLM Supervisor

File: `src/minisweagent/run/avo/supervisor.py`. Three public functions form the
pipeline: **`build_bundle` → `run_supervisor` → `apply_directive`**.

### 4.1 `build_bundle` — read-only evidence

```python
def build_bundle(signal, lineage, step_dir, output_dir) -> str:
    bundle = {
        "stagnation": {"reason": signal.reason, "counters": signal.counters},
        "lineage_summary": lineage.summary(last_n=8),
        "current_direction": lineage.current_direction(),
        "best_speedup": lineage.best_speedup,
        "recent_attempts": _recent_attempts(lineage, limit=10),   # tail of attempts.jsonl
        "strategy_state": _read_strategy_state(output_dir),       # P-mem-1 shared file
        "profile_bottleneck": _read_profile_bottleneck(output_dir),
    }
    return json.dumps(bundle, indent=2, default=str)
```

- `strategy_state` is read from the **run-wide** `.optimization_strategies.md`
  — the same file the variation agents write to (P-mem-1) — so the supervisor
  sees the real tried/failed history, not a per-step view.
- `profile_bottleneck` extracts a `bottleneck` / `limiter` / `summary` field
  from `profile.json` (best-effort; "unknown" if absent).

### 4.2 `run_supervisor` — LLM-first, self-validated, fallback

```python
def run_supervisor(bundle, avo_config, *, model=None) -> dict:
    if model is not None:
        directive = _run_llm_supervisor(bundle, model)
        if directive is not None:
            return _validate_directive(directive, bundle)
        logger.warning("supervisor: LLM path failed/unparseable; using fallback taxonomy.")
    return _fallback_directive(bundle)
```

Three guarantees of "always returns a usable directive":

1. **LLM path** (`_run_llm_supervisor`): one `model.query(messages)` using the
   `avo-supervisor` system prompt, parsed tolerantly by `_parse_directive`
   (accepts a fenced ```json block or a bare `{...}`; returns `None` on any
   failure).
2. **Self-validation** (`_validate_directive`, §4.3).
3. **Deterministic fallback** (`_fallback_directive`, §4.4).

### 4.3 `_validate_directive` — turn a subjective LLM output into a checked one

```python
def _validate_directive(directive, bundle) -> dict:
    tried = _tried_set(bundle)          # failed ∪ successful ∪ exploring ∪ skipped ∪ current
    novel, seen = [], set()
    for strat in directive.get("new_strategies") or []:
        key = str(strat.get("name", "")).strip().lower()
        if not key or key in tried or key in seen:
            continue                    # drop already-tried / duplicate proposals
        seen.add(key); novel.append(strat)
    if not novel:                       # nothing novel ⇒ splice one fallback direction
        novel = (_fallback_directive(bundle).get("new_strategies") or [])[:1]
    directive["new_strategies"] = novel
    return directive
```

This is the objectivity guard: the run never re-assigns a direction the
strategy state already knows is dead — **without** an extra model call.

### 4.4 `_fallback_directive` — deterministic taxonomy

When no model is available or the output is unparseable, rotate through 8
generic, language-agnostic directions, skipping those already failed/current:

```python
_FALLBACK_TAXONOMY = [
    {"name": "memory_coalescing",     "expected": "improve global memory access pattern"},
    {"name": "vectorized_load_store", "expected": "reduce memory transactions"},
    {"name": "shared_memory_tiling",  "expected": "reduce redundant global loads"},
    {"name": "loop_unrolling",        "expected": "expose more ILP"},
    {"name": "occupancy_tuning",      "expected": "rebalance registers / block size"},
    {"name": "pipeline_overlap",      "expected": "hide latency by overlapping stages"},
    {"name": "warp_specialization",   "expected": "split warp roles to overlap work"},
    {"name": "reduce_synchronization","expected": "remove unnecessary barriers/fences"},
]
```

This is what keeps a multi-day run progressing even if the supervisor LLM is
down for the entire run.

### 4.5 `apply_directive` — execute on GEAK's existing state machine

```python
def apply_directive(directive, lineage, strategy_file, *, supervisor_cycle, repo=None) -> None:
    from minisweagent.tools.strategy_manager import StrategyManager
    manager = StrategyManager(str(strategy_file)) if strategy_file.exists() else None

    if manager is not None:
        _mark_failed_by_name(manager, directive.get("mark_failed") or [], diagnosis)
        for strat in directive.get("new_strategies") or []:
            manager.add_strategy(name=..., description=..., expected=..., target=strat.get("priority"))
        manager.add_note(f"[avo-supervisor] {diagnosis}")

    next_strategy = str((directive.get("new_strategies") or [{}])[0].get("name", ""))
    lineage.set_direction(next_strategy, assigned_by="supervisor", supervisor_cycle=supervisor_cycle)

    if directive.get("backtrack_to_id"):                         # optional P2 backtrack
        if lineage.set_best_pointer(str(backtrack_to)) and repo is not None:
            lineage.reset_worktree_to(repo, str(backtrack_to))

    _log_supervisor(lineage, directive, supervisor_cycle)        # supervisor_log.jsonl
```

The supervisor's intelligence lands entirely **on top of** GEAK's existing
`strategy_manager` state machine and the lineage. It never edits a kernel or
runs the GPU.

### 4.6 Directive schema

```json
{
  "diagnosis": "memory-bound; shared-mem double buffering regressed due to bank conflicts",
  "mark_failed": ["shared_mem_double_buffer"],
  "new_strategies": [
    { "name": "vectorized_global_load", "priority": "high", "expected": "reduce memory transactions" },
    { "name": "warp_specialization", "priority": "normal", "expected": "overlap MMA/softmax" }
  ],
  "backtrack_to_id": null
}
```

---

## 5. The `avo-supervisor` subagent contract

Directory: `subagents/avo-supervisor/`.

`SUBAGENT.yaml` (key fields):

| Field | Value | Rationale |
|-------|-------|-----------|
| `execution_mode` | `inprocess` | no extra process for a single planning call |
| `agent.step_limit` | `12` | planning is short; bounded |
| `agent.cost_limit` | `0.0` | disabled (run budget governs) |
| `model.model_kwargs.temperature` | `0.0` | determinism for a planning decision |
| `tools.profiling` | `false` | read-only; never runs the GPU |
| `tools.rag` | `true` | may consult the knowledge base while planning |

`SYSTEM_PROMPT.md` instructs the model to (1) **diagnose** in one evidence-grounded
paragraph, (2) `mark_failed` dead ends, (3) propose **2–3 NEW** strategies
consistent with the bottleneck, (4) optionally `backtrack_to_id`, and to output
**only** the JSON directive.

> Implementation note: `supervisor.py` does not run this subagent through the
> subagent registry; it reads `SYSTEM_PROMPT.md` as the system prompt and issues
> one `model.query` (`_load_supervisor_system_prompt` → `_run_llm_supervisor`).
> `SUBAGENT.yaml` documents the contract and parameters.

---

## 6. Control flow (controller integration)

File: `src/minisweagent/run/avo/controller.py`, inside the main loop:

```python
signal = detector.evaluate(result, committed)
if signal.level == StagnationLevel.NUDGE:
    pending_nudge = f"Progress is stalling ({signal.reason}). Try a different angle ..."
elif signal.level == StagnationLevel.REDIRECT:
    _do_redirect(signal, lineage, step_dir, output_dir, strategy_file, detector, model_factory, work_repo)
elif signal.level == StagnationLevel.ESCALATE:
    _do_escalate(lineage, output_dir, config, model_factory, verify_ctx, work_repo, base_task=task)
    detector.reset(partial=False)
```

`_do_redirect` wires L1 → L2:

```python
def _do_redirect(signal, lineage, step_dir, output_dir, strategy_file, detector, model_factory, repo) -> None:
    detector.note_supervisor_cycle()
    cycle = detector.supervisor_cycles_without_commit
    bundle = build_bundle(signal, lineage, step_dir, output_dir)
    model = None
    try:
        model = model_factory()
    except Exception as exc:                       # even model build failure is non-fatal
        logger.warning("supervisor: could not build model (%s); using fallback taxonomy.", exc)
    directive = run_supervisor(bundle, {}, model=model)
    apply_directive(directive, lineage, strategy_file, supervisor_cycle=cycle, repo=repo)
    detector.reset(partial=True)
```

- **NUDGE**: no LLM; the reason text is injected into the *next* step's task body.
- **REDIRECT**: cycle++ → bundle → supervisor → apply → partial reset.
- **ESCALATE**: hand off to the diversified rescue (§7), then full reset.

The interaction between L1 and L2 around `supervisor_cycles_without_commit` is
the escalation ratchet: each REDIRECT increments it (and only a *commit* clears
it), so after `supervisor_cycles_without_commit` fruitless redirects, L1 returns
ESCALATE instead.

---

## 7. ESCALATE — diversified rescue

When redirects keep failing, `controller._do_escalate` runs a small population
of *distinct* directions in parallel-style worker dirs and keeps the best
verified one — the closest AVO gets to GEAK's parallel exploration:

```python
rescue_round = 9000 + len(lineage.committed)              # unique round id
for k, strat in enumerate(_diversified_directions(n_workers)):   # distinct taxonomy dirs
    run_variation_step(..., direction={"strategy": strat, "assigned_by": "escalate", ...},
                        avo_config=_with_patch_dir(avo_cfg, rescue_dir / f"rescue-worker-{k}"))
round_eval = evaluate_round_best(verify_ctx, rescue_round, rescue_dir)   # best of N, verified
lineage.commit_from_round(round_eval, repo=repo)          # fold best back via the commit gate
```

It reuses GEAK's multi-candidate evaluator (`evaluate_round_best`) and the same
commit gate, so a rescue result enters the lineage only if independently
verified.

---

## 8. Persistence & observability

| Artifact | Writer | Purpose |
|----------|--------|---------|
| `avo_state/direction.json` | `apply_directive` → `lineage.set_direction` | the strategy assigned to the next step |
| `avo_state/supervisor_log.jsonl` | `_log_supervisor` | every directive (ts, cycle, full JSON) |
| `.optimization_strategies.md` (run-wide) | `strategy_manager` (agents + supervisor) | shared tried/failed/pending state (P-mem-1) |
| `avo_state/attempts.jsonl` | `lineage.record_attempts` | the attempt tail surfaced in the bundle |
| `trajectory.json` (finalize) | `controller._write_trajectory` | supervisor-intervention count + per-strategy stats |

Detector counters are **in-memory only**: on resume a fresh `StagnationDetector`
starts at zero (a deliberate, conservative choice — see §10).

---

## 9. Anti-"lying-flat" guarantee matrix

| Failure mode | Mechanism | Layer |
|--------------|-----------|-------|
| Agent declares "done" early | outer loop treats any termination as "step done" | controller |
| Grinds a dead direction | stall thresholds → REDIRECT | L1 |
| Re-emits the same patch | `patch_hash_repeat` → REDIRECT | L1 |
| Slow-but-real direction killed early | `_is_climbing` trend rescue | L1 |
| LLM supervisor stalls/repeats | deterministic detector fires anyway; repeated cycles → ESCALATE | L1 |
| LLM proposes already-tried directions | `_validate_directive` drops them | L2 |
| LLM unavailable / unparseable | `_fallback_directive` taxonomy | L2 |
| Model construction fails | `_do_redirect` try/except → fallback | controller |
| Redirects keep failing | `supervisor_cycles_without_commit` → ESCALATE diversify | L1+controller |
| Fake progress (unverified speedup) | only `committed` (verified) counts as progress | commit gate |

---

## 10. Configuration

`avo.stagnation` in `src/minisweagent/config/geak_avo.yaml` (deep-merged over
`geak.yaml`):

```yaml
avo:
  stagnation:
    steps_without_commit: 80
    wall_time_without_commit_s: 2700      # 45 min
    consecutive_correctness_failures: 5
    consecutive_no_improvement: 8
    patch_hash_repeat: 3
    supervisor_cycles_without_commit: 3
    trend_window: 3                       # 0 disables trend awareness
  escalate:
    enabled: true
    rescue_mode: planned
    rescue_workers: 4
```

The L2 model is configured by `subagents/avo-supervisor/SUBAGENT.yaml` (and
inherits the run's model via `model_factory` when invoked through the
controller).

---

## 11. Residual gaps (by design)

- **Static, global thresholds.** They are not per-kernel adaptive; a genuinely
  slow-but-real direction is partially protected by `trend_window` but the
  hard caps are fixed. Adaptive (EWMA-of-delta) thresholds are a candidate
  extension.
- **No explicit novelty pressure.** Direction novelty is only "not in the
  failed/current set"; there is no behavioral-diversity archive (MAP-Elites).
  Acceptable for the single-lineage regime.
- **Detector state is not persisted across resume.** A restart re-arms the
  alarm from zero; conservative (won't prematurely ESCALATE after a restart).
- **Single-shot L2.** No multi-sample self-consistency voting on the directive;
  `_validate_directive` provides the determinism guard instead.

---

## 12. Extension points

- **Tune intervention aggressiveness**: edit `avo.stagnation` thresholds.
- **Add fallback directions**: extend `_FALLBACK_TAXONOMY` (keep names
  snake_case; they double as `direction.json` strategy names).
- **Richer diagnosis input**: add fields to `build_bundle` (the schema is a
  plain dict) — e.g. per-shape regressions or a longer attempt tail.
- **Persist detector state**: serialize the counters into `avo_state/` and
  reload in `__post_init__` if cross-resume continuity is desired.

---

## 13. References

- [`avo_design.md`](avo_design.md) — full AVO design (Supervisor = §8).
- `src/minisweagent/run/avo/stagnation.py` — L1 detector.
- `src/minisweagent/run/avo/supervisor.py` — L2 bundle/run/apply.
- `src/minisweagent/run/avo/controller.py` — dispatch (`_do_redirect`,
  `_do_escalate`).
- `subagents/avo-supervisor/{SUBAGENT.yaml,SYSTEM_PROMPT.md}` — L2 contract.
- `src/minisweagent/tools/strategy_manager.py` — the state machine directives
  are applied to.
