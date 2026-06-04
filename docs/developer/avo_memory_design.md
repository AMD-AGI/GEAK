# AVO Memory — Module Design

A multi-day AVO (Agentic Variation Operators) run must remember what it has
tried, what worked, and *why* — across hundreds of variation steps — while
keeping every LLM prompt **bounded** so the context window never grows with run
length. This document specifies AVO's memory subsystem as a self-contained
module: the constraint that shapes it, the three reconstruction layers, every
data structure and function, how a single step assembles its memory surface, the
boundedness guarantees, and the residual gaps vs the AVO paper.

> Scope: a deep-dive companion to [`avo_design.md`](avo_design.md) §13 (P-mem)
> and §15.1 (boundedness) and §17.9 (evolution log). The memory code lives in
> `src/minisweagent/run/avo/variation_step.py` + `controller.py`, reuses
> `minisweagent/memory/working_notebook.py` and
> `minisweagent/tools/strategy_manager.py`, and modifies no GEAK core module.

---

## 1. The constraint: why memory must be *reconstructed*

The AVO paper uses **one long-running agent with continuous conversation
memory** across the whole run. GEAK's `OptimizationAgent` does not fit that
shape, and AVO deliberately does not force it to:

1. `OptimizationAgent.run()` **resets `messages` every call** — there is no
   native cross-call conversation.
2. AVO starts a **fresh agent every variation step** and
   `reset_worktree_to_best`s the tree back to the committed best before each step.

These two choices are what make a 7-day run safe:

- **Bounded context.** Run length never accumulates into one growing
  conversation.
- **No error accumulation.** Each step re-bases on the clean current best, so a
  bad step is at worst wasted — never a regression of the working tree.

The cost: there is **no native continuous memory between steps**. AVO therefore
*reconstructs* memory from durable on-disk state and injects a **bounded**
summary into each step. The reconstruction has three layers:

| Layer | Name | Store | Injected | Role |
|-------|------|-------|----------|------|
| **P-mem-1** | Unified strategy file | `.optimization_strategies.md` (run-wide) | via the agent's tool | shared tried/failed/pending **state machine** |
| **P-mem-2** | Cross-step working notebook | `avo_state/notebook/` (`WorkingNotebook`) | summary in each prompt | "what was done, and the outcome" |
| **P-mem-3** | Causal evolution log | `avo_state/evolution_log/step_*.json` | bounded block in each prompt | "why it changed, and how the bottleneck moved" |

A guiding rule (`avo_design.md` §9.3): memory is **not** domain knowledge. These
layers carry *run state*, not optimization know-how (which lives in skills/RAG).

---

## 2. P-mem-1 — Unified strategy file (shared state machine)

### 2.1 Problem

`OptimizationAgent` normally writes its strategy file under the per-step
`patch_output_dir`. With a fresh agent per step, that state would be thrown away
every step — the run could not remember which strategies are already failed.

### 2.2 Mechanism

`_build_agent` pins the strategy file to a single **run-wide absolute path**:

```python
cfg = {
    ...
    "use_strategy_manager": True,
    # P-mem-1: one run-wide strategy file shared across steps AND with the
    # supervisor. An absolute path makes OptimizationAgent._get_strategy_file
    # ignore the per-step patch_output_dir. Lives outside the repo worktree so
    # it never leaks into kernel patches.
    "strategy_file_path": str((output_dir / ".optimization_strategies.md").resolve()),
}
```

Consequences:

- **Absolute path** ⇒ `OptimizationAgent._get_strategy_file` ignores the
  per-step dir; every step's agent reads/writes the *same* file.
- The file lives under `output_dir` (outside the repo worktree), so it never
  contaminates a kernel patch / lineage diff.
- The **supervisor shares this exact file**: `build_bundle._read_strategy_state`
  reads it; `apply_directive` writes it (mark failed, add strategies). It is the
  shared memory channel between the variation agents and the supervisor.

### 2.3 Effect

Even though each agent is brand new, on startup it can query `strategy_manager`
and see *"memory_coalescing (failed), shared_mem_tiling (no gain), …"* — so it
does not redo dead work, and the supervisor re-plans against the real history.

---

## 3. P-mem-2 — Cross-step working notebook (compressed summary)

Store: one run-wide `WorkingNotebook` at `avo_state/notebook/`, reusing GEAK's
existing facility (`minisweagent/memory/working_notebook.py`).

### 3.1 Write (end of each step, controller)

```python
def _record_to_notebook(notebook_root, step_index, result, committed) -> None:
    try:
        from minisweagent.memory.working_notebook import WorkingNotebook
        nb = WorkingNotebook(notebook_root, writer_id="avo")
        nb.record_attempt(strategy=result.strategy, change_category=None, step=step_index)
        nb.record_round_evaluation(
            round_num=step_index,
            best_task=result.strategy,
            verified_speedup=result.best_speedup,   # verified value, never self-report
            baseline_ms=None, candidate_ms=None, per_shape_speedups=None,
        )
        if not committed and not result.best_correct:
            nb.append_event("result", strategy=result.strategy, tag="FAIL",
                            message=f"step {step_index}: no committable candidate ({result.exit_status})",
                            step=step_index, returncode=1)
    except Exception as exc:                         # memory I/O is best-effort
        logger.debug("notebook record failed (non-fatal): %s", exc)
```

Every memory write is wrapped in `try/except`: a notebook failure must never
interrupt the evolution loop.

### 3.2 Read + inject (start of each step, variation_step)

```python
def _read_memory_summary(notebook_root):
    summary = WorkingNotebook.summarize_dir(notebook_root)
    return summary or None                          # None on step 1 (no events ⇒ prompt unchanged)
```

### 3.3 The summary is hard-capped (the boundedness key)

`WorkingNotebook.summarize_dir` aggregates *all* events into a fixed-size
summary, regardless of how many steps have run:

```python
lines.append("WHAT WORKED: " + "; ".join(winner_strs[:3]))          # top 3 winners
...
ranked = sorted(tried.items(), ...)[:5]                              # Tried families: top 5
...
lines.append("Dead ends: " + "; ".join(reversed(uniq_dead)))        # last 4 dead ends
...
for shape, (speedup, label) in list(sorted(best_shape...))[:2]:      # per-shape: top 2
```

Resulting shape (≈10–15 lines, independent of step count):

```text
--- Within-Session Working Notebook ---
Baseline: kernel=<cat> | bottleneck=<type> | baseline=<ms>ms
Best so far: 1.23x via shared_mem_tiling
WHAT WORKED: shared_mem_tiling (1.23x); vectorized_load (1.10x)
Tried families: shared_mem_tiling (3 try, best 1.2300x, IMPROVED); loop_unroll (2 try, best 0.98x, REGRESSED — bank conflicts)
Dead ends: loop_unroll: 0.9800x; naive_pad: 0.9500x
<shapeA> best 1.40x via shared_mem_tiling; <shapeB> weak 0.95x via loop_unroll
```

---

## 4. P-mem-3 — Causal evolution log (option C)

P-mem-1/2 capture *what was tried and the outcome*. P-mem-3 adds the missing
**causal chain** from the paper's continuous memory — *why* a change was made,
*how* the bottleneck moved, and the agent's own reasoning — in a bounded form.

### 4.1 The carried signal (on `VariationResult`)

```python
@dataclass
class VariationResult:
    ...
    rationale: str = ""        # agent's last substantive assistant message (its own account)
    raw_tail: str = ""         # short verbatim tail of recent assistant/tool turns
    profiling: dict[str, float] = field(default_factory=dict)   # this step's profiler metrics
```

- `rationale` / `raw_tail` are extracted from the finished agent's `messages` by
  `_capture_agent_memory`, truncated to `_RATIONALE_MAX = 800` and
  `_RAW_TAIL_MAX = 1800` chars.
- `profiling` is filled by `controller._read_profile_metrics`, which scans
  `round_{N}_evaluation.json` for scalar fields whose key looks like a perf
  metric (`occupancy`, `bandwidth`, `tflops`, `latency`, `register`, `lds`, …),
  capped at 8 entries (cross-backend best-effort).

### 4.2 Write (end of each step)

```python
def write_evolution_entry(output_dir, result, committed) -> None:
    entry = {
        "step_index": result.step_index,
        "strategy": result.strategy,
        "committed": bool(committed),
        "verified_speedup": result.best_speedup,
        "per_shape": result.per_shape_speedups,
        "profiling": result.profiling,
        "rationale": result.rationale,
        "failure": None if result.best_correct else f"no committable candidate ({result.exit_status})",
        "raw_tail": result.raw_tail,
    }
    (log_dir / f"step_{result.step_index:04d}.json").write_text(json.dumps(entry, indent=2, default=str))
```

### 4.3 Read + inject (start of each step)

`build_evolution_log` produces a bounded "causal history" block: the most recent
`k_recent` (default 2) steps verbatim (their `raw_tail`), older steps collapsed
to a one-liner, capped at `max_versions` (default 8):

```python
blocks = ["## Evolution log (causal history — what changed, why, and the effect)"]
older  = entries[:-k_recent][-max_versions:]
recent = entries[-k_recent:]
prev_prof = {}
for e in older:
    delta = _fmt_profiling_delta(e["profiling"], prev_prof)    # e.g. "occupancy 0.4→0.6, bandwidth ..."
    line  = f"- step {e['step_index']}: {e['strategy']} → {sp}x [{committed|rejected}]"
    if delta: line += f" | {delta}"
    if note:  line += f" | {rationale_or_failure[:140]}"
    blocks.append(line)
for e in recent:
    blocks.append(f"### step {e['step_index']} (recent): {e['strategy']} → {sp}x [{flag}]")
    if e["raw_tail"]:
        blocks.append(f"```\n{e['raw_tail']}\n```")
```

`_fmt_profiling_delta` diffs the current vs previous entry's profiler metrics, so
the agent sees **how the bottleneck moved across versions** — the part of
continuous memory that drives cumulative micro-architectural reasoning.

---

## 5. How a single step assembles its memory surface

`compose_task` prefixes all memory (except P-mem-1, which the agent reads via its
tool at runtime) in a fixed order:

```python
def compose_task(base_task, lineage, direction, memory_summary=None, exemplar=None,
                 profiling_enabled=True, hardware=None, lineage_context=None, evolution_log=None) -> str:
    parts = [contract]                              # AVO step contract (lineage.summary last 5)
    if hardware:        parts.append(hardware)      # re-grounded GPU facts (D1)
    parts.append(_PROFILING_NOTE if profiling_enabled else _STRUCTURAL_NOTE)
    if exemplar:        parts.append(exemplar)      # current-best diff (Kernel-Smith)
    if lineage_context: parts.append(lineage_context)  # other prior implementations + per-shape
    if evolution_log:   parts.append(evolution_log)    # P-mem-3 causal log
    if memory_summary:  parts.append("## Cross-step memory ...\n" + memory_summary)  # P-mem-2
    parts.append(base_task)
    return "\n\n".join(parts)
```

So the per-step "memory surface" is:

- **lineage summary** (last 5 commits) + **best exemplar** (current-best diff) +
  **lineage context** (top-K other committed versions + per-shape) — these come
  from the `LineageStore`, the durable record of *committed* versions `P_t`;
- **P-mem-2** notebook summary;
- **P-mem-3** causal evolution log;
- **P-mem-1** shared strategy state machine (consulted at runtime via the tool).

> Relationship to the lineage: the `LineageStore` is the authoritative record of
> *committed, verified* versions (`P_t`); P-mem-1/2/3 capture the *search
> trajectory around it* (including non-committed attempts, rationale, and
> outcomes). The exemplar/lineage-context injectors are documented in
> `avo_design.md` §16.2 / §17.6; this module covers the trajectory memory.

---

## 6. Boundedness guarantees (why a multi-day run never blows the window)

The defining constraint of the whole subsystem (`avo_design.md` §15.1):

| Memory source | Bound |
|---------------|-------|
| Within-step tool transcript | `step_limit` (default 200); each observation `truncate_observation`'d (10000 chars, head+tail) |
| Lineage summary | last 5 commits |
| P-mem-2 notebook summary | WHAT WORKED `[:3]`, Tried `[:5]`, Dead ends `[:4]`, per-shape `[:2]` → ~10–15 lines |
| P-mem-3 evolution log | last `k_recent` steps verbatim (`raw_tail` ≤1800 chars each) + one-liner × ≤`max_versions` |
| Best exemplar / lineage diffs | truncated to `_EXEMPLAR_DIFF_MAX = 4000` / `_LINEAGE_DIFF_MAX = 1500` |

On-disk stores (`attempts.jsonl`, notebook events, `evolution_log/`) grow without
limit, **but are never injected verbatim** — only their bounded summaries are.
Both axes are bounded: **across steps** (per-step reset + fixed-size summaries)
and **within a step** (`step_limit` + observation truncation).

---

## 7. Persistence map

| Artifact | Writer | Reader | Layer |
|----------|--------|--------|-------|
| `<output_dir>/.optimization_strategies.md` | variation agents (tool) + supervisor `apply_directive` | agents (tool), `build_bundle` | P-mem-1 |
| `avo_state/notebook/events/*.jsonl` | `_record_to_notebook` | `summarize_dir` → each prompt | P-mem-2 |
| `avo_state/evolution_log/step_*.json` | `write_evolution_entry` | `build_evolution_log` → each prompt | P-mem-3 |
| `avo_state/lineage.json` + `patches/` | `LineageStore.maybe_commit` | exemplar / lineage-context injectors | committed `P_t` |

All memory writes are best-effort (`try/except`), so a corrupt/missing store
degrades the prompt rather than crashing the run. On resume, P-mem-1/2/3 are all
re-read from disk, so cross-step memory survives restarts (unlike the in-memory
`StagnationDetector` counters).

---

## 8. Configuration

`avo.*` knobs in `src/minisweagent/config/geak_avo.yaml`:

```yaml
avo:
  evolution_log_enabled: true     # P-mem-3 on/off
  evolution_log_recent: 2         # recent steps shown verbatim (raw_tail)
  evolution_log_max_versions: 8   # older steps as structured one-liners
  inject_best_exemplar: true      # current-best diff into the prompt
  lineage_context_k: 3            # top-K other prior implementations injected (0 = off)
```

P-mem-1 (`strategy_file_path`) and P-mem-2 (notebook) are always on; their
content is bounded by `summarize_dir`, not by config.

Truncation constants are module-level in `variation_step.py`
(`_RATIONALE_MAX`, `_RAW_TAIL_MAX`, `_EXEMPLAR_DIFF_MAX`, `_LINEAGE_DIFF_MAX`).

---

## 9. Residual gaps vs the paper (by design)

- **Option C, not option B.** AVO carries the *causal signal* of a continuous
  session (rationale + raw tail + bottleneck Δ), not a literal persistent single
  agent (option B). Option B conflicts with the per-step worktree reset and
  needs core changes — out of scope.
- **No uncompressed continuous context.** There is no verbatim full history of
  compiler/profiler transcripts and the agent's complete cross-version reasoning
  chain; P-mem-3 approximates it within a fixed budget.
- **No cross-session knowledge base.** `GEAK_SAVE_TO_KNOWLEDGE_BASE` (memory
  *across runs*) is not yet wired into AVO; enabling it would let insights
  persist between runs — a future extension.

---

## 10. Extension points

- **Carry more recent raw context**: raise `evolution_log_recent` /
  `_RAW_TAIL_MAX` (watch the prompt budget, §6).
- **Richer causal metrics**: extend `_PROFILE_METRIC_HINTS` so more profiler
  fields enter the bottleneck-delta line.
- **Tune summary caps**: the `[:3]/[:5]/[:4]/[:2]` caps live in
  `WorkingNotebook.summarize_dir`.
- **Cross-run memory**: wire `GEAK_SAVE_TO_KNOWLEDGE_BASE` into finalize to
  persist committed insights keyed by kernel metadata.

---

## 11. References

- [`avo_design.md`](avo_design.md) — full AVO design: P-mem (§13), boundedness
  (§15.1), evolution log (§17.9), exemplar/lineage context (§16.2/§17.6).
- [`avo_supervisor_design.md`](avo_supervisor_design.md) — the supervisor, which
  shares P-mem-1.
- `src/minisweagent/run/avo/variation_step.py` — `_capture_agent_memory`,
  `build_evolution_log`, `write_evolution_entry`, `_read_memory_summary`,
  `compose_task`, the truncation constants.
- `src/minisweagent/run/avo/controller.py` — `_record_to_notebook`,
  `_read_profile_metrics`.
- `src/minisweagent/memory/working_notebook.py` — `WorkingNotebook` +
  `summarize_dir` (P-mem-2).
- `src/minisweagent/tools/strategy_manager.py` — the P-mem-1 state machine.
