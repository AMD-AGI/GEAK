# AVO + Supervisor on GEAK — Design

This document describes how to implement **AVO (Agentic Variation Operators)**
on top of GEAK with **minimal intrusion** into the existing core. It maps the
AVO paper (NVIDIA, *Agentic Variation Operators for Autonomous Evolutionary
Search*, arXiv:2603.24517) onto GEAK's existing building blocks, specifies the
new modules, the **Supervisor** that keeps a long run from stalling, the
**skills** strategy, and a phased plan with a feature-status table.

> **Scope.** AVO is an **additive layer**. It reuses GEAK's preprocess,
> `OptimizationAgent`, `save_and_test`, `strategy_manager`, RAG, `RunBudget`,
> and git-worktree isolation. It does **not** modify `run/unified.py`,
> `agents/optimization_agent.py`, or any core scheduler. New code lives under
> `src/minisweagent/run/avo/`, `subagents/avo-supervisor/`, and
> `skills/avo-evolution/`.
>
> **Implementation note.** The AVO package is placed under
> `src/minisweagent/run/avo/` rather than a top-level `scripts/avo/` so that the
> `geak-avo` console script and the unit tests import it like every other GEAK
> entry point (`minisweagent.*`). The repo's `setuptools` config only packages
> `src/minisweagent*`, so a top-level `scripts/` package would not be
> installable. Skills and subagents stay at the repo root because they are
> discovered at runtime via `get_repo_root()`, not imported.

---

## 1. What AVO is, in GEAK terms

AVO replaces the classical `Vary(P_t) = Generate(Sample(P_t))` decomposition
with a single self-directed agent run:

```text
Vary(P_t) = Agent(P_t, K, f)
```

| AVO concept | GEAK realization | New or reused |
|-------------|------------------|---------------|
| `P_t` — lineage of committed solutions | `avo_state/lineage.json` + git tags `avo-v{N}` | **New** (`LineageStore`) |
| `K` — domain knowledge base | RAG MCP (`query`/`optimize`) + `skills/` | Reused |
| `f` — scoring function (correctness + throughput) | `COMMANDMENT.md` + `save_and_test` | Reused |
| `Agent(...)` — one variation step | one `OptimizationAgent.run()` in a worktree | Reused |
| Supervisor — stall/cycle intervention | deterministic detector + `avo-supervisor` subagent | **New** |
| 7-day continuous evolution | `RunBudget(total_s=...)` + outer controller loop | Reused + **New** |

The **only** conceptually new pieces are the **outer controller loop**, the
**lineage store**, and the **two-layer supervisor**. Everything that touches a
kernel, a test, or the GPU is already in GEAK.

---

## 2. Architecture

```text
                       ┌────────────────────────────────────────────┐
                       │  AVO Controller   (scripts/avo/controller)  │
                       │  deterministic outer loop; never "gives up" │
                       └───────────────┬────────────────────────────┘
            preprocess (reused)         │ per variation step
   ┌──────────────────────────┐        │
   │ run_preprocess_v3(...)    │        ▼
   │  → COMMANDMENT.md (= f)   │   ┌──────────────────────────────┐
   │  → baseline / profile     │   │ VariationStep                │
   │  harness-generator loads  │   │  - reset worktree → best ref │
   │  skills: triton / hip     │   │  - inject avo-evolution skill│
   └──────────────────────────┘   │  - OptimizationAgent.run()   │
                                   │    (save_and_test, RAG, ...) │
                                   └───────────────┬──────────────┘
                                                   │ attempts + result
                          ┌────────────────────────▼─────────────────────┐
                          │ LineageStore  (avo_state/)                    │
                          │  commit gate: correctness && speedup≥best·(1-ε)│
                          └────────────────────────┬─────────────────────┘
                                                   │ signals
                          ┌────────────────────────▼─────────────────────┐
                          │ StagnationDetector (deterministic)            │
                          │  level 0..4 — the "alarm clock"               │
                          └────────────────────────┬─────────────────────┘
                                level ≥ 3           │
                          ┌────────────────────────▼─────────────────────┐
                          │ avo-supervisor subagent (read-only LLM)       │
                          │  diagnose → mark failed → new directions      │
                          │  Controller executes the directive            │
                          └───────────────────────────────────────────────┘
                                                   │ budget exhausted
                                                   ▼
                          finalize (reused: auto_finalize / finalize_run)
```

---

## 3. Directory layout (all additive)

```text
src/minisweagent/run/avo/
├── __init__.py            # re-exports VariationResult / AttemptRecord
├── result.py              # VariationResult / AttemptRecord (pure dataclasses)
├── controller.py          # AVO outer loop + Typer CLI (geak-avo)
├── variation_step.py      # wraps one OptimizationAgent run + skill injection
├── lineage_store.py       # P_t persistence + git tags + commit gate
├── stagnation.py          # deterministic StagnationDetector
└── supervisor.py          # bundle builder + LLM/fallback directive + applier

src/minisweagent/config/
└── geak_avo.yaml          # extends geak.yaml; AVO knobs + long "full" budget

subagents/avo-supervisor/
├── SUBAGENT.yaml          # read-only, short step budget
└── SYSTEM_PROMPT.md

skills/avo-evolution/
├── SKILL.md
└── docs/
    ├── variation_step_contract.md
    ├── lineage_usage.md
    ├── stagnation_self_check.md
    └── kernel_adaptation.md

skills/attention-microarch-optimization/   # OPTIONAL — only for CUDA attention
├── SKILL.md
└── docs/...

tests/run/avo/
├── test_lineage_store.py  # commit gate + persistence (GPU-free)
└── test_stagnation.py     # detector levels (GPU-free)
```

Run artifacts (written under the normal run output dir, alongside
`COMMANDMENT.md`):

```text
optimization_logs/<kernel>_<ts>/
├── COMMANDMENT.md            # reused (= f)
├── baseline_metrics.json     # reused
├── profile.json              # reused
├── avo_state/                # NEW
│   ├── lineage.json          # committed versions (P_t)
│   ├── attempts.jsonl        # every attempt, incl. failures
│   ├── supervisor_log.jsonl  # supervisor interventions
│   ├── direction.json        # current assigned strategy
│   ├── stagnation_counters.json
│   └── heartbeat.json        # liveness + resume anchor
├── variation_0001/           # one dir per variation step
│   ├── agent.log
│   ├── .optimization_strategies.md
│   └── events/*.jsonl        # WorkingNotebook
└── final_report.json         # reused
```

---

## 4. Data model

### 4.1 `lineage.json`

```json
{
  "best_id": "v3",
  "committed": [
    {
      "id": "v0",
      "parent_id": null,
      "patch": "patches/v0_baseline.patch",
      "git_ref": "avo-v0",
      "strategy": "baseline",
      "score": { "speedup": 1.0, "latency_ms": 12.4, "verified": true },
      "committed_at": "2026-06-02T03:00:00Z"
    },
    {
      "id": "v3",
      "parent_id": "v2",
      "patch": "patches/v3.patch",
      "git_ref": "avo-v3",
      "strategy": "register_reallocation",
      "score": { "speedup": 1.08, "latency_ms": 11.5, "verified": true },
      "committed_at": "2026-06-02T05:21:00Z"
    }
  ]
}
```

### 4.2 Commit gate (aligns with paper §3.2)

A candidate enters `committed[]` **iff**:

1. correctness passes (`save_and_test` returns success), **and**
2. `verified_speedup >= best_speedup * (1 - epsilon)` where the speedup is the
   **independently verified** `FULL_BENCHMARK` value (the same value
   `post_round_evaluate` trusts — never the agent's self-report).

Failed or non-improving attempts are appended to `attempts.jsonl` only. This is
the GEAK-faithful version of "unsuccessful attempts stay in the search
trajectory but are not added to the committed lineage."

---

## 5. The Controller (outer loop)

`scripts/avo/controller.py` owns the loop. Its single responsibility is: **never
stop because an agent stopped.** A variation step may end via `Submitted`,
`LimitsExceeded`, an exception, or a deadline — the controller treats all of
these as "this step is done" and proceeds.

```python
"""AVO continuous-evolution controller.

Drives a single-lineage AVO run on top of GEAK's preprocess + OptimizationAgent.
The controller is a deterministic outer loop: it owns the lineage, the budget,
the stagnation detector, and supervisor dispatch. It deliberately does NOT
subclass or modify any GEAK core agent or scheduler.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from minisweagent.run.budget import BudgetSpec, RunBudget

from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.stagnation import StagnationDetector, StagnationLevel
from minisweagent.run.avo.supervisor import apply_directive, build_bundle, run_supervisor
from minisweagent.run.avo.variation_step import run_variation_step

logger = logging.getLogger(__name__)


def run_avo(
    *,
    repo: Path,
    task: str,
    output_dir: Path,
    avo_config: dict,
    budget: RunBudget,
) -> dict:
    """Run one AVO single-lineage evolution and return the final report dict."""
    lineage = LineageStore(output_dir / "avo_state")
    lineage.seed_from_baseline(output_dir)             # v0 from preprocess artifacts
    detector = StagnationDetector(avo_config["stagnation"])

    step_idx = 0
    while not budget.soft_stopped():
        step_idx += 1
        step_dir = output_dir / f"variation_{step_idx:04d}"
        lineage.reset_worktree_to_best(repo)

        result = run_variation_step(
            repo=repo,
            task=task,
            step_dir=step_dir,
            lineage=lineage,
            direction=lineage.current_direction(),
            avo_config=avo_config,
            deadline=budget.deadline,
        )

        lineage.record_attempts(result)
        lineage.maybe_commit(result)                   # commit gate (§4.2)

        signal = detector.evaluate(lineage, result)
        logger.info("variation %d: stagnation level=%s", step_idx, signal.level)
        if signal.level >= StagnationLevel.REDIRECT:
            bundle = build_bundle(signal, lineage, step_dir)
            directive = run_supervisor(bundle, avo_config)
            apply_directive(directive, lineage)
            detector.reset(partial=True)

    return _finalize(output_dir, lineage)


def _finalize(output_dir: Path, lineage: LineageStore) -> dict:
    """Reuse GEAK's auto_finalize so final_report.json keeps its canonical shape."""
    from minisweagent.run.postprocess.results import auto_finalize

    ctx = lineage.build_postprocess_ctx(output_dir)
    return auto_finalize(ctx)
```

Key reuse points (import, never fork):

- `RunBudget` / `BudgetSpec` for the wall-clock cap and `soft_stop`.
- `auto_finalize` for the final report — keeps `final_report.json` parseable by
  existing tooling.
- Worktree reset uses the same git plumbing GEAK already relies on.

---

## 6. The Variation Step

`run/avo/variation_step.py` wraps a single `OptimizationAgent` run. It does
**not** subclass it; it constructs it the same way GEAK does and injects the
AVO contract via the task body + skills.

```python
"""One AVO variation step = one OptimizationAgent run in an isolated worktree."""

from __future__ import annotations

import logging
from pathlib import Path

from minisweagent.skills.skill_runtime import SkillRuntime

logger = logging.getLogger(__name__)


def _select_skills(kernel_language: str, task: str) -> list[str]:
    """Pick which GEAK skills to inject for this step (Controller-driven)."""
    skills = ["avo-evolution"]                          # always
    if kernel_language == "flydsl":
        skills.append("flydsl")
    if "gemm" in task.lower():
        skills.append("fp8-gemm-tuning-sglang-aiter")
    return skills


def _inject_skill_bodies(task_body: str, skills: list[str]) -> str:
    """Force-inject skill bodies into the prompt instead of relying on self-load.

    In a multi-day run the model cannot be trusted to emit a ``use_skill``
    action every step, so the Controller reads the SKILL.md bodies via the
    existing SkillRuntime discovery and prepends them. No core change.
    """
    rt = SkillRuntime()
    bodies = []
    for name in skills:
        desc = rt.skills.get(name)
        if desc is not None:
            bodies.append((desc.path / "SKILL.md").read_text(encoding="utf-8"))
    return "\n\n".join(bodies + [task_body])


def run_variation_step(*, repo, task, step_dir, lineage, direction, avo_config, deadline):
    """Build + run one OptimizationAgent, return a structured result."""
    from minisweagent.agents.optimization_agent import OptimizationAgent
    # ... construct model/env exactly as GEAK does (see run/dispatch.py),
    #     set use_skills=True, step_limit=avo_config["variation_step_limit"]
    task_body = _compose_avo_task(task, lineage, direction)         # §7 contract
    task_body = _inject_skill_bodies(task_body, _select_skills(lineage.language, task))
    # agent.run(task_body) ... parse save_and_test logs into a VariationResult
```

**Skill loading is the key reuse decision** (see §9): the Controller reads
`SKILL.md` bodies through `SkillRuntime` and force-injects them, while still
leaving `use_skills=True` so the model's self-load path remains as a backstop.

---

## 7. AVO task contract (prompt addendum)

Every variation step's task body is prefixed with a contract so the agent knows
it is one link in a chain, not the whole job:

```text
## AVO Variation Step Contract
- You are executing ONE variation step in a continuous evolution run.
- Lineage best so far: {best_id} ({best_speedup}x). Parent: {parent_id}.
- Active direction (assigned by supervisor): "{direction}".
- You MUST call save_and_test after each meaningful edit.
- If this direction shows no improvement after 3 attempts, mark it failed via
  strategy_manager and request the next strategy.
- DO NOT declare the overall optimization complete — only THIS step ends.
- Only submit when you have a verified improvement OR you have exhausted this
  direction and documented why.
```

This text is the runtime mirror of `skills/avo-evolution/docs/variation_step_contract.md`.

---

## 8. Supervisor — two layers

The supervisor is what keeps the run from "lying flat" (摆烂). It has a
**deterministic layer** (always fires) and an **LLM layer** (re-plans). The
deterministic layer is essential: a pure-LLM supervisor can stall together with
the main agent, so a hard outer alarm guarantees intervention.

### 8.1 Deterministic `StagnationDetector`

`run/avo/stagnation.py`:

```python
from __future__ import annotations

import enum
from dataclasses import dataclass


class StagnationLevel(enum.IntEnum):
    NONE = 0       # continue
    NUDGE = 1      # inject a reminder into the next step's prompt
    INTERRUPT = 2  # force-end the current step early
    REDIRECT = 3   # call the avo-supervisor subagent, switch direction, reset worktree
    ESCALATE = 4   # run one GEAK parallel "rescue" round (mode="planned")


@dataclass(frozen=True)
class StagnationSignal:
    level: StagnationLevel
    reason: str
    counters: dict


@dataclass
class StagnationDetector:
    """Deterministic stall/cycle detector. No LLM. Thresholds from YAML."""

    config: dict

    def evaluate(self, lineage, result) -> StagnationSignal:
        """Inspect counters and return the highest triggered level."""
        ...
```

Default thresholds (in `geak-avo.yaml`, tunable):

| Counter | Default | Triggers |
|---------|---------|----------|
| `steps_without_commit` | 80 | wasted effort inside a step |
| `wall_time_without_commit_s` | 2700 | 45 min with no commit |
| `consecutive_correctness_failures` | 5 | repeated compile/correctness fails |
| `consecutive_no_improvement` | 8 | speedup ≤ 1.001 repeatedly |
| `patch_hash_repeat` | 3 | identical diff → cycle |
| `supervisor_cycles_without_commit` | 3 | supervisor tried, still no progress → ESCALATE |

### 8.2 LLM `avo-supervisor` subagent

A standard GEAK subagent (`subagents/avo-supervisor/SUBAGENT.yaml`), **read-only,
short budget**. It receives a Controller-built bundle and returns a JSON
directive. It never edits code or runs the GPU — the Controller executes its
decisions through `strategy_manager` and git.

`SUBAGENT.yaml` (follows the format in `subagents/README.md`):

```yaml
name: avo-supervisor
description: >-
  Use to re-plan an AVO evolution run that has stalled. Reads lineage,
  failed attempts, strategy state, and profile bottleneck; outputs a JSON
  directive (diagnosis, strategies to mark failed, new strategies, optional
  backtrack target). Read-only: proposes directions, never edits kernels.

execution_mode: inprocess

parameters:
  - name: bundle
    type: string
    description: "Stagnation bundle JSON (lineage summary, failures, profile)."
    required: true

agent:
  system_template_file: SYSTEM_PROMPT.md
  instance_template: |
    Stagnation bundle:
    {{bundle}}
  step_limit: 12
  cost_limit: 0.0

model:
  model_class: amd_llm
  model_name: claude-opus-4.6
  api_key: null
  model_kwargs:
    temperature: 0.0
    max_tokens: 8000

env:
  env:
    PAGER: cat
  timeout: 600

tools:
  profiling: false
  rag: true
```

Directive schema (returned by the subagent, applied by `supervisor.apply_directive`):

```json
{
  "diagnosis": "memory-bound; shared-mem double buffering regressed due to bank conflicts",
  "mark_failed": ["shared_mem_double_buffer"],
  "new_strategies": [
    { "name": "vectorized_global_load", "priority": "high",
      "expected": "reduce memory transactions" },
    { "name": "warp_specialization", "priority": "normal", "expected": "overlap MMA/softmax" }
  ],
  "backtrack_to_id": null
}
```

`apply_directive` then calls the existing `strategy_manager` API
(`mark <x> failed`, `add ...`), updates `direction.json`, optionally
`git checkout avo-v{backtrack}`, and appends to `supervisor_log.jsonl`. The
supervisor's intelligence lands entirely on top of GEAK's existing strategy
state machine.

### 8.3 Anti-"lying-flat" guarantees

| Mechanism | Where | Guarantee |
|-----------|-------|-----------|
| Outer loop survives any agent termination | `controller.py` | agent "done" ≠ run done |
| Commit gate | `lineage_store.py` | no false-positive "improvement" enters P_t |
| Per-step `step_limit` + INTERRUPT | detector + step config | no infinite grinding on one direction |
| REDIRECT resets worktree + new direction | controller + supervisor | a dead direction is abandoned |
| Fallback strategy taxonomy | `supervisor.py` | if the LLM supervisor fails, a fixed rotation continues |
| `min_commits_before_stop` | `geak-avo.yaml` | budget-remaining runs may not stop "with nothing" |
| ESCALATE rescue round | `controller.py` → `run_pipeline(mode="planned")` | single-lineage stall borrows GEAK parallel exploration |
| Heartbeat + resume | `avo_state/heartbeat.json` | crash/restart resumes from lineage |

### 8.4 ESCALATE — borrowing GEAK parallelism (the one deep reuse)

When the supervisor has intervened `supervisor_cycles_without_commit` times with
no commit, the controller runs **one** GEAK round in `planned` mode as a rescue,
then folds the best verified result back into the lineage:

```python
from minisweagent.run.unified import PipelineContext, run_pipeline
from minisweagent.run.postprocess.results import post_round_evaluate

ctx = lineage.build_pipeline_context(...)      # reuse PipelineContext as-is
run_pipeline(ctx, mode="planned")              # 1 round, N planned workers
best = post_round_evaluate(ctx_dict, round_num=1, output_dir=...)
if best and best.verified_speedup and best.verified_speedup > lineage.best_speedup:
    lineage.commit_from_round(best)
```

This is the only place AVO calls deep into GEAK orchestration, and even here it
**calls** the public entry point rather than editing it.

---

## 9. Skills usage

AVO reuses GEAK's skill system rather than re-encoding domain knowledge. See the
prior analysis; the rules below are normative for implementers.

### 9.1 Reuse map

| Skill | AVO role | Loaded at |
|-------|----------|-----------|
| `triton` | builds the harness = `f` | preprocess (`harness-generator`) — automatic |
| `hip` | builds the harness = `f` | preprocess (`harness-generator`) — automatic |
| `flydsl` | FlyDSL optimization knowledge = `K` | variation step (force-injected) |
| `fp8-gemm-tuning-sglang-aiter` | GEMM tuning playbook = `K` | variation step (GEMM tasks) |
| `pytorch2flydsl-translation` | reference for adaptation steps | variation step (migration) |
| `avo-evolution` | **NEW** — evolution methodology | variation step (always) |
| `attention-microarch-optimization` | **NEW, optional** — CUDA attention | variation step (attention) |

### 9.2 Loading rule (important)

GEAK skills are normally **model-self-loaded** via a ` ```skills ` action
(`skill_runtime.py`). For a multi-day AVO run that is unreliable. The Controller
therefore:

1. discovers skills via the existing `SkillRuntime` (no new discovery code), and
2. **force-injects** the selected `SKILL.md` bodies into each step's task body
   (`variation_step._inject_skill_bodies`), while
3. keeping `use_skills=True` so the self-load path remains a backstop.

This is a **read-only reuse** of `SkillRuntime` — no change to the runtime or to
any existing skill.

### 9.3 What is NOT a skill

Supervisor thresholds, lineage I/O, and the outer loop are **not** domain
knowledge and must not be encoded as skills. They live in
`src/minisweagent/run/avo/` (deterministic) and `subagents/avo-supervisor/`
(LLM re-planning). A variation agent must not "supervise itself" through a skill.

---

## 10. Configuration

`scripts/avo/config/geak-avo.yaml` extends `geak.yaml` (deep-merged, same
mechanism as `--config`):

```yaml
# Inherits model/env/tools from geak.yaml; only AVO-specific knobs here.
run:
  mode: full
  budgets:
    full:
      total_s: 604800        # 7 days; override per run with --total-budget-s
      preprocess_soft_cap_s: 1800
      finalize_grace_s: 600

avo:
  variation_step_limit: 200      # OptimizationAgent step_limit per variation
  variation_cost_limit: 0.0
  commit_epsilon: 0.001          # min relative speedup to enter lineage
  min_commits_before_stop: 5     # budget-remaining runs may not stop empty-handed

  stagnation:
    steps_without_commit: 80
    wall_time_without_commit_s: 2700
    consecutive_correctness_failures: 5
    consecutive_no_improvement: 8
    patch_hash_repeat: 3
    supervisor_cycles_without_commit: 3

  escalate:
    enabled: true
    rescue_mode: planned
    rescue_workers: 4
```

CLI entry (one line in `pyproject.toml`, mirrors `geak` / `geak-gemm-tuning`):

```toml
[project.scripts]
geak-avo = "scripts.avo.controller:app"
```

---

## 11. Code style — follow GEAK

New AVO code must match GEAK conventions so it reads like the rest of the repo:

- **Module headers**: a triple-quoted module docstring describing purpose and,
  where relevant, *why* a non-obvious choice was made (see `run/budget.py`,
  `run/unified.py` for the house style of long explanatory docstrings).
- `from __future__ import annotations` at the top of every module.
- **Dataclasses** for structured state (`LineageStore` entries, `StagnationSignal`),
  matching `Strategy` / `BudgetSpec` style.
- **Type hints everywhere**; `pathlib.Path` for paths; `str | None` unions
  (Python 3.10+ target).
- **Logging** via `logger = logging.getLogger(__name__)`; no `print` in library
  code (CLI user-facing output may use `rich.Console`, as in `run/mini.py`).
- **Typer** for the `geak-avo` CLI, `rich` for console tables — same stack as
  `subagent_cli.py`.
- **Reuse, don't fork**: import `RunBudget`, `auto_finalize`, `post_round_evaluate`,
  `PipelineContext`, `SkillRuntime`, `OptimizationAgent`, `strategy_manager`.
  Never copy their bodies into `run/avo/`.
- **Subagent/skill files** follow `subagents/README.md` and `skills/README.md`
  exactly (frontmatter keys, `SUBAGENT.yaml` sections).
- **Tests** mirror `tests/run/` layout; prefer `pytest` with the existing
  `conftest.py` fixtures. Pure-logic modules (`stagnation.py`, `lineage_store.py`)
  should be unit-testable without a GPU.

---

## 12. Intrusiveness assessment

| Change | Required? | Notes |
|--------|-----------|-------|
| Modify `run/unified.py` | **No** | AVO uses its own controller |
| Modify `OptimizationAgent` | **No** | constructed as-is; behavior via prompt + config |
| Modify `strategy_manager` | **No** | called through its existing API |
| Modify `SkillRuntime` | **No** | read-only reuse |
| New `src/minisweagent/run/avo/` modules | Yes | additive |
| New `src/minisweagent/config/geak_avo.yaml` | Yes | additive (packaged via existing `config/**/*`) |
| New `subagents/avo-supervisor/` | Yes | standard subagent extension |
| New `skills/avo-evolution/` | Yes | standard skill extension |
| New `geak-avo` CLI line | Yes (1 line) | `pyproject.toml [project.scripts]` |

**Net: 0 edits to core logic; 1-line CLI registration in `pyproject.toml`; the
rest is new additive files.**

---

## 13. Feature status

Tracks which AVO features exist and how each is realized on GEAK. Update the
**Status** column as development proceeds.

| AVO feature (paper §) | GEAK realization | Status |
|-----------------------|------------------|--------|
| Variation operator = full coding agent (§3.1) | `OptimizationAgent` per step (`variation_step.py`) | ✔ done |
| Single-lineage `P_t` (§3.3) | `LineageStore` + git tags `avo-v{N}` | ✔ done |
| Knowledge base `K` (§3.1) | RAG MCP + skills | ✅ reused |
| Scoring `f` = correctness + throughput (§3.1) | `COMMANDMENT.md` + `save_and_test` | ✅ reused |
| Commit gate (§3.2) | `LineageStore.maybe_commit` (verified speedup, ε) | ✔ done (unit-tested) |
| Edit→evaluate→diagnose loop (§3.2) | OptimizationAgent step loop | ✅ reused |
| Continuous evolution / persistence (§3.3) | `controller.run_avo` + `RunBudget` + heartbeat | ✔ done |
| Supervisor — deterministic layer (§3.3) | `StagnationDetector` (levels 0–4) | ✔ done (unit-tested) |
| Supervisor — LLM re-planning (§3.3) | `avo-supervisor` subagent + `supervisor.run_supervisor` | ✔ done |
| Supervisor fallback taxonomy | `supervisor._fallback_directive` | ✔ done |
| Strategy backtracking (§4.4) | `direction.json` + directive `backtrack_to_id` | 🟡 wired (checkout TODO) |
| ESCALATE rescue round (§8.4) | `controller._do_escalate` → `post_round_evaluate` | 🟡 wired (planned-round dispatch TODO) |
| Variant transfer e.g. MHA→GQA (§4.3) | `skills/avo-evolution/docs/kernel_adaptation.md` | ✔ doc done |
| Attention micro-arch patterns (§5) | `attention-microarch-optimization` skill | ⬜ optional (not yet created) |

Legend: ✅ reused (already in GEAK) · ⬜ planned (to build) · 🟡 partial/wired · ✔ done.

### Verified in this implementation

- `tests/run/avo/test_lineage_store.py` — commit gate accepts improvements,
  rejects regressions / incorrect / unverified candidates; JSON persistence
  round-trips; `direction.json` round-trips.
- `tests/run/avo/test_stagnation.py` — `NONE` on commit, `NUDGE` before
  thresholds, `REDIRECT` on no-improvement / correctness-failure / patch-cycle,
  `ESCALATE` after repeated supervisor cycles, partial vs full counter resets.

These two modules are pure-stdlib and run without a GPU or model. The
GPU/model-bound modules (`controller`, `variation_step`, `supervisor` LLM path)
are syntax/`py_compile`-validated; full execution requires an installed GEAK
environment (`make install`).

### Known follow-ups (left as explicit TODOs in code)

- `controller._do_escalate` currently calls `post_round_evaluate` over a
  `results/round_1/` directory; wiring the actual `run_pipeline(mode="planned")`
  rescue dispatch + folding its best patch back into the lineage is the next
  step.
- `supervisor.apply_directive` logs a requested `backtrack_to_id`; performing the
  `git checkout avo-v{N}` worktree backtrack is not yet executed.
- `variation_step._correctness_passed` uses log-marker heuristics; for stricter
  verification it can be swapped to GEAK's `post_round_evaluate` per step.

---

## 14. Implementation plan

Each phase is independently verifiable and leaves `main` green. Phases 0–3 are
**implemented** in this change; Phase 4 is future work.

### Phase 0 — scaffolding ✔ done (zero core edits)
- `src/minisweagent/run/avo/` package; `config/geak_avo.yaml`.
- `LineageStore` read/write (`lineage.json`, `attempts.jsonl`).
- `run_avo()`: preprocess (via `geak --preprocess-only`) → variation steps →
  finalize (`auto_finalize`).

### Phase 1 — continuous loop + commit gate ✔ done
- Outer `while` + `RunBudget` integration; `heartbeat.json`.
- Commit gate via verified speedup; git tag `avo-v{N}`; worktree reset to best.
- Force-injects `avo-evolution` skill body (+ language/task-matched skills).
- Verified by `tests/run/avo/test_lineage_store.py`.

### Phase 2 — supervisor ✔ done (core of the design)
- `StagnationDetector` with all levels; `note_supervisor_cycle` / partial reset.
- `subagents/avo-supervisor/` + `apply_directive` over `strategy_manager`;
  `supervisor_log.jsonl`; deterministic fallback taxonomy.
- Verified by `tests/run/avo/test_stagnation.py`.

### Phase 3 — long-run + escalate 🟡 mostly done
- 7-day `full` budget in `geak_avo.yaml`; resume from existing `lineage.json`.
- `geak-avo` CLI entry registered.
- ESCALATE hook present; full `run_pipeline(mode="planned")` rescue dispatch is a
  documented TODO (see §13).

### Phase 4 — evaluation & skills polish (future)
- Optional `attention-microarch-optimization` skill.
- Compare single-lineage AVO vs GEAK round-loop (commits/hour, speedup/hour).
- Consider exposing `--mode avo` only after Phases 1–3 prove stable in real runs.

### Running it

```bash
geak-avo --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel. Metric: latency (lower is better)." \
  --mode full          # 7-day cap; use --mode quick (1h) for smoke runs
# resumes automatically if avo_state/lineage.json already exists in -o dir
```

---

## 15. References

- AVO paper: *Agentic Variation Operators for Autonomous Evolutionary Search*,
  NVIDIA, arXiv:2603.24517.
- GEAK building blocks referenced above: `run/budget.py`, `run/unified.py`,
  `agents/optimization_agent.py`, `tools/strategy_manager.py`,
  `skills/skill_runtime.py`, `subagents/subagent_registry.py`,
  `run/postprocess/results.py`.
- Extension conventions: `subagents/README.md`, `skills/README.md`,
  [Contribution guidelines](contribution_guidelines.md).
