# AVO + Supervisor on GEAK — Design

This document describes how to implement **AVO (Agentic Variation Operators)**
on top of GEAK with **minimal intrusion** into the existing core. It maps the
AVO paper (NVIDIA, *Agentic Variation Operators for Autonomous Evolutionary
Search*, arXiv:2603.24517) onto GEAK's existing building blocks, specifies the
new modules, the **Supervisor** that keeps a long run from stalling, the
**skills** strategy, and a phased plan with a feature-status table.

> **Companion module docs.** This is the top-level design. Subsystems have their
> own deep-dive documents:
> [`avo_control_flow_design.md`](avo_control_flow_design.md) (execution model:
> the two-level loop, per-step lifecycle, `submit` vs commit gate, exit
> conditions), [`avo_supervisor_design.md`](avo_supervisor_design.md) (two-layer
> anti-stall supervisor), [`avo_memory_design.md`](avo_memory_design.md) (bounded
> cross-step memory), and
> [`avo_evaluation_design.md`](avo_evaluation_design.md) (verified scoring &
> profiling). Usage/run instructions live in
> [`run/avo/README.md`](../../src/minisweagent/run/avo/README.md).

> **Scope.** AVO is an **additive layer**. It reuses GEAK's preprocess,
> `OptimizationAgent`, `save_and_test`, `strategy_manager`, RAG, `RunBudget`,
> and git-worktree isolation. It does **not** modify `run/unified.py`,
> `agents/optimization_agent.py`, or any core scheduler. New code lives under
> `src/minisweagent/run/avo/`, `subagents/avo-supervisor/`, and
> `skills/avo-evolution/`.
>
> **Repo isolation (important).** AVO **never touches the user's original
> repository** — not even its `.git` (refs/tags/commits). After preprocess, the
> controller builds a **fully independent clone** at `<output_dir>/avo_repo` via
> `_prepare_work_repo`:
>
> - **Git repos:** `git clone --local` (objects hardlinked → ~no extra disk;
>   the clone gets its **own** `.git`), then the *live* tree is synced
>   (dirty-tracked + untracked + JIT `.so` symlinks + nested repos) using GEAK's
>   existing worktree helpers so the snapshot matches what preprocess profiled.
> - **Non-git directories:** a full copy bootstrapped into a throwaway git repo.
>
> `avo_repo` holds the lineage (`.git` + `avo-v{N}` tags). Because the clone has
> a **separate `.git`**, the `avo-v{N}` tags never appear in the user's `git tag`
> list — a multi-day run cannot mutate, clean, or pollute the user's repository
> in any way.
>
> **Per-step ephemeral worktree (GEAK-aligned).** The agent does **not** run
> directly in `avo_repo`. Each variation step (and each ESCALATE rescue worker)
> runs in a fresh `git worktree` created from the current best
> (`variation_NNNN/repo`), discarded right after the step — exactly like GEAK's
> per-round/per-slot worktrees. This keeps `avo_repo` clean (the agent's scratch
> — assembly dumps, test variants, build outputs — cannot accumulate or leak into
> the lineage) and gives verification a pristine `avo-v{best}` base. The agent's
> patch is saved under `results/round_{step}/`, so discarding the worktree loses
> nothing (`_make_step_worktree` / `_remove_step_worktree`).
>
> **Patch hygiene + anti "test hacking".** `save_and_test` drops non-source noise
> from captured patches — binary `Binary files … differ` stubs and compiler
> intermediates (`*.s`, `*.ll`, `*.ptx`, `*.cubin`, `*.bc`) — so the verified
> `best→candidate` patch applies cleanly and the lineage stays source-only. AVO
> also sets `GEAK_PROTECT_TEST_FILES=1`, which strips edits to test/harness files
> (`conftest.py`, `test_*.py`, `*_test.py`, `*harness*.py`, `task_runner.py`) from
> the patch: since verification re-applies the patch onto a clean base, an agent
> that tries to pass correctness by weakening the test/reference never affects the
> verified score or the lineage. (Both are env-gated and additive; standard GEAK
> is unchanged unless `GEAK_PROTECT_TEST_FILES=1` is set.)
>
> **Resume protection.** The `avo-v{N}` commits/tags live ONLY in
> `avo_repo/.git`. Re-cloning on resume would pull from the (clean) user repo
> and lose them, breaking `reset_worktree_to_best`. So `_prepare_work_repo`
> reuses an existing `avo_repo/.git` as-is instead of recreating it, preserving
> the accumulated lineage history across restarts.
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
                       │  AVO Controller   (run/avo/controller)      │
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
├── avo_repo/                 # NEW — independent clone (agent edits here, not the user's repo)
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

> **Module deep-dive:** how the verified speedup that feeds this gate is computed
> (per-shape geomean, profiling stages/timeouts, cost guards) is detailed in
> [`avo_evaluation_design.md`](avo_evaluation_design.md).

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

`run/avo/controller.py` owns the loop. Its single responsibility is: **never
stop because an agent stopped.** A variation step may end via `Submitted`,
`LimitsExceeded`, an exception, or a deadline — the controller treats all of
these as "this step is done" and proceeds.

> The block below is a **condensed** illustration of the real loop. The shipped
> `run_avo` additionally takes `model_name` / `gpu_ids` / `kernel_language`,
> builds a verify context, runs `_apply_verified_score` after each step (P0),
> records the working notebook (P-mem-2), and handles `NUDGE` / `ESCALATE` as
> well as `REDIRECT`. Signatures below match the actual API.

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
    while not budget.soft_stop.is_set():
        step_idx += 1
        step_dir = output_dir / f"variation_{step_idx:04d}"
        lineage.reset_worktree_to_best(repo)

        result = run_variation_step(
            repo=repo,
            base_task=task,
            step_dir=step_dir,
            lineage=lineage,
            direction=lineage.current_direction(),
            output_dir=output_dir,
            avo_config=avo_config,
            model_factory=model_factory,
            notebook_root=notebook_root,
        )

        lineage.record_attempts(result)
        _apply_verified_score(result, verify_ctx, step_idx, output_dir)  # P0: verified geomean
        committed = lineage.maybe_commit(result, repo=repo)              # commit gate (§4.2)

        signal = detector.evaluate(result, committed)
        if signal.level >= StagnationLevel.REDIRECT:
            bundle = build_bundle(signal, lineage, step_dir, output_dir)
            directive = run_supervisor(bundle, {}, model=model_factory())
            apply_directive(directive, lineage, strategy_file, supervisor_cycle=cycle, repo=repo)
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
- `_prepare_work_repo` builds the independent clone `avo_repo` (separate `.git`)
  so the loop operates on `work_repo`, never the user's original `repo`. The
  verify context's `repo_root` also points at `work_repo` so the verified
  `best → candidate` patch applies onto the same base the agent edited.

---

## 6. The Variation Step

`run/avo/variation_step.py` wraps a single `OptimizationAgent` run. It does
**not** subclass it; it constructs it the same way GEAK does and injects the
AVO contract via the task body + skills.

```python
"""One AVO variation step = one OptimizationAgent run in the repo (reset to best)."""

from __future__ import annotations

import logging
from pathlib import Path

from minisweagent.skills.skill_runtime import SkillRuntime

logger = logging.getLogger(__name__)


def select_skills(kernel_language: str, task: str) -> list[str]:
    """Pick which GEAK skills to inject for this step (Controller-driven)."""
    skills = ["avo-evolution"]                          # always
    if kernel_language == "flydsl":
        skills.append("flydsl")
    if "gemm" in task.lower():
        skills.append("fp8-gemm-tuning-sglang-aiter")
    return skills


def inject_skill_bodies(task_body: str, skills: list[str]) -> str:
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


def run_variation_step(*, repo, base_task, step_dir, lineage, direction,
                       output_dir, avo_config, model_factory,
                       deadline=None, nudge=None, notebook_root=None):
    """Build + run one OptimizationAgent, return a structured result."""
    from minisweagent.agents.optimization_agent import OptimizationAgent
    # ... construct model/env exactly as GEAK does (see run/dispatch.py),
    #     set use_skills=True, step_limit=avo_config["variation_step_limit"],
    #     patch_output_dir = output_dir/results/round_{step}/avo-worker (P0),
    #     strategy_file_path = output_dir/.optimization_strategies.md (P-mem-1)
    memory = _read_memory_summary(notebook_root)                    # P-mem-2
    task_body = compose_task(base_task, lineage, direction, memory_summary=memory)  # §7
    task_body = inject_skill_bodies(task_body, select_skills(lineage.language, base_task))
    # agent.run(task_body) ... light-parse worker dir into a VariationResult;
    # the controller then overwrites best_speedup with the verified geomean.
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

> **Module deep-dive:** see [`avo_supervisor_design.md`](avo_supervisor_design.md)
> for the full supervisor module design (every counter, the directive pipeline,
> control flow, and the anti-stall guarantee matrix). This section is the
> overview.

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

    def evaluate(self, result, committed: bool) -> StagnationSignal:
        """Inspect counters and return the highest triggered level.

        ``committed`` is whether the commit gate accepted a new version this
        step (the single source of "progress").
        """
        ...
```

Default thresholds (in `geak_avo.yaml`, tunable):

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
| `min_commits_before_stop` | `geak_avo.yaml` | budget-remaining runs may not stop "with nothing" |
| ESCALATE diversified rescue | `controller._do_escalate` | several distinct directions run + evaluated together; best verified folded in |
| Heartbeat + resume | `avo_state/heartbeat.json` | crash/restart resumes from lineage |

### 8.4 ESCALATE — diversified rescue (implemented)

When the supervisor has intervened `supervisor_cycles_without_commit` times with
no commit, the controller runs a **diversified rescue**: it launches
`rescue_workers` variation steps under *distinct* generic directions (from the
fallback taxonomy) into separate worker dirs of one rescue round, then reuses
GEAK's multi-candidate evaluator to pick and verify the best, and folds it back
into the lineage.

```python
# controller._do_escalate (condensed)
from minisweagent.run.postprocess.evaluation import evaluate_round_best

rescue_round = 9000 + len(lineage.committed)         # unique round id
for k, strat in enumerate(_diversified_directions(n_workers)):
    run_variation_step(..., direction={"strategy": strat, "assigned_by": "escalate", ...},
                        avo_config=_with_patch_dir(avo_cfg, rescue_dir / f"rescue-worker-{k}"))
round_eval = evaluate_round_best(verify_ctx, rescue_round, rescue_dir)  # selects best of N
lineage.commit_from_round(round_eval, repo=repo)     # verified geomean + commit gate
```

This reuses `evaluate_round_best` (which already selects the best among multiple
worker dirs in a round) instead of constructing a full `PipelineContext`,
keeping the rescue self-contained. It is the closest AVO gets to GEAK's
parallel exploration, without editing any core scheduler.

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
   (`variation_step.inject_skill_bodies`), while
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

`src/minisweagent/config/geak_avo.yaml` extends `geak.yaml` (deep-merged, same
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
  commit_epsilon: 0.001          # tolerance vs current best (don't regress)
  min_commit_speedup: 1.0        # anti-lazy-opt floor; a commit must EXCEED this vs baseline (§16.1)
  min_per_shape_speedup: 0.0     # per-shape regression guard; 0=off, e.g. 0.95 (§17.3)
  min_commits_before_stop: 5     # budget-remaining runs may not stop empty-handed
  verify_each_step: true         # FULL_BENCHMARK + per-shape geomean per step (P0)
  inject_best_exemplar: true     # inject current-best diff + metrics into each step (§16.2)
  profiling_after_step: 3        # delayed profiling: structural-first for N steps, then profiling-guided (§16.3)

  verify_repeats: 1              # B1 noise-robust median over N re-measures (§17.7)
  commit_significance_margin: 0.0  # B1 noise floor above best (§17.7)
  skip_verify_on_no_gain: true   # C3 skip FULL_BENCHMARK when agent self-reports no gain (§17.5)
  evolution_log_enabled: true    # P-mem-3 causal cross-step memory (§17.9)
  evolution_log_recent: 2        # recent steps shown verbatim
  evolution_log_max_versions: 8  # older steps as structured one-liners

  stagnation:
    steps_without_commit: 80
    wall_time_without_commit_s: 2700
    consecutive_correctness_failures: 5
    consecutive_no_improvement: 8
    patch_hash_repeat: 3
    supervisor_cycles_without_commit: 3
    trend_window: 3              # rescue a climbing direction from false-stall (§17.8); 0=off

  escalate:
    enabled: true
    rescue_mode: planned
    rescue_workers: 4
```

CLI entry (one line in `pyproject.toml`, mirrors `geak` / `geak-gemm-tuning`):

```toml
[project.scripts]
geak-avo = "minisweagent.run.avo.controller:app"
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
| Strategy backtracking (§4.4) | `set_best_pointer` + `reset_worktree_to`; supervisor `backtrack_to_id` executed | ✔ done |
| ESCALATE rescue round (§8.4) | `controller._do_escalate` → diversified workers → `evaluate_round_best` → `commit_from_round` | ✔ done |
| Multi-config per-shape geomean scoring (§4.1) | `_apply_verified_score` reuses `evaluate_round_best` (per-shape geomean) | ✔ done |
| Independent FULL_BENCHMARK verification (§3.2) | `_apply_verified_score` (worktree apply + FULL_BENCHMARK), `verify_each_step` toggle | ✔ done |
| Variant transfer e.g. MHA→GQA (§4.3) | `skills/avo-evolution/docs/kernel_adaptation.md` | ✔ doc done |
| Persistent memory (§4.1) — strategy state | run-wide `.optimization_strategies.md` shared by agents + supervisor | ✔ done (P-mem-1) |
| Persistent memory (§4.1) — attempt history | run-wide `WorkingNotebook` summary injected each step | ✔ done (P-mem-2) |
| Persistent memory (§4.1) — continuous causal context | evolution log: rationale + profiling Δ + raw tail (§17.9, option C) | ✔ done (option C; option B by design out) |
| Anti-lazy-optimization commit floor | `min_commit_speedup` (Kernel-Smith §16.1) | ✔ done |
| Best-program exemplar in prompt | `build_best_exemplar` (Kernel-Smith §16.2) | ✔ done |
| Delayed profiling injection | stage note + `profiling_after_step` (CuTeGen §16.3) | ✔ done |
| Kernel-class menus + anti-simplify rules | `optimization_playbook.md` (CuTeGen §16.4) | ✔ done |
| Lineage tag ≡ verified patch | `_materialize_and_tag` + baseline tag (§17.1) | ✔ done |
| Isolated work repo (original repo fully untouched, incl. .git) | `_prepare_work_repo` → independent clone `avo_repo` (separate .git) + resume reuse; loop uses `work_repo` | ✔ done |
| Worktree clean between steps | `git clean -fd` in `_checkout` (§17.2) | ✔ done |
| Per-shape regression guard | `min_per_shape_speedup` (§17.3) | ✔ done |
| Hardware grounding per step | `hardware_summary` injection (§17.4) | ✔ done |
| Multi-version lineage context in prompt (AVO §3.2) | `build_lineage_context` + per-node per_shape (§17.6) | ✔ done |
| Noise-robust progress signal | `verify_repeats` median + `commit_significance_margin` (§17.7) | ✔ done |
| Trend-aware stall detection | `trend_window` rescue of climbing directions (§17.8) | ✔ done |
| Supervisor directive self-validation | `_validate_directive` (§17.8) | ✔ done |
| Verified dedup cache (C2) / skip-on-no-gain (C3) | `verify_cache.json` + `skip_verify_on_no_gain` (§17.5) | ✔ done |
| Trajectory observability | `trajectory.json` at finalize (§17.5) | ✔ done |
| Best-of-K parallel generations (C1) | main-loop population (§17.5) | ⬜ deferred by choice |
| Population / MAP-Elites diversity | single-lineage by design (§16.5) | ⬜ optional (future) |
| Pattern registry + whole-graph composition | cross-session reuse / module mode (FACT §16.5) | ⬜ optional (future) |
| Attention micro-arch patterns (§5) | `attention-microarch-optimization` skill | ⬜ optional (not yet created) |

Legend: ✅ reused (already in GEAK) · ⬜ planned (to build) · 🟡 partial/wired · ✔ done.

### Verified in this implementation

- `tests/run/avo/test_lineage_store.py` — commit gate accepts improvements,
  rejects regressions / incorrect / unverified candidates; JSON persistence
  round-trips; `direction.json` round-trips; active-best pointer advances on
  commit; backtrack (`set_best_pointer`) + branch-from-pointer; `commit_from_round`
  prefers verified speedup and rejects patchless rounds.
- `tests/run/avo/test_stagnation.py` — `NONE` on commit, `NUDGE` before
  thresholds, `REDIRECT` on no-improvement / correctness-failure / patch-cycle,
  `ESCALATE` after repeated supervisor cycles, partial vs full counter resets.

These two modules are pure-stdlib and run without a GPU or model. The
GPU/model-bound modules (`controller`, `variation_step`, `supervisor` LLM path)
are syntax/`py_compile`-validated; full execution requires an installed GEAK
environment (`make install`).

### Scoring & verification (P0 — implemented)

> **Module deep-dive:** [`avo_evaluation_design.md`](avo_evaluation_design.md)
> covers the full verification flow, profiling stages/timeouts, and cost guards.

Each variation step writes its patches + `best_results.json` into GEAK's
canonical `results/round_{step}/avo-worker/` layout. After the step, the
controller calls `_apply_verified_score`, which reuses
`evaluate_round_best(ctx, step, results/round_{step})`: it applies the best
patch in a temp worktree, runs `FULL_BENCHMARK` + `PROFILE`, and computes a
**per-shape geomean** verified speedup — the same machinery GEAK's round loop
trusts. That verified value (not the agent's self-report or a heuristic log
scrape) feeds the commit gate. Set `avo.verify_each_step: false` to fall back to
the lightweight log parse when per-step FULL_BENCHMARK is too expensive.

### ESCALATE rescue (P1 — implemented)

`controller._do_escalate` runs `rescue_workers` variation steps under distinct
generic directions (from the fallback taxonomy) into
`results/round_{9000+n}/rescue-worker-k/`, then calls `evaluate_round_best` —
which already selects the best among multiple worker dirs — and folds the best
verified result into the lineage via `LineageStore.commit_from_round`. This
reuses GEAK's multi-candidate evaluator instead of constructing a full
`PipelineContext`, keeping the rescue self-contained.

### Backtracking (P2 — implemented)

`LineageStore` carries an explicit `active_best_id` "tip" pointer.
`set_best_pointer(id)` moves it to an earlier committed version and
`reset_worktree_to(repo, id)` checks out that version; the supervisor's
`backtrack_to_id` directive triggers both. Subsequent commits gate against — and
branch from — the backtracked node (single-lineage semantics; no archive/tree).

### Memory mechanism (P-mem — implemented)

> **Module deep-dive:** the three memory layers (strategy file / working notebook
> / causal evolution log) and their boundedness are detailed in
> [`avo_memory_design.md`](avo_memory_design.md).

The paper uses **one long-running agent with continuous conversation memory**
across the whole run. GEAK's `OptimizationAgent` resets `messages` per run, so
AVO runs a fresh agent each variation step and **reconstructs** cross-step memory
instead. Two pieces make that reconstruction coherent:

- **P-mem-1 — unified strategy file.** Each step's agent pins its
  `strategy_file_path` to a single run-wide `output_dir/.optimization_strategies.md`
  (absolute path → `OptimizationAgent._get_strategy_file` ignores the per-step
  `patch_output_dir`). The supervisor reads and writes the **same** file. So the
  "tried / failed / pending" strategy state persists across steps and is shared
  between the variation agents and the supervisor. The file lives outside the
  repo worktree, so it never leaks into kernel patches.
- **P-mem-2 — cross-step working notebook.** A single run-wide
  `WorkingNotebook` at `avo_state/notebook/` records each step's attempt +
  verified outcome; its `summarize_dir` summary (best-so-far, what worked, dead
  ends, recent evidence) is injected into every step's task body. On the first
  step there is no summary, so the prompt is unchanged.

Both reuse existing GEAK facilities (`strategy_manager`, `WorkingNotebook`) with
no core change. **Remaining gap vs the paper:** there is still no continuous raw
conversation context (compiler/profiler transcripts and the agent's own
cross-version reasoning chain) — the reconstruction is a compact summary, not the
full history. Closing that fully (P-mem-3, a rolling compressed agent context)
is future work.

### Remaining optional follow-up

- `skills/attention-microarch-optimization/` is not yet created (only needed to
  reproduce the paper's CUDA attention experiments on NVIDIA hardware).
- P-mem-3: carry a rolling compressed agent context across steps to approach the
  paper's continuous-memory model.
- Cross-session knowledge base (`GEAK_SAVE_TO_KNOWLEDGE_BASE`) is not wired into
  AVO; enabling it would let optimization insights persist across runs.

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

### Phase 3 — long-run + escalate ✔ done
- 7-day `full` budget in `geak_avo.yaml`; resume from existing `lineage.json`.
- `geak-avo` CLI entry registered.
- ESCALATE implemented as a diversified rescue (§8.4): distinct directions →
  `evaluate_round_best` → `commit_from_round`.

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

## 15. Long-horizon robustness

This section answers four questions about running AVO for hours/days: does the
LLM context window stay bounded; does the search avoid oscillation and
"lying-flat"; how is a bad direction detected and stopped; and how are new
directions designed. It states what the current implementation guarantees and
where the residual gaps are.

### 15.1 Context window stays bounded

The dominant design choice — **a fresh `OptimizationAgent` per variation step,
with `messages` reset each step** — is what keeps the window bounded over a
multi-day run. Run length does **not** accumulate into a single growing context.

| Source of per-step context | Bound |
|----------------------------|-------|
| Within-step tool transcript | `step_limit` (default 200) steps; each observation capped by `truncate_observation` (`OBSERVATION_MAX_LEN = 10000` chars, head+tail elision) |
| Injected skill bodies | fixed set (`avo-evolution` + language/task-matched); static size |
| Lineage summary | `LineageStore.summary(last_n=5)` — last 5 commits only |
| Cross-step memory summary | `WorkingNotebook.summarize_dir` is **hard-capped**: WHAT WORKED `[:3]`, Tried families `[:5]`, Dead ends `[:4]`, Per-shape `[:2]`, Recent evidence `[-3:]` → ~10–15 lines regardless of step count |
| COMMANDMENT / codebase context | fixed, from preprocess |

So both axes are bounded: **across steps** (reset + fixed-size summaries) and
**within a step** (`step_limit` + observation truncation). The on-disk
`attempts.jsonl` / notebook events grow, but they are never injected verbatim —
only their bounded summary is.

**Residual gap (P-mem-3):** the flip side of resetting per step is no continuous
*raw* context across steps. This is now mitigated by the bounded **evolution log**
(§17.9, option C): rationale + profiler delta + a short verbatim recent tail are
carried across steps. A literal persistent agent (option B) remains out of scope
by design (conflicts with the per-step worktree reset).

### 15.2 No oscillation, no lying-flat

| Failure mode | Mechanism that prevents it |
|--------------|----------------------------|
| **Patch cycling** (re-emitting the same diff) | `StagnationDetector.patch_hash_repeat` → `REDIRECT` |
| **Drift / corruption accumulation** | every step `reset_worktree_to_best` — each step starts from the clean current best, not from a half-broken prior attempt |
| **Fake progress** (claimed but unverified speedup) | commit gate uses the independently verified per-shape geomean (§13 P0); self-reported numbers never enter the lineage |
| **Agent "declares done" early** | outer loop treats any agent termination as "step done" and continues until the budget's `soft_stop` |
| **Budget burned with nothing** | `min_commits_before_stop` warning; the loop keeps producing steps while budget remains |
| **Supervisor itself stalls** | deterministic detector fires regardless of the LLM; repeated supervisor cycles without a commit → `ESCALATE` |

The worktree reset is the key anti-oscillation property: because each step
re-bases on the committed best, a bad step cannot poison the next one — at worst
it is a wasted step, never a regression of the working tree.

**Residual gap:** at the *macro* level, the supervisor could in principle keep
proposing directions in a neighborhood that never commits; this is bounded by
`supervisor_cycles_without_commit → ESCALATE` (which diversifies) and ultimately
by the wall-clock budget, but there is no "abandon the whole run early" signal —
by design AVO uses the full budget.

### 15.3 Detecting an unreliable direction and stopping it

Two layers decide a direction is not worth continuing:

1. **Agent self-pivot** (`skills/avo-evolution/docs/stagnation_self_check.md`):
   after 3 no-improvement attempts / 2 same-root-cause failures / an unchanged
   profiler bottleneck, the agent itself marks the strategy `failed` via
   `strategy_manager` and requests the next one.
2. **Deterministic `StagnationDetector`** (the backstop that always fires):

   | Counter | Default | "Direction unreliable" signal |
   |---------|---------|-------------------------------|
   | `consecutive_no_improvement` | 8 | speedup ≤ 1.001 repeatedly |
   | `consecutive_correctness_failures` | 5 | can't even compile/pass |
   | `steps_without_commit` | 80 | effort with nothing to show |
   | `wall_time_without_commit_s` | 2700 | 45 min with nothing |
   | `patch_hash_repeat` | 3 | going in circles |

   Reaching any of these → `REDIRECT`: the LLM supervisor marks the dead
   direction `failed` and assigns a new one; counters reset (`partial`).

**Residual gap:** thresholds are **static and global**. A "slow-but-real"
direction (tiny positive deltas) is treated like a truly dead one. A useful
refinement is an *adaptive / trend-aware* criterion (e.g. EWMA of per-step delta,
or "no commit AND negative trend") so promising-but-slow paths are not cut
prematurely. Not yet implemented.

### 15.4 Designing new optimization directions

When a direction is retired, a new one is produced by, in order of preference:

1. **`avo-supervisor` LLM**, grounded in evidence — it receives the lineage
   summary, recent attempts, the run-wide strategy state (P-mem-1), and the
   profiler bottleneck, and must propose 2–3 new strategies **consistent with
   the bottleneck** and **not already tried** (enforced by the system prompt and
   `strategy_state.failed`).
2. **RAG / skills** as the knowledge base `K` the agent consults while
   implementing a direction.
3. **Deterministic fallback taxonomy** (`supervisor._FALLBACK_TAXONOMY`, 8
   generic directions) when the LLM is unavailable/unparseable — guarantees the
   run always has a next move.
4. **ESCALATE diversification**: a rescue round runs several *distinct* taxonomy
   directions in parallel worker dirs and keeps the best verified one.

**Residual gap:** direction *novelty* is only "not in the failed/current set".
There is no explicit behavioral-diversity pressure (e.g. MAP-Elites niches) —
acceptable for the paper's single-lineage regime, but a candidate extension if
exploration breadth becomes the bottleneck.

### 15.5 Summary

| Concern | Status |
|---------|--------|
| LLM window bounded over a long run | ✔ guaranteed (per-step reset + capped summaries + observation truncation) |
| Continuous raw cross-step context | 🟡 reconstructed summary only (P-mem-3 future) |
| Oscillation / cycling | ✔ patch-hash detector + worktree reset |
| Lying-flat / fake progress | ✔ verified commit gate + loop survives agent exit + ESCALATE |
| Detect & stop a bad direction | ✔ agent self-pivot + deterministic thresholds (🟡 static, not adaptive) |
| Design new directions | ✔ supervisor (evidence-grounded) + fallback taxonomy + ESCALATE diversify (🟡 no explicit novelty pressure) |

---

## 16. Insights adopted from related work

This section records ideas borrowed from adjacent papers, what was implemented,
and what was deliberately left out.

### Kernel-Smith (arXiv:2603.28342)

**Kernel-Smith** (*A Unified Recipe for Evolutionary Kernel Optimization*,
Shanghai AI Lab + MetaX) studies the same evolutionary-kernel-optimization
problem with a population (island + MAP-Elites) agent and an evolution-oriented
post-training recipe. Two of its engineering findings are directly applicable to
AVO and are now implemented:

### 16.1 Anti-"lazy optimization" commit floor (implemented)

Kernel-Smith documents a failure mode where strong models produce a
correct-but-trivial rewrite (e.g. re-expressing an elementwise add in Triton)
that passes compilation/correctness at ≈1.0x but has no real value ("lazy /
advanced hacking"). Our original commit gate (`candidate >= best·(1-ε)`) would,
when `best = 1.0` (the baseline), admit such a 1.0x candidate as the first
"improvement".

**Fix:** `LineageStore.min_commit_speedup` (config `avo.min_commit_speedup`,
default `1.0`). A candidate must **exceed** this verified speedup over the
baseline to enter the lineage — a ≈1.0x trivial rewrite is rejected. Set it
higher (e.g. `1.05`) to require a minimum gain per commit. Verified by
`tests/run/avo/test_lineage_store.py::test_commit_gate_rejects_lazy_no_gain` and
`::test_commit_gate_respects_custom_min_speedup`.

### 16.2 Best-program exemplar in the step prompt (implemented)

Kernel-Smith injects the **top-performing program(s)** plus structured metrics
as exemplars each iteration, so the model edits from the best implementation
rather than re-deriving it. AVO already resets the worktree to the best version,
but previously the prompt carried only a *text summary*. `build_best_exemplar`
now injects the **current-best diff + its verified speedup + the strategy that
produced it** (config `avo.inject_best_exemplar`, default on), truncated to keep
the prompt bounded (§15.1). Verified by `tests/run/avo/test_variation_exemplar.py`.

### CuTeGen (arXiv:2604.01489)

**CuTeGen** (*An LLM-Based Agentic Framework … using CuTe*, U. Toronto) does
single-kernel progressive refinement (no population) with three findings; two
are adopted:

#### 16.3 Delayed profiling injection (implemented)

CuTeGen shows (with an ablation) that exposing profiler feedback too early on
structurally complex kernels biases the model toward premature parameter tuning
(tile-size sweeps) and poor local optima; withholding it until the structure is
sound yields a better trajectory. AVO now stamps each step's prompt with an
**Optimization stage**: `STRUCTURAL (profiling withheld)` for the first
`avo.profiling_after_step` steps (default 3) — flipped on early if a real commit
already landed — then `PROFILING-GUIDED`. `compose_task` injects the matching
note; the `avo-evolution` skill instructs the agent to honor it. Set
`profiling_after_step: 0` for simple/elementwise kernels (profile early).

#### 16.4 Kernel-class menus + non-negotiable anti-simplify rules (implemented)

CuTeGen's optimization/debugging prompts (Appendices A/B) first **classify** the
kernel (GEMM / elementwise / other) and offer a class-specific optimization
menu, and enforce a **non-negotiable rule**: never "fix" or "optimize" by
simplifying away the kernel structure or falling back to PyTorch/cuBLAS. This is
the *debug/optimize-side* complement to §16.1's commit-side anti-lazy floor.
Captured in `skills/avo-evolution/docs/optimization_playbook.md` and summarized
in `SKILL.md` (loaded into every variation step).

### 16.5 Considered but not adopted (kept as future work)

- **Population / MAP-Elites + island diversity** (Kernel-Smith). A *diverse*
  archive (feature space = kernel complexity × score) sampling top + diverse
  exemplars would address the "no explicit novelty pressure" gap (§15.4) but
  changes AVO's single-lineage regime (chosen by the AVO paper to isolate the
  operator). Optional extension.
- **Dynamic pattern registry + whole-graph multi-pattern composition** (FACT,
  *Framework for Agentic CUTLASS Transpilation*, arXiv:2604.26666). FACT indexes
  realized kernels by `(rule, dtype, arch)` and reuses them across workloads, and
  optimizes *multiple* subgraphs of a whole model then composes them. For AVO
  this maps to (a) persisting committed patches keyed by kernel metadata into the
  cross-session knowledge base for cross-run reuse, and (b) a module-level
  multi-kernel mode. Both are larger, scenario-level extensions — left as future.
- **Evolution-oriented post-training** (Kernel-Smith step-centric SFT/RL). AVO
  uses a frozen frontier model via `OptimizationAgent`; training a local improver
  is out of scope for this additive layer.
- **Measurement stability** (Kernel-Smith / CuTeGen: warm-up, repeated runs +
  outlier removal, CUDAGraph → <1% noise). AVO relies on GEAK's harness /
  `evaluate_round_best`; the dependency is noted in `skills/avo-evolution` so
  harness authors keep timing stable (noise compounds across an evolutionary run).

---

## 17. Expert-review hardening

A kernel-agent-systems review surfaced correctness/efficiency gaps; the
high-leverage, low-risk ones are implemented (the rest are scoped follow-ups).

### 17.1 Lineage integrity — tag ≡ verified patch (A1, done)

`save_and_test` produces an **incremental** `git diff` against the current HEAD,
and each step starts from `reset_worktree_to_best`, so a step's patch is
`best → new`. The old `_tag_git` committed the agent's *final dirty worktree*,
which can differ from the verified-best patch — so `avo-v{N}` could point at a
worse/abandoned attempt and the next step would resume from the wrong code.
Fixed: `seed_from_baseline(repo=...)` commits + tags the baseline as `avo-v0`,
and `_materialize_and_tag` reconstructs each commit by checking out the
parent-best, `git apply`-ing the **verified** incremental patch (the same diff
`evaluate_round_best` benchmarked), then committing + tagging. Result: `avo-v{N}`
≡ the verified patch. Verified by `tests/run/avo/test_lineage_git.py`.

### 17.2 Worktree hygiene (A2, done)

`reset_worktree_to_best` / `_checkout` now run `git clean -fd` after
`checkout -f` so untracked junk from an aborted step cannot leak into the next
step's diff. Legitimate new files are preserved because each commit captures them
via `git add -A`.

### 17.3 Per-shape regression guard (B2, done)

`evaluate_round_best` already computes per-shape verified speedups; the geomean
commit gate could still hide a regressed shape. `min_per_shape_speedup` (config,
default `0.0` = off) rejects a commit if any shape falls below the floor. Set
`0.95` to forbid commits that regress any shape >5%. Verified by
`test_per_shape_guard_*`.

### 17.4 Hardware grounding in every step (D1, done)

Because the agent is reset each step, GPU facts must be re-grounded per prompt.
`variation_step.hardware_summary` (cached) probes ROCm (`rocminfo`/`detect_gpu_arch`)
then NVIDIA (`nvidia-smi`) and injects a `## Target hardware` block into each
step, so tiling/occupancy/tensor-core decisions are arch-aware.

### 17.6 Multi-version lineage context (AVO §3.2 alignment, done)

The AVO paper notes the agent "frequently examines multiple prior implementations
in P_t within a single variation step, comparing their profiling characteristics."
Previously AVO injected only the single best (the exemplar) plus a text summary.
`build_lineage_context` now also injects the **top-K other committed versions**
(config `avo.lineage_context_k`, default 3; 0=off) — each with its verified
speedup, **per-shape** profile (worst-shape-first, to show where to improve), and
a truncated diff, with a pointer that full source is available via
`git show avo-v{id}`. Per-shape speedups are persisted on each `LineageNode`
(`score.per_shape`). Diffs are capped (1500 chars each) to keep the prompt bounded
(§15.1). This closes the largest remaining §3.2 gap; the residual difference is
that backtracking to an earlier P_t version is supervisor-driven, not
agent-autonomous mid-step (an autonomy choice, not a missing capability).

### 17.5 Scoped follow-ups

- **Best-of-K parallel generations (C1) — NOT implemented (deferred by choice).**
  The normal loop is single-lineage and sequential, underusing multi-GPU.
  Generalizing the ESCALATE "diversified workers + `evaluate_round_best` +
  `commit_from_round`" into the main loop (K parallel directions per generation,
  keep best) is the biggest throughput + exploration win, but changes the control
  flow — deferred for careful design.
- **Verified-result dedup cache (C2) — done.** `avo_state/verify_cache.json` keyed
  by patch content hash; `_apply_verified_score` reuses a cached verified speedup
  instead of re-benchmarking an identical diff.
- **Skip re-verification on no-gain (C3) — done.** `skip_verify_on_no_gain`
  (default on): when the agent's self-reported best is ≤ the commit floor (cannot
  commit anyway), the per-step FULL_BENCHMARK is skipped.
- **Trajectory observability — done.** `_write_trajectory` emits
  `trajectory.json` at finalize (running-best curve, commits, total attempts,
  commit rate, supervisor interventions, per-strategy stats).

### 17.7 Measurement significance — noise-robust progress signal (B1, done)

The detector's "did we progress?" input is only as objective as the benchmark is
stable; noise can drive false commits (resetting stall counters) or false stalls
(premature redirects). Two opt-in knobs make the verified-speedup signal
noise-robust:

- `verify_repeats` (default `1`): `_apply_verified_score` re-runs
  `evaluate_round_best` N times (each a fresh worktree + FULL_BENCHMARK of the
  *same* candidate → independent measurements) and uses the **median** speedup
  and per-shape **medians** (`_median_per_shape`), suppressing outlier spikes.
  Cost is N× benchmark, so default 1; set 3 on noisy harnesses.
- `commit_significance_margin` (default `0.0`): the commit gate requires
  `candidate ≥ best · (1 + margin)` (the stricter of this and the epsilon
  tolerance), i.e. a candidate must clear the current best by a noise floor
  rather than merely "match" it. Set e.g. `0.01` to require a real >1% gain.

Verified by `tests/run/avo/test_lineage_store.py::test_significance_margin_*`.
This raises the *objectivity* of the progress signal that feeds the
deterministic supervisor; harness-side warm-up/repeat discipline (noted in
`skills/avo-evolution`) remains complementary.

### 17.8 Supervision reliability upgrades (done)

Three changes raise the reliability/objectivity of the supervisor loop beyond the
B1 progress signal:

- **Trend-aware stall detection.** `StagnationDetector` tracks recent verified
  speedups; a non-committing step that sets a new intra-direction high (vs a
  `trend_window` of recent steps) is treated as "still climbing" and does **not**
  accrue the no-improvement stall — rescuing slow-but-real directions from
  premature redirect (false-stall). Verified by `test_trend_rescues_climbing_direction`.
- **Supervisor directive self-validation.** `run_supervisor._validate_directive`
  deterministically drops proposed directions already in `strategy_state`
  (failed/successful/exploring/skipped) or equal to the current one, de-dups, and
  splices a fallback-taxonomy direction if nothing novel remains — turning a
  subjective single-shot LLM proposal into a checked one (no extra model calls).
  Verified by `tests/run/avo/test_supervisor.py`.
- **Cost guards (C2/C3).** Dedup cache + skip-on-no-gain (see §17.5) cut wasted
  re-benchmarks so the budget buys more genuine exploration.

Residual (intentional): direction *generation* is still LLM-authored; multi-sample
self-consistency voting (needs sampling temperature) is left out as marginal.

### 17.9 Continuous causal memory across steps (P-mem-3, option C, done)

The paper's "persistent conversation history" cannot be carried raw across a
multi-day run (any window overflows), so AVO carries the **causal signal** of a
continuous session in a bounded form. After each step, `write_evolution_entry`
persists `avo_state/evolution_log/step_{N}.json` with the agent's **rationale**
(its last substantive assistant message), a short **verbatim raw tail** (last
assistant/tool turns, capped), the **verified speedup / committed flag**, and a
small dict of **profiler metrics** (`_read_profile_metrics`, cross-backend
best-effort). `build_evolution_log` then injects a bounded
`## Evolution log (causal history)` block each step: the most recent
`evolution_log_recent` steps verbatim, older steps collapsed to one-liners
(`strategy → speedup [committed/rejected] | bottleneck Δ | rationale/failure`),
capped at `evolution_log_max_versions`. This gives the agent the *why/what/effect*
chain across versions (the part of continuous memory that drives cumulative
micro-arch reasoning) while staying O(1) in prompt size (§15.1).

This is **option C** of the P-mem-3 proposal — not a literal persistent single
agent (option B), which conflicts with the per-step worktree reset and needs
core changes. Verified by `tests/run/avo/test_evolution_log.py`. The residual
gap vs option B (true uncompressed continuity) is accepted by design.

---

## 18. References

- AVO paper: *Agentic Variation Operators for Autonomous Evolutionary Search*,
  NVIDIA, arXiv:2603.24517.
- Kernel-Smith: *A Unified Recipe for Evolutionary Kernel Optimization*,
  Shanghai AI Lab + MetaX, arXiv:2603.28342.
- CuTeGen: *An LLM-Based Agentic Framework for Generation and Optimization of
  High-Performance GPU Kernels using CuTe*, U. Toronto, arXiv:2604.01489.
- FACT: *Compositional Kernel Synthesis with a Three-Stage Agentic Workflow*,
  Virginia Tech, arXiv:2604.26666.
- GEAK building blocks referenced above: `run/budget.py`, `run/unified.py`,
  `agents/optimization_agent.py`, `tools/strategy_manager.py`,
  `skills/skill_runtime.py`, `subagents/subagent_registry.py`,
  `run/postprocess/results.py`.
- Extension conventions: `subagents/README.md`, `skills/README.md`,
  [Contribution guidelines](contribution_guidelines.md).
