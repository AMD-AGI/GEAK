# GEAK Unification Plan

**Companion to**: `GEAK_codebase_audit.md` (the verified current-state baseline).

**Goal**: make the codebase **simpler, modular, easy to read, easy to maintain, easy to extend to a new kernel language**, without regressing any behavior observed in the audit.

**Core insight from the audit**: `StrategyInteractiveAgent` is ALREADY language-agnostic (same class, same config file, same prompts used for both Triton and HIP runs). The coupling lives in **three narrow places**: (a) mode routing at `mini.py:498-512`, (b) the duplicate ThreadPoolExecutor dispatch branch inside `ParallelAgent`, (c) the missing per-language `KernelLanguage` object that would hold language-specific content. Fixing those three is the unification.

---

## Table of contents
1. [Naming — make it readable for anyone new](#1-naming)
2. [Core concept — homo is hetero with a fixed prompt](#2-core-concept)
3. [One data type: `KernelLanguage`](#3-one-data-type-kernellanguage)
4. [One round loop: `run_pipeline`](#4-one-round-loop-run_pipeline)
5. [What gets deleted](#5-what-gets-deleted)
6. [What gets renamed](#6-what-gets-renamed)
7. [What gets added](#7-what-gets-added)
8. [Final directory tree](#8-final-directory-tree)
9. [Adding a new kernel language — step by step](#9-adding-a-new-kernel-language)
10. [Phased migration](#10-phased-migration)

---

## 1. Naming

Every name in the plan serves this test: *a new engineer opens the file, reads the name, and knows what the thing does without consulting a glossary.*

### 1.1 Before → After name table

| Today's name | Confusion | New name | Why |
|---|---|---|---|
| `kernel_type: str` | Overlaps with Python type system; "type" is vague | `language: str` (a `KernelLanguage.name`) | It's a programming language for kernels. Call it that. |
| `heterogeneous: bool` | What does True/False mean at a glance? | `mode: Literal["single", "multi", "auto"]` | `single`=one fixed prompt/N slots; `multi`=planner makes N prompts; `auto`=mix |
| `replicate` / `plan` flavors (from earlier drafts) | Internal jargon | `FixedPrompt` / `PlannedPrompt` | What they LITERALLY are: a fixed prompt vs a planner-produced prompt |
| `Mixture(replicate=x, plan=y)` | Same jargon | `PromptMix(fixed=x, planned=y)` | |
| `LanguagePlugin` (from earlier drafts) | "Plugin" is Java-ish overhead | `KernelLanguage` | It IS a kernel language. Name = thing. (User request.) |
| `StrategyInteractiveAgent` (4-level inheritance chain) | Long, hides purpose | `OptimizationAgent` | This class optimizes kernels; say so. |
| `ParallelAgent` | Misleading (not a sub-agent, it's a runner) | `ParallelRunner` | Truthful. |
| `run_homogeneous_agent` / `run_heterogeneous_orchestrator` | Two routines for the same thing | `run_pipeline(mode, language, ...)` | One function, mode is a parameter. |
| `AgentTask` | Why "agent" prefix? | `Task` | Short, sufficient. |
| `_normalize_kernel_type` / `_infer_kernel_type` / `_infer_kernel_language` | 3 functions for detection | `KernelLanguage.detect(path) -> float` | One method, one behavior. |
| `TASKGEN_SYSTEM_PROMPT` (Triton-biased, in prompts.py) | Language leak in a "language-agnostic" location | `PLANNER_SYSTEM_PROMPT` (generic) + `KernelLanguage.planner_strategy_hints` | Separate concerns. |
| `SYSTEM_PROMPT` (hetero orchestrator prompt, Triton-biased) | Same | `ORCHESTRATOR_SYSTEM_PROMPT` (generic) + `KernelLanguage.system_prompt` (per-language sub-agent role) | Two distinct system prompts; name each by its consumer. |

### 1.2 Mode → flavor mixture mapping (user-facing)

```yaml
mode: single    →  PromptMix(fixed=N, planned=0), max_rounds=1       # homo today
mode: multi     →  PromptMix(fixed=0, planned=N), max_rounds=K       # hetero today
mode: auto      →  controller picks the mix each round, max_rounds=K # new default
```

Reads as: *single-prompt × N, multi-prompt × N, or automatic.* No Greek vocabulary.

---

## 2. Core concept — homo is hetero with a fixed prompt

Two ways to produce N task bodies for a round. That's the only axis of variation. Everything else is shared.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   N parallel slots each run OptimizationAgent.run(body).   │
│                                                             │
│   Where does `body` come from?                              │
│                                                             │
│   (a) FixedPrompt flavor                                    │
│       body = compose(                                       │
│                KernelLanguage.system_prompt,                │
│                KernelLanguage.optimization_prompt,  ← FIXED │
│                starting_patch?,                             │
│                kernel_analysis?,                            │
│              )                                              │
│       All N slots get the SAME body.                        │
│       Diversity emerges from stochastic LLM sampling.       │
│       0 planner LLM calls.                                  │
│                                                             │
│   (b) PlannedPrompt flavor                                  │
│       planner_bodies = planner_llm(                         │
│                          KernelLanguage.planner_strategy_hints,│
│                          profile, history, KB,              │
│                          max_tasks=N,                       │
│                        )                                    │
│       body[i] = compose(                                    │
│                  KernelLanguage.system_prompt,              │
│                  planner_bodies[i],                         │
│                  starting_patch?,                           │
│                  kernel_analysis?,                          │
│                )                                            │
│       N slots get N DIFFERENT bodies.                       │
│       1 planner LLM call.                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Homogeneous mode = FixedPrompt × N, 1 round. Heterogeneous mode = PlannedPrompt × N, K rounds. Auto mode = controller splits between FixedPrompt and PlannedPrompt each round.

---

## 3. One data type: `KernelLanguage`

This is the only object that knows anything language-specific. Add a new language = create one `KernelLanguage` instance. Nothing else in the codebase branches on language.

```python
@dataclass(frozen=True)
class KernelLanguage:
    # ═══ Identity ═══
    name: str                            # "triton" | "hip" | "flydsl" | ...
    file_extensions: set[str]
    detection_priority: int = 100
    
    # How do I tell if a file is this language?
    detect: Callable[[Path], float]      # returns 0.0..1.0 confidence
    
    # ═══ The 3 prompt fragments (this is what actually differs between languages) ═══
    
    system_prompt: str
    """The agent's role + environment + tools contract + conventions.
    Used verbatim in EVERY task body (both FixedPrompt and PlannedPrompt).
    
    Example sections: Role ('You are a <language> optimizer for MI300X...'),
    Environment (build step? rocprofv3? harness flags?), Tools (bash, save_and_test...),
    Hard rules (never modify harness; always pass correctness), Workflow."""
    
    optimization_prompt: str
    """The DEFAULT task statement used by FixedPrompt flavor.
    Answers: what are we optimizing, what knobs does this language offer?
    
    Used verbatim by FixedPrompt flavor.
    Used as scaffolding by PlannedPrompt flavor (planner can reference it)."""
    
    planner_strategy_hints: str
    """Language-specific strategy taxonomy fed to the planner LLM when producing
    PlannedPrompt bodies. Example: Triton's 'autotune / split-K / warp-specialized /
    persistent-kernel', HIP's 'LDS tiling / MFMA intrinsics / __launch_bounds__'."""
    
    # ═══ Commands for preprocess + eval (flow into COMMANDMENT.md) ═══
    harness_template: str                # Jinja template for universal HarnessBuilder
    commandment_template: str            # Jinja template for COMMANDMENT.md
    test_runner_command: str             # "python3 {harness_path}" or "./{binary}"
    profiler_command: str                # "rocprofv3 --kernel-trace --"
    patch_apply_strategy: str = "git_3way"
    
    # ═══ Runtime environment ═══
    eval_env: Callable[[Path], dict[str, str]]  # PYTHONPATH, LD_LIBRARY_PATH, HIPCC flags
    
    # ═══ Knowledge ═══
    kb_namespace: str                    # filter tag for KB + RAG
    
    # ═══ Optional controller hint ═══
    prefer_single_to_multi_ratio: float | None = None  # sparse-KB languages tilt toward planned
```

That's **one dataclass with 13 fields**. Triton, HIP, FlyDSL, and future languages are instances of this dataclass. Every language-specific decision in the codebase is a field access on this object.

---

## 4. One round loop: `run_pipeline`

One function. No `if heterogeneous:` split. No separate CLI entry. `mode` is a user-facing parameter that controls what `PromptMix` the controller picks.

```python
def run_pipeline(
    ctx: PreprocessContext,
    mode: Literal["single", "multi", "auto"] = "auto",
    max_rounds: int = 5,
    *,
    num_parallel: int,
    gpu_ids: list[int],
    controller_config: ControllerConfig = DEFAULT_CONTROLLER,
) -> FinalReport:
    """The one and only round loop. Called from mini.py for every run.
    
    mode='single' → one fixed prompt × N slots, 1 round (was: homogeneous mode)
    mode='multi'  → planner prompt × N slots, K rounds (was: heterogeneous mode)
    mode='auto'   → controller picks a mix of single + multi each round (new default)
    """
    history = RoundHistory()
    best_global = None
    
    for round_num in range(1, max_rounds + 1):
        # 1. Controller picks mixture (trivial for single/multi modes, smart for auto)
        mix = pick_prompt_mix(mode, history, ctx, controller_config, N=num_parallel)
        # mix example: PromptMix(fixed=1, planned=3)
        
        # 2. Build tasks (the ONE function where language matters)
        tasks = generate_tasks(
            mix=mix,
            language=ctx.language,                       # KernelLanguage
            kernel_path=ctx.kernel_path,
            kernel_code=ctx.kernel_code,
            round_num=round_num,
            history=history,
            starting_patch=best_global.patch if best_global else None,
            kernel_analysis=ctx.kernel_analysis,
        )
        
        # 3. Dispatch (fully language-agnostic)
        results = run_pool(
            tasks,
            gpu_ids=gpu_ids,
            env_factory=lambda: make_env(ctx.language.eval_env(ctx.kernel_path)),
            per_task_timeout=ctx.per_task_timeout,
        )
        
        # 4. Evaluate (universal COMMANDMENT runner)
        round_eval = evaluate_round(results, ctx.commandment_path, ctx.language.eval_env)
        
        # 5. Sticky starting_patch (global-max, never demote)
        if round_eval.best_patch and (
            best_global is None or
            round_eval.verified_speedup > best_global.verified_speedup
        ):
            best_global = round_eval
        
        # 6. KB write (with cross-validation + baseline fingerprint)
        if round_eval.verified_speedup >= config.min_store_speedup:
            if cross_validate(round_eval.best_patch, ctx):
                record_experience(round_eval, ctx)
        
        history.append(round_eval)
        
        # 7. Controller decide
        if controller_config.decide(round_eval, history) != CONTINUE:
            break
    
    return build_final_report(history, best_global)
```

### 4.1 `generate_tasks` — the only language-aware function

```python
def generate_tasks(
    *,
    mix: PromptMix,                        # {fixed: int, planned: int}
    language: KernelLanguage,              # ← ONE argument that carries all language context
    kernel_path: Path,
    kernel_code: str,
    round_num: int,
    history: RoundHistory,
    starting_patch: Path | None,
    kernel_analysis: str | None,
) -> list[Task]:
    tasks: list[Task] = []
    
    # FixedPrompt branch — 0 LLM calls
    if mix.fixed > 0:
        body = _compose(
            system_prompt=language.system_prompt,
            task_statement=language.optimization_prompt,
            starting_patch=starting_patch,
            kernel_analysis=kernel_analysis,
        )
        for i in range(mix.fixed):
            tasks.append(Task(
                label=f"fixed_r{round_num}_s{i}",
                body=body,
                flavor="fixed",
                kernel_code=kernel_code,
                kernel_path=kernel_path,
                language=language.name,
                round_num=round_num,
            ))
    
    # PlannedPrompt branch — 1 LLM call produces N distinct bodies
    if mix.planned > 0:
        planner_outputs = _run_planner_llm(
            strategy_hints=language.planner_strategy_hints,
            profile=ctx.profile_json,
            history=history,
            kb_hits=kb_retriever(...),
            max_tasks=mix.planned,
        )
        for label, strategy_body in planner_outputs:
            body = _compose(
                system_prompt=language.system_prompt,
                task_statement=strategy_body,  # planner's output replaces optimization_prompt
                starting_patch=starting_patch,
                kernel_analysis=kernel_analysis,
            )
            tasks.append(Task(
                label=label,
                body=body,
                flavor="planned",
                kernel_code=kernel_code,
                kernel_path=kernel_path,
                language=language.name,
                round_num=round_num,
            ))
    
    return tasks


def _compose(
    *,
    system_prompt: str,              # KernelLanguage.system_prompt
    task_statement: str,             # language.optimization_prompt OR planner output
    starting_patch: Path | None = None,
    kernel_analysis: str | None = None,
) -> str:
    parts = [system_prompt, "", task_statement]
    if starting_patch:
        parts += ["", "## Starting patch (sticky global-max, build on this)",
                  f"```diff\n{starting_patch.read_text()}\n```"]
    if kernel_analysis:
        parts += ["", "## Kernel analysis (primary evidence)", kernel_analysis]
    return "\n".join(parts)
```

This is the **entire language-aware surface area** of the runtime: `generate_tasks` and `_compose`. Adding a new language doesn't require editing either.

---

## 5. What gets deleted

| File / section | LoC | Why | Risk |
|---|---|---|---|
| `agents/homogeneous/homogeneous_agent.py` | 196 | Replaced by `run_pipeline(mode="single")` | low (thin wrapper) |
| `agents/homogeneous/__init__.py` + dir | — | Empty dir goes away | none |
| `agents/parallel_agent.py` homogeneous branch (lines 414-609) | 196 | Duplicate of `run/utils/parallel_helpers.run_pool` | low (behavior-equivalent once callers switched) |
| `agents/heterogeneous/orchestrator.py` | 493 | Logic moves to `run/unified.py::run_pipeline` (smaller due to no LLM-driven tool loop) | medium (careful port) |
| `agents/heterogeneous/prompts.py` language-specific content (~260 of 379 LoC) | 260 | Moves to per-language `system_prompt.md` files; ~120 LoC of generic framing stays as `ORCHESTRATOR_SYSTEM_PROMPT` | low |
| `agents/heterogeneous/tools.py` | 410 | Tools (generate_tasks, dispatch_tasks, collect_results, finalize) move to `tools/` top-level; the tool-calling pattern itself is replaced by direct `run_pipeline` calls | medium |
| `agents/heterogeneous/workload_guidance.py` | 307 | `_HIP_SEARCH_HINT_PATTERNS` etc. replaced by per-language `planner_strategy_hints` field | low |
| `agents/heterogeneous/result_scanning.py` | 210 | Only ~40 LoC actually used; fold into task_generator | low |
| `agents/heterogeneous/schemas.py` | 96 | Tool-call schemas → `tools/orchestrator_schemas.py` | low |
| `agents/heterogeneous/__init__.py` + dir | — | Empty dir goes away | none |
| `agents/interactive.py` | 181 | Merged into `default.py` via `mode: str` config | low |
| `agents/strategy_agent.py` | 156 | Merged into `optimization_agent.py` via `use_strategy_manager: bool` config | low |
| `agents/interactive_textual.py` | 450 | Moved to `dev_tools/` (not in prod build) | none |
| `agents/shape_fixer_agent.py` | 7 | Stub; import real one directly | none |
| `agents/unit_test_agent.py` | 251 | Duplicate of `run/preprocess/unit_test_agent.py` | low |
| `config/mini_unit_test_agent.yaml` (8KB) | — | Duplicate of `run/preprocess/config/mini_unit_test_agent.yaml` | none |
| `config/mini_reverse_kl.yaml` (15KB) | — | Unreferenced; verify with grep first | none (after grep-confirm) |
| `tools/validate_commandment.py` (7 LoC stub) | 7 | Duplicate of `run/preprocess/validate_commandment.py` | none |
| `run/orchestrator.py` | 288 | Redundant CLI entry; merge into `mini.py` | medium |
| `_normalize_kernel_type` (`mini.py:61-67`) | 7 | `KernelLanguage.detect` replaces all detection | low |
| `_infer_kernel_type` (`task_generator.py:74-105`) | 32 | Same | low |
| `_infer_kernel_language` (`discovery_types.py:247-253`) | 7 | Same | low |
| `mini.py:498-512` mode auto-detect branch | 15 | Replaced by `mode="auto"` controller | low |
| `mini.py:514-560` heterogeneous branch + `mini.py:562-630` homogeneous branch | ~130 | Both replaced by single `run_pipeline(mode=..., language=...)` call | medium (merged replacement) |

**Total deletion**: ~3,300 LoC removed, 13 files gone, 2 directories removed, 3 duplicate helpers collapsed.

---

## 6. What gets renamed

| Old name | New name |
|---|---|
| `StrategyInteractiveAgent` | `OptimizationAgent` |
| `ParallelAgent` | `ParallelRunner` (and it's not an Agent subclass anymore; it's a dispatcher) |
| `AgentTask` | `Task` |
| `kernel_type: str` (everywhere) | `language: str` or `KernelLanguage.name` |
| `heterogeneous: bool` (arg) | `mode: Literal["single","multi","auto"]` |
| `run_homogeneous_agent` | (deleted; shim during Phase 2) |
| `run_heterogeneous_orchestrator` | (deleted; shim during Phase 2) |
| `LanguagePlugin` (from earlier design drafts) | `KernelLanguage` (user-facing simpler name) |
| `heterogeneous/` directory | (flattened into `run/unified.py` + top-level `task_generator.py`) |
| `homogeneous/` directory | (deleted) |
| `parallel_i/` output subdir (homo's naming) | `<flavor>_r<N>_s<i>/` (uniform with hetero) |

---

## 7. What gets added

| New file/module | LoC | Purpose |
|---|---|---|
| `plugins/base.py` | 150 | `KernelLanguage` dataclass, `Task`, `PromptMix`, `RoundHistory`, `PreprocessContext` (with `__post_init__` absolute-path invariant), registry |
| `plugins/languages/triton/plugin.py` | 40 | Triton `KernelLanguage` instance |
| `plugins/languages/triton/system_prompt.md` | ~60 lines | Triton role + env + tools |
| `plugins/languages/triton/optimization_prompt.md` | ~40 lines | Triton default "optimize this" + knob list |
| `plugins/languages/triton/planner_strategy_hints.md` | ~30 lines | Triton-specific strategy taxonomy for the planner LLM |
| `plugins/languages/triton/commandment.j2` | ~40 lines | Jinja template for COMMANDMENT.md (Triton SETUP block) |
| `plugins/languages/triton/harness.j2` | ~30 lines | Jinja template for universal harness (Triton scaffolding) |
| `plugins/languages/hip/*` | same 6 files | HIP equivalents |
| `plugins/languages/flydsl/*` | same 6 files | FlyDSL equivalents |
| `run/unified.py::run_pipeline` | 180 | THE round loop |
| `run/task_generator.py::generate_tasks` | 200 | Slim from today's 997 LoC; two clear branches (Fixed / Planned) |
| `run/compose.py::_compose` | 30 | THE single place where `KernelLanguage` fields become task body text |
| `plugins/controllers/auto_mix.py` | 80 | The "auto" mode mixture picker (adaptive_ratio_v1) |
| `plugins/preprocess/harness_builder.py` | 120 | Universal `HarnessBuilder` agent (one for all languages) |
| `plugins/preprocess/kernel_analysis.py` | 80 | Generates `kernel_analysis.md` via [A]-[D] rubric |
| `scripts/check_function_length.py` | 40 | CI gate (≤ 80 LoC per function, fail on violations) |
| `scripts/check_inline_imports.py` | 30 | CI gate (no inline imports) |
| `scripts/check_language_leaks.py` | 40 | CI gate: grep `kernel_type ==` / `if triton` etc. in core code (not plugins/) → fail |

**Total new code**: ~1,300 LoC of focused new code. Net: ~-2,000 LoC removed, 6 agent classes → 1 (OptimizationAgent) + 1 dispatcher (ParallelRunner), 16+ language-branching sites → 1 per language.

---

## 8. Final directory tree

```
src/minisweagent/
├── agents/
│   ├── __init__.py
│   ├── agent_spec.py                   # Task dataclass (with invariants)
│   ├── default.py                      # DefaultAgent (absorbs InteractiveAgent via mode)
│   ├── optimization_agent.py           # was StrategyInteractiveAgent; THE kernel optimizer
│   ├── select_patch_agent.py           # unchanged
│   └── parallel_runner.py              # was parallel_agent; dispatcher (homo branch deleted)
│
├── plugins/                            # NEW
│   ├── base.py                         # KernelLanguage, PromptMix, Registry
│   ├── languages/
│   │   ├── triton/                     # 1 plugin.py + 5 small templates
│   │   ├── hip/
│   │   └── flydsl/
│   ├── preprocess/
│   │   ├── harness_builder.py          # universal HarnessBuilder agent
│   │   ├── kernel_analysis.py          # [A]-[D] rubric producer
│   │   └── unit_test_agent.py          # consolidated UTA (was in 2 places)
│   ├── retrievers/                     # KB / RAG / skill / codebase (moved from memory/cross_session + rag_postprocessor)
│   ├── evaluators/                     # commandment / cross_validator / regression_gate
│   ├── controllers/
│   │   └── auto_mix.py                 # "auto" mode mixture picker
│   └── reporters/                      # file_reporter, kb_reporter
│
├── run/
│   ├── mini.py                         # CLI (thin; ~350 LoC after shedding homo/hetero branches)
│   ├── unified.py                      # run_pipeline() — THE round loop
│   ├── task_generator.py               # generate_tasks (2 branches: Fixed / Planned)
│   ├── compose.py                      # _compose — the ONE language-aware function
│   ├── dispatch.py                     # task_file_to_agent_task (cleanup only)
│   ├── task_file.py                    # unchanged (path-normalization fix already on main)
│   ├── model_factory.py                # split out of pipeline_helpers
│   ├── env_factory.py                  # split out of pipeline_helpers
│   ├── preprocess/                     # universal preprocessor (commandment for ALL languages)
│   └── postprocess/                    # unchanged shape
│
├── tools/                              # native tools (save_and_test, bash, strategy_manager, ...)
├── memory/                             # cross_session KB (unchanged semantically; baseline_fingerprint added)
├── models/                             # unchanged
├── environments/                       # unchanged
└── config/
    ├── geak.yaml                       # slim; references KernelLanguage by name
    └── mini.yaml                       # slim; agent defaults only
```

**What's gone**:
- `agents/homogeneous/` — empty directory, deleted
- `agents/heterogeneous/` — empty directory, deleted
- `agents/interactive.py`, `agents/strategy_agent.py`, `agents/interactive_textual.py` — merged or moved
- `run/orchestrator.py` — merged into `mini.py`
- Duplicate `agents/unit_test_agent.py` (kept the `plugins/preprocess/` version)
- Duplicate `config/mini_unit_test_agent.yaml` (kept the `run/preprocess/config/` version)

---

## 9. Adding a new kernel language

A complete recipe. Runs after Phase 2 ships. No edits to any file outside `plugins/languages/<x>/`.

### Step 1 — Create the directory

```
mkdir -p src/minisweagent/plugins/languages/X/
cd src/minisweagent/plugins/languages/X/
```

### Step 2 — Write `X/plugin.py`

```python
from pathlib import Path
from minisweagent.plugins.base import KernelLanguage

def _x_confidence(path: Path) -> float:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return 0.0
    if "<MARKER_FOR_X>" in text or path.suffix in {".x", ".xl"}:
        return 0.9
    return 0.0

XKernelLanguage = KernelLanguage(
    name="x",
    file_extensions={".x", ".xl"},
    detection_priority=100,
    detect=_x_confidence,
    
    # 3 prompt files in this directory
    system_prompt           = (Path(__file__).parent / "system_prompt.md").read_text(),
    optimization_prompt     = (Path(__file__).parent / "optimization_prompt.md").read_text(),
    planner_strategy_hints  = (Path(__file__).parent / "planner_strategy_hints.md").read_text(),
    
    # 2 Jinja templates
    harness_template        = (Path(__file__).parent / "harness.j2").read_text(),
    commandment_template    = (Path(__file__).parent / "commandment.j2").read_text(),
    
    test_runner_command     = "python3 {harness_path}",          # or "./{binary}"
    profiler_command        = "rocprofv3 --kernel-trace --",
    patch_apply_strategy    = "git_3way",
    
    eval_env = lambda path: {
        # Any env vars your language runtime needs:
        # "PYTHONPATH": "...",
        # "LD_LIBRARY_PATH": "...",
    },
    
    kb_namespace = "x",
)
```

### Step 3 — Write `X/system_prompt.md`

```markdown
# Role
You are a <Language X> kernel optimizer for AMD GPUs.

# Environment
- Tests run via: <how tests run>
- Profiler: rocprofv3 --kernel-trace
- Harness exposes: --correctness, --benchmark, --full-benchmark
- Report speedups via "GEAK_RESULT_SPEEDUP=<float>"

# Tools
bash, str_replace_editor, save_and_test, submit, strategy_manager,
profile_kernel, query, optimize.

# Hard rules
- Never modify the test harness
- Always pass correctness BEFORE claiming a speedup
- Patches apply via git 3-way fallback

# Workflow
1. Read kernel, COMMANDMENT.md, baseline_metrics.json, profile.json
2. For each idea: edit → save_and_test → check → decide
3. Use strategy_manager to track what you've tried
4. submit when verified speedup is maximized
```

### Step 4 — Write `X/optimization_prompt.md`

```markdown
# Task
Optimize this <Language X> kernel. Preserve correctness; improve FULL_BENCHMARK speedup.

# Optimization knobs specific to <Language X>
- <knob 1 with brief description>
- <knob 2>
- <knob 3>
- <algorithm patterns common in X>

# What NOT to touch
- Function signatures the harness calls
- Build flags the test harness depends on
- Correctness tolerances

# Evaluation
Speedup = baseline_latency_ms / candidate_latency_ms on FULL_BENCHMARK.
Must be >= 1.10x with correctness PASS to be recorded to KB.
```

### Step 5 — Write `X/planner_strategy_hints.md`

```markdown
# Strategy taxonomy for <Language X>
When proposing diverse optimization strategies for this language, consider:
- <strategy family 1> (when applicable: <condition>)
- <strategy family 2>
- <strategy family 3>
- Language-specific intrinsics: <list>
- Memory-hierarchy patterns: <list>
- Algorithm rewrites: <list>
```

### Step 6 — Write `X/commandment.j2`

Jinja template with placeholders `{{ eval_env_exports }}`, `{{ test_runner_command }}`, `{{ profiler_command }}`. See `plugins/languages/triton/commandment.j2` for reference. 40-ish lines.

### Step 7 — Write `X/harness.j2`

Jinja template providing the skeleton the `HarnessBuilder` fills in. Must produce a harness that exposes `--correctness`, `--benchmark`, `--full-benchmark`, `--profile`. 30-ish lines.

### Step 8 — Register (only if not using entry points)

```python
# src/minisweagent/plugins/languages/__init__.py
from .x.plugin import XKernelLanguage
```

Or, in `pyproject.toml`:

```toml
[project.entry-points."geak.languages"]
x = "minisweagent.plugins.languages.x.plugin:XKernelLanguage"
```

### Step 9 — Smoke test

```bash
geak "optimize this X kernel" --kernel-url path/to/example.x
```

Expected in log:
```
Language detected: x (confidence=0.9)
--- Step 7/7: Commandment ---
  COMMANDMENT.md generated (from KernelLanguage.commandment_template)
Running Pipeline (mode=auto, max_rounds=5)
Round 1: PromptMix(fixed=2, planned=2)
Sub-agent (fixed_r1_s0) started on GPU 0
Sub-agent (fixed_r1_s1) started on GPU 1
Sub-agent (planned_r1_s0, label=<planner-label>) started on GPU 2
Sub-agent (planned_r1_s1, label=<planner-label>) started on GPU 3
...
```

**Total new files for a typical language**: 6 (plugin.py + 5 content files). **Total edits to core code**: 0.

---

## 10. Phased migration

Everything threaded so no single PR lands a big-bang rewrite. Each phase is independently releasable.

### Phase 0 — Quick wins (week 1; 7 small PRs)

| # | Change | Evidence anchor |
|---|---|---|
| 1 | Add `GEAK_TASK_TIMEOUT` to homo ThreadPool | slot_a stuck compile 90+ min |
| 2 | Add `consecutive_no_improvement` early stop to hetero orchestrator | slot_b R3-R5 plateau |
| 3 | Route FlyDSL to hetero (1-line change) | FlyDSL never gets verified speedup |
| 4 | Resolve PR #153/#155 `--target-language` conflict | two PRs define same flag differently |
| 5 | CI smoke tests for 3 kernels | regression prevention |
| 6 | `task_file.py` absolute-path normalization | **already on main (8578d62b)** |
| 7 | Readability CI scaffolding (warning mode): function length, inline imports, language leaks | setup for Phase 1+ |

### Phase 1 — `KernelLanguage` foundation (weeks 2-3)

- Add `plugins/base.py` with `KernelLanguage` dataclass + `PromptMix` + `Task`
- Create 3 language instances: `TritonKernelLanguage`, `HipKernelLanguage`, `FlyDSLKernelLanguage` + their template files
- Replace `_normalize_kernel_type` + `_infer_kernel_type` + `_infer_kernel_language` with `KernelLanguage.detect`
- **No behavior change yet**. Plugin system coexists with old code.

Acceptance: `grep -r "_normalize_kernel_type\|_infer_kernel_type" src/` returns 0 production matches; detection uses `KernelLanguage.detect` exclusively.

### Phase 2 — `run_pipeline` + delete duplicates (weeks 4-7)

- Ship `run/unified.py::run_pipeline` behind `GEAK_UNIFIED_PIPELINE=1` flag
- Ship `run/task_generator.py::generate_tasks` (slim; 2 branches)
- Ship `run/compose.py::_compose`
- Write shims: `run_homogeneous_agent` → `run_pipeline(mode="single")`; `run_heterogeneous_orchestrator` → `run_pipeline(mode="multi")`
- **Delete**: `ParallelAgent.run_parallel` homogeneous branch (~200 LoC); `agents/heterogeneous/workload_guidance.py`; `agents/heterogeneous/result_scanning.py`; `agents/interactive_textual.py`; `agents/shape_fixer_agent.py`
- **Delete duplicates**: `agents/unit_test_agent.py`; `config/mini_unit_test_agent.yaml`; `tools/validate_commandment.py` stub
- Merge `InteractiveAgent` → `DefaultAgent` (via `mode: str` config); `StrategyAgent` → `OptimizationAgent` (via `use_strategy_manager: bool` config)
- Rename `StrategyInteractiveAgent` → `OptimizationAgent`; `ParallelAgent` → `ParallelRunner`

Acceptance:
- Byte-identical `round_N_evaluation.json` for 5 Triton kernels (mode="multi") vs today's hetero path
- Functionally equivalent result on 3 HIP kernels (mode="single") PLUS NEW: `round_1_evaluation.json` is now produced (homo today doesn't)
- `grep "if heterogeneous" src/` returns 0 matches
- `grep "if kernel_type ==" src/` returns 0 matches in core code (only in plugin detection functions)
- Sub-agent KB injection never produces `target_code=0B` on any test kernel

### Phase 3 — auto mode default (weeks 8-9)

- Add `plugins/controllers/auto_mix.py` (the adaptive_ratio_v1 controller)
- Flip default from `GEAK_UNIFIED_PIPELINE=0` → `=1`
- Default `mode="auto"`
- Add sticky `starting_patch` (global-max)
- Add `STOP_NO_VALID_CANDIDATE` controller decision (for both-arms-stuck cases like fused_mxfp4_quant_moe_sort)

Acceptance: on a 10-kernel benchmark, auto mode produces ≥ same verified speedup as single-mode or multi-mode for every kernel, while saving ≥ 15% wall-clock via early stop.

### Phase 3.5 — KB stale-entry validation (week 10)

- `scripts/kb_validate.py` periodic job flags stale entries
- Retriever filters `validation_status=stale` by default

### Phase 4 — NL-driven preprocess (weeks 11-13)

- Opt-in `PreprocessAgent` for NL intent parsing
- Deprecate `--kernel-url`, `--target-language`, `--heterogeneous` flags

### Phase 5 — Remove old code (weeks 14-15)

Delete remaining shims:
- `run_homogeneous_agent` and its shim
- `run_heterogeneous_orchestrator` shim
- `run/orchestrator.py` (redundant CLI)
- `_normalize_kernel_type` (dead after Phase 1)
- `GEAK_UNIFIED_PIPELINE=0` fallback
- `config/mini_reverse_kl.yaml` (if grep-confirmed unused)

Acceptance:
- Total LoC: ≤ 22,000 (from today's ~28,000)
- 0 functions > 120 LoC
- 0 `if kernel_type ==` or `if heterogeneous` branches in core code
- Tagged `pre-phase-5-removal` release for historical reproducibility

---

## 10.5 Determinism & seeding contract

Reproducible correctness tests and bit-stable benchmark latencies require explicit seeding at the start of every mode. The existing GEAK Triton harnesses (e.g. `gemm_a16w16_atomic/test_kernel_harness.py`) set `torch.manual_seed(42)` at the top of `run_correctness`, `run_benchmark`, and `run_profile`. The refactored `harness.j2` templates codify this + add cudnn determinism + multi-RNG seeding.

### 10.5.1 Universal contract (added to `plugins/base.py`)

```python
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    """MUST be called at the top of every harness mode (correctness, benchmark,
    full-benchmark, profile) to guarantee bit-reproducible tests and latencies."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 10.5.2 Harness template requirements (all languages)

Every `harness.j2` MUST call `_seed_everything(42)` (an inline copy of `seed_everything`, since the harness is a standalone file) at the start of:
- `run_correctness(shapes)`
- `run_benchmark(shapes, iters, warmup)`
- `run_full_benchmark(shapes, iters, warmup)` (if distinct from benchmark)
- `run_profile(shapes)`

Per-shape seeding uses `seed=42 + shape_index` so different shapes get different-but-reproducible inputs.

### 10.5.3 Wrapping harnesses (HIP pattern)

When `harness.j2` is a wrapper around an existing project test file (e.g. HIP's `task_runner.py`), the seeding must happen INSIDE the delegated file. The `builder_hints.md` instructs the HarnessBuilder LLM to either:
1. Patch the delegated file to add seeding at mode-start, OR
2. Refuse to produce a wrapper if the delegated file's determinism can't be guaranteed

### 10.5.4 Contract validator enforces determinism

```python
def validate_harness(harness_path: Path) -> list[str]:
    errors = [...]
    # Run --benchmark twice; latencies must match within 2% (small jitter allowed
    # for thermal state, scheduling — anything larger indicates broken seeding).
    def _run_bench() -> float:
        r = subprocess.run([sys.executable, str(harness_path), "--benchmark"],
                           capture_output=True, text=True, timeout=180)
        m = OUTPUT_PROTOCOL_REGEX["latency_ms"].search(r.stdout)
        if not m:
            raise RuntimeError("--benchmark did not emit GEAK_RESULT_LATENCY_MS")
        return float(m.group(1))
    lat1 = _run_bench(); lat2 = _run_bench()
    drift = abs(lat1 - lat2) / min(lat1, lat2)
    if drift > 0.02:
        errors.append(f"Latency varies {drift*100:.1f}% across runs "
                      f"({lat1:.4f} vs {lat2:.4f}); seeding is broken")
    return errors
```

A harness that fails this determinism check is rejected at preprocess time, before optimization starts. The `baseline_fingerprint` captured by preprocess is meaningful only when the harness is deterministic.

---

## 10.6 Four-path compatibility matrix (refactor regression prevention)

`mode × language` = 4 combinations. Today's code enables 2 (auto-selected) and allows 2 more (user-forced via `--heterogeneous`). The refactor must work correctly for all 4. Reference baselines grounded in observed logs:

### Path 1: Triton + hetero (→ `mode=planned`, language=Triton)
**Today**: default for Triton. Works well. Hetero orchestrator, 5-round loop, LLM task_generator, run_pool, post_round_evaluate, KB write. `COMMANDMENT.md` generated and correctly enforced.
**Evidence**: `gemm_a16w16_atomic_canonical-rocm700_memon_20260422_083415.log` live run @ 08:51.
**After refactor**: `run_pipeline(mode="planned", language=TritonKernelLanguage, max_rounds=5)`. Same behavior + baseline_fingerprint + sticky starting_patch + STOP_NO_VALID_CANDIDATE safety.
**Regression risk**: LOW — this is the most-exercised path.

### Path 2: HIP + homo (→ `mode=fixed`, language=HIP)
**Today**: default for non-Triton. Preprocess runs steps 1, 5, 6, 7 (2-4 skipped). Homo runner dispatches N identical agents via ThreadPoolExecutor. `COMMANDMENT.md` generated but has **3 bugs** (wrong profile target, nested unescaped quotes, BENCHMARK == FULL_BENCHMARK identical); never actually enforced — runner falls back to `--task` test_command. `task_runner.py performance` emits `"Performance: 0.1234 ms (shape_N)"` which is not the GEAK protocol; no speedup number exists. No per-round verification. No KB write.
**Evidence**: `hip_ab_v3/assign_score_withk_mem_20260415_180639.log` + `hip2hip_others_three_nn_20260414_230906_logs/COMMANDMENT.md`.
**After refactor**: `run_pipeline(mode="fixed", language=HipKernelLanguage, max_rounds=1)`. REQUIRES:
- `HipKernelLanguage.harness.j2` producing a Python wrapper around `task_runner.py` that emits `GEAK_RESULT_LATENCY_MS=<float>` and `GEAK_RESULT_SPEEDUP=<float>` (with speedup computed against `BASELINE_LATENCY_MS` captured at preprocess)
- `HipKernelLanguage.commandment.j2` with corrected quoting, correct profile target, distinct BENCHMARK vs FULL_BENCHMARK
- Preprocess captures `baseline_latency_ms` by running baseline `task_runner.py performance`
- `run_pipeline` enforces commandment via `evaluate_round_best` on every round
- `record_experience` called per qualifying round
**Regression risk**: MEDIUM — needs the HarnessBuilder + corrected commandment to produce the right output. Contract validators catch bugs at preprocess time.

### Path 3: Triton + homo (→ `mode=fixed`, language=Triton) [rare]
**Today**: requires `--heterogeneous=False` for a Triton kernel. Rarely used. Preprocess completes all 7 steps; Triton `COMMANDMENT.md` is well-formed but not enforced by homo runner. Loses the LLM planner (the hetero path's strength). N parallel identical Triton agents; SelectPatchAgent at end; no KB write.
**Evidence**: no live log captured (rare path). Extrapolated from code.
**After refactor**: `run_pipeline(mode="fixed", language=TritonKernelLanguage, max_rounds=1)`. Triton `commandment.j2` is already well-formed; KB write + FULL_BENCHMARK verification come "for free" from unification. Still gets the language-specific `system_prompt` and `optimization_prompt` (today gets neither — uses generic `mini_kernel_strategy_list.yaml`).
**Regression risk**: LOW — upgrades a rare path without changing the default-used path.

### Path 4: HIP + hetero (→ `mode=planned`, language=HIP) [BROKEN today]
**Today**: requires `--heterogeneous=True` for HIP kernel. Orchestrator uses Triton-biased `SYSTEM_PROMPT` + `TASKGEN_SYSTEM_PROMPT` (mentions `@triton.jit`, `tl.reshape`, `BLOCK_S`, `num_warps` throughout). Planner emits Triton-flavored task bodies for HIP code — **language mismatch**. `post_round_evaluate` tries to run HIP `COMMANDMENT.md` — fails due to the 3 bugs. Verification errors, rounds produce no verified speedups, budget wasted.
**Evidence**: not observed in logs (users don't force this combination because it doesn't work).
**After refactor**: `run_pipeline(mode="planned", language=HipKernelLanguage, max_rounds=5)`. REQUIRES ALL OF:
- `HipKernelLanguage.system_prompt` (HIP agent role, replaces today's Triton-biased SYSTEM_PROMPT for HIP runs)
- `HipKernelLanguage.planner_strategy_hints` (HIP strategy taxonomy replaces Triton-centric TASKGEN_SYSTEM_PROMPT)
- `HipKernelLanguage.harness.j2` (same as Path 2 — produces protocol-emitting wrapper)
- `HipKernelLanguage.commandment.j2` (same as Path 2 — corrected commandment)
- Everything else Path 2 needs
**Regression risk**: LOW in absolute terms (the current path doesn't work; refactor only makes it work). But this is the biggest improvement: unlocks HIP for planned mode for the first time.

### Summary

| Combination | Today | After refactor | Requires |
|---|---|---|---|
| Triton + planned (was hetero) | works | works + extras | nothing new |
| HIP + fixed (was homo) | partial (no verification/KB/speedup) | works fully | HIP harness.j2 + commandment.j2 + BASELINE_LATENCY_MS |
| Triton + fixed (was homo, rare) | partial | works fully | unification's universal verification + KB write |
| HIP + planned (was hetero, broken) | BROKEN | works | HIP system_prompt + planner_strategy_hints + Path 2 deliverables |
| any + auto (new) | doesn't exist | works | controller + adaptive_ratio_v1 |

The refactor upgrades all 4 language×mode combinations AND adds `mode=auto`. Every combination is reachable via `geak --mode <fixed|planned|auto> --language <name>` (or auto-detected). No combination regresses.

---

## 11. Summary — before/after in numbers

| Metric | Today | After Phase 5 |
|---|---|---|
| Total LoC in `src/minisweagent/` | ~28,000 | ~22,000 (-20%) |
| Production agent classes | 8 (DefaultAgent, InteractiveAgent, StrategyAgent, StrategyInteractiveAgent, ParallelAgent, UnitTestAgent×2, SelectPatchAgent, _TextualAgent) | 4 (DefaultAgent, OptimizationAgent, UnitTestAgent, SelectPatchAgent) + 1 runner (ParallelRunner) |
| CLI entry points | 2 (`geak`, `geak-orchestrate`) | 1 (`geak`) |
| Language-branching sites | 16+ | 1 per language (in `plugins/languages/<x>/`) |
| Files to edit to add a language | 9 (PR #155 evidence) | 0 |
| Files to create to add a language | 0 + changes to 9 | 6 (1 plugin.py + 5 template files) |
| Duplicate modules | 4 pairs (`unit_test_agent` x2, `validate_commandment` x2, `discovery_types` x2, `benchmark_parsing` x2) | 0 |
| Dispatch code paths | 2 (ThreadPoolExecutor in homo, run_pool in hetero) | 1 (run_pool only) |
| Final-report schemas | 2 (4-key homo, 16+-key hetero) | 1 (unified, optional fields) |
| KB write paths | Hetero only | Universal (any mode, any language) |
| Per-round FULL_BENCHMARK verification | Hetero only | Universal |
| Function > 120 LoC | 12 | 0 |
| `if kernel_type ==` in core code | ~35 occurrences | 0 |

---

**End of unification plan.** See `GEAK_codebase_audit.md` for the verified baseline this plan refactors from.
