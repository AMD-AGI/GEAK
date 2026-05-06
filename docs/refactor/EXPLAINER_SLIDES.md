# GEAK Refactor, Explained

> One-page-per-slide deck for newcomers to GEAK.
> Read top-to-bottom; each section is a slide.

---

## Slide 1 — The big idea in one sentence

Before: **two agents, glued to Triton conventions.**
After: **one agent, language plugs in as data.**

```
BEFORE                             AFTER
┌──────────────────────┐           ┌────────────────────────────┐
│ HomogeneousAgent     │           │                            │
│ + language hardcoded │           │     OptimizationAgent      │
└──────────────────────┘           │                            │
┌──────────────────────┐   ──►     │  + KernelLanguage(triton)  │
│ HeterogeneousAgent   │           │  + KernelLanguage(hip)     │
│ + language hardcoded │           │  + KernelLanguage(...)     │
└──────────────────────┘           └────────────────────────────┘
```

---

## Slide 2 — What the two "old" agents actually were

They were **not two different algorithms**.
They were **two different task prompts** fed to the same optimization loop.

```
┌─── Homogeneous ────────────────────┐
│                                    │
│  Task:                             │
│    "Optimize this kernel."         │
│                                    │
│  Copies: N                         │
│  Each worker sees the SAME prompt. │
│                                    │
└────────────────────────────────────┘

┌─── Heterogeneous ──────────────────┐
│                                    │
│  Task #1:  "Optimize via fusing"   │
│  Task #2:  "Optimize via tiling"   │
│  Task #3:  "Optimize via split-k"  │
│  Task #4:  "Optimize via rewriting"│
│                                    │
│  One worker per task.              │
│                                    │
└────────────────────────────────────┘
```

The worker code was **identical**. Only the task TEXT differed.
So why have two agent classes?

---

## Slide 3 — The insight

Homogeneous is just **heterogeneous with N copies of the same task**.

```
heterogeneous mode  ──► [ task_1, task_2, task_3, task_4 ]
                                   │
                                   ▼
homogeneous mode    ──► [ task_1, task_1, task_1, task_1 ]
                         (same body, replicated)
```

Same dispatcher. Same worker class. Same logs. Same patches.
**No need for two agents — they're one with different task bodies.**

---

## Slide 4 — So we deleted one agent

Now there is exactly **ONE agent**:

```python
class OptimizationAgent:
    """The one and only worker.  Runs any optimization task body."""
    def run(self, task_body: str) -> Patch:
        ...
```

Three "modes" now just describe how the task LIST is built:

```
mode = "fixed"    ──► N identical task bodies         (was: homogeneous)
mode = "planned"  ──► N planner-generated task bodies (was: heterogeneous)
mode = "auto"     ──► adaptive mix, per round
```

All three modes instantiate the same `OptimizationAgent` on each worker.

---

## Slide 5 — Second problem: language was hardcoded

The old code had Triton and HIP details scattered everywhere.

```
BEFORE — language knowledge leaked into the agent:
┌────────────────────────────────────────────────────┐
│ agents/heterogeneous/                              │
│   workload_guidance.py                             │
│     if kernel_type == "triton":  ...  # hardcoded  │
│     if kernel_type == "hip":     ...  # hardcoded  │
│                                                    │
│   prompts.py                                       │
│     TRITON_SYSTEM_PROMPT = "You are a Triton..."   │
│     HIP_SYSTEM_PROMPT    = "You are a HIP..."      │
│                                                    │
│   task_generator.py                                │
│     _infer_kernel_type()                           │
│     _is_triton_like_kernel()                       │
│     _is_hip_like_kernel()                          │
│                                                    │
│   ...and many more                                 │
└────────────────────────────────────────────────────┘
```

Adding a new language = hunting through the agent code for
if-branches, copy-pasting, and hoping you got them all.

---

## Slide 6 — The fix: pull language out as DATA

Every language-specific thing moves into a folder you can read end-to-end.

```
AFTER — language is a data bundle:
┌────────────────────────────────────────────────────┐
│ kernel_languages/                                  │
│   triton/                                          │
│     kernel_language.py  ← metadata                 │
│     harness.j2          ← Jinja harness template   │
│     commandment.j2      ← Setup/Correctness/Bench  │
│     system_prompt.md    ← worker role              │
│     optimizer_hints.md  ← "prefer tl.dot over..."  │
│     idioms.md           ← style guidance           │
│     builder_hints.md    ← HarnessBuilder tips      │
│     memory_hints.md     ← KB extraction hints      │
│                                                    │
│   hip/                                             │
│     (same 8 files, hip-specific content)           │
│                                                    │
│   <new_language>/                                  │
│     (drop in 8 files, done — no agent edits)       │
└────────────────────────────────────────────────────┘
```

The agent code NEVER mentions "triton" or "hip".
It says `ctx.language.system_prompt`, `ctx.language.harness_template`, etc.

---

## Slide 7 — Side-by-side: file structure

```
BEFORE                                  AFTER
───────────────────────────────────     ─────────────────────────────────
src/minisweagent/                       src/minisweagent/
├─ agents/                              ├─ agents/
│  ├─ homogeneous/                      │  └─ optimization_agent.py
│  │  └─ homogeneous_agent.py           │    (the one agent class)
│  │    (worker + dispatch tangled)     │
│  │                                    ├─ kernel_languages/
│  └─ heterogeneous/                    │  ├─ base.py      (KernelLanguage)
│     ├─ orchestrator.py                │  ├─ triton/...
│     ├─ workload_guidance.py           │  │  ├─ harness.j2
│     │  (Triton/HIP if-branches)       │  │  ├─ commandment.j2
│     ├─ prompts.py                     │  │  ├─ system_prompt.md
│     │  (Triton/HIP prompt variants)   │  │  └─ ...
│     ├─ task_generator.py              │  └─ hip/...
│     │  (Triton/HIP detection)         │     └─ (same shape as triton)
│     └─ ...                            │
│                                       ├─ run/
├─ run/                                 │  ├─ unified.py         (_run_fixed,
│  ├─ orchestrator.py (heterogen only)  │  │                      _run_planned)
│  └─ <homogeneous path in cli.py>      │  └─ compose.py          (task body)
│                                       │
├─ run/preprocess/                      ├─ run/preprocess/
│  └─ preprocessor.py                   │  └─ phases/
│    (4000+ lines, monolithic,          │     ├─ discovery.py
│     Triton/HIP branches everywhere)   │     ├─ harness.py    (7 layers)
│                                       │     ├─ baseline.py
                                        │     ├─ explore.py
                                        │     └─ translation.py
                                        │     (each ≈ 200 lines,
                                        │      one job, one file)
```

---

## Slide 8 — Side-by-side: pipeline flow

```
BEFORE (homogeneous)                   BEFORE (heterogeneous)
────────────────────                   ────────────────────────
cli.py                                 geak-orchestrate
  │                                      │
  ├─ parse_task_info                     ├─ load preprocess_ctx
  ├─ run_preprocessor (monolith)         ├─ run_orchestrator
  │   ├─ resolve kernel                  │   └─ run_heterogeneous_
  │   ├─ discover tests                  │       orchestrator
  │   ├─ build harness (UTA+fixer)       │       ├─ generate tasks (LLM)
  │   ├─ write commandment               │       ├─ dispatch workers
  │   └─ baseline                        │       └─ collect + evaluate
  │                                      │
  └─ run_homogeneous_agent               └─ (different path, different
      └─ N identical copies                    reports, different logs)
      └─ ParallelAgent
          └─ run_pool


AFTER (every mode — fixed, planned, auto, translate)
─────────────────────────────────────────────────────
geak -t "..."
  │
  ├─ parse_task_info
  │
  ├─ preprocess pipeline (5 phases — one job each)
  │   ├─ TranslationPhase  (conditional; runs if target != source lang)
  │   ├─ DiscoveryPhase    (resolve kernel + codebase context)
  │   ├─ HarnessPhase      (7-layer resolution, first hit wins)
  │   ├─ BaselinePhase     (measure unoptimized perf)
  │   └─ ExplorePhase      (render commandment, cross-session memory)
  │
  ├─ run_pipeline (ctx, mode)
  │   ├─ _resolve_auto_mode      (if mode == "auto")
  │   ├─ _run_fixed              (if mode == "fixed")
  │   │   └─ build_fixed_tasks   (N identical task bodies)
  │   │       └─ run_pool ──► OptimizationAgent × N
  │   └─ _run_planned            (if mode == "planned")
  │       └─ planner LLM ──► N task bodies
  │           └─ run_pool ──► OptimizationAgent × N
  │
  └─ final report
```

One path, one CLI, one agent class, one pool scheduler.

---

## Slide 9 — Side-by-side: what a task body looks like

```
FIXED MODE (identical copies) — the task body:
┌──────────────────────────────────────────────────────┐
│ # System                                             │
│ {triton.system_prompt}                               │
│                                                      │
│ # User                                               │
│ Optimize the kernel at /path/to/kernel.py.           │
│ Use the harness at /path/to/harness.py.              │
│                                                      │
│ {commandment rendered from triton/commandment.j2}    │
│                                                      │
│ {triton.optimizer_hints}                             │
│                                                      │
│ {cross-session memory (if enabled)}                  │
│                                                      │
│ {user's extra addenda, e.g. "tune for gfx950"}       │
└──────────────────────────────────────────────────────┘

Replicated N times.  All N workers get the same body.
Variance comes from LLM sampling (temperature > 0).


PLANNED MODE (diverse strategies) — the task bodies:
┌──────────────────────────────────────────────────────┐
│ Task 1 body = (same header as above)                 │
│             + "Strategy: fuse stage1 + stage2 into   │
│                one persistent kernel."               │
├──────────────────────────────────────────────────────┤
│ Task 2 body = (same header)                          │
│             + "Strategy: rewrite top-k as a bitonic  │
│                sort with in-SMEM exchange."          │
├──────────────────────────────────────────────────────┤
│ Task 3 body = (same header)                          │
│             + "Strategy: split-k with warp-level     │
│                reduction and num_warps=16."          │
├──────────────────────────────────────────────────────┤
│ Task 4 body = (same header)                          │
│             + "Strategy: eliminate stage2 by using   │
│                torch.topk fallback on small rows."   │
└──────────────────────────────────────────────────────┘

Header is the same across all 4.  ONLY the strategy
hint differs.  Same agent class runs all 4.
```

This is the key insight: **homogeneous = "copy the task", planned = "vary the task"**.
Same worker either way.

---

## Slide 10 — Side-by-side: adding a new language

```
BEFORE — "Add FlyDSL support"
─────────────────────────────
  1. Edit agents/heterogeneous/workload_guidance.py
     add  _is_flydsl_like_kernel()
  2. Edit agents/heterogeneous/prompts.py
     add  FLYDSL_SYSTEM_PROMPT
  3. Edit agents/heterogeneous/task_generator.py
     add  _infer_kernel_type(... "flydsl")
  4. Edit run/preprocess/preprocessor.py
     add  if kernel_type == "flydsl": ...
  5. Edit harness_utils.py
     add  _detect_flydsl_kernel_definitions()
  6. Edit memory/cross_session/extractor.py
     add  "flydsl" to path-heuristic list
  7. Edit ...8 more files...

Touch points: 14+ files.   Risk of missed branch: high.


AFTER — "Add FlyDSL support"
─────────────────────────────
  1. mkdir src/minisweagent/kernel_languages/flydsl/
  2. Drop in 8 files:
       kernel_language.py
       harness.j2
       commandment.j2
       system_prompt.md
       optimizer_hints.md
       idioms.md
       builder_hints.md
       memory_hints.md
  3. Register in kernel_languages/__init__.py:
       from .flydsl import flydsl_language
       REGISTRY.register(flydsl_language)

Touch points: 1 new folder + 1 registry line.
Agent code untouched.  Nothing to miss.
```

---

## Slide 11 — What stayed the same

Not everything changed. A lot of battle-tested code just moved:

```
UNCHANGED OR MINIMALLY CHANGED:
┌──────────────────────────────────────────────────────┐
│ • ParallelAgent.run_parallel   (the GPU-pool shell)  │
│ • run_pool / AgentTask         (the scheduler)       │
│ • Patch application + git worktree isolation         │
│ • run_harness (correctness/profile/benchmark/full)   │
│ • SelectPatchAgent (best-patch picker)               │
│ • Cross-session memory retrieval                     │
│ • RAG integration                                    │
│ • Strategy manager                                   │
│ • All MCP tools (profiler, cross-session, RAG, ATD)  │
└──────────────────────────────────────────────────────┘
```

What changed is **organization + naming + one agent class**.
The underlying optimization + verification machinery is the same.

---

## Slide 12 — Bonus: harness is now a seed loop

Small but powerful corner fix.  Before, when a user passed their own harness:

```
BEFORE                                  AFTER
───────────────────────────────         ──────────────────────────────────
1. Validate all 4 modes strictly        1. Validate all 4 modes strictly
2. If any mode fails ──► raise          2. If any mode fails ──► stash as SEED
3. Pipeline crashes or                  3. Fall through to HarnessBuilder
   falls through silently                  with user's harness in the prompt
4. User's domain knowledge              4. LLM retries for up to 30 min,
   (shapes, refs) lost                      preserving user's shapes/refs
                                        5. Converges on a contract-compliant
                                           harness fast
```

"User's harness is a suggestion, not a final deliverable — we iterate on it."

---

## Slide 13 — Net wins (what this buys us)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. ONE agent class                                         │
│     → less code, same behavior, easier to test              │
│                                                             │
│  2. Language = data, not code                               │
│     → new language in minutes, zero agent edits             │
│                                                             │
│  3. Preprocess split into 5 small phases                    │
│     → each file does ONE thing, ~200 LoC max                │
│                                                             │
│  4. Single CLI entry: `geak -t "..."`                       │
│     → no more "which script do I run?"                      │
│                                                             │
│  5. Verified speedups on BOTH branches                      │
│     → refactor matches main's results in parity tests       │
│                                                             │
│  6. Universal harness contract enforced in one place        │
│     → Jinja template per language, parser normalizes names  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 14 — Glossary (for brand-new readers)

| Term                  | What it means                                                          |
|-----------------------|------------------------------------------------------------------------|
| **Kernel**            | A GPU function (Triton, HIP, CUDA, ...) we want to make faster.        |
| **Harness**           | A Python script that runs the kernel with `--correctness/--benchmark/--full-benchmark/--profile` flags and prints `GEAK_RESULT_LATENCY_MS=...`. |
| **Commandment**       | A markdown file listing shell commands: Setup, Correctness, Benchmark, Full Benchmark, Profile. Rendered from a per-language Jinja template. |
| **OptimizationAgent** | The (now-single) worker class that takes a task body and produces a patch. |
| **Mode**              | How the task list is built: `fixed` (N copies), `planned` (N planner strategies), `auto` (pick per round). |
| **KernelLanguage**    | A frozen dataclass describing everything language-specific (templates, prompts, hints) for ONE language. |
| **Verified speedup**  | Latency-on-optimized / latency-on-baseline, measured by actually running `--full-benchmark` on both and dividing. Not self-reported by the agent. |
| **Parity run**        | Same kernel, same task, main branch vs refactor branch, in parallel. Success = similar verified speedups. |

---

## Slide 15 — TL;DR (one paragraph)

We noticed `HomogeneousAgent` and `HeterogeneousAgent` were running the
**same worker code**; only the task **text** differed (one copy vs. N
planner-generated strategies). We collapsed them into one class,
`OptimizationAgent`, and now the "mode" just decides how the task list is
built. Separately, we pulled Triton/HIP-specific knowledge out of the agent
code into per-language folders under `kernel_languages/`, so adding a new
language means dropping in 8 files, not editing 14. The scheduler,
patch apply, benchmarking, memory, and RAG machinery are unchanged. Parity
tests show the refactor produces matching verified speedups to main on the
same kernels — so the simplification is real and safe.
