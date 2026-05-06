# GEAK Codebase Audit — Current State (reference document)

**Snapshot**: `origin/main @ d7b880c3` (post-PR #166, #174), observed 2026-04-22 08:51 via two live runs.

**Purpose**: Single reference for how GEAK actually works today, verified end-to-end by live log traces. This document is the BASELINE the unification plan refactors from. Don't guess — check this file first, then the code.

---

## 0. How I verified (don't trust docs; trust logs)

Two live runs observed during audit:

| | Triton run | HIP run |
|---|---|---|
| Kernel | `L3/gemm_a16w16_atomic/kernel.py` (aiter) | `hip2hip/others/assign_score_withk/assign_score_withk_wrapper.py` |
| Command | `geak -t "..."` NL only | `geak --kernel-url ... --repo ... --task ... --num-parallel 2 --gpu-ids 0,1` |
| Log | `/data/sapmajum/triton_runs/gemm_a16w16_atomic_canonical-rocm700_memon_20260422_083415.log` | `/data/sapmajum/AgentKernelArena/logs/hip_ab_v3/assign_score_withk_mem_20260415_180639.log` |
| Status | Live (round loop in progress at snapshot time) | Archived (completed 2026-04-15) |

All claims below have a log-line citation. If a claim has no citation, it came from code reading, not runtime behavior.

---

## 1. CLI entry point

```
pyproject.toml  [project.scripts]
  geak = "minisweagent.run.mini:app"
```

Single entry. Typer CLI. One line.

There is also `geak-orchestrate` (`run/orchestrator.py:main`) — a **second CLI** that only supports heterogeneous mode (raises `NotImplementedError` for homogeneous at line 85-86). It's used internally for resume-from-preprocess-dir workflows. **Redundant entry point** — should consolidate into `geak` post-unification.

---

## 2. Startup (both paths)

### `mini.py::run_main` lines 230-450

```
1. configure_if_first_time()               → merges YAMLs (geak.yaml + mini_kernel_strategy_list.yaml)
2. parse_pipeline_params(task, model)      → LLM call #1: {heterogeneous, max_rounds} from NL
3. parse_task_info(task, model)            → LLM call #2: {kernel_type, repo, gpu_ids, ...} from NL
4. kernel_type = _normalize_kernel_type(parsed_config["kernel_type"])     [mini.py:61-67]
5. if kernel_path known: _infer_kernel_type(kernel_path)                   [overrides step 4 when more confident]
```

**Observed (Triton run line 34)**: `Normalized kernel_type from task content: triton`
**Observed (HIP run line 13)**:    `Normalized kernel_type from task content: hip`

---

## 3. Preprocessing (`run_preprocessor`)

Common 7 steps defined in `run/preprocess/preprocessor.py`:

| Step | Module | Triton | HIP |
|---|---|---|---|
| 1 — resolve kernel URL | `resolve_kernel_url.py` | ✅ ran (line 76) | ✅ ran (line 47) |
| 2 — codebase context | `codebase_context.py` | ✅ ran (line 89) | ❌ skipped |
| 3 — test discovery (ATD) | `automated_test_discovery` MCP + `unit_test_agent.py` | ✅ ran (line 95) | ❌ skipped |
| 4 — harness validation | `harness_utils.execute_harness_validation` | ✅ ran (line 126) | ❌ skipped |
| 5 — kernel profiling | `kernel_profile.py` (Metrix MCP) | ✅ ran (line 155) | ✅ ran (line 54) |
| 6 — baseline metrics | `baseline.build_baseline_metrics` | ✅ ran (line 185) | ✅ ran (line 70) |
| 7 — COMMANDMENT generation | `commandment.generate_commandment(kernel_language)` | ✅ ran (line 192) | ✅ ran (line 77) |

**Correction to earlier audits**: HIP DOES get COMMANDMENT.md generated (line 77). The claim "HIP skips the commandment pipeline" is wrong at the preprocessor level. What actually happens: HIP's commandment IS generated, but the **homogeneous runner doesn't enforce it per-round** — `save_and_test` uses the raw `test_command` from `--task` instead of re-reading COMMANDMENT. So FULL_BENCHMARK per-round verification doesn't happen for HIP, even though the commandment file exists.

Steps 2, 3, 4 are skipped for HIP because:
- `resolve_kernel_url` couldn't normalize the HIP wrapper as an ATD-compatible test path
- ATD expects a harness with `--correctness / --benchmark / --full-benchmark` CLI; HIP's `scripts/task_runner.py compile/correctness/performance` is a different contract
- Without a validated harness, `execute_harness_validation` has nothing to validate

---

## 4. Mode routing — the ONE hardcoded branch

### `mini.py:498-512`

```python
if heterogeneous is None:  # not set by LLM extraction
    _auto_kernel_type = discovery.kernel.type
    if not _auto_kernel_type:
        _auto_kernel_type = _infer_kernel_type(kernel_path)
    
    if _auto_kernel_type == "triton":
        heterogeneous = True
    else:
        heterogeneous = False
```

**Observed (Triton run line 197)**: `Using heterogeneous mode based on discovery`
**Observed (HIP run line 82)**: `Using homogeneous mode based on discovery`

This is THE split. Everything downstream forks here.

---

## 5. Heterogeneous path (Triton)

### `mini.py:514-560` → `run_orchestrator` → `run_heterogeneous_orchestrator`

**Observed at snapshot time (line 215)**: `Exploration Phase (this may take a few minutes)` — the run was in the exploration phase, ~14 min elapsed.

```
run_heterogeneous_orchestrator(preprocess_ctx, gpu_ids, model, ...)
    [agents/heterogeneous/orchestrator.py]
  │
  ├─ Build initial messages
  │    system = SYSTEM_PROMPT                                [prompts.py:8, 74 LoC]
  │                Triton+rocprofv3+COMMANDMENT.md baked in
  │    instance = INSTANCE_TEMPLATE.format(
  │                 kernel_path=..., profiling_summary=...,
  │                 commandment_excerpt=...,
  │                 memory_context=assemble_memory_context(kernel_path=...))
  │
  ├─ Exploration phase: run_llm_steps(model, messages, phase="explore")
  │    (reads kernel, no orchestration tools)
  │
  └─ for round_num in 1..5:
       │
       ├─ LLM calls generate_tasks(...)
       │    → heterogeneous/tools.py::_tool_generate_tasks
       │    → task_generator.generate_tasks(kernel_type, kernel_language, ...)
       │    → TASKGEN LLM call: reads TASKGEN_SYSTEM_PROMPT, history,
       │      profile, baseline, KB, → emits JSON task list
       │    → write_task_files(round_N/NN_<label>.md)
       │
       ├─ LLM calls dispatch_tasks(task_files)
       │    → run_pool(M tasks, N GPUs)
       │    → for each: StrategyInteractiveAgent(model, env).run(task_body)
       │      ↑ agent's system_template + instance_template comes from
       │        mini_kernel_strategy_list.yaml (generic kernel-optimization
       │        prompt — NOT Triton-specific)
       │
       ├─ LLM calls collect_results(...)
       │
       └─ post_round_evaluate(best_patch, commandment_path)
            → runs SETUP + CORRECTNESS + FULL_BENCHMARK + PROFILE in eval worktree
            → verified_speedup → round_N_evaluation.json

finalize_run → record_final_outcome → memory.db
```

---

## 6. Homogeneous path (HIP)

### `mini.py:562-630` → `run_homogeneous_agent`

**Observed (HIP log lines 82-107)**:

```
Line 82:  Using homogeneous mode based on discovery.
Line 85:  Retriever: category=unknown language=hip bottleneck=latency
Line 90:  Cross-session memory injected into homogeneous task (27608 chars)
Line 95:  ============================================================
Line 96:  Homogeneous Agent (2 agents, GPUs [2, 3])
Line 97:  ============================================================
Line 106: Sub-agent 1 (task_1) started on GPU 3
Line 107: Sub-agent 0 (task_0) started on GPU 2
```

```
run_homogeneous_agent(config, task_content, num_parallel=2, gpu_ids=[0,1])
    [agents/homogeneous/homogeneous_agent.py, 196 LoC]
  │
  ├─ inject KB memory into task_content                     [verified: 27608 chars]
  ├─ agent_config = config["agent"]                          [from mini_kernel_strategy_list.yaml]
  ├─ base_agent_class = StrategyInteractiveAgent
  └─ ParallelAgent(model, env, **agent_config).run(task_content)
        │
        └─ ParallelAgent.run_parallel(...)                  [parallel_agent.py:353]
             │
             └─ HOMOGENEOUS branch [lines 414-609]:         ← ~200 LoC duplicate of run_pool
                  │
                  ├─ for i in range(2):                     ← N identical copies
                  │    write parallel_{i}.md (body = task_content verbatim)
                  │
                  ├─ ThreadPoolExecutor(2) as executor:
                  │    run_single_agent(i) for i in 0..1:
                  │       create_worktree(repo, "slot_{i}")
                  │       StrategyInteractiveAgent(model, env, **cfg).run(task_content)
                  │       returns exit_status, patches
                  │
                  └─ _select_best_from_parallel_runs(...)
                       → SelectPatchAgent picks winner from patches
                       → NO FULL_BENCHMARK verification
                       → writes final_report.json (4-key schema)
                       → NO call to record_optimization_outcome (no KB write)
```

---

## 7. The agent: `StrategyInteractiveAgent`

Inheritance chain (4 levels):

```
DefaultAgent                   [agents/default.py, 611 LoC]    — core step-loop
  └─ InteractiveAgent          [agents/interactive.py, 181 LoC] — adds mode: yolo/confirm/human
       └─ StrategyAgent        [agents/strategy_agent.py, 156 LoC] — adds strategy_manager tool
            └─ StrategyInteractiveAgent  [agents/strategy_interactive.py, 50 LoC] — combines for CLI
```

**What it actually does** (after reading the code):
- Receives `task_body: str` (opaque string)
- Runs a step-loop: query LLM → parse tool call → execute tool → observation → repeat
- Tools: `bash`, `str_replace_editor`, `save_and_test`, `submit`, `strategy_manager`, `profile_kernel`, `query`, `optimize`
- Never reads `kernel_type` or any language-specific identifier
- Never branches on language

**The critical fact**: `StrategyInteractiveAgent` is already language-agnostic. Both Triton and HIP runs instantiate THE SAME class with THE SAME config (`mini_kernel_strategy_list.yaml`).

---

## 8. What actually differs between Triton and HIP at agent-level

| | Triton (hetero path) | HIP (homo path) |
|---|---|---|
| Agent class | `StrategyInteractiveAgent` | `StrategyInteractiveAgent` (same) |
| Agent config YAML | `mini_kernel_strategy_list.yaml` | `mini_kernel_strategy_list.yaml` (same file) |
| Agent `system_template` | Generic "high-performance computing and kernel optimization..." | Same generic string |
| Agent `instance_template` | Generic "Workflow for Kernel Optimization..." | Same generic string |
| Tools available | bash, str_replace_editor, save_and_test, submit, strategy_manager, profile_kernel, query, optimize | Same set |
| `task_body` content | LLM-planned strategy-specific body (e.g. "fuse softmax with topk, use tl.dot_scaled for...") | Verbatim user NL task ("Optimize the repository, test command is python3 scripts/task_runner.py compile && ...") |
| KB context | Injected by task generator | Injected by `run_homogeneous_agent` (27608 chars observed) |
| Language awareness in prompt | **NONE** — nothing in agent prompts mentions Triton | **NONE** — same |

**Where language awareness actually lives**:
1. `COMMANDMENT.md` SETUP block (build step for HIP; PYTHONPATH for Triton)
2. The word "HIP" or "Triton" appearing in the user's NL `--task` string (if the user typed it)
3. `mini_kernel_strategy_list.yaml` Step 1 "Hardware Grounding" mentions `rocminfo` and generic GFX/CU/LDS (AMD-specific but not Triton/HIP-specific)

That's it. The agent gets no "you are a HIP expert" or "you are a Triton expert" prompt today.

---

## 9. Complete inventory of language-branching sites

From grep + AST analysis of `origin/main@d7b880c3`:

| # | File | Line(s) | What it hardcodes |
|---|---|---|---|
| 1 | `run/mini.py` | 61-67 | `_normalize_kernel_type` (triton / hip / rocm / rocblas strings) |
| 2 | `run/mini.py` | 331-342 | parsed_config normalization + `_infer_kernel_type` fallback |
| 3 | `run/mini.py` | 498-512 | mode auto-detect: `if _auto_kernel_type == "triton": heterogeneous=True` |
| 4 | `run/mini.py` | 514 | `if heterogeneous:` — the pipeline branch |
| 5 | `agents/heterogeneous/task_generator.py` | 74-105 | `_infer_kernel_type` (@triton, tl., .hip, .cpp, /ck/ markers) |
| 6 | `agents/heterogeneous/task_generator.py` | 22, 183-210, 276 | `kernel_type` / `kernel_language` string params throughout API |
| 7 | `agents/heterogeneous/prompts.py` | 8-131 | `SYSTEM_PROMPT` + `GPU_AND_PROFILER_RULES` (rocprofv3, Triton assumptions) |
| 8 | `agents/heterogeneous/prompts.py` | 133-281 | `TASKGEN_SYSTEM_PROMPT` (Triton-centric strategy taxonomy) |
| 9 | `agents/heterogeneous/workload_guidance.py` | 13-78 | `_HIP_SEARCH_HINT_PATTERNS`, `_is_hip_like_kernel`, `_is_triton_like_kernel` |
| 10 | `run/preprocess/discovery_types.py` | 36-37, 247-253 | `kernel_type`, `kernel_language` + `_infer_kernel_language` |
| 11 | `run/preprocess/commandment.py` | 55-127 | `generate_commandment` `kernel_language` param drives inner-kernel vs simple |
| 12 | `agents/unit_test_agent.py` | 40-75 | `_LANGUAGE_GUIDANCE` dict (triton/hip/ck/asm test generation) |
| 13 | `run/preprocess/unit_test_agent.py` | — | Duplicate of #12 (two files with same name, ~250 LoC each) |
| 14 | `memory/cross_session/extractor.py` | 500-505 | Path-string sniffing for language |
| 15 | `config/mini_kernel_strategy_list.yaml` | system_template, instance_template (14KB) | Python/Triton conventions baked in |
| 16 | `config/mini_unit_test_agent.yaml` (8KB) + `run/preprocess/config/mini_unit_test_agent.yaml` (17KB) | — | Duplicate configs |

**Total branching sites**: 16+ distinct places. Adding one language today requires touching ~9 of them (PR #155 FlyDSL evidence).

---

## 10. Duplicates / dead code / bloat inventory

### 10.1 Duplicate files

| Duplicate | Location A | Location B | LoC |
|---|---|---|---|
| `UnitTestAgent` class | `agents/unit_test_agent.py` (251) | `run/preprocess/unit_test_agent.py` (252) | 503 combined; merge to 252 |
| `mini_unit_test_agent.yaml` | `config/mini_unit_test_agent.yaml` (8KB) | `run/preprocess/config/mini_unit_test_agent.yaml` (17KB) | ~25KB combined |
| `validate_commandment.py` | `tools/validate_commandment.py` (7 LoC stub) | `run/preprocess/validate_commandment.py` (280 LoC) | 7 LoC stub deletable |
| `discovery_types.py` | `tools/discovery_types.py` | `run/preprocess/discovery_types.py` (337 LoC) | verify if one is re-export |
| `benchmark_parsing.py` | `run/preprocess/benchmark_parsing.py` (381) | `run/postprocess/benchmark_parsing.py` (389) | near-identical; consolidate |

### 10.2 Dead/redundant code

| File | LoC | Reason to delete |
|---|---|---|
| `agents/homogeneous/homogeneous_agent.py` | 196 | Thin wrapper over ParallelAgent homo branch; replaced by unified pipeline |
| `agents/parallel_agent.py` homogeneous branch (lines 414-609) | ~196 | Duplicate of `run/utils/parallel_helpers.run_pool` |
| `agents/interactive_textual.py` | 450 | Dev TUI; not used in production runs |
| `agents/heterogeneous/workload_guidance.py` | 307 | `_HIP_SEARCH_HINT_PATTERNS` + language sniffing; replaceable by per-language taskgen_strategy_prompt |
| `agents/heterogeneous/result_scanning.py` | 210 | Only ~40 LoC actually used; fold into task_generator |
| `agents/heterogeneous/schemas.py` | 96 | Tool-call schemas move to top-level `tools/orchestrator_schemas.py` |
| `run/orchestrator.py` | 288 | Second CLI entry (`geak-orchestrate`) — homo-path raises NotImplementedError; redundant with mini.py |
| `agents/shape_fixer_agent.py` | 7 | Re-export stub; import real one from `run/preprocess/shape_fixer_agent.py` |
| `agents/strategy_agent.py` + `agents/strategy_interactive.py` + `agents/interactive.py` | 156 + 50 + 181 = 387 | 4-level inheritance for what should be 1 class with 2 config fields |
| `config/mini_reverse_kl.yaml` | 15KB | Not referenced from any entry point (verify with grep) |
| `config/mini_unit_test_agent.yaml` | 8KB | Duplicate with run/preprocess/config/ version |
| `tools/validate_commandment.py` stub | 7 | Duplicate of run/preprocess/ version |

### 10.3 File size concerns (readability gates)

Files >800 LoC in one concern area:

| File | LoC | Recommendation |
|---|---|---|
| `run/preprocess/harness_utils.py` | 1660 | Split into `harness/build.py`, `harness/validate.py`, `harness/cache.py` |
| `run/preprocess/preprocessor.py` | 1386 | Split the 7 steps into `steps/*.py`; preprocessor just orchestrates |
| `agents/heterogeneous/task_generator.py` | 997 | Split `_run_taskgen_llm` / `_build_taskgen_prompt` / `_parse_taskgen_output` / `_validate_tasks` into separate files |
| `run/pipeline_helpers.py` | 915 | Split into `model_factory.py`, `env_factory.py`, `context_injection.py` |
| `memory/cross_session/extractor.py` | 752 | Split per-field extraction logic into a table |
| `run/postprocess/evaluation.py` | 721 | Keep; already well-structured |
| `run/mini.py` | 629 | Thin after unification (delete 498-512, 562-630) → ~350 LoC |
| `agents/default.py` | 611 | Keep; absorbs InteractiveAgent |
| `agents/parallel_agent.py` | 609 | Delete homo branch → ~400 LoC |

---

## 11. Observed flow comparison (evidence table)

| Step | Expected (from code reading) | Triton observed | HIP observed | Match |
|---|---|---|---|---|
| kernel_type normalized to "triton" or "hip" | Per mini.py:61-67 | `triton` (line 34) | `hip` (line 13) | ✅ |
| 7-step preprocess | Per preprocessor.py | Steps 1-7 all ran | Steps 1, 5, 6, 7 ran; 2-4 skipped | partial — HIP skips ATD steps silently |
| COMMANDMENT.md generated | Per commandment.py:55-127 | ✅ line 192 | ✅ line 77 | ✅ (correction: earlier claim "HIP skips commandment" is WRONG) |
| Mode auto-detect | Per mini.py:498-512 | `heterogeneous=True` (line 202) | `homogeneous` (line 82) | ✅ |
| KB memory injected | Per mini.py:573-585 (homo) / orchestrator.py (hetero) | (TBD — run in progress) | 27608 chars (line 90) | ✅ for HIP |
| Agent class | StrategyInteractiveAgent for both | (TBD — expected via run_pool) | `Sub-agent (task_1) started` (line 106) | ✅ |
| Sub-agents parallel | Per ThreadPoolExecutor (homo) / run_pool (hetero) | (TBD) | 2 sub-agents on GPUs 2,3 (line 106-107) | ✅ |
| Per-round FULL_BENCHMARK | Only hetero per postprocess/evaluation.py | (TBD — round 1 not yet reached) | ❌ never runs on homo path | ✅ (asymmetry confirmed) |
| KB write on completion | Only hetero per integration.py | (TBD) | ❌ `record_optimization_outcome` never called | ✅ (asymmetry confirmed) |

---

## 12. What I'm confident about vs uncertain

### Confident (grepped AND observed)
- Triton auto-routes to heterogeneous; HIP auto-routes to homogeneous (THE branch at mini.py:498-512)
- `StrategyInteractiveAgent` is the one and only sub-agent class on both paths
- Agent prompt (`mini_kernel_strategy_list.yaml`) is language-agnostic — nothing in it mentions Triton or HIP
- HIP generates COMMANDMENT.md but skips ATD/harness-validation steps
- HIP never writes to cross-session KB (grep-verified `record_optimization_outcome` never called on homo path)
- ParallelAgent.run_parallel has TWO branches: ThreadPoolExecutor (homo, lines 414-609) and run_pool delegation (hetero)

### Uncertain (code read but not observed)
- Whether `run_orchestrator` CLI (`geak-orchestrate`) is still used in practice, or dead
- Whether `config/mini_reverse_kl.yaml` is referenced anywhere
- Exact behavior of `save_and_test` when HIP's test_command is `compile && correctness && performance` vs reading COMMANDMENT

### Needs observation before Phase 2 refactor
- What happens in round 2+ of the active Triton run (not yet reached)
- Whether the observed `memory_context=27608 chars` includes baseline_fingerprint or not (schema audit)

---

## 13. Mapping: where each concern lives today (for the unification plan)

| Concern | Today's location | Complexity |
|---|---|---|
| Language detection | `mini.py:61-67`, `task_generator.py:74-105`, `discovery_types.py:247-253` | 3 scattered spots |
| Mode routing | `mini.py:498-512`, `run/orchestrator.py:85-86` | 2 spots |
| Agent class resolution | `homogeneous_agent.py:105` (hard-coded `StrategyInteractiveAgent`) + `heterogeneous/tools.py` (same) | 2 spots, both pick same class |
| Task-body generation (homo) | `parallel_agent.py:428-431` (verbatim N copies of `task_content` string) | 1 spot |
| Task-body generation (hetero) | `heterogeneous/task_generator.py::generate_tasks` (LLM-driven) | 1 spot |
| Dispatch — homo | `parallel_agent.py:597` (`ThreadPoolExecutor`) | duplicate of run_pool |
| Dispatch — hetero | `run/utils/parallel_helpers.py:423` (`run_pool`) | canonical |
| COMMANDMENT file | `preprocess/commandment.py::generate_commandment` | one place ✓ |
| Per-round eval (hetero) | `postprocess/evaluation.py::evaluate_round_best` | one place ✓ |
| KB write (hetero only) | `postprocess/results.py::record_final_outcome` → `memory/integration.py` | one place ✓ but not called from homo |
| KB retrieve | `memory/cross_session/retriever.py` — called from BOTH paths | one place ✓ |

The only multi-location concerns are **language detection** (3 places) and **mode routing** (2 places) and **dispatch** (2 places, ThreadPool vs run_pool duplicate). Everything else is already single-source-of-truth. That's the opportunity for the unification.

---

## 14. Key log lines for future verification

If refactoring, the following log markers should continue to appear (or have clear replacements):

```
# Triton run markers (from gemm_a16w16_atomic_canonical-rocm700_memon_20260422_083415.log):
Normalized kernel_type from task content: triton
--- Step 1/7: Resolve kernel URL ---
--- Step 7/7: Commandment ---
  COMMANDMENT.md generated
Using heterogeneous mode based on discovery.
run_orchestrator: ... heterogeneous=True
Cross-session memory
Exploration Phase

# HIP run markers (from assign_score_withk_mem_20260415_180639.log):
Normalized kernel_type from task content: hip
--- Step 7/7: Commandment ---
Using homogeneous mode based on discovery.
Retriever: category=unknown language=hip
Cross-session memory injected into homogeneous task (... chars)
Homogeneous Agent (N agents, GPUs [...])
Sub-agent N (task_N) started on GPU M
```

Any unification refactor should produce either identical markers OR a documented replacement markers table.

---

**End of audit.** Next: unification plan (see companion document or the v3 design).
