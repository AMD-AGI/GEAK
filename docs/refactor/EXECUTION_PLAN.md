# GEAK Refactor — Execution Plan (v3, grounded in live runs)

### How to read this document

- **§0** — the evidence register: what was observed live, what speedups must be preserved, what bugs each PR addresses. Start here.
  - §0.1 log markers • §0.2 speedups • §0.3 bugs • §0.4 branch state
  - §0.5 **current flow diagram** (the baseline the refactor preserves semantically)
  - §0.6 **AgentKernelArena test procedure** (how the 18 Triton + 16 HIP kernels are actually run)
- **§1-§7** — the *target architecture*: where every file lives, what gets deleted, what the CLI looks like.
- **§8** — the *PR sequence* (**3 PRs**, ~6-10 weeks; restructured from 38 after user direction). Each PR has a What / Risk / Rollback / LoC / Dependencies line.
- **§9-§10** — the *agent architecture*: orchestrator pseudocode, deterministic vs LLM subagent tradeoffs.
- **§10.5-§10.7** — deep dive on `CrossSessionMemoryAnalysisAgent`, review of what else could be a subagent (discussion; not added), and how RAG + cross-session memory coexist.
- **§11-§12** — *safety net*: smoke tests, CI gates, risk matrix with historical precedent.
- **§13-§14** — scope boundaries, next action, rollback posture, **flag matrix**.
- **§15** — **reviewer response log** (verified-evidence audit of external review).
- **§16** — **implementation blueprint**: every class/dataclass signature, end-to-end call flow, per-week build plan, fixture-corpus spec, CI gate summary, day-1 dev walkthrough.
- **Terminology block** (right below this nav): 1-line definitions + recommended Python construct for `deterministic function`, `tool`, `MCP tool`, `KernelLanguage` (frozen dataclass), `subagent` (subclass of `SubagentBase` ABC), `main agent` (`OptimizationAgent` plain class, peer of SubagentBase), plus the decision rule.

### Terminology (1-line definitions + Python construct)

- **Deterministic function** — pure code; same input → same output; no LLM.
  *Python*: module-level `def` or `@staticmethod`. (e.g. Jinja render, regex validators, retriever math.)

- **Tool** — function the main agent calls mid-reasoning via the tool-call protocol; deterministic or thin wrapper.
  *Python*: callable class with `__call__(**kwargs) -> dict` returning `{output, returncode}` (existing `ResolveKernelUrlTool` pattern); registered in `tools/tools_runtime.py`. (e.g. `bash`, `save_and_test`, `str_replace_editor`.)

- **MCP tool** — same shape as a tool, but implementation lives in an external server (Model Context Protocol). Used when the capability needs its own process / dependencies / shared state.
  *Python*: external package under `mcp_tools/<name>-mcp/`; bridged into the agent via `tools/mcp_bridge.py`. (e.g. `query`/`optimize` via `rag-mcp`; `profile_kernel` via Metrix.)

- **KernelLanguage** — central data object describing everything language-specific about a kernel language (Triton / HIP / FlyDSL / any new language). Holds paths to prompts, Jinja templates, test-runner command, profiler command, `kb_namespace`, idioms, translation hints.
  *Python*: `@dataclass(frozen=True)` in `kernel_languages/base.py` — full schema in §16.1. Each language ships as a folder `kernel_languages/<name>/` with `kernel_language.py` instantiating it. Discovered at runtime via `registry.detect_best(path)` in `kernel_languages/__init__.py`.

- **Subagent** — narrow-purpose LLM "employee" with ONE job, called by orchestrator code (not the main agent), produces a structured artifact. Short-lived (1-5 LLM turns), own budget, black-box to callers.
  *Python*: `class MySubagent(SubagentBase)` — inherits from `SubagentBase` (an `abc.ABC` in `subagents/base.py`). Each subclass **overrides exactly ONE of two methods**: `run(**inputs)` (one-shot) or `loop(max_attempts, verify_fn, **inputs)` (multi-round verify-retry). CI gate `check_subagent_base_contract.py` enforces "exactly one". (e.g. `HarnessBuilder` uses `run()`; `TranslationLoop` uses `loop()`.)

- **Main agent (`OptimizationAgent`)** — long-running LLM loop that uses tools+MCP to iteratively optimize a kernel over many rounds. ONE class (no inheritance chain).
  *Python*: plain `class OptimizationAgent:` in `agents/optimization_agent.py` — **NOT** a `SubagentBase` subclass (peer class, not parent or child). Shares only `AgentConfig` + exceptions (`Submitted`, `LimitsExceeded`, etc.) from `agents/agent_spec.py`. Subagents COMPOSE `OptimizationAgent` (via `SubagentBase._make_optimization_agent()` helper) when they need a tool loop; they never inherit from it.

- **Decision rule**: pure logic → function; dynamic mid-reasoning decision → tool/MCP; narrow reasoning task with structured output → subagent (inherits from `SubagentBase`); long iterative reasoning → main agent (`OptimizationAgent`).

---

**Branch**: `refactor-test` (off `origin/main @ d7b880c3`, as of 2026-04-22)
**Scope**: Full-liberty refactor on a fork. Unify GEAK's homo+hetero pipelines; collapse the 4-layer agent inheritance into one class; delete ~8,000 LoC of mini-swe-agent heritage, duplicates, and dead code (current: **34,301 LoC across 143 Python files**; target: ~26,000 LoC); introduce first-class `KernelLanguage` objects with a user-facing `geak add-language` scaffolder; add a `TranslationLoop` subagent (Translation phase of preprocessing) that converts kernels between any registered language pair; add a `CrossSessionMemoryAnalysisAgent` (per-round subagent) that writes a structured markdown memory context when cross-session memory is enabled; keep `geak -t "prompt"` as the single entry point in one CLI file. AgentKernelArena is expected to be updated in parallel to match the new interface.
**Principle**: modular + simple + no band-aids. Every PR is independently rollback-safe. Every claim grounded in live run evidence.
**Reference docs**:
  - `GEAK_codebase_audit.md` — current state, VERIFIED by two live runs (Triton `gemm_a16w16_atomic`, HIP `assign_score_withk`)
  - `GEAK_unification_plan.md` — full architecture rationale
  - `GEAK_hardened_refactor.md` — regression-safety analysis

---

## §0 — Verified evidence from live GEAK runs

This plan is grounded in concrete observations from two live runs captured 2026-04-22 plus KB state from 60+ historical runs. Every decision below has a citation. If a claim has no citation, it came from code reading, not runtime behavior.

### §0.1 Exact log markers that will be smoke-tested

These strings were observed live. The refactor must either preserve them verbatim OR provide a documented replacement mapping (tracked in PR-42).

**Triton + heterogeneous path** (from `/data/sapmajum/triton_runs/gemm_a16w16_atomic_canonical-rocm700_memon_20260422_083415.log`):
```
Normalized kernel_type from task content: triton         [line 34]
--- Step 1/7: Resolve kernel URL ---                     [line 76]
--- Step 2/7: Codebase Context ---                       [line 89]
--- Step 3/7: Test Discovery ---                         [line 95]
--- Step 4/7: Harness Validation ---                     [line 126]
--- Step 5/7: Kernel Profiling ---                       [line 155]
--- Step 6/7: Baseline Metrics ---                       [line 185]
--- Step 7/7: Commandment ---                            [line 192]
Using heterogeneous mode based on discovery.             [line 197]
run_orchestrator: ... heterogeneous=True                 [line 202]
Cross-session memory                                     [present]
Exploration Phase (this may take a few minutes)          [line 215]
```

**HIP + homogeneous path** (from `/data/sapmajum/AgentKernelArena/logs/hip_ab_v3/assign_score_withk_mem_20260415_180639.log`):
```
Normalized kernel_type from task content: hip            [line 13]
--- Step 1/7: Resolve kernel URL ---                     [line 47]
--- Step 5/7: Kernel Profiling ---                       [line 54]
--- Step 6/7: Baseline Metrics ---                       [line 70]
--- Step 7/7: Commandment ---                            [line 77]
Using homogeneous mode based on discovery.               [line 82]
Retriever: category=unknown language=hip                 [line 85]
Cross-session memory injected into homogeneous task (27608 chars)  [line 90]
Homogeneous Agent (2 agents, GPUs [2, 3])                [line 96]
Sub-agent 1 (task_1) started on GPU 3                    [line 106]
Sub-agent 0 (task_0) started on GPU 2                    [line 107]
```

HIP skips Steps 2-4 silently (no-op; see §0.3 bug #1). Refactor must surface this as an explicit skip decision, not a silent fall-through.

### §0.2 Best-observed speedups (regression thresholds)

From the live knowledge base at `src/minisweagent/memory/cross_session/knowledge_base.json` (version 7, 60+ recorded experiences). Every PR must maintain or improve these speedups on the listed kernels; any regression below threshold fails CI.

| Kernel | Best observed speedup | Source | Regression threshold (PR gate) |
|---|---|---|---|
| `gemm_a16w16_atomic` | 3.9183x | KB line 962 | ≥ 3.5x |
| `llama_ff_triton` | 5.2364x | KB line 5075 | ≥ 4.7x |
| `three_nn` | 5.3813x | KB line 1909 | ≥ 4.8x |
| `knn` | 3.0095x | KB line 1811 | ≥ 2.7x |
| `fused_mxfp4_quant_moe_sort` | 2.463x (peak 3.51x historical) | KB line 57 | ≥ 2.2x |
| `fused_rms_fp8` | 2.2312x (peak) | KB line 4928 | ≥ 2.0x |
| `gemm` (generic) | 1.5889x | KB line 639 | ≥ 1.4x |
| `topk` | 1.3394x | KB line 1518 | ≥ 1.2x |
| `gather_points` | 1.3646x | KB line 2007 | ≥ 1.2x |
| `moe_routing_sigmoid_top1` | 1.2138x | KB line 1249 | ≥ 1.1x |
| `ball_query` | 1.18x | KB line 1713 | ≥ 1.1x |
| `fused_qkv_rope` | 1.131x | KB line 5000 | ≥ 1.08x |
| `fused_qk_rope_cache_mla` | 1.1008x | KB line 285 | ≥ 1.05x |

Plus 5 other kernels in `/data/sapmajum/triton_runs/` that have run but not yet stored in KB (`ff_backward`, `gemm_a16wfp4`, `lean_atten_paged`, `mla_decode`, `refk_fp8_blockwise_mm`) — these are tracked in `tests/regression/baseline_speedups.yaml` starting PR-00.

### §0.3 Bugs observed in live runs (each addressed by a specific PR)

| # | Bug | Evidence | Fix PR |
|---|---|---|---|
| 1 | **HIP commandment silent no-op** — `COMMANDMENT.md` IS generated in preprocess step 7, but the homogeneous runner ignores it and uses raw `test_command` from `--task` instead. FULL_BENCHMARK never runs for HIP per-round. | Audit §3 note, line 77 vs homo runner source | PR-24 (Jinja commandment + `run_pipeline` universal enforcement) |
| 2 | **`target_code=0B` in sub-agent retrieval** — Sub-agents passed non-existent relative paths to retriever → `_read_target_code` returned empty → `code_sim=0` → KB ranking was random. Fixed in PR #166 commit 8578d62b via `write_task_file` absolute-path normalization. | Historical incident in `/data/sapmajum/triton_runs/fused_qkv_rope_*.log` | Already fixed on `refactor-test`; PR-10 carries it forward |
| 3 | **Patch apply failures** — `save_and_test.apply_patch` failed on ~30% of agent-claimed speedups due to patch-lineage mismatches. Fixed in PR #161 with `git apply --3way` fallback. | Historical in 20+ round logs | Already fixed on `refactor-test`; PR-00 carries it forward |
| 4 | **Retriever bottleneck-mismatch penalty** — `fused_rms_fp8` 2.23x peak KB entry was outranked by irrelevant matches because runtime profiling reclassified bottleneck type → bottleneck-mismatch penalty promoted wrong entries. Fixed by removing the penalty and switching to code-similarity-first ranking. | Historical in `/data/sapmajum/triton_runs/fused_rms_fp8_*_memon_*.log` | Already fixed on `refactor-test`; PR-70 (CrossSessionMemoryAnalysisAgent) replaces remaining regex-based scoring |
| 5 | **Baseline drift / stale KB** — When docker/framework versions change, stored baselines become stale; identical-code retrievals return patches that no longer apply cleanly or suggest strategies that no longer optimize. | Historical in `fused_mxfp4_quant_moe_sort` runs after aiter upgrade | Three-layer mitigation (no version-hash fingerprint needed): (a) `save_and_test --3way` fallback (PR #161, already merged) handles patch-lineage mismatches at apply time; (b) structured insights file (PR-70) gives the subagent full KB content (baseline_code, winning_diff, profiler traces) so it can detect staleness from content patterns — e.g., "stored profile says compute-bound, current profile says memory-bound → strategy may not transfer"; (c) docker image pinning in nightly CI prevents env drift in regression runs. An earlier draft of this plan added a `baseline_fingerprint` hash field; removed because it was not load-bearing (see §15 Issue 4). |
| 6 | **HIP never writes to KB** — `record_optimization_outcome` is called in hetero path only; homo path never writes to cross-session memory. | Audit §12 "Confident (grepped AND observed)" | PR-40 (`run_pipeline` unified) — KB write is mode-agnostic |
| 7 | **`_PARAM_PATTERNS` language leak in `formatter.py`** — 10 Triton/HIP-hardcoded regex patterns in core memory code. | `src/minisweagent/memory/cross_session/formatter.py:35-50` | PR-71 moves patterns to `kernel_languages/<lang>/memory_hints.md` |
| 8 | **16+ language-branching sites** — Language-specific `if kernel_type == "triton"` scattered across 16 files. | Audit §9 inventory table | PR-10 (normalizer shim) + PR-42 (default flip) + CI gate `check_language_leaks.py` |
| 9 | **`DefaultAgent` import fallback branches in `parallel_helpers.py`** — 844-LoC parallel runner has conditional imports that will break after PR-50 agent collapse. | `run/utils/parallel_helpers.py:530` | PR-52 splits `parallel_helpers.py` and drops the fallback branches |
| 10 | **Identical-duplicate files** — 7 file pairs with content 0-90% overlap (most extreme: `debug_runtime.py` identical in two locations). | Verified via `diff -q` during audit | PR-00 deletes 6 stubs/identical-dups immediately |

### §0.4 Summary of refactor-test branch state (as of planning)

- `refactor-test` branch cut from `origin/main @ d7b880c3`.
- **Code size (verified by `wc -l src/minisweagent/**/*.py`)**: 34,301 LoC across 143 Python files. Target after refactor: ~26,000 LoC (−8,000 net).
- Patch-apply fix (PR #161) already merged to main.
- Cross-session memory + RAG + path-normalization fixes (PR #166) landed.
- `memory/cross_session/knowledge_base.json` at version 7 with 60+ experiences.
- CI already runs pylint + ruff clean on both PR #161 and #166.
- Docker image `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` is canonical for Triton runs.
- GPUs 0-3 available for experiments; GPUs 4-7 not available in current environment. This caps parallelism for nightly regression and ablation.
- AgentKernelArena coupling: JSON/YAML only (`final_report.json`, `task_result.yaml`) — verified by inspection. No log-marker parsing in AKA. AKA uses `GEAK_CONFIG_NAME` env var (e.g., `heterogeneous_memory_on`) not `--heterogeneous` CLI flag — verified with `rg` across `/data/sapmajum/AgentKernelArena/`. AKA will be updated in parallel with this refactor (per user direction; tracked in PR-09b).

### §0.5 Current end-to-end flow (the baseline the refactor must preserve semantically)

Reconstructed from the two live runs in §0.1 plus code reading. Every refactor PR in §8 is named relative to which arrow or box in this diagram it rewrites. **The refactor does not invent a new flow** — it unifies the two split paths and makes the phase boundaries explicit.

#### §0.5 (a) — BEFORE the refactor (what lives on `origin/main @ d7b880c3`)

```
                       ┌──────────────────────┐
                       │  geak -t '...prompt' │  pyproject.toml → minisweagent.run.mini:app
                       └──────────┬───────────┘   pyproject.toml actually ships **10** CLI scripts today:
                                  │       ∙ geak (primary; NL prompt)
                                  │       ∙ geak-orchestrate (2nd CLI; homo unsupported — raises NotImplementedError)
                                  │       ∙ geak-preprocess, kernel-profile, commandment, validate-commandment,
                                  │         run-harness, baseline-metrics, codebase-context, resolve-kernel-url,
                                  │         task-generator (9 auxiliary :main entry points from preprocess + tools modules)
                                  │       ALL 9 auxiliary entries are DELETED in PR-1 per user direction; only `geak` survives.
                                  │
             configure_if_first_time → YAML merge (mini_kernel_strategy_list.yaml + geak.yaml)
                                  │
                       parse_pipeline_params (LLM #1)    ← heterogeneous true/false/null, max_rounds
                                  │
                       parse_task_info      (LLM #2)    ← kernel_type, repo, gpu_ids, num_parallel
                                  │
                       _normalize_kernel_type           ← 16+ scattered language-branching sites
                                  │
      ┌─────────────── run_preprocessor — SHARED 7 STEPS ────────────────────┐
      │ 1 resolve_url+split           (resolve_kernel_url.py, 548 LoC)        │
      │ 2 CODEBASE_CONTEXT.md         (codebase_context.py, 499 LoC)          │
      │ 3 discovery+harness+UTA       (ATD MCP + harness_utils.py, 1660 LoC!) │
      │ 4 harness_validation          (harness_utils, Triton-only today)      │
      │ 5 Metrix profile              (kernel_profile.py, Metrix MCP)         │
      │ 6 baseline_metrics.json       (baseline.py build_baseline_metrics)    │
      │ 7 COMMANDMENT.md              (commandment.py, 530 LoC of branching)  │
      │   ※ HIP: steps 2-4 SKIPPED silently (bug #1 in §0.3)                  │
      └──────────────────────────────────┬───────────────────────────────────┘
                                         │
                  auto-detect if heterogeneous is None:
                     discovery.kernel.type == "triton" ? → TRUE : FALSE
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                                                                  │
  heterogeneous=True (Triton)                         heterogeneous=False (HIP/other)
   run_orchestrator →                                 run_homogeneous_agent
   run_heterogeneous_orchestrator.py (493 LoC)        homogeneous_agent.py (196 LoC)
                                                             │
   ▸ Exploration phase                                 ▸ inject KB into task_content (27608 chars
     — LLM reads kernel; no orchestration tools          observed; assemble_memory_context)
                                                        │
   ▸ for r in 1..5 (default):                        ▸ ParallelAgent.run_parallel (609 LoC)
     ∙ LLM→generate_tasks(...)                         ∙ for i in 0..N: parallel_{i}.md (same body)
         heterogeneous/task_generator.py (997 LoC)     ∙ ThreadPoolExecutor(N)  ← DUPLICATE of run_pool
         + heterogeneous/prompts.py (379 LoC           ∙ StrategyInteractiveAgent(model, env).run(task)
            Triton-biased SYSTEM_PROMPT)                 (same agent class as hetero path)
         + heterogeneous/workload_guidance.py (307)    ∙ _select_best_from_parallel_runs
         + heterogeneous/result_scanning.py (210)         (SelectPatchAgent LLM picks winner)
     ∙ LLM→dispatch_tasks(tasks)                       ∙ writes final_report.json
         → run_pool(M tasks, N GPUs; queued if M>N)
         → for each: StrategyInteractiveAgent(...)    ▸ NO FULL_BENCHMARK per-round verify
           uses 4-layer chain:                        ▸ NO record_optimization_outcome call
             DefaultAgent(611) → InteractiveAgent(181)   → HIP never writes to memory.db!
             → StrategyAgent(156) → StrategyInteractiveAgent(50)
           tool set: bash, str_replace_editor,
                     save_and_test, submit,
                     strategy_manager, profile_kernel,
                     query, optimize, sub_agent,
                     resolve_kernel_url, baseline_metrics
                     ↑ resolved by tools_runtime.py:120-162
                       with an `allowed:set[str]` that
                       the 4-layer chain decides
     ∙ LLM→collect_results(...)
     ∙ post_round_evaluate
         → SETUP + CORRECTNESS + FULL_BENCHMARK + PROFILE
         → round_N_evaluation.json

   ▸ finalize_run / auto_finalize
   ▸ merge_round_evaluation_into_final_report
   ▸ record_final_outcome → memory.db (SQLite)
                                         │
                      ┌──────────────────┴──────────────────┐
                      │ Memory retrieve (BEFORE every run)  │
                      │ assemble_memory_context(kernel,     │
                      │   profiler_summary, bottleneck):    │
                      │  ├── cross_session/retriever.py     │
                      │  │    — top-k by code similarity    │
                      │  │    + Jaccard-on-normalized-lines │
                      │  └── cross_session/formatter.py     │
                      │       — regex-extract num_warps,    │
                      │         tl.constexpr_dtype,         │
                      │         hipLaunchKernelGGL (LEAK!)  │
                      │       — dump up to 40 KB raw into   │
                      │         task body (_MAX_CONTEXT_FULL)│
                      │                                     │
                      │ + RAG MCP (query, optimize tools)   │
                      │   per PR #90                        │
                      └─────────────────────────────────────┘
```

**Key asymmetries this diagram surfaces** (each becomes a PR in §8):

| Asymmetry / Issue | Fix PR |
|---|---|
| `heterogeneous is None` auto-detects pipeline by language string comparison | PR-10 / PR-41: `--mode {fixed,planned,auto}` controls pipeline; language becomes a `KernelLanguage` data object |
| HIP silently skips preprocessor steps 2-4 (bug #1) | PR-20: phases make the skip explicit; PR-24 Jinja commandment enforces step coverage |
| Heterogeneous path does FULL_BENCHMARK per round; homo path does not | PR-24: `run_pipeline` universally enforces COMMANDMENT per round for both modes |
| Heterogeneous writes to `memory.db`; homo does NOT (`record_optimization_outcome` never called) | PR-40: `run_pipeline(ctx, mode)` calls `record_optimization_outcome` regardless of mode |
| Two CLI entry points (`geak`, `geak-orchestrate`); two pipeline entry fns (`run_orchestrator`, `run_homogeneous_agent`); duplicated `run_pool` / `ThreadPoolExecutor` logic | PR-53: single `cli.py`; PR-52 single `run/pool_runner.py`; homo becomes `mode="fixed"` |
| Memory injection uses 40 KB raw unstructured dump from `formatter.py`; regex extraction is Triton/HIP-hardcoded (leak) | PR-70: `CrossSessionMemoryAnalysisAgent` writes ~5-25 KB **structured** `cross_session_memory_insights.md` (analysis + ranked recommendations + full reference KB entries with diffs/profiler/dead-ends); PR-71 moves regex patterns to `kernel_languages/<lang>/memory_hints.md` |
| 4-layer agent inheritance (`DefaultAgent` → `InteractiveAgent` → `StrategyAgent` → `StrategyInteractiveAgent`) | PR-50 collapses to `OptimizationAgent`; PR-49 rewrites tests first |
| `tools_runtime.py` tool selection decided implicitly by the 4-layer chain | PR-40 moves decision to `run/unified.py::_resolve_tools(ctx, mode)`; CI gate `check_tool_resolve_single_site.py` enforces |
| `heterogeneous/task_generator.py` (997 LoC) + `prompts.py` (379) + `workload_guidance.py` (307) are Triton-biased | PR-31 unified `run/task_generator.py` (with A/B equivalence gate); PR-51 deletes the directory |

#### §0.5 (b) — AFTER the refactor (end state after PR-73)

```
                        ┌────────────────────────────┐
     geak -t '...'      │  src/minisweagent/cli.py   │  pyproject.toml → minisweagent.cli:app
     geak translate ──► │  (THE ONE ENTRY FILE)      │   — enforced by check_one_cli_file.py
     geak add-language  │  @app.command()s:          │
     geak test-language │    optimize / translate /  │
     geak resume        │    add-language / resume   │
                        └──────────────┬─────────────┘
                                       │
             configure_if_first_time → YAML merge (same as before)
                                       │
             parse_pipeline_params (LLM #1)    ← mode: auto (default) / fixed / planned / translate
             parse_task_info       (LLM #2)    ← kernel_path, target_language (optional), gpu_ids, num_parallel
                                       │
             registry.detect_best(path)        ← one detection site; returns KernelLanguage instance
             ctx.language = KernelLanguage     ← detected source language (data object, no strings)
             ctx.target_language = ctx.language IF --target-language not passed
                                   registry.detect_best_by_name(user_arg) OTHERWISE
             (default: target == language → Translation phase SKIPPED for normal optimize runs)
                                       │
   ┌────────── PreprocessOrchestrator.run() — preprocess/orchestrator.py (~30 LoC) ──────────┐
   │                                                                                          │
   │  ┌── Translation phase (CONDITIONAL: only when target_language ≠ language;
   │  │                      default target_language == language, so this phase SKIPPED) ─┐  │
   │  │   preprocess/phases/translation.py                                │          │
   │  │   a. run source harness → golden tensors (reference)                      │          │
   │  │   b. TranslationLoop.loop(max_attempts=3, verify_fn=golden_match)         │          │
   │  │        — subagents/translation/translator.py                              │          │
   │  │        — each attempt: OptimizationAgent.run(compose_task_body(           │          │
   │  │            mode="translate", src_lang=..., tgt_lang=...,                │          │
   │  │            hints=_translation/<src>_to_<tgt>.md OR _fallback.md))         │          │
   │  │        — verify: tensor allclose(src_out, tgt_out, atol=1e-5)             │          │
   │  │   c. validate_translation_performance (0.5x fail / 0.8x warn per pair)    │          │
   │  │   d. ctx.kernel_path = translated_path;  ctx.language = target_language   │          │
   │  │   ※ if translate_only: return ctx (geak translate exits here)             │          │
   │  └───────────────────────────────────────────────────────────────────────────┘          │
   │                                       │                                                  │
   │  ┌── Discovery phase ────────────────────────────────────────────────────────┐           │
   │  │   preprocess/phases/discovery.py                                         │           │
   │  │   ∙ resolve_kernel_url()  (unchanged)                                     │           │
   │  │   ∙ codebase_context.py  → CODEBASE_CONTEXT.md                            │           │
   │  │   ∙ test_discovery (ATD MCP)  → ctx.discovery                             │           │
   │  └───────────────────────────────────────────────────────────────────────────┘           │
   │                                       │                                                  │
   │  ┌── Harness phase ─────────────────────────────────────────────────────────┐           │
   │  │   preprocess/phases/harness.py                                            │           │
   │  │   ∙ harness_decision:                                                     │           │
   │  │       if no user tests: UnitTestAgent.run()  [SubagentBase.run()]         │           │
   │  │                          — subagents/preprocess/unit_test_agent.py         │           │
   │  │   ∙ HarnessBuilder.run()  [SubagentBase.run()]                            │           │
   │  │       — subagents/preprocess/harness_builder.py                           │           │
   │  │       — reads language.harness_template_path (Jinja) +                    │           │
   │  │              language.builder_hints_path                                  │           │
   │  │       — outputs harness.py conforming to universal contract               │           │
   │  │   ∙ validate_harness(path, language)  [kernel_languages/contract.py]      │           │
   │  │       — checks --correctness/--benchmark/--full-benchmark/--profile flags │           │
   │  │       — checks GEAK_RESULT_LATENCY_MS / GEAK_RESULT_SPEEDUP emit          │           │
   │  │       — ≥29/30 pass rate on fixture corpus required (PR-22)               │           │
   │  └───────────────────────────────────────────────────────────────────────────┘           │
   │                                       │                                                  │
   │  ┌── Baseline phase ────────────────────────────────────────────────────────┐           │
   │  │   preprocess/phases/baseline.py                                           │           │
   │  │   ∙ baseline_execute: run_harness(--correctness + --benchmark)            │           │
   │  │   ∙ profile (Metrix MCP)  → profile.json                                  │           │
   │  │   ∙ build_metrics  → baseline_metrics.json                                │           │
   │  └───────────────────────────────────────────────────────────────────────────┘           │
   │                                       │                                                  │
   │  ┌── Explore phase ─────────────────────────────────────────────────────────┐           │
   │  │   preprocess/phases/explore.py                                              │           │
   │  │   ∙ render_commandment  [Jinja: language.commandment_template_path]       │           │
   │  │   ∙ validate_commandment(path, language)  [same contract module]          │           │
   │  │   ∙ KernelAnalysisAgent.run()  [SubagentBase.run()]                       │           │
   │  │       — subagents/preprocess/kernel_analysis.py                           │           │
   │  │       — emits [A]-[D] rubric markdown (Primitives / Shapes / Hotspots /   │           │
   │  │         Attack Surfaces)                                                  │           │
   │  │       → ctx.kernel_analysis_md                                            │           │
   │  └───────────────────────────────────────────────────────────────────────────┘           │
   └──────────────────────────────────────┬───────────────────────────────────────────────────┘
                                          │
   ┌── run_pipeline(ctx, mode: {fixed|planned|auto|translate}) — run/unified.py ──────────┐
   │                                                                                       │
   │   _resolve_tools(ctx, mode)                                                           │
   │     — SINGLE tool-resolution site (CI gate check_tool_resolve_single_site.py)         │
   │     — starts from ctx.language.tool_set; adds mode-specific tools;                    │
   │       applies CLI overrides last                                                      │
   │     — hands final list to OptimizationAgent (agent never re-decides)                  │
   │                                                                                       │
   │   for r in 1..ctx.max_rounds:                                                         │
   │     retrieve = cross_session.retrieve(ctx)                                            │
   │       — top-k by code similarity (retriever ranking unchanged from today)             │
   │                                                                                       │
   │     IF GEAK_USE_CROSS_SESSION_MEMORY=1 (default; opt-out with =0):                    │
   │       insights_path = CrossSessionMemoryAnalysisAgent(ctx.language).run(              │
   │                   target_code=current_kernel, target_profile=current_profile,         │
   │                   retrieved=top_k,                                                    │
   │                   out_path=ctx.artifacts_dir/"cross_session_memory_insights.md")      │
   │         — subagents/memory/cross_session_memory_analysis.py  [SubagentBase.run()]     │
   │         — ALWAYS this path. No fast/slow branching.                                   │
   │         — Emits cross_session_memory_insights.md (~5-25 KB). Content (not just a      │
   │           distilled summary — FULL reference info for each retrieved experience):     │
   │            ∙ Analysis Summary (applicability assessment, priority ranking)           │
   │            ∙ Top Recommended Strategies (ranked, with reasoning + concrete patterns) │
   │            ∙ Avoid / Known Dead-Ends (with regression evidence from KB)              │
   │            ∙ Reference: Full KB Entries (for each retrieved experience):              │
   │                 - baseline_code (full source of the KB kernel)                       │
   │                 - winning_diff (unified patch that produced the speedup)              │
   │                 - profiler_before / profiler_after (hotspots, roofline)              │
   │                 - strategies_tried (with per-strategy speedup/regression)            │
   │                 - dead_ends (with reasons)                                           │
   │                 - code_sim score + subagent-derived staleness signal                │
   │            ∙ none_applicable: true/false (subagent may say "nothing transfers")      │
   │         — file contents read into task_body by compose_task_body()                   │
   │     ELSE IF GEAK_USE_CROSS_SESSION_MEMORY=0:                                          │
   │       no insights file written; agent explores from scratch                           │
   │                                                                                       │
   │     tasks = build_tasks_for_round(ctx, context, mode)                                 │
   │       — uses compose_task_body(language, mode=mode, ctx, context)                     │
   │       — mode="fixed":   one prompt × N copies (was homogeneous)                       │
   │       — mode="planned": N planned strategies via task_generator (was heterogeneous)   │
   │       — mode="auto":    controller picks a {fixed, planned} mixture per round         │
   │                                                                                       │
   │     results = pool_run(tasks, tools) — run/pool_runner.py (ONE path, both modes)      │
   │       — for each: OptimizationAgent(config, model, env, tools).run(task_body)         │
   │         (THE one agent class; no inheritance chain)                                   │
   │                                                                                       │
   │     eval = evaluate_round_best                                                        │
   │       — SETUP + CORRECTNESS + FULL_BENCHMARK + PROFILE for EVERY round in both modes  │
   │       — round_N_evaluation.json                                                       │
   │                                                                                       │
   │     if eval.verified_speedup > best.speedup: best = eval                              │
   │                                                                                       │
   │   if best.verified:                                                                   │
   │     record_optimization_outcome(ctx, best)   ← called for BOTH modes now (bug #6 fix) │
   │                                              ← KB write is mode-agnostic               │
   │                                                                                       │
   │   finalize_report(ctx, best) → final_report.json                                      │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │ Memory retrieve ALWAYS uses:                │
                   │   ∙ ctx.language.kb_namespace (clean partition)│
                   │   ∙ code-similarity top-k (retriever math)  │
                   │   ∙ RAG MCP (query, optimize tools)         │
                   │   ∙ KB write: classify_kernel_category      │
                   │     (moved to memory/cross_session/__init__)│
                   └─────────────────────────────────────────────┘

  CI gates active everywhere:
  ├── check_one_cli_file.py                    FAIL on Typer outside cli.py
  ├── check_language_leaks.py                  FAIL on kernel_type=="X" outside kernel_languages/
  ├── check_subagent_location.py               FAIL on SubagentBase outside subagents/
  ├── check_subagent_base_contract.py          FAIL if subagent overrides both run/loop
  ├── check_no_agent_inheritance.py            FAIL if anyone subclasses legacy agent classes
  ├── check_tool_resolve_single_site.py        FAIL on ToolRuntime(allowed=..) outside _resolve_tools
  ├── harness_corpus_gate.py                   ≥29/30 HarnessBuilder fixtures pass before PR-42 flip
  ├── ab_task_generator.py                     PR-31 A/B equivalence gate before PR-42 flip
  └── check_baseline_speedups.py               Nightly §0.2 kernel regression
```

#### §0.5 (c) — Side-by-side diff of what changed

| Before | After | Cause (PR) |
|---|---|---|
| **10 CLI entry points** today: `geak`, `geak-orchestrate`, `geak-preprocess`, `kernel-profile`, `commandment`, `validate-commandment`, `run-harness`, `baseline-metrics`, `codebase-context`, `resolve-kernel-url`, `task-generator` | **1 CLI file** (`cli.py`) with `@app.command()` subcommands; **all 9 auxiliary entry points DELETED** per user direction — only `geak -t "..."` (+ subcommands `add-language`, `test-language`, `translate`, `resume`, `optimize`) survives | PR-1 |
| 7 flat preprocess steps, HIP silently skips 2-4 | 4 named phases (Discovery, Harness, Baseline, Explore) + 1 conditional (Translation); skips are explicit | PR-20, PR-24, PR-61 |
| 16+ language-branching sites in core code | 0 in core; all behavior in `KernelLanguage` object | PR-10, PR-11, PR-71, CI gates |
| 4-layer agent inheritance | 1 `OptimizationAgent` class (no inheritance) | PR-49, PR-50 |
| Tool set decided by implicit 4-layer chain | `run/unified.py:_resolve_tools(ctx, mode)` single site | PR-40 (enforced by CI PR-50) |
| Hetero writes KB, homo doesn't | Both modes write via `run_pipeline` + `record_optimization_outcome` | PR-40 |
| Hetero does FULL_BENCHMARK per round, homo doesn't | Both modes do FULL_BENCHMARK per round | PR-24 |
| 40 KB raw unstructured KB dump injected pre-agent | Single path: `CrossSessionMemoryAnalysisAgent` writes `cross_session_memory_insights.md` (structured: analysis + ranked recs + FULL reference KB entries with baseline code, winning diffs, profiler traces, dead-ends); gated by `GEAK_USE_CROSS_SESSION_MEMORY` flag | PR-70, PR-73 |
| `_PARAM_PATTERNS` Triton/HIP regex in `memory/formatter.py` | Moved to `kernel_languages/<lang>/memory_hints.md` | PR-71 |
| `task_generator.py` (997 LoC, Triton-biased) + `workload_guidance.py` + `result_scanning.py` + `prompts.py` (Triton SYSTEM_PROMPT) | `run/task_generator.py` (~700 LoC, language-agnostic) + per-language `planner_strategy_hints.md` | PR-31 (A/B-gated before PR-42) + PR-51 |
| `ParallelAgent.run_parallel` ThreadPool branch + `run_pool` hetero function (duplicated logic) | `run/pool_runner.py` single path | PR-52 |
| `geak add-language X` does not exist | `geak add-language X` scaffolds 7 files under `kernel_languages/X/`; zero core edits for new languages | PR-07 |
| `geak translate` does not exist | `geak translate --source ... --target-language ...` via Translation phase + `TranslationLoop` subagent | PR-60 → PR-63 |
| `baseline_fingerprint` does not exist | **Left as-is — not needed.** Earlier drafts added a version hash to detect stale KB entries. Removed after audit: the subagent reasons about staleness from KB content (profiler traces, code patterns, winning diffs), `save_and_test --3way` (PR #161) handles patch-apply mismatches, and docker pinning in nightly CI prevents env drift. Adding a hash was speculative machinery with no load-bearing consumer. See §15 Issue 4. | — |
| `classify_kernel_category` in dead-looking `memory/cross_session_memory.py` (but actually imported 3×) | Moved to `memory/cross_session/__init__.py` + 3 callsites updated + 1-release deprecation shim | PR-19 |

**What the refactor leaves unchanged** (already correct):
- `geak -t "..."` invocation syntax — backward compatible
- `configure_if_first_time` YAML merge
- `parse_pipeline_params` + `parse_task_info` NL-to-config extraction (2 LLM calls at startup)
- Core tool ecosystem (`save_and_test`, `str_replace_editor`, `bash`, `profile_kernel`, MCP bridges)
- Cross-session memory retrieval math (code similarity + scaled success)
- RAG MCP (`query`, `optimize` tools) per PR #90 — integration unchanged
- AKA invocation contract (`geak --kernel-url ... --test-command ... -t ...`) — AKA updated separately per PR-09b

### §0.6 How GEAK kernels are actually tested (AgentKernelArena + docker)

AgentKernelArena (AKA) at `/data/sapmajum/AgentKernelArena/` is the harness that runs GEAK on all 18 Triton kernels and 16 HIP kernels in isolated workspaces. Every kernel run in §0.2 came through this harness. The refactor must continue to work with AKA without changes on the AKA side.

#### The 18 Triton kernels (canonical set)

| # | Kernel | Level | Config | @triton.jit |
|---|---|---|---|---|
| 1 | `llama_ff_triton` | L1 | 3 | direct |
| 2 | `fused_append_shared_experts` | L1 | 18 | direct |
| 3 | `moe_routing_sigmoid_top1` | L1 | 34 | direct |
| 4 | `mla_decode` | L1 | 320 | wrapper (aiter) |
| 5 | `ff_backward` | L1 | 4 | direct |
| 6 | `refk_identity` | L1 | self-contained | direct |
| 7 | `refk_fp8_blockwise_mm` | L1 | self-contained | direct |
| 8 | `fast_rms_layernorm` | L2 | 1 | direct |
| 9 | `topk` | L2 | 80 | direct |
| 10 | `lean_atten_paged` | L2 | 7 | direct |
| 11 | `rope` | L2 | 6480 | direct |
| 12 | `gemm_a16w16_atomic` | L3 | 13 | direct |
| 13 | `gemm` | L3 | 13 | wrapper (aiter) |
| 14 | `fused_qkv_rope` | L3 | 1200 | direct |
| 15 | `fused_mxfp4_quant_moe_sort` | L3 | 24 | wrapper (aiter) |
| 16 | `fused_moe_mxfp4` | L3 | 15 | direct |
| 17 | `fused_qk_rope_cache_mla` | L3 | 128 | direct |
| 18 | `fused_rms_fp8` | L3 | 25 | direct |

"Wrapper" kernels import Triton kernels from aiter submodules; GEAK detects them via import-following.

#### The 16 HIP kernels (canonical set, from `config_geak_hip.yaml`)

- L1 hip2hip (5): `ball_query`, `knn`, `matrix_multiplication`, `silu`, `three_nn`
- L2 hip2hip (7): `assign_score_withk`, `furthest_point_sample`, `gather_points`, `points_in_boxes`, `roiaware_pool3d`, `roipoint_pool3d`, `three_interpolate`
- L3 rocPRIM (4): `block_radix_rank`, `device_binary_search`, `device_merge_sort`, `device_search_n`

#### How a single run actually looks (verified command)

For each kernel, AKA's `launch_agent.py` (for `geak_v3_triton` or `geak_v3`) invokes:

```bash
geak --kernel-url <kernel.py> \
     --test-command 'python3 <harness.py>' \
     --gpu-ids <ids> \
     --num-parallel <N> \
     --yolo \
     --exit-immediately \
     -t <task_prompt.md> \
     -o <logs_dir>
```

Inside the docker container (`geak-agent-<user>`), GEAK then runs the §0.5 flow, writes `final_report.json` and `round_N_evaluation.json` into `<logs_dir>`, and AKA reads those JSON files directly (no re-benchmarking).

#### Parallel execution (verified: 2 streams × 4 GPUs)

For full-benchmark runs, AKA splits kernels across two 4-GPU streams via `scripts/run_geak_triton.sh`:

```bash
# Stream A: GPUs 0-3, half the kernels (indices 0, 2, 4, ...)
docker exec -e GEAK_GPU_IDS=0,1,2,3 <container> python3 main.py --config_name .tmp_config_stream_a.yaml
# Stream B: GPUs 4-7, the other half (indices 1, 3, 5, ...)
docker exec -e GEAK_GPU_IDS=4,5,6,7 <container> python3 main.py --config_name .tmp_config_stream_b.yaml
```

Both streams run concurrently inside the same container. A single kernel run with `--num-parallel 4` fans out to 4 sub-agents on the 4 GPUs of that stream.

#### Required environment (docker)

```
Container:       geak-agent-<user>                     (from GEAK/scripts/run-docker.sh)
Base image:      lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x
Required ver:    rocm 7.0+, torch, triton 3.4+, aiter pinned commit 22122345c03991cb8026947b8df05e02f50d1f88
API key:         AMD_LLM_API_KEY (injected via docker exec -e)
GPU arch:        PYTORCH_ROCM_ARCH=gfx950 (MI300-class)
Defaults:        GEAK_MAX_ROUNDS=5, GEAK_MODEL=claude-opus-4.6,
                 GEAK_MODEL_ENSEMBLE=gpt-5.2,claude-opus-4.6,
                 GEAK_BENCHMARK_ITERATIONS=30
```

#### Memory on/off control (the ablation knob)

AgentKernelArena exposes `GEAK_CONFIG_NAME` to flip memory state:

- `heterogeneous_memory_off` — baseline runs (no KB read, no KB write)
- `heterogeneous_memory_on` — KB reads (retrieves top-k) + writes (successful patches go to KB)

§0.2 regression thresholds come from `heterogeneous_memory_on` runs. §11.6 nightly regression tests must run both modes and require `memory_on ≥ memory_off` per kernel.

#### Result harvesting (how we verify a run)

```bash
# Per-task result (AKA layer)
for f in workspace_*/run_*/*/task_result.yaml; do
  grep speedup_ratio "$f"
done

# Per-round GEAK internals
for d in workspace_*/run_*/*_logs/final_report.json; do
  python3 -c "
import json; d=json.load(open('$d'))
fb=(d.get('round_evaluation') or {}).get('full_benchmark') or {}
print(f'verified={fb.get(\"verified_speedup\",\"N/A\")}x benchmark={d.get(\"round_evaluation\",{}).get(\"benchmark_speedup\",\"N/A\")}x')
"
done
```

The regression gates in §11.6 parse these two fields: `verified_speedup` (from FULL_BENCHMARK) is authoritative; `benchmark_speedup` (from agent's in-loop timing) is advisory.

#### What the refactor changes vs AKA (none)

The refactor is internal to GEAK. AKA continues to invoke `geak --kernel-url ... --test-command ... -t ...` unchanged. Translations (`geak translate`) are additive subcommands — AKA does not depend on them until AKA is extended to benchmark translation workflows (separate future effort).

---

## North-star architecture

```
                       ┌───────────────────────────────────────────────┐
     geak -t "..."     │  src/minisweagent/cli.py  ← THE ONE ENTRY POINT│
     geak add-language │                                               │
     geak translate    │  detect language  →  KernelLanguage instance  │
     geak --mode {...} │  PreprocessOrchestrator(ctx).run()            │
     ────────────────► │  run_pipeline(ctx, mode | "translate")        │
                       │  write FinalReport                            │
                       └────────────┬──────────────────────────────────┘
                                    │
   ┌────────────────────────────────┼─────────────────────────────────────┐
   ▼                                ▼                                     ▼
┌─────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────┐
│ PreprocessOrch      │ │ KernelLanguage (data obj)    │ │ run_pipeline             │
│                     │ │ ├─ name                      │ │                          │
│ 11 steps:           │ │ ├─ file_extensions           │ │ for r in 1..K:           │
│  · resolve_kernel   │ │ ├─ system_prompt             │ │   tasks =                │
│  · codebase_ctx     │ │ ├─ optimization_prompt       │ │     compose_tasks(       │
│  · test_discovery   │◄┤ ├─ planner_strategy_hints    │ │       language,          │
│  · harness_builder* │ │ ├─ harness_template (j2)     │ │       mode={fixed|     │
│  · validate_harness │ │ ├─ commandment_template (j2) │ │         planned|trans},  │
│  · run_baseline     │ │ ├─ builder_hints             │ │       ...)               │
│  · profile          │ │ ├─ (no per-language command fields; │   results =              │
│  · build_metrics    │ │ │  commandment.j2 is single-src-of-truth) │ run_pool(tasks)      │
│  · render_cmdment   │◄┤ ├─ eval_env (pip/dockerfile) │ │   eval = evaluate(...)   │
│  · validate_cmdment │ │ ├─ kb_namespace              │ │   if win: kb_write       │
│  · kernel_analysis* │ │ ├─ translation_hints {dst→md}│ │                          │
│                     │ │ └─ detect(path) → float      │ │                          │
│  * = LLM subagent   │ └──────────────────────────────┘ │                          │
│  others = deterministic                                │                          │
└─────────────────────┘                                  └────────────┬─────────────┘
                                                                      │
                                      ┌───────────────────────────────┴──────┐
                                      ▼                                      ▼
                    ┌─────────────────────────────────┐   ┌────────────────────────────────┐
                    │ OptimizationAgent               │   │ SubagentBase (and 6 subclasses)│
                    │  (MAIN agent; standalone class) │   │   PEER of OptimizationAgent,   │
                    │                                 │   │   NOT a subclass of it.        │
                    │ .run(task_body: str)            │   │                                │
                    │  never imports KernelLanguage   │   │  each subagent COMPOSES        │
                    │  never branches on language     │   │  a short-lived OptimizationAg  │
                    │  instantiated by:               │   │  via _make_optimization_agent()│
                    │   1. run_pipeline (main loop)   │   │                                │
                    │   2. SubagentBase._make_...     │   │ subclasses:                    │
                    │      (subagents composing it)   │   │  HarnessBuilder, KernelAnalysis│
                    └─────────────────────────────────┘   │  UnitTestAgent, ShapeFixer,    │
                                                          │  CrossSessionMemoryAnalysisAgt │
                                                          │  TranslationLoop (multi-round) │
                                                          └────────────────────────────────┘
```

**Principle**: TWO peer class families. Homogeneous vs heterogeneous is a **`mode`** parameter on `run_pipeline` + `compose_task_body` handled by the MAIN `OptimizationAgent`. Narrow, bounded, often-structured LLM tasks (harness synthesis, kernel analysis, KB synthesis, translation) are handled by subclasses of `SubagentBase` that COMPOSE `OptimizationAgent` when they need its tool loop. Adding HIP, Triton, FlyDSL, Metal, SYCL, CUDA is adding a `kernel_languages/<name>/` folder — nothing else.

---

## §1 — Full file audit (every file, with a verdict)

This is the input to the refactor. Each file in `src/minisweagent/` gets one of 5 verdicts: **KEEP** (stays as-is), **KEEP+rename** (content stays, file moves/renames), **SPLIT** (too big, gets broken up), **MERGE** (consolidate with another file), **DELETE** (dead code / mini-swe-agent heritage / duplicate / unused).

**Current total (verified via `wc -l`)**: 34,301 LoC across 143 Python files.
**Target total after refactor**: ~25,000 LoC across ~90 Python files.
**Reduction**: ~9,000 LoC / ~50 files.

### Dead code / mini-swe-agent heritage — DELETE (~2,262 LoC, 16 files)

| File | LoC | Reason | PR |
|---|---|---|---|
| `run/extra/swebench.py` | 266 | mini-swe-agent SWE-bench evaluator; nothing GEAK-related | PR-53 |
| `run/extra/swebench_single.py` | 79 | same | PR-53 |
| `run/extra/config.py` | 130 | same | PR-53 |
| `run/extra/utils/batch_progress.py` | 178 | same | PR-53 |
| `run/mini_extra.py` | 44 | Typer meta-dispatcher (only launches swebench cmds) | PR-53 |
| `run/hello_world.py` | 36 | mini-swe-agent tutorial | PR-00 |
| `run/inspector.py` | 212 | trajectory TUI; only used by dev, depends on `interactive_textual` | PR-53 |
| `agents/interactive_textual.py` | 450 | textual TUI; only used by inspector | PR-53 |
| `config/extra/swebench.yaml` | 230 | SWE-bench config | PR-53 |
| `config/extra/swebench_roulette.yaml` | 232 | SWE-bench roulette config | PR-53 |
| `models/extra/roulette.py` | 62 | model-roulette wrapper; only used by swebench | PR-53 |
| `environments/extra/bubblewrap.py` | 112 | bubblewrap sandbox; only in tests | PR-53 |
| `environments/extra/swerex_docker.py` | 47 | swerex container; only in tests | PR-53 |
| `environments/singularity.py` | 97 | singularity sandbox; only in tests | PR-53 |
| `models/anthropic_model.py` | 35 | wrapper to litellm anthropic; unused in prod paths | PR-53 |
| `models/test_models.py` | 52 | `DeterministicModel`; only used by unit tests | (move to `tests/fixtures/models.py`) PR-53 |

**Net delete**: ~2,262 LoC, 16 files.

### Exact duplicates / near-duplicates — MERGE or DELETE shim (~900 LoC waste)

**VERIFIED via `diff -q`, `rg` import search, and `Read`** (reviewer caught 2 misclassifications in an earlier draft — now fixed):

| Pair | Verified status | Action | PR |
|---|---|---|---|
| `debug_runtime.py` (root, 70) vs `run/preprocess/debug_runtime.py` (70) | `diff -q` = identical; neither imported from root | Delete root copy | PR-00 |
| `tools/resolve_kernel_url.py` (57) vs `run/preprocess/resolve_kernel_url.py` (548) | **NOT a shim** — `Read` confirmed it's a `ResolveKernelUrlTool` adapter class; **registered at `tools_runtime.py:147`** as the `resolve_kernel_url` tool. Wraps preprocess function into `{output, returncode}` contract. | **KEEP** as-is (tool adapter) | — (stays) |
| `tools/discovery_types.py` (7) | `Read` confirmed pure re-export stub; no adapter logic | Delete (callers import from preprocess) | PR-00 |
| `tools/validate_commandment.py` (7) | Pure re-export stub | Delete; move real 280-LoC version → `kernel_languages/contract.py` | PR-04 |
| `agents/shape_fixer_agent.py` (7) | Pure re-export stub | Delete (real is 241-LoC in preprocess) | PR-00 |
| `agents/unit_test_agent.py` (251) vs `run/preprocess/unit_test_agent.py` (252) | 83-line diff; both actively imported | Merge → `subagents/preprocess/unit_test_agent.py` | PR-21 |
| `run/preprocess/benchmark_parsing.py` (381) vs `run/postprocess/benchmark_parsing.py` (389) | 54-line diff; both imported | Merge → `run/benchmark_parsing.py` | PR-20 |
| `memory/cross_session_memory.py` (27) | **NOT stale** — exports `classify_kernel_category`; **verified imported at 3 production callsites**: `agents/heterogeneous/orchestrator.py:291`, `run/postprocess/results.py:175`, `run/postprocess/results.py:537`. Hits on every heterogeneous run + every postprocess. | **MOVE** function to `memory/cross_session/__init__.py` + update 3 callsites → **THEN** delete file | **PR-19 (new; before Phase 2)** |
| `memory/integration.py` (104) vs `memory/cross_session/__init__.py` (134) | Overlap; both imported | Merge `integration.py` → `cross_session/__init__.py`; delete `integration.py` | PR-21 |

**Net delete/merge**: ~850 LoC, 9 file reductions (revised down from 10 after keeping `tools/resolve_kernel_url.py`).

### Agent hierarchy — COLLAPSE (4 classes → 1)

Today we have a 4-layer inheritance chain that nobody outside GEAK understands:
```
DefaultAgent (611 LoC)
  └── InteractiveAgent (181 LoC)  [adds console rich output]
        └── StrategyAgent (156 LoC)  [adds strategy_manager hook]
              └── StrategyInteractiveAgent (50 LoC)  [adds notify_strategy_changed]
```
All four are used only at the final leaf. `InteractiveAgent.input_allowed=False` is set everywhere; nobody uses the interactivity. `StrategyAgent` just calls `notify_strategy_changed` on one callback.

**Collapse to ONE class**: `agents/optimization_agent.py` (~500 LoC) — has the step loop, tool calling, exception handling, and the strategy-manager callback. No inheritance chain. Exceptions (`Submitted`, `NonTerminatingException`, `LimitsExceeded`, `TerminatingException`) + `AgentConfig` dataclass stay; they're re-exported from `optimization_agent.py`.

| File | LoC | Verdict | PR |
|---|---|---|---|
| `agents/default.py` | 611 | MERGE into `optimization_agent.py` (keep step loop, exceptions, AgentConfig); drop the `DefaultAgent` name | PR-50 |
| `agents/interactive.py` | 181 | MERGE into `optimization_agent.py` (drop; input_allowed always False in GEAK) | PR-50 |
| `agents/strategy_agent.py` | 156 | MERGE into `optimization_agent.py` (the strategy-manager callback is ~30 LoC) | PR-50 |
| `agents/strategy_interactive.py` | 50 | MERGE into `optimization_agent.py` (rich console → always on) | PR-50 |
| `agents/homogeneous/homogeneous_agent.py` | 196 | DELETE (thin wrapper that `run_pipeline` subsumes) | PR-50 |
| `agents/heterogeneous/` (whole dir) | 2,908 | DELETE (logic moves to `run/unified.py`, `run/task_generator.py`, `kernel_languages/`, `tools/orchestrator_*.py`) | PR-51 |
| `agents/select_patch_agent.py` | 224 | KEEP (post-eval patch selector) | — |
| `agents/parallel_agent.py` | 609 | SPLIT (the homo-branch gets deleted in PR-52; remaining hetero-branch stays as `run/pool_runner.py`) | PR-52 |
| `agents/unit_test_agent.py` | 251 | MERGE with preprocess copy into `subagents/preprocess/unit_test_agent.py` | PR-21 |
| `agents/shape_fixer_agent.py` | 7 | DELETE stub | PR-00 |
| `agents/agent_spec.py` | 185 | KEEP (protocol + helpers) | — |
| `agents/__init__.py` | 1 | KEEP (re-exports) | — |

**Net**: 4-layer inheritance + 2 subdirectories → 1 file (`optimization_agent.py`) + preserved helpers. Saves ~2,100 LoC; gains readability.

### Preprocess monolith — SPLIT

| File | LoC | Verdict | PR |
|---|---|---|---|
| `run/preprocess/preprocessor.py` | 1386 | SPLIT into `preprocess/orchestrator.py` (100) + `preprocess/steps/step_NN_*.py` (11 files × ~100-200) | PR-20 |
| `run/preprocess/harness_utils.py` | 1660 | SPLIT: the runtime helpers stay as `run/harness_runtime.py` (~400); the template-generation logic moves into `kernel_languages/<lang>/harness.j2` + subagent (~0 LoC in Python) | PR-22 |
| `run/preprocess/commandment.py` | 530 | DELETE after PR-24 (replaced by `kernel_languages/<lang>/commandment.j2` + `preprocess/steps/step_10_render_commandment.py`) | PR-24 |
| `run/preprocess/codebase_context.py` | 499 | KEEP as a step (`preprocess/steps/step_02_codebase_context.py`, unchanged) | — |
| `run/preprocess/kernel_profile.py` | 471 | KEEP as a step | — |
| `run/preprocess/resolve_kernel_url.py` | 548 | KEEP as a step | — |
| `run/preprocess/baseline.py` | 300 | KEEP as a step | — |
| `run/preprocess/discovery_types.py` | 337 | KEEP as a shared type module | — |
| `run/preprocess/run_harness.py` | 338 | KEEP (harness execution wrapper) | — |
| `run/preprocess/benchmark_parsing.py` | 381 | MERGE with postprocess sibling into `run/benchmark_parsing.py` | PR-20 |
| `run/preprocess/testcase_cache.py` | 250 | KEEP | — |
| `run/preprocess/unit_test_agent.py` | 252 | MERGE with agents/ sibling → `subagents/preprocess/unit_test_agent.py` | PR-21 |
| `run/preprocess/shape_fixer_agent.py` | 241 | MOVE to `subagents/preprocess/shape_fixer.py` | PR-21 |
| `run/preprocess/validate_commandment.py` | 280 | MOVE to `kernel_languages/contract.py` (generalize to per-language) | PR-04 |
| `run/preprocess/context.py` | 113 | MERGE into `run/context_types.py` | PR-11 |
| `run/preprocess/debug_runtime.py` | 70 | KEEP (canonical copy) | — |
| `run/preprocess/repo_paths.py` | 27 | KEEP | — |
| `run/preprocess/config_loader.py` | 38 | KEEP | — |
| `run/preprocess/INSTRUCTIONS.md` | 700 | KEEP (documentation) | — |
| `run/preprocess/config/mini_unit_test_agent.yaml` | 344 | MOVE under `subagents/preprocess/configs/` | PR-21 |
| `run/preprocess/config/mini_shape_fixer.yaml` | 65 | same | PR-21 |

### Run/ flat files — rename, split, consolidate

| File | LoC | Verdict | PR |
|---|---|---|---|
| `run/mini.py` | 629 | MERGE into `cli.py` (slim to ~250); orchestration logic to `run/unified.py` + `preprocess/orchestrator.py` | PR-06, PR-42, PR-53 |
| `run/orchestrator.py` | 288 | DELETE (redundant `geak-orchestrate` CLI; merge any needed resume-from-preprocess logic into `cli.py`) | PR-53 |
| `run/github_issue.py` | 91 | DELETE (GitHub issue import never used in GEAK) | PR-53 |
| `run/pipeline_helpers.py` | 915 | SPLIT 4 ways: `run/model_factory.py` (~250), `run/env_factory.py` (~200), `run/context_injection.py` (~200), `run/agent_filter.py` (~100) + ~150 dead helpers deleted | PR-20 |
| `run/pipeline_types.py` | 170 | RENAME → `run/context_types.py` (merge in `preprocess/context.py`) | PR-11 |
| `run/task_file.py` | 567 | KEEP as library | — |
| `run/dispatch.py` | 483 | KEEP as library | — |
| `run/utils/parallel_helpers.py` | 844 | SPLIT: runner logic → `run/pool_runner.py` (~400), thread-safe logger → `run/pool_logger.py` (~150); remove DefaultAgent/InteractiveAgent import fallback branches | PR-52 |
| `run/utils/config_editor.py` | 673 | KEEP (config mutation helpers; could simplify later but not in-scope) | — |
| `run/utils/generated_artifacts.py` | 457 | KEEP | — |
| `run/utils/task_parser.py` | 404 | KEEP | — |
| `run/utils/prompts.py` | 121 | MERGE into `kernel_languages/<lang>/*.md` + `run/unified.py` | PR-30 |
| `run/utils/save.py` | 76 | KEEP | — |
| `run/utils/git_safe_env.py` | 42 | KEEP | — |
| `run/postprocess/evaluation.py` | 721 | KEEP | — |
| `run/postprocess/results.py` | 561 | KEEP | — |
| `run/postprocess/benchmark_parsing.py` | 389 | MERGE with preprocess sibling → `run/benchmark_parsing.py` | PR-20 |

### Tools, models, memory, environments — mostly KEEP

| Bucket | Verdict |
|---|---|
| `tools/save_and_test.py` (1095) | KEEP (patch apply + benchmark; core of the agent loop) |
| `tools/profiling_tools.py` (798) | KEEP |
| `tools/editor_tool.py` (765) | KEEP |
| `tools/strategy_manager.py` (742) | KEEP (the strategy-list JSON store) |
| `tools/mcp_bridge.py` (354) + `tools/mcp_client/` | KEEP (MCP is in use) |
| `tools/rag_postprocessor.py` (132) | KEEP |
| `tools/tools_runtime.py` (280) | KEEP (central tool registry runtime) |
| `tools/baseline_metrics_tool.py` (73) | KEEP |
| `tools/sub_agent_tool.py` (169) | KEEP (general subagent-call tool) |
| `tools/check_compat.py` (98) | KEEP |
| `tools/bash_command.py` (171) | KEEP |
| `tools/prompt_for_profiling_analyzer.py` (67) | KEEP |
| `tools/str_replace_editor.py` (81) | KEEP |
| `tools/registry.py` (56) | KEEP |
| `tools/submit.py` (17) | KEEP |
| `models/*.py` | KEEP (amd_base, amd_claude, amd_gemini, amd_llm, amd_openai, litellm_model, __init__) |
| `models/anthropic_model.py` (35) | DELETE if unused after PR-53 grep confirms |
| `models/test_models.py` (52) | MOVE to `tests/fixtures/models.py` |
| `memory/cross_session/*` | KEEP (this is where most of our KB R&D lives) |
| `memory/working_memory.py` (815) + `working_notebook.py` (404) | KEEP |
| `memory/integration.py` (104) | MERGE into `cross_session/__init__.py` |
| `memory/cross_session_memory.py` (27) | **MOVE** `classify_kernel_category` to `memory/cross_session/__init__.py` in PR-19 + update 3 callsites; keep as DeprecationWarning re-export shim for one release; **delete shim in PR-53**. (Reviewer-caught: NOT safe to delete in PR-00; see §15 Tier-0 Bug A) |
| `environments/docker.py`, `local.py`, `protected_files.py` | KEEP |
| `environments/singularity.py`, `environments/extra/*` | DELETE (only in tests) |

### NEW files added by the refactor (+~3,000 LoC)

| File | Purpose | PR |
|---|---|---|
| `src/minisweagent/cli.py` | THE Typer app; `geak`, `geak add-language`, `geak translate`, `geak resume` | PR-06 → PR-53 |
| `src/minisweagent/kernel_languages/base.py` | `KernelLanguage` dataclass | PR-01 |
| `src/minisweagent/kernel_languages/__init__.py` | Registry + `detect_best(path)` | PR-05 |
| `src/minisweagent/kernel_languages/contract.py` | `validate_harness`, `validate_commandment` | PR-04 |
| `src/minisweagent/kernel_languages/triton/*` | 7-file Triton bundle | PR-01 |
| `src/minisweagent/kernel_languages/hip/*` | 7-file HIP bundle | PR-01 |
| `src/minisweagent/kernel_languages/scaffolder.py` | `geak add-language <name>` logic | PR-07 |
| `src/minisweagent/kernel_languages/_templates/` | Jinja scaffolds for new-language bootstrap | PR-07 |
| `src/minisweagent/kernel_languages/_translation/<src>_to_<dst>.md` | Translation hint packs | PR-60 |
| `src/minisweagent/preprocess/orchestrator.py` | 11-step pipeline | PR-20 |
| `src/minisweagent/preprocess/steps/step_NN_*.py` | Per-step functions | PR-20 |
| `src/minisweagent/subagents/preprocess/harness_builder.py` | LLM subagent for harness production | PR-22 |
| `src/minisweagent/subagents/preprocess/kernel_analysis.py` | LLM subagent for [A]-[D] rubric | PR-25 |
| `src/minisweagent/subagents/preprocess/unit_test_agent.py` | Merged UTA | PR-21 |
| `src/minisweagent/subagents/preprocess/shape_fixer.py` | Merged shape fixer | PR-21 |
| `src/minisweagent/subagents/translation/translator.py` | `TranslationLoop` (multi-round) for `geak translate` | PR-61 |
| `src/minisweagent/subagents/memory/cross_session_memory_analysis.py` | KB top-k synthesis subagent | PR-70 |
| `src/minisweagent/run/unified.py` | `run_pipeline(ctx, mode)` | PR-40 |
| `src/minisweagent/run/compose.py` | `compose_task_body()` | PR-03 |
| `src/minisweagent/run/task_generator.py` | Moved/simplified planner | PR-30 |
| `src/minisweagent/run/context_types.py` | Renamed+merged from `pipeline_types.py`+`preprocess/context.py` | PR-11 |
| `src/minisweagent/run/benchmark_parsing.py` | Merged from pre+postprocess | PR-20 |
| `src/minisweagent/run/model_factory.py` etc. | Split from `pipeline_helpers.py` | PR-20 |
| `src/minisweagent/run/pool_runner.py` | Split from `parallel_helpers.py`; single path (homo+hetero both use it) | PR-52 |
| `src/minisweagent/agents/optimization_agent.py` | THE collapsed agent class | PR-02, PR-50 |
| `src/minisweagent/tools/orchestrator_tools.py` | Moved from `heterogeneous/tools.py` | PR-51 |
| `src/minisweagent/tools/orchestrator_schemas.py` | Moved from `heterogeneous/schemas.py` | PR-51 |
| `scripts/check_language_leaks.py` | CI: no `kernel_type == "triton"` in core | PR-00 |
| `scripts/check_one_cli_file.py` | CI: only `cli.py` has `typer.Typer()` | PR-00 |
| `scripts/check_no_agent_inheritance.py` | CI: no class subclasses `DefaultAgent`/`InteractiveAgent`/`StrategyAgent` | PR-50 |
| `tests/smoke/test_hip_homo_invariants.py` | 6 log markers | PR-00 |
| `tests/smoke/test_triton_hetero_invariants.py` | 7 log markers | PR-00 |
| `tests/smoke/test_add_language_scaffolder.py` | Scaffolds "noop" language, runs 1 round, passes smoke test | PR-07 |
| `tests/smoke/test_translate_triton_to_hip.py` | Translates a small kernel, verifies golden match | PR-61 |

---

## §2 — Target directory tree (final state)

```
src/minisweagent/
├── cli.py                          ← THE entry point (Typer app; all @app.command()s)
├── __init__.py
├── __main__.py
│
├── agents/
│   ├── __init__.py
│   ├── optimization_agent.py       ← THE agent class (no inheritance chain)
│   ├── select_patch_agent.py       ← post-eval patch selector
│   └── agent_spec.py               ← Protocol + AgentConfig + exceptions
│
├── subagents/                      ← ALL LLM subagents live here
│   ├── __init__.py
│   ├── base.py                     ← SubagentBase (base for all subagents; peer of OptimizationAgent, NOT a subclass)
│   │
│   ├── preprocess/                 ← ONE-SHOT subagents (run before the round loop)
│   │   ├── __init__.py
│   │   ├── harness_builder.py      ← adapt user tests → standard harness
│   │   ├── kernel_analysis.py      ← produce [A]-[D] rubric markdown
│   │   ├── unit_test_agent.py      ← generate tests when no user tests exist
│   │   ├── shape_fixer.py          ← fix kernel shape mismatches
│   │   └── configs/
│   │       ├── harness_builder.yaml
│   │       ├── kernel_analysis.yaml
│   │       ├── unit_test_agent.yaml
│   │       └── shape_fixer.yaml
│   │
│   ├── memory/                     ← PER-ROUND subagents for KB/memory
│   │   ├── __init__.py
│   │   ├── cross_session_memory_analysis.py  ← synthesizes top-k KB experiences
│   │   └── configs/
│   │       └── cross_session_memory_analysis.yaml
│   │
│   └── translation/                ← STANDALONE subagent for `geak translate`
│       ├── __init__.py
│       ├── translator.py
│       └── configs/
│           └── translator.yaml
│
├── kernel_languages/               ← ALL language-specific logic lives here
│   ├── __init__.py                 ← registry + detect_best()
│   ├── base.py                     ← KernelLanguage dataclass
│   ├── contract.py                 ← validate_harness, validate_commandment
│   ├── scaffolder.py               ← `geak add-language <name>` implementation
│   ├── _templates/                 ← Jinja scaffolds for new-language bootstrap
│   │   ├── kernel_language.py.j2
│   │   ├── system_prompt.md.j2
│   │   ├── optimization_prompt.md.j2
│   │   ├── planner_strategy_hints.md.j2
│   │   ├── harness.j2.j2
│   │   ├── builder_hints.md.j2
│   │   └── commandment.j2.j2
│   ├── _translation/               ← Translation hint packs (src→dst)
│   │   ├── triton_to_hip.md
│   │   ├── hip_to_triton.md
│   │   └── _fallback.md            ← generic "rewrite in target" guidance
│   ├── triton/
│   │   ├── kernel_language.py      ← TritonKernelLanguage = KernelLanguage(...)
│   │   ├── system_prompt.md
│   │   ├── optimization_prompt.md
│   │   ├── planner_strategy_hints.md
│   │   ├── harness.j2
│   │   ├── builder_hints.md
│   │   └── commandment.j2
│   └── hip/
│       └── (same 7 files for HIP)
│
├── preprocess/
│   ├── __init__.py
│   ├── orchestrator.py             ← runs phases in order (~30 LoC)
│   ├── phases/                     ← 4 phases + 1 conditional (not 11 flat steps)
│   │   ├── __init__.py
│   │   ├── translation.py       ← CONDITIONAL (runs only if target_language != language)
│   │   │                            ← calls subagents/translation/translator.py
│   │   ├── discovery.py          ← resolve_kernel + codebase_context + test_discovery
│   │   ├── harness.py            ← harness_decision + harness_builder + validate_harness
│   │   │                            ← calls subagents/preprocess/{unit_test_agent,harness_builder}.py
│   │   ├── baseline.py           ← baseline_execute + profile + build_metrics
│   │   └── explore.py              ← render_commandment + validate_commandment + kernel_analysis
│   │                                ← calls subagents/preprocess/kernel_analysis.py
│   ├── baseline.py
│   ├── codebase_context.py
│   ├── discovery_types.py
│   ├── kernel_profile.py
│   ├── resolve_kernel_url.py
│   ├── run_harness.py
│   ├── testcase_cache.py
│   ├── repo_paths.py
│   ├── debug_runtime.py
│   ├── config_loader.py
│   └── INSTRUCTIONS.md
│
├── run/                            ← PURE library modules (no Typer imports)
│   ├── __init__.py
│   ├── unified.py                  ← run_pipeline(ctx, mode) — the round loop
│   ├── compose.py                  ← compose_task_body()
│   ├── task_generator.py           ← planner output assembly
│   ├── dispatch.py                 ← task_file_to_agent_task
│   ├── task_file.py                ← task I/O + worktree helpers
│   ├── context_types.py            ← dataclasses (PreprocessContext, Task, etc.)
│   ├── model_factory.py            ← split from pipeline_helpers
│   ├── env_factory.py              ← split from pipeline_helpers
│   ├── context_injection.py        ← split from pipeline_helpers
│   ├── agent_filter.py             ← split from pipeline_helpers
│   ├── pool_runner.py              ← split from parallel_helpers (one path for homo+hetero)
│   ├── pool_logger.py              ← split from parallel_helpers
│   ├── benchmark_parsing.py        ← merged pre+postprocess
│   ├── harness_runtime.py          ← runtime helpers split from harness_utils
│   ├── postprocess/
│   │   ├── evaluation.py
│   │   └── results.py
│   └── utils/
│       ├── config_editor.py
│       ├── generated_artifacts.py
│       ├── task_parser.py
│       ├── save.py
│       └── git_safe_env.py
│
├── tools/                          ← agent tool ecosystem
│   ├── save_and_test.py
│   ├── profiling_tools.py
│   ├── editor_tool.py
│   ├── strategy_manager.py
│   ├── mcp_bridge.py
│   ├── mcp_client/
│   ├── rag_postprocessor.py
│   ├── tools_runtime.py
│   ├── bash_command.py
│   ├── baseline_metrics_tool.py
│   ├── sub_agent_tool.py
│   ├── check_compat.py
│   ├── str_replace_editor.py
│   ├── prompt_for_profiling_analyzer.py
│   ├── orchestrator_tools.py       ← moved from heterogeneous/
│   ├── orchestrator_schemas.py     ← moved from heterogeneous/
│   ├── registry.py
│   └── submit.py
│
├── memory/
│   ├── __init__.py
│   ├── working_memory.py
│   ├── working_notebook.py
│   └── cross_session/
│       ├── __init__.py             ← top-level API (merged integration.py)
│       ├── schemas.py
│       ├── extractor.py
│       ├── formatter.py
│       ├── retriever.py
│       ├── consolidation.py
│       ├── fingerprint.py
│       ├── config.py
│       ├── cli.py
│       ├── knowledge_base.json
│       ├── backends/
│       └── server/
│
├── models/
│   ├── __init__.py
│   ├── amd_base.py, amd_claude.py, amd_gemini.py, amd_llm.py, amd_openai.py
│   ├── litellm_model.py
│   └── utils/
│
├── environments/
│   ├── __init__.py
│   ├── docker.py
│   ├── local.py
│   └── protected_files.py
│
├── config/
│   ├── __init__.py
│   ├── geak.yaml                   ← default runtime config
│   ├── mini.yaml                   ← shared mini-agent settings
│   ├── mini_kernel_strategy_list.yaml
│   ├── mini_select_patch.yaml
│   └── mini_reverse_kl.yaml
│
└── utils/
    └── log.py
```

**Diff vs today**:
- Deleted: `run/extra/`, `run/mini.py`, `run/orchestrator.py`, `run/github_issue.py`, `run/hello_world.py`, `run/inspector.py`, `run/mini_extra.py`, `agents/default.py`, `agents/interactive.py`, `agents/strategy_agent.py`, `agents/strategy_interactive.py`, `agents/interactive_textual.py`, `agents/homogeneous/`, `agents/heterogeneous/`, `agents/shape_fixer_agent.py`, `agents/unit_test_agent.py`, `memory/cross_session_memory.py`, `memory/integration.py`, `models/test_models.py`, `models/anthropic_model.py`, `models/extra/`, `environments/singularity.py`, `environments/extra/`, `tools/resolve_kernel_url.py`, `tools/discovery_types.py`, `tools/validate_commandment.py`, `debug_runtime.py` (root), `config/extra/swebench*.yaml`.
- Moved: `run/preprocess/` → `preprocess/` (sibling of `run/` — preprocess is not part of the round loop).
- New: `cli.py`, `kernel_languages/`, `subagents/` (top-level; ALL LLM subagents live here), `preprocess/orchestrator.py`, `preprocess/steps/`, `run/unified.py`, `run/compose.py`, `agents/optimization_agent.py`.

---

## §3 — ONE CLI file principle

The **only** Typer entry point is `src/minisweagent/cli.py`. All subcommands live as `@app.command()` decorators in the same file.

```python
# src/minisweagent/cli.py (canonical, final state)

import typer
from pathlib import Path

app = typer.Typer(no_args_is_help=True)

@app.command()
def optimize(
    task: str = typer.Option(..., "-t", "--task"),
    mode: str = typer.Option("auto", "--mode"),           # fixed|planned|auto
    num_parallel: int = typer.Option(1, "--num-parallel"),
    gpu_ids: str = typer.Option("0", "--gpu-ids"),
    max_rounds: int = typer.Option(5, "--max-rounds"),
    rag: bool = typer.Option(True, "--rag/--no-rag"),
    kb: bool = typer.Option(True, "--kb/--no-kb"),
    # ... remaining flags from today's mini.py
):
    """Optimize a kernel (default behavior; what `geak -t "..."` does today)."""
    ctx = build_preprocess_context(task, ...)
    ctx = PreprocessOrchestrator(ctx).run()
    run_pipeline(ctx, mode=mode)

@app.command("add-language")
def add_language(
    name: str = typer.Argument(...),
    extensions: str = typer.Option(..., help="comma-separated file extensions, e.g. '.metal,.msl'"),
):
    """Scaffold a new KernelLanguage. Creates kernel_languages/<name>/ with 7 files."""
    from minisweagent.kernel_languages.scaffolder import scaffold_language
    scaffold_language(name, extensions=extensions.split(","))

@app.command()
def translate(
    source: Path = typer.Option(..., "--source"),
    source_language: str = typer.Option(None, "--source-language"),
    target_language: str = typer.Option(..., "--target-language"),
    output: Path = typer.Option(..., "--output"),
    test: Path = typer.Option(..., "--test", help="harness that runs on source to produce golden output"),
    max_rounds: int = typer.Option(8, "--max-rounds"),
):
    """Translate a kernel from one language to another, preserving semantics."""
    from minisweagent.preprocess.subagents.translator import run_translation
    run_translation(source, source_language, target_language, output, test, max_rounds)

@app.command()
def resume(
    preprocess_dir: Path = typer.Option(..., "--preprocess-dir"),
    mode: str = typer.Option("auto", "--mode"),
):
    """Resume optimization from a pre-computed preprocess output directory."""
    ctx = load_preprocess_context(preprocess_dir)
    run_pipeline(ctx, mode=mode)

@app.command()
def test_language(name: str):
    """Smoke-test a newly scaffolded KernelLanguage."""
    from minisweagent.kernel_languages.scaffolder import smoke_test_language
    smoke_test_language(name)

# Default: `geak -t "..."` (no subcommand) routes to `optimize` for backward compat.
@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context, task: str = typer.Option(None, "-t", "--task")):
    if ctx.invoked_subcommand is None and task:
        ctx.invoke(optimize, task=task)
```

### `pyproject.toml` entry

```toml
[project.scripts]
geak = "minisweagent.cli:app"
# No second CLI entry. Ever.
```

### CI gate

`scripts/check_one_cli_file.py` greps the repo for `typer.Typer(` and `@app.command(`. If any match appears outside `src/minisweagent/cli.py`, CI fails.

---

## §4 — ONE agent class principle

Today's 4-layer chain (DefaultAgent → InteractiveAgent → StrategyAgent → StrategyInteractiveAgent, plus `HomogeneousAgent` wrapper) is collapsed into a single `agents/optimization_agent.py`:

```python
# src/minisweagent/agents/optimization_agent.py

from dataclasses import dataclass
from minisweagent.agents.agent_spec import AgentConfig, Submitted, NonTerminatingException, LimitsExceeded, TerminatingException

class OptimizationAgent:
    """The MAIN agent class in GEAK. A PEER of SubagentBase (not a parent, not a child).

    - No inheritance chain (PR-50 collapses the old 4-layer DefaultAgent→...→StrategyInteractiveAgent).
    - NOT a subclass of SubagentBase. Does NOT inherit anything from subagents.
    - Language-agnostic: never imports KernelLanguage, never branches on kernel_type.
    - Task body + tool set + model are passed in; the agent loops until a Submitted exception
      or a round/step limit is hit.
    - Two distinct callers instantiate this class (both produce the same per-call behavior):
      1. `run/unified.py::run_pipeline` — the main optimization loop (long-lived, N rounds).
      2. `subagents/base.py::SubagentBase._make_optimization_agent` — a subagent composing
         a short-lived OptimizationAgent for a single narrow-task LLM turn.
    - **Tool decisions happen BEFORE this class**: the caller (either the main pipeline's
      `_resolve_tools(ctx, mode)` or the subagent's own tool list) resolves the final set.
      The agent receives a resolved tool list and never re-decides what tools to use.
    """

    def __init__(self, config: AgentConfig, model, env, tools: list, on_strategy_changed=None):
        self.config = config
        self.model = model
        self.env = env
        self.tools = tools           # already resolved by caller
        self.on_strategy_changed = on_strategy_changed  # optional callback

    def run(self, task_body: str) -> dict:
        """Execute the task. Returns a final-report dict."""
        # (~300 LoC: step loop, tool calling, exception handling, strategy callback)
        ...
```

### Tool registration (reviewer-raised Issue 5 — explicit answer)

Today, tool selection happens in `tools/tools_runtime.py:120-162` behind an `allowed: set[str]` parameter, with `resolve_kernel_url`, `baseline_metrics`, `check_kernel_compatibility`, `sub_agent`, MCP tools registered conditionally. The 4-layer chain (DefaultAgent → InteractiveAgent → StrategyAgent → StrategyInteractiveAgent) decides what to pass. After PR-50 that chain is gone, so we must specify a new decider.

**Decision** (locked in by PR-40, enforced by CI after PR-50): tool selection is a **`run_pipeline` responsibility**, composed from three inputs in precedence order:

```python
# src/minisweagent/run/unified.py (after PR-40)

def _resolve_tools(ctx: PreprocessContext, mode: str) -> list:
    """Build the final tool list for OptimizationAgent. Called once per task.

    Ordered precedence:
    1. KernelLanguage defaults: ctx.language.tool_set          (all tasks get these)
    2. Mode-specific additions:
         - mode == "planned"  -> orchestrator_tools (generate_tasks, dispatch_tasks, collect_results)
         - mode == "fixed"    -> no additions
         - mode == "auto"     -> planner decides per round
         - mode == "translate"-> translation_tools (run_golden_match, emit_report)
    3. User / task-level explicit opts (CLI --tool enable/disable)

    Each tool is registered through tools_runtime once (lazy singleton); the
    agent receives a resolved list and never asks for more.
    """
    tools = list(ctx.language.tool_set)          # per-language default set
    if mode == "planned":
        tools += ORCHESTRATOR_TOOLS
    elif mode == "translate":
        tools += TRANSLATION_TOOLS
    return _apply_cli_overrides(tools, ctx.cli_overrides)
```

New field on `KernelLanguage` (added in PR-01): `tool_set: frozenset[str]` — the minimum tool names every task in that language gets. For Triton, that's today's `{bash, str_replace_editor, save_and_test, submit, strategy_manager, profile_kernel, query, optimize, baseline_metrics, resolve_kernel_url, sub_agent}`.

### What OptimizationAgent does NOT have access to during round-loop optimization

Reviewer-raised: **the OptimizationAgent during optimization rounds must NOT have direct read access to the cross-session KB.** The single path is enforced by tool curation — the agent's `tool_set` does NOT include any tool that reads from `memory/cross_session/knowledge_base.json` directly. Specifically:

| Tool | Available to main OptimizationAgent? | Reason |
|---|---|---|
| `query` (RAG MCP — generic docs) | **YES** | RAG is for generic documentation lookup (Triton docs, HIP programming guide, kernel patterns). Safe and complementary to cross-session KB. |
| `optimize` (RAG MCP — optimization patterns) | **YES** | Same — generic corpus of optimization techniques, not kernel-specific past runs. |
| `profile_kernel` / Metrix MCP | **YES** | Per-round profiling is essential. |
| `baseline_metrics` | **YES** | Read pre-computed baseline metrics from artifacts. |
| `strategy_manager` | **YES** | The agent's own strategy state (strategy_list.json), NOT cross-session. |
| `save_and_test` | **YES** | Core patch-apply loop. |
| `resolve_kernel_url` | **YES** (kept as `ResolveKernelUrlTool`) | URL→path resolution. |
| `bash`, `str_replace_editor`, `submit`, `sub_agent` | **YES** | Core tool set. |
| *Direct cross-session KB read / retriever call* | **NO** | The only interface to cross-session KB during rounds is the `cross_session_memory_insights.md` file written by the pre-round subagent. Contents are injected via `compose_task_body()` — they are **data**, not a **tool call**. The agent reads the file content in its task body prompt; it cannot issue a new query. |
| *Direct KB write* | **NO** | KB writes happen only post-round via `record_optimization_outcome(ctx, best)` in `run_pipeline`, not via an agent tool. |

**Why this boundary?** Three reasons:

1. **Single path for memory**: If the agent could directly call the retriever, the pre-round subagent's work (curation, ranking, dead-end flagging) would be bypassed and we'd be back to 40 KB raw dumps.
2. **Determinism of the memory context within a round**: The insights file is computed ONCE at round start; the agent works against a stable memory context for the whole round. A direct-retrieval tool would let the agent re-query mid-round with a shifting top-k, making behavior harder to reproduce.
3. **Security / token budget**: Generic tools `query`/`optimize` (RAG) have predictable corpus scope and response shape. Raw KB access would let the agent burn context on unbounded retrieval.

CI gate `check_agent_tool_set.py` (added in PR-50) enforces: no tool in the resolved tool list may import from `minisweagent.memory.cross_session.retriever` or `.knowledge_base`. The subagent's use is fine because it lives in `subagents/memory/`, not in the main agent's tool table.

### CI gate

`scripts/check_no_agent_inheritance.py` greps for `class.*\(DefaultAgent\)` / `InteractiveAgent` / `StrategyAgent` — any hit fails CI. After PR-50 the old base classes are gone, so the grep defends against regressions.

A second CI gate `scripts/check_tool_resolve_single_site.py` greps for any call to `ToolRuntime(allowed=...)` outside `run/unified.py:_resolve_tools` — any hit fails. This prevents someone re-introducing scattered tool-decision logic after PR-50.

---

## §5 — Adding a new KernelLanguage (user cookbook)

This is the **primary test of modularity**: can an external user add a new kernel language (Metal / SYCL / CUDA / FlyDSL / OpenCL / WGPU) without touching any file outside `kernel_languages/<their_name>/`?

### Step-by-step: adding FlyDSL

```bash
# 1. Scaffold (one command)
geak add-language flydsl --extensions '.fly,.flydsl'

# What this creates under src/minisweagent/kernel_languages/flydsl/:
#   kernel_language.py            <- python glue (15-line KernelLanguage instance)
#   system_prompt.md              <- "You are a FlyDSL expert..."
#   optimization_prompt.md        <- "Optimize this FlyDSL kernel..."
#   planner_strategy_hints.md     <- strategy menu for the planner
#   harness.j2                    <- Jinja template for `harness.py`
#   builder_hints.md              <- hints fed to the HarnessBuilder subagent
#   commandment.j2                <- Jinja template for `COMMANDMENT.md`
#
# The scaffolder also:
#   - appends `from .flydsl.kernel_language import FlyDSLKernelLanguage`
#     to kernel_languages/__init__.py
#   - registers it in the _Registry (single-line append)

# 2. Fill in the 7 files with FlyDSL-specific content.
#    - kernel_language.py: file extensions, kb_namespace, detect_hints, env_setup (pip/docker hints)
#      (test/benchmark/profile commands go in commandment.j2 — NOT duplicated here)
#    - *.md: domain prompts
#    - *.j2: harness/commandment layout + universal CLI contract hooks

# 3. Validate (runs contract checks + 1-round smoke test with a toy FlyDSL kernel)
geak test-language flydsl
# Expected output:
#   [ok]  kernel_language.py imports and validates
#   [ok]  harness.j2 renders and passes validate_harness()
#   [ok]  commandment.j2 renders and passes validate_commandment()
#   [ok]  registry.detect("test_fixture.fly") → flydsl (score 0.95)
#   [ok]  1-round smoke optimize on fixture kernel succeeded
#   READY: flydsl is registered. Run with: geak -t "Optimize <path>.fly ..."

# 4. Use
geak -t "Optimize /path/to/my_kernel.fly, test: python3 flydsl_runner.py ..."
# Language auto-detected. No core code modified.
```

### What the user has to write (concrete sizes)

| File | Typical size | What it defines |
|---|---|---|
| `kernel_language.py` | 15-30 LoC | `FlyDSLKernelLanguage = KernelLanguage(name="flydsl", file_extensions={".fly",".flydsl"}, ...)` |
| `system_prompt.md` | 30-80 lines | "You are a FlyDSL expert. The idioms are ..." |
| `optimization_prompt.md` | 10-30 lines | Task framing for the round loop |
| `planner_strategy_hints.md` | 30-60 lines | Strategy menu for the planner (tiling, fusion, etc.) |
| `harness.j2` | 60-120 lines | How to wrap a user's FlyDSL kernel into a standard harness |
| `builder_hints.md` | 20-40 lines | Hints for the HarnessBuilder LLM subagent |
| `commandment.j2` | 30-60 lines | How to render `COMMANDMENT.md` for FlyDSL |

**Total: ~250-400 LoC, all inside `kernel_languages/flydsl/`.**

### What the user does NOT modify

- `cli.py` (zero change)
- `agents/optimization_agent.py` (zero change)
- `run/unified.py` (zero change)
- `preprocess/orchestrator.py` (zero change)
- `preprocess/steps/*.py` (zero change)
- Any tool module (zero change)

If the user finds they need to modify any of the above, that's a **bug in the KernelLanguage abstraction** and we fix the abstraction — not let the user patch core.

### Inspected by CI

`scripts/check_language_leaks.py` fails if any file outside `kernel_languages/` contains `== "flydsl"` or `== "triton"` etc. This forces all language-specific behavior through the `KernelLanguage` data object.

### Contract validators (`kernel_languages/contract.py`)

Every rendered `harness.py` and `COMMANDMENT.md` is validated. Required properties:

```python
# validate_harness(path):
#   - imports cleanly under the language's eval_env
#   - exposes argparse with --correctness, --benchmark, --full-benchmark, --profile
#   - emits GEAK_RESULT_LATENCY_MS=<float> on stdout
#   - emits GEAK_RESULT_SPEEDUP=<float> on --full-benchmark
#   - returns nonzero exit code on mismatch

# validate_commandment(path):
#   - is a valid markdown file
#   - contains exactly these sections (by header regex): "## Test Command", "## Benchmark Command", "## Full Benchmark", "## Profile Command"
#   - each section's code block parses as shell
#   - the commands reference the same harness.py path
```

These validators are deterministic; they run in every preprocess step 6 and step 11, and also in `geak test-language`. A broken template can't land.

---

## §6 — Translation subagent framework

### Goal

Translate a kernel from language A to language B while preserving semantics. Examples:
- PyTorch → FlyDSL (the one PR #153 tried)
- Triton → HIP
- HIP → Triton
- CUDA → Triton (after CUDA is added)
- Any src → any dst, once both languages are registered.

### Placement: Translation phase of preprocessing (NOT a separate pipeline)

Translation runs **before** the Discovery phase. Reason: after translation, `ctx.kernel_path` points to the TRANSLATED file, and every downstream phase must operate on that. If translation ran after Discovery, codebase context + test discovery + profiling would all target the wrong (source) file.

Two user-facing modes, both routed through the same Translation phase:

| Mode | CLI | Behavior |
|---|---|---|
| **Standalone translation** | `geak translate --source X --target-language Y --output Z` | Preprocess orchestrator runs the Translation phase, then exits (`translate_only=True`). Just writes the translated file + report. |
| **Translate + optimize** | `geak -t "..." --target-language Y` | Translation phase runs first, then Discovery/Harness/Baseline/Explore run on the translated kernel, then `run_pipeline` optimizes. End-to-end pipeline. |

The `TranslationLoop` is a **multi-round verify-and-retry loop that composes `OptimizationAgent`** (reviewer-raised Issue 6: not the same class — the verification criterion + exit condition + loop structure are genuinely different). Structure:

```python
# src/minisweagent/subagents/translation/translator.py (sketch)

class TranslationLoop(SubagentBase):
    """Multi-round translation loop. Composes OptimizationAgent for each attempt.

    NOT the same class as OptimizationAgent — has a different exit condition
    (golden-match instead of Submitted/speedup) and a bespoke retry loop.
    """

    def __init__(self, src_lang: KernelLanguage, tgt_lang: KernelLanguage, config):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config

    def translate(self, src_kernel_path, test_harness_path, max_attempts=3) -> TranslationResult:
        golden = run_harness(src_kernel_path, test_harness_path, mode="correctness")  # reference outputs
        for attempt in range(max_attempts):
            task_body = compose_task_body(
                language=self.tgt_lang,
                mode="translate",
                src_lang=self.src_lang,
                src_code=read(src_kernel_path),
                golden_ref_path=golden.outputs_path,
            )
            # Inner OptimizationAgent call — generates one target-language draft
            draft = OptimizationAgent(
                config=self.config,
                model=...,
                env=...,
                tools=[editor, bash, run_harness],
            ).run(task_body)
            target_result = run_harness(draft.output_path, test_harness_path, mode="correctness")
            if tensors_allclose(golden.outputs, target_result.outputs, atol=1e-5):
                perf = validate_translation_performance(golden.latency, target_result.latency)
                return TranslationResult(ok=True, path=draft.output_path, attempt=attempt, perf=perf)
            # feedback loop — pass diff summary back into next task_body
        return TranslationResult(ok=False, reason="all attempts failed golden match")
```

Three concrete differences from `OptimizationAgent`:

1. **Task body**: `compose_task_body(..., mode="translate", ...)` includes both languages' `system_prompt` + `idioms` + the pair-specific `_translation/<src>_to_<tgt>.md`.
2. **Verification criterion**: tensor-by-tensor `allclose` against golden reference outputs, not `Submitted` exception and not speedup threshold.
3. **Exit condition**: exits on first golden match within `max_attempts` (default 3) — NOT after fixed N rounds. Optional: `validate_translation_performance()` (PR #153's 0.5x fail / 0.8x warn thresholds adapted; thresholds per-pair via `translation_hints`).

**Why this matters for §10 / §14**: `TranslationLoop` is listed separately in the subagent table because its *lifecycle and success criterion* differ. `HarnessBuilder` is a one-shot LLM call; `TranslationLoop` is a multi-round verify-and-retry loop of its own. The two are both "subagents" (narrow-task LLM-using components extending `SubagentBase`) but they are genuinely different kinds of computation.

### Flow

```
                      ┌─────────────────────────────────────────────┐
                      │  geak translate  OR                         │
                      │  geak -t "..." --target-language hip        │
                      └─────────────────┬───────────────────────────┘
                                        │
                                        ▼
                      ┌─────────────────────────────────────────────┐
                      │  cli.py:                                    │
                      │    ctx = build_preprocess_context(...)      │
                      │    ctx.target_language = registry.get(hip)  │
                      │    ctx.translate_only = True (if subcommand)│
                      │    PreprocessOrchestrator(ctx).run()        │
                      └─────────────────┬───────────────────────────┘
                                        │
                                        ▼
                      ┌─────────────────────────────────────────────┐
                      │  Translation phase                          │
                      │  ────────────────────                       │
                      │  a. Source harness runs on source kernel    │
                      │     → captures golden reference tensors     │
                      │  b. Load translation hints:                 │
                      │     kernel_languages/_translation/          │
                      │       <src>_to_<tgt>.md or _fallback.md     │
                      │  c. TranslationLoop.loop(...)               │
                      │     → retry loop (max_attempts=3)           │
                      │     → each attempt: draft + run target      │
                      │       harness + compare to golden           │
                      │  d. validate_translation_performance(       │
                      │       src_latency, tgt_latency, thresholds) │
                      │  e. ctx.kernel_path = translated_path       │
                      │     ctx.language = target_language          │
                      │                                             │
                      │  IF translate_only: return ctx, exit        │
                      └─────────────────┬───────────────────────────┘
                                        │
                                        ▼
                      ┌─────────────────────────────────────────────┐
                      │  Discovery/Harness/Baseline/Explore phases  │
                      │  run on translated kernel                   │
                      │   then run_pipeline(ctx, mode) optimizes    │
                      └─────────────────────────────────────────────┘
```

### New fields on `KernelLanguage`

```python
@dataclass(frozen=True)
class KernelLanguage:
    name: str
    file_extensions: frozenset[str]
    system_prompt: str
    optimization_prompt: str
    planner_strategy_hints: str
    harness_template: str             # Jinja source for harness.py skeleton
    commandment_template: str         # Jinja source; SINGLE SOURCE OF TRUTH for Setup/Correctness/
                                       # Benchmark/FullBenchmark/Profile commands (per-language quirks
                                       # like HIP's `make &&` or Triton's `python3` live here)
    builder_hints: str
    eval_env: dict                    # pip packages, dockerfile hints (setup-time, not runtime command)
    kb_namespace: str
    # NEW for translation:
    idioms: str                       # "tl.load = ...; @triton.jit is..." (~10 KB)
    builtin_types: set[str]           # for parser sanity checks
```

**Note**: earlier drafts had `test_runner_command` and `profiler_command` fields. Removed — they duplicated what `commandment.j2` already encodes, creating a drift risk (two sources of truth for the same information). The harness CLI surface is UNIVERSAL (`python3 {harness} --correctness|--benchmark|--full-benchmark|--profile` for every language — enforced by `HarnessBuilder` + `validate_harness()` contract), so there's no language-level runner-command to factor out. Language-specific setup (e.g., HIP's `make`) and profiler invocation (`rocprof` vs `metrix`) live in `commandment.j2`'s Setup and Profile sections respectively.

### Translation hint packs

`kernel_languages/_translation/triton_to_hip.md` (curated per-pair when needed):

```markdown
# Triton → HIP translation hints

## Common mappings
- `tl.program_id(0)` → `hipBlockIdx_x` (HIP inline)
- `tl.arange(0, BLOCK)` → index arithmetic in a HIP kernel
- `@triton.jit` function → `__global__ void` HIP kernel
- `tl.load(ptr + offsets, mask)` → guarded global load with predicate
- `tl.store(ptr + offsets, x, mask)` → guarded global store

## Pitfalls
- Triton uses implicit warp masking; HIP requires explicit masking in thread loops.
- Triton fp8/fp4 types: map to __hip_fp8_e4m3 / __hip_fp4.
- Triton tl.dot -> use HIP WMMA / MFMA intrinsics (architecture-dependent).

## Benchmark parity
- Include a warmup loop of 10 iterations (both languages) before timing to amortize compile.
- Use torch.cuda.synchronize() / hipDeviceSynchronize() before timer start/stop.
```

If no pair-specific pack exists, `_fallback.md` loads generic guidance + the two languages' `system_prompt.md` and `idioms` fields, and the LLM subagent bootstraps from there.

### CLI: two entry points, one subcommand

```bash
# Mode 1: standalone translation (runs the Translation phase only, exits)
geak translate --source foo.py --target-language hip --output foo_hip.cpp --test test.py

# With hints from KB (semi-supervised translation using past Triton→HIP pairs)
geak translate \
  --source foo.py \
  --source-language triton \
  --target-language hip \
  --output foo_hip.cpp \
  --test test.py \
  --max-attempts 3 \
  --use-kb                # pulls prior triton→hip translation records

# Batch translation
geak translate --batch translation_list.yaml  # each entry = {source, target_lang, output, test}

# Mode 2: integrated translate + optimize (runs Translation phase → Discovery → Harness → Baseline → Explore → run_pipeline)
geak -t "Optimize /path/to/foo.py. Tests: python3 test.py ..." \
     --target-language hip \
     --max-rounds 8
# Translation phase converts foo.py → foo_translated.hip; then Discovery/Harness/Baseline/Explore + optimization run on the translated file.
```

Both routes go through the SAME `cli.py` (no separate `geak-translate` entry — PR #153's ambiguity avoided).

### Testing the translator

`tests/smoke/test_translate_triton_to_hip.py`:
1. Takes `tests/fixtures/triton/tiny_add.py` (50-LoC add kernel with known outputs).
2. Runs `geak translate --source tiny_add.py --target-language hip --output /tmp/tiny_add.hip --test tiny_add_test.py`.
3. Asserts: (a) `/tmp/tiny_add.hip` compiles, (b) outputs match the Triton baseline within 1e-5, (c) `translation_report.json` exists and has `"golden_match": true`.

---

## §7 — Principles for every PR

1. **Pure additive first, deletion last.** Phases 0-4 ADD code or gate behind a flag. Phases 5-6 delete.
2. **One concern per PR.** No "while I'm in there" scope creep. Each PR has a single-sentence summary.
3. **Rollback < 5 min.** Every additive PR can be reverted with `git revert` or a flag flip.
4. **Smoke tests required.** HIP-homo + Triton-hetero invariants must pass on every PR.
5. **No band-aids.** Hacks get a `FIXME(PR-XX)` marker and a linked issue.
6. **LLM subagents only where reasoning is needed.** `HarnessBuilder`, `KernelAnalysis`, `UnitTestAgent`, `ShapeFixer`, `TranslationLoop` (multi-round), `CrossSessionMemoryAnalysisAgent` are the 6 LLM subagents. Everything else is deterministic (templates, validators, glue).
7. **Contract validators gate every template output.** Broken templates fail fast at preprocess time.
8. **ONE CLI FILE.** `src/minisweagent/cli.py` is the only Typer module.
9. **Two agent class families (NOT one).** There are exactly two:
   - `OptimizationAgent` — the main agent that runs the optimization round loop. No inheritance chain. Standalone class. NOT a subclass of `SubagentBase`.
   - `SubagentBase` and its subclasses (`HarnessBuilder`, `KernelAnalysisAgent`, `UnitTestAgent`, `ShapeFixerAgent`, `CrossSessionMemoryAnalysisAgent`, `TranslationLoop`) — all narrow-task LLM subagents. Each extends `SubagentBase`.
   - **Relationship**: subagents COMPOSE `OptimizationAgent` (when they need a full LLM tool-loop, e.g., HarnessBuilder, TranslationLoop internals) via a `SubagentBase._make_optimization_agent(tools)` helper. They do NOT inherit from it. CI gate `check_no_agent_inheritance.py` enforces the no-inheritance boundary.
10. **No language branching in core code.** Any `if language.name == "..."` outside `kernel_languages/` fails CI.
11. **Adding a language costs zero core edits.** If a new language forces a core file edit, the abstraction is wrong and we fix the abstraction.
12. **Flags are for staged rollout, not permanent dual paths.** Every env flag introduced (`GEAK_USE_HARNESS_BUILDER`, `GEAK_USE_JINJA_COMMANDMENT`, `GEAK_UNIFIED`, `GEAK_USE_CROSS_SESSION_MEMORY`, `GEAK_TRANSLATION_SELF_REVIEW`) has an explicit removal PR in this plan (see §14 flag matrix). Without a removal PR, a "1-line rollback" becomes a "10-line runtime conditional forever." Reviewer-flagged as a structural risk; addressed.

---


## §8 — PR sequence (3 PRs, ~6-10 weeks)

Restructured from 38 small PRs to **3 large coherent PRs** per user direction. Tradeoff:

- **What we gain**: fewer merges, single-shot review cycles, internal consistency enforced within each PR, much faster calendar time, matches the reality that this is a fork on `refactor-test` that can iterate.
- **What we lose**: fine-grained rollback (each PR is all-or-nothing), no 2-week observation window between flag flips, larger review surface per PR.
- **Mitigation**: each PR still ships behind feature flags where possible; each has its own CI gates; each is independently rollbackable via `git revert <pr>`; the §0.2 nightly regression + smoke tests + CI gates catch regressions regardless of PR granularity.

| # | PR | Scope | Risk | Owner | Est. LoC delta |
|---|---|---|---|---|---|
| **1** | **Foundation + Cleanup + CLI Consolidation** | Single `cli.py` (deletes 10 old CLI entry points incl. `geak-orchestrate`, `geak-preprocess`, `commandment`, etc.); `KernelLanguage` class + registry + `kernel_languages/{triton,hip}/` bundles; `SubagentBase` peer class; `geak add-language` scaffolder; CI gates; smoke tests; regression baselines yaml; **~8,000 LoC of mini-swe-agent heritage, duplicates, and dead code DELETED**; `classify_kernel_category` safely relocated; language-detection collapsed to one site | medium (lots of deletions; mitigated by gates + smoke tests) | Either | −5,000 net (−8,000 dead + 3,000 new) |
| **2** | **Preprocessing Refactor** | `PreprocessOrchestrator` + 4 phases (`discovery`, `harness`, `baseline`, `explore`) + 1 conditional (`translation`); 4 preprocess subagents (`HarnessBuilder`, `KernelAnalysisAgent`, `UnitTestAgent`, `ShapeFixer`) + 1 standalone subagent (`TranslationLoop`); Jinja commandment rendering; contract validators; **`geak translate` subcommand in `cli.py`**; HIP commandment-enforcement fix (bug #1); **deletes `preprocessor.py` 1386 LoC, `commandment.py` 530 LoC, `harness_utils.py` 1660 LoC monoliths** | medium-high (HarnessBuilder is a new LLM subagent; contract validator + 30-fixture corpus gate) | Yue | −1,500 net (−3,576 old + 2,000 new phases/subagents) |
| **3** | **Orchestration + Optimization Refactor** | `OptimizationAgent` single class (collapse 4-layer chain); `run_pipeline(ctx, mode)` unified round loop; `_resolve_tools(ctx, mode)` single tool-resolution site; unified `run/task_generator.py` with A/B equivalence gate; single `run/pool_runner.py` (collapse homo/hetero split); `CrossSessionMemoryAnalysisAgent` subagent + structured `cross_session_memory_insights.md`; uniform per-round FULL_BENCHMARK + `record_optimization_outcome` (fixes bug #1 + bug #6); **deletes `agents/heterogeneous/` 2,908 LoC + `agents/homogeneous/` 196 LoC + 4-layer agent files ~998 LoC**; A/B gate validates old-vs-new task_generator equivalence on §0.2 kernels before PR-3 merges | HIGH (the speedup-critical reasoning path) | Saptarshi | −2,500 net (−4,200 old + 1,700 new) |

**Total LoC delta**: −9,000 (target: ~25,000 LoC, was 34,301).

**Dependencies**: PR-1 → PR-2 → PR-3 (sequential). PR-1 provides foundation (CLI, `SubagentBase`, `KernelLanguage`) that PR-2 and PR-3 depend on. PR-2 provides preprocess artifacts that PR-3's round loop consumes.

**Gating**: each PR green against:
- CI gates in §16.6
- Smoke tests in §11 (12 Triton markers + 11 HIP markers + add-language scaffolder + translate smoke + memory-analysis unit)
- `scripts/check_baseline_speedups.py` — §0.2 regression table green for PR-3 (the speedup-critical one)
- HarnessBuilder ≥29/30 fixture corpus green for PR-2

**Observation window**: after PR-3 merges, 1-week watch on `refactor-test` branch before cutting a tag. No 2-week freeze as in the 38-PR version — the 3-PR cadence doesn't have flag flips to observe separately.

---

### PR-1 · Foundation + Cleanup + CLI Consolidation

**Single-line summary**: Collapse multiple entry points + delete dead code + introduce `KernelLanguage` / `SubagentBase` foundation + CI safety net.

**Files created** (`+` = new):
```
+ src/minisweagent/cli.py                                           (~500 LoC — THE single Typer app)
+ src/minisweagent/kernel_languages/base.py                         (KernelLanguage dataclass)
+ src/minisweagent/kernel_languages/__init__.py                     (registry + detect_best)
+ src/minisweagent/kernel_languages/contract.py                     (validate_harness, validate_commandment)
+ src/minisweagent/kernel_languages/scaffolder.py                   (geak add-language + test-language)
+ src/minisweagent/kernel_languages/_templates/*.j2                 (7 scaffolder templates)
+ src/minisweagent/kernel_languages/triton/{kernel_language.py,*.md,*.j2}  (7 files)
+ src/minisweagent/kernel_languages/hip/{kernel_language.py,*.md,*.j2}     (7 files)
+ src/minisweagent/subagents/__init__.py
+ src/minisweagent/subagents/base.py                                (SubagentBase — peer of OptimizationAgent)
+ src/minisweagent/subagents/{preprocess,memory,translation}/__init__.py
+ docs/refactor/{EXECUTION_PLAN.md,CODEBASE_AUDIT.md,INVARIANTS.md}
+ tests/smoke/test_triton_hetero_invariants.py                      (12 markers)
+ tests/smoke/test_hip_homo_invariants.py                           (11 markers)
+ tests/smoke/test_cross_session_memory_analysis.py                 (unit, no GPU — stub for PR-3)
+ tests/smoke/test_add_language_scaffolder.py
+ tests/regression/baseline_speedups.yaml                           (§0.2 table)
+ tests/regression/test_kb_speedups.py                              (nightly-only)
+ scripts/check_one_cli_file.py
+ scripts/check_language_leaks.py                                    (WARN initially, FAIL after PR-2)
+ scripts/check_subagent_location.py
+ scripts/check_no_agent_inheritance.py                              (WARN initially, FAIL after PR-3)
+ scripts/check_tool_resolve_single_site.py                          (WARN initially, FAIL after PR-3)
+ scripts/check_subagent_base_contract.py
+ scripts/check_agent_tool_set.py                                    (WARN initially, FAIL after PR-3)
+ scripts/check_no_capability_loss.py                                (gates §13.1 audit)
```

**Files deleted** (`-` = removed):
```
# mini-swe-agent heritage (confirmed unreferenced in production)
- src/minisweagent/run/hello_world.py                               (36 LoC — tutorial)
- src/minisweagent/run/inspector.py                                 (212 LoC — only uses deleted interactive_textual)
- src/minisweagent/agents/interactive_textual.py                    (450 LoC — only used by inspector)
- src/minisweagent/run/mini_extra.py                                (44 LoC — meta-dispatcher)
- src/minisweagent/run/github_issue.py                              (91 LoC — not used in GEAK)
- src/minisweagent/run/extra/ (entire dir)                          (475 LoC — SWE-bench evaluator)
- src/minisweagent/environments/singularity.py                      (97 LoC — only in tests)
- src/minisweagent/environments/extra/bubblewrap.py                 (112 LoC — only in tests)
- src/minisweagent/environments/extra/swerex_docker.py              (47 LoC — only in tests)
- src/minisweagent/models/anthropic_model.py                        (35 LoC — verified unused)
- src/minisweagent/models/extra/roulette.py                         (62 LoC — SWE-bench-only)
- src/minisweagent/config/extra/swebench*.yaml                      (462 LoC — SWE-bench-only)
- src/minisweagent/models/test_models.py                            (52 LoC — move to tests/fixtures/models.py)

# duplicates (verified identical or re-export stubs)
- src/minisweagent/debug_runtime.py (root)                          (70 LoC — byte-identical to preprocess copy)
- src/minisweagent/tools/discovery_types.py                         (7 LoC — pure re-export stub)
- src/minisweagent/agents/shape_fixer_agent.py                      (7 LoC — pure re-export stub)

# second CLI entry + 9 auxiliary CLI entries (per user direction: only `geak -t "..."` survives)
- src/minisweagent/run/orchestrator.py                              (288 LoC — `geak-orchestrate`, homo unsupported)
- pyproject.toml entries for:
    geak-preprocess, kernel-profile, commandment, validate-commandment, run-harness,
    baseline-metrics, codebase-context, resolve-kernel-url, task-generator
  (The modules they point to stay — they're called from within the pipeline —
   but are no longer exposed as standalone CLI entries. Reviewer-raised: validate-commandment
   points to tools/validate_commandment.py which PR-1 also deletes safely because no
   callsites remain after the pyproject entry is removed.)
- src/minisweagent/tools/validate_commandment.py                    (7 LoC — re-export stub)
```

**Files renamed/moved**:
```
src/minisweagent/memory/cross_session_memory.py::classify_kernel_category
    → src/minisweagent/memory/cross_session/__init__.py::classify_kernel_category
   (+ update 3 callsites: agents/heterogeneous/orchestrator.py:291,
    run/postprocess/results.py:175, run/postprocess/results.py:537)
   (cross_session_memory.py stays as a DeprecationWarning re-export shim; deleted in PR-3)
```

**pyproject.toml final state**:
```toml
[project.scripts]
geak = "minisweagent.cli:app"   # ONLY entry point
```

**Behavior preserved**: `geak -t "..."` invocation works identically to today. `geak -t "..." --kernel-url ... --test-command ... --num-parallel N --gpu-ids ... --yolo --exit-immediately -o <logs_dir>` — same flags, same semantics. AKA continues to work unchanged.

**`cli.py` subcommands** (all via `@app.command()` — single file):
- `geak -t "..."` (default, routes to `optimize`) — today's behavior, kept
- `geak optimize ...` — explicit form
- `geak add-language NAME --extensions '.ext,...'` — scaffolder (PR-1)
- `geak test-language NAME` — scaffolder smoke test (PR-1)
- `geak translate --source X --target-language Y --output Z --test T` — added in PR-2
- `geak resume --preprocess-dir PATH` — added in PR-2 (replaces former `geak-orchestrate --resume`)

**Contract** (PR-1 asserts):
- All §11 smoke tests pass against pre-PR-1 behavior (baseline green)
- CI gates are WARN-only for things that land in later PRs (flip to FAIL after PR-3)
- `check_one_cli_file.py` is FAIL from PR-1 merge (no Typer apps outside `cli.py`)
- `check_subagent_base_contract.py` is FAIL from PR-1 merge (`SubagentBase` subclasses must override exactly one of `run`/`loop` — enforced even with zero subclasses yet, via empty-set check)
- §13.1 capability audit green (every non-aux capability preserved; aux CLI entries are explicitly marked DELETE per user direction)

**Rollback**: `git revert <PR-1 merge commit>`. All 10 old CLI entry points come back (by reverting `pyproject.toml`), and all deleted files restore from git history. ~10-minute rollback.

**Risk items + mitigations**:
| Risk | Likelihood | Mitigation |
|---|---|---|
| Deleting `tools/validate_commandment.py` breaks callers | low | Verified 0 production callers via `rg`; only the pyproject entry referenced it, and we delete that entry in the same PR |
| Deleting `memory/cross_session_memory.py` breaks 3 production callers | low (already handled by PR-1 move) | `classify_kernel_category` moved to `memory/cross_session/__init__.py`; 3 callsites updated in same commit; shim stays for one release cycle |
| `geak add-language` scaffolder produces malformed language bundles | low | `geak test-language` asserts contract validators pass on scaffolded output; smoke test `test_add_language_scaffolder.py` creates + validates + deletes a noop language |
| Someone has a script calling `geak-orchestrate` | low (only internal resume-from-preprocess path) | `geak resume --preprocess-dir PATH` subcommand in `cli.py` replaces it (added in PR-2) |
| Someone has a script calling the other 8 aux entries (`geak-preprocess`, etc.) | unknown | Per-user direction: they're not needed — only `geak -t "..."` matters. If anyone hits this, fix is to call the underlying Python function OR invoke `geak` which runs the full pipeline. |

---

### PR-2 · Preprocessing Refactor

**Single-line summary**: Replace 1386-LoC `preprocessor.py` monolith with `PreprocessOrchestrator` + 4 phases + 1 conditional Translation phase + 5 LLM subagents + Jinja commandment + contract validators.

**Depends on**: PR-1 merged (needs `KernelLanguage`, `SubagentBase`, `cli.py`).

**Files created**:
```
+ src/minisweagent/preprocess/orchestrator.py                       (~30 LoC — runs 4 phases + 1 conditional)
+ src/minisweagent/preprocess/phases/__init__.py
+ src/minisweagent/preprocess/phases/discovery.py                   (~200 LoC — resolve_kernel + codebase_context + test_discovery)
+ src/minisweagent/preprocess/phases/harness.py                     (~250 LoC — harness_decision + HarnessBuilder + validate_harness)
+ src/minisweagent/preprocess/phases/baseline.py                    (~200 LoC — baseline_execute + profile + build_metrics)
+ src/minisweagent/preprocess/phases/explore.py                     (~150 LoC — render_commandment + validate_commandment + kernel_analysis)
+ src/minisweagent/preprocess/phases/translation.py                 (~200 LoC — CONDITIONAL: calls TranslationLoop)
+ src/minisweagent/subagents/preprocess/harness_builder.py          (~350 LoC — LLM subagent, SubagentBase.run())
+ src/minisweagent/subagents/preprocess/kernel_analysis.py          (~300 LoC — [A]-[D] rubric, SubagentBase.run())
+ src/minisweagent/subagents/preprocess/unit_test_agent.py          (~260 LoC — merged from 2 copies, SubagentBase.run())
+ src/minisweagent/subagents/preprocess/shape_fixer.py              (~241 LoC — moved, SubagentBase.run())
+ src/minisweagent/subagents/preprocess/configs/*.yaml              (4 configs)
+ src/minisweagent/subagents/translation/translator.py              (~350 LoC — TranslationLoop, SubagentBase.loop())
+ src/minisweagent/subagents/translation/configs/translator.yaml
+ src/minisweagent/kernel_languages/_translation/triton_to_hip.md   (~150 LoC — hint pack)
+ src/minisweagent/kernel_languages/_translation/hip_to_triton.md
+ src/minisweagent/kernel_languages/_translation/_fallback.md       (generic src→tgt guidance)
+ src/minisweagent/run/benchmark_parsing.py                         (merged from pre+postprocess)
+ tests/fixtures/harness_corpus/                                    (30+ fixtures: 10 triton_direct, 6 triton_wrapper, 8 hip_raw, 4 hip_pybind, 2 edge)
+ tests/smoke/test_translate_triton_to_hip.py
+ scripts/harness_corpus_gate.py                                    (≥29/30 pass rate gate)
```

**Files deleted**:
```
- src/minisweagent/run/preprocess/preprocessor.py                   (1386 LoC — monolith replaced by orchestrator+phases)
- src/minisweagent/run/preprocess/commandment.py                    (530 LoC — replaced by Jinja commandment.j2 templates)
- src/minisweagent/run/preprocess/harness_utils.py                  (1660 LoC — template-generation parts deleted; runtime helpers move to run/harness_runtime.py)
- src/minisweagent/run/preprocess/benchmark_parsing.py              (381 LoC — merged into run/benchmark_parsing.py)
- src/minisweagent/run/postprocess/benchmark_parsing.py             (389 LoC — merged; 54-LoC diff reconciled)
- src/minisweagent/run/preprocess/unit_test_agent.py                (252 LoC — merged into subagents/preprocess/unit_test_agent.py)
- src/minisweagent/agents/unit_test_agent.py                        (251 LoC — same merge)
- src/minisweagent/run/preprocess/shape_fixer_agent.py              (241 LoC — moved to subagents/preprocess/shape_fixer.py)
- src/minisweagent/run/preprocess/validate_commandment.py           (280 LoC — moved to kernel_languages/contract.py in PR-1)
- src/minisweagent/memory/integration.py                            (104 LoC — merged into memory/cross_session/__init__.py)
- src/minisweagent/run/preprocess/context.py                        (113 LoC — merged into run/context_types.py)
```

**Files renamed**:
```
src/minisweagent/run/preprocess/ → src/minisweagent/preprocess/     (dir move; sibling of run/ not under it)
src/minisweagent/run/pipeline_types.py → src/minisweagent/run/context_types.py
src/minisweagent/run/pipeline_helpers.py (915 LoC) →
  src/minisweagent/run/model_factory.py      (~250)
  src/minisweagent/run/env_factory.py        (~200)
  src/minisweagent/run/context_injection.py  (~200)
  src/minisweagent/run/agent_filter.py       (~100)
  (~150 LoC dead helpers deleted)
```

**`geak translate` subcommand** (new in `cli.py`):
```bash
# Standalone translation: runs Translation phase only, exits
geak translate --source foo.py --target-language hip --output foo.hip --test test.py --max-attempts 3

# Integrated: translate + optimize
geak -t "Optimize /path/to/foo.py..." --target-language hip --max-rounds 8
```

Phase 0 (Translation phase) runs when `ctx.target_language != ctx.language` (only when user passes `--target-language`); else skipped.

**HIP bug #1 fix**: `ExplorePhase.validate_commandment()` + universal commandment enforcement in PR-3's `run_pipeline`. PR-2 produces the commandment correctly via Jinja; PR-3 wires the per-round execution contract.

**Log markers (reviewer-raised)**: PR-2 emits BOTH old markers (`--- Step 1/7: Resolve kernel URL ---` ... `--- Step 7/7: Commandment ---`) AND new phase markers (`Phase: Discovery` ... `Phase: Explore`) for a transition window. AKA is updated in parallel to consume the new markers. After AKA migration, old markers can be dropped (separate cleanup commit).

**HarnessBuilder gate** (PR-2 merge condition): fixture corpus ≥29/30 pass rate via `scripts/harness_corpus_gate.py`. Fixture corpus spec in §16.7.

**Risk items + mitigations**:
| Risk | Likelihood | Mitigation |
|---|---|---|
| HarnessBuilder produces malformed harnesses on real kernels | medium | 30-fixture corpus + `validate_harness()` contract checker; PR-2 gated on ≥29/30 pass |
| Jinja commandment byte-diverges from old Python commandment output | medium | Byte-level diff check via `scripts/pr2_commandment_diff.py` on §0.2 kernel set; must match within cosmetic whitespace |
| 5 new LLM subagents inflate per-run cost/latency | low-medium | Each subagent has `step_limit` in its YAML config; `HarnessBuilder` + `KernelAnalysisAgent` run ONCE per preprocess; `UnitTestAgent` + `ShapeFixer` run CONDITIONALLY; `TranslationLoop` only when user asks |
| `TranslationLoop` produces wrong-semantics kernel | low-medium | Golden-match tensor allclose verification; `validate_translation_performance()` with 0.5x fail / 0.8x warn thresholds; max_attempts=3 retry loop with feedback |
| Preprocess phases break hidden dependencies in old preprocessor.py | medium | Full 12-marker Triton + 11-marker HIP smoke tests; §0.2 regression suite green before merge |

---

### PR-3 · Orchestration + Optimization Refactor (the speedup-critical path)

**Single-line summary**: Single `OptimizationAgent` class + `run_pipeline(ctx, mode)` + `CrossSessionMemoryAnalysisAgent` + uniform KB write across modes + delete hetero/homo dirs.

**Depends on**: PR-1 + PR-2 merged.

**Files created**:
```
+ src/minisweagent/agents/optimization_agent.py                     (~500 LoC — collapsed 4-layer chain)
+ src/minisweagent/agents/agent_spec.py                             (kept; slight cleanup)
+ src/minisweagent/run/unified.py                                   (~400 LoC — run_pipeline(ctx, mode))
+ src/minisweagent/run/compose.py                                   (~300 LoC — compose_task_body)
+ src/minisweagent/run/task_generator.py                            (~700 LoC — unified planner; A/B-gated)
+ src/minisweagent/run/pool_runner.py                               (~400 LoC — single path for homo+hetero)
+ src/minisweagent/run/pool_logger.py                               (~150 LoC — split from parallel_helpers)
+ src/minisweagent/subagents/memory/cross_session_memory_analysis.py (~300 LoC — writes structured insights file)
+ src/minisweagent/subagents/memory/configs/cross_session_memory_analysis.yaml
+ src/minisweagent/kernel_languages/triton/memory_hints.md          (~100 LoC — regex patterns moved here)
+ src/minisweagent/kernel_languages/hip/memory_hints.md
+ scripts/ab_task_generator.py                                      (PR-3 A/B gate)
+ scripts/memory_ablation.py                                        (optional post-merge ablation)
+ tests/agents/test_optimization_agent.py                           (rewrites 3 old test files)
+ tests/agents/test_tool_resolve.py
```

**Files deleted**:
```
- src/minisweagent/agents/default.py                                (611 LoC)
- src/minisweagent/agents/interactive.py                            (181 LoC)
- src/minisweagent/agents/strategy_agent.py                         (156 LoC)
- src/minisweagent/agents/strategy_interactive.py                   (50 LoC)
- src/minisweagent/agents/homogeneous/ (entire dir)                 (196 LoC — replaced by mode="fixed")
- src/minisweagent/agents/heterogeneous/ (entire dir)               (2,908 LoC — replaced by mode="planned" + moved files)
- src/minisweagent/agents/parallel_agent.py homo branch             (~196 LoC of lines 414-609)
- src/minisweagent/memory/cross_session_memory.py                   (27 LoC — deprecation shim from PR-1 finally removed)
- old tests/agents/test_default.py + test_interactive.py + test_homogeneous_agent.py  (superseded by new test_optimization_agent.py)
- src/minisweagent/run/utils/prompts.py                             (121 LoC — folded into kernel_languages/*/*.md + compose.py)

# moved from heterogeneous/ to top-level tools/
src/minisweagent/agents/heterogeneous/tools.py          → src/minisweagent/tools/orchestrator_tools.py
src/minisweagent/agents/heterogeneous/schemas.py        → src/minisweagent/tools/orchestrator_schemas.py

# memory formatter: Triton/HIP regex patterns (~150 LoC) moved into per-language memory_hints.md files
src/minisweagent/memory/cross_session/formatter.py                  (slimmed from ~400 to ~80 LoC; just high-confidence fast-path removed)
```

**`cli.py` changes**: add `--mode {fixed,planned,auto}` flag (default: `auto`). Deprecate `--heterogeneous` with DeprecationWarning (maps `true→planned`, `false→fixed`). Add `GEAK_USE_CROSS_SESSION_MEMORY=1/0` env flag (default `1` on PR-3 merge — single-path design, memory on by default).

**A/B gate** (PR-3 merge condition — reviewer Tier-1 Issue 2 fix):
- `scripts/ab_task_generator.py` runs OLD `heterogeneous/task_generator.py` vs NEW `run/task_generator.py` on §0.2's 13 kernels × 3 seeds × 5 rounds = **195 runs per arm × 2 arms = 390 runs** × ~15 min / 4 GPUs ≈ **~1 wall-clock day on 4 GPUs**.
- Pass criterion: new generator's best-of-3-seeds speedup ≥ 0.95× old generator's per kernel, AND geomean ≥ 0.98× old across 13 kernels.
- Gate evidence attached to PR-3 description as `docs/refactor/PR3_AB_REPORT.md`.

**Memory ablation** (post-merge, not a gate — run on a dedicated weekend slot to validate the `GEAK_USE_CROSS_SESSION_MEMORY=1` default):
- 6-kernel subset × 3 seeds × **2 arms** (`no_kb` vs `subagent_kb`) × 5 rounds = **180 runs ≈ 0.5 GPU-day on 4 GPUs**.
- Pass criterion: subagent_kb geomean ≥ 1.05× no_kb on 6 kernels AND strictly better on rounds-to-first-speedup in ≥4/6 kernels.
- If ablation fails, revert `GEAK_USE_CROSS_SESSION_MEMORY` default to `0` (1-line env var change); the subagent path stays available as opt-in.

**Risk items + mitigations**:
| Risk | Likelihood | Mitigation |
|---|---|---|
| Collapsing 4-layer agent chain loses subtle inheritance behavior | medium | 3 old test files rewritten into `test_optimization_agent.py` against new contract; full §0.2 regression; A/B gate catches task_generator regressions |
| New `task_generator.py` underperforms old Triton-centric version | medium-high | **Hard A/B gate before PR-3 merges** — 13 kernels × 3 seeds × 2 arms, geomean ≥ 0.98× required. Rollback = revert PR-3 |
| `CrossSessionMemoryAnalysisAgent` produces poor recommendations | low-medium | Structured output spec (§10.5); single path avoids regression-window complexity of earlier fast/slow router design; ablation validates `GEAK_USE_CROSS_SESSION_MEMORY=1` default |
| Uniform per-round FULL_BENCHMARK in homo path breaks HIP | medium | Bug #1 fix verified against ≥3 HIP kernels with observable FULL_BENCHMARK output; smoke test extended |
| Uniform KB write across modes floods KB with low-quality entries | low | Same `GEAK_MEMORY_MIN_SPEEDUP=1.10` threshold as today; bug #6 fix just makes homo path also respect it |
| Memory LLM-subagent cost becomes the bottleneck | low-medium | Subagent output file is capped by the YAML `step_limit`; if LLM cost becomes an issue, `GEAK_USE_CROSS_SESSION_MEMORY=0` returns to no-memory mode (permanent opt-out) |

---

### End-state (after PR-1 + PR-2 + PR-3)

- **1 CLI file** (`cli.py`) — enforced by CI. **Only `geak -t "..."` and its subcommands survive**; all 9 auxiliary CLI entries deleted per user direction.
- **1 agent class** (`OptimizationAgent`) — no inheritance; enforced by CI.
- **1 pipeline** (`run_pipeline(ctx, mode)`) — unified across Triton + HIP + future languages.
- **1 subagent folder** (`src/minisweagent/subagents/`) with 3 purpose subfolders — enforced by CI.
- **6 LLM subagents**: HarnessBuilder, KernelAnalysisAgent, UnitTestAgent, ShapeFixer, TranslationLoop, CrossSessionMemoryAnalysisAgent. All in `subagents/`, all extending `SubagentBase`.
- **4 preprocess phases** (Discovery, Harness, Baseline, Explore) + 1 conditional (Translation) — NOT 11 flat steps.
- **1 way to add a language**: `geak add-language X` + fill 7 files in `kernel_languages/X/`.
- **1 way to translate**: `geak translate ...` (standalone) OR `geak -t "..." --target-language Y` (integrated).
- **0 `kernel_type == "..."` branches in core** — enforced by `check_language_leaks.py` FAIL.
- **~25,000 LoC** (down from 34,301; −9,000 net).
- **Nightly §0.2 regression suite** (13 kernels, memory on/off) + 5 smoke tests + 7 CI gates.---

## §9 — Preprocess orchestrator code sketch (what it looks like after PR-2)

### §9.1 — Language-branching-sites audit (reviewer Issue 2.3 fix)

The plan claims "16+ language-branching sites in core code". Exact incantation to reproduce:

```bash
# Narrow pattern (literal equality): finds 3
rg -n 'kernel_type\s*==\s*"(triton|hip)"|kernel_language\s*==\s*"(triton|hip)"' src/minisweagent/

# Broad pattern (semantic sites — prompt templates, dict dispatch, @triton decorator, tl., rocprof, hipcc, _LANGUAGE_GUIDANCE maps, strategy YAML language mentions): finds 20 files with matches
rg -c 'kernel_type|kernel_language|@triton\.jit|\btl\.|rocprof|hipcc|rocminfo|_HIP_|_TRITON_|_LANGUAGE_GUIDANCE' src/minisweagent/ | grep -v ':0$'
```

From `GEAK_codebase_audit.md` §9, the full enumeration (**16 distinct semantic decision points across 14 files**) is:

1. `run/mini.py:61-67` — `_normalize_kernel_type` string normalizer
2. `run/mini.py:331-342` — parsed_config normalization + `_infer_kernel_type` fallback
3. `run/mini.py:498-512` — mode auto-detect: `if _auto_kernel_type == "triton": heterogeneous=True`
4. `run/mini.py:514` — `if heterogeneous:` pipeline branch
5. `agents/heterogeneous/task_generator.py:74-105` — `_infer_kernel_type`
6. `agents/heterogeneous/task_generator.py:22,183-210,276` — `kernel_type` / `kernel_language` params through API
7. `agents/heterogeneous/prompts.py:8-131` — `SYSTEM_PROMPT` Triton-biased
8. `agents/heterogeneous/prompts.py:133-281` — `TASKGEN_SYSTEM_PROMPT` Triton-centric
9. `agents/heterogeneous/workload_guidance.py:13-78` — `_HIP_SEARCH_HINT_PATTERNS` + `_is_hip_like_kernel`
10. `run/preprocess/discovery_types.py:36-37, 247-253` — `_infer_kernel_language`
11. `run/preprocess/commandment.py:55-127` — `generate_commandment(kernel_language)` drives inner-kernel vs simple
12. `agents/unit_test_agent.py:40-75` — `_LANGUAGE_GUIDANCE` dict (triton/hip/ck/asm)
13. `run/preprocess/unit_test_agent.py` — duplicate of #12 (same dict, two files, ~250 LoC each)
14. `memory/cross_session/extractor.py:500-505` — path-string sniffing for language
15. `config/mini_kernel_strategy_list.yaml` (system_template + instance_template, 14 KB) — Python/Triton conventions baked in
16. `config/mini_unit_test_agent.yaml` (8 KB) + `run/preprocess/config/mini_unit_test_agent.yaml` (17 KB) — duplicate configs with language guidance

After the refactor, every one of these 16 sites either:
- moves to `kernel_languages/<name>/` as per-language data (templates, prompt markdown), OR
- is deleted (when the decision point collapses into a single `ctx.language.X` read).

Zero `kernel_type == "triton"|"hip"` literal equalities remain in core code — enforced by `check_language_leaks.py` (FAIL-strict after PR-2).

### §9.2 — Orchestrator code sketch (post-PR-2)

```python
# src/minisweagent/preprocess/orchestrator.py — the WHOLE orchestrator

class PreprocessOrchestrator:
    """Runs the preprocess phases for any KernelLanguage.

    4 mandatory phases + 1 conditional translation phase.
    No language branching; every phase reads ctx.language and dispatches via
    templates or the registered subagent.
    """

    def __init__(self, ctx: PreprocessContext):
        self.ctx = ctx

    def run(self) -> PreprocessContext:
        ctx = self.ctx

        # Translation phase — CONDITIONAL. Runs only when translation is requested.
        # After this phase, ctx.kernel_path points to the TRANSLATED file; all
        # downstream phases operate on that, not the original.
        # Runs only when the user explicitly asked for a DIFFERENT target language.
        # Default: ctx.target_language == ctx.language, so the phase is skipped
        # with zero ceremony for regular optimization runs.
        if ctx.target_language != ctx.language:
            ctx = TranslationPhase(ctx).run()
            if ctx.translate_only:
                return ctx                      # `geak translate` exits here

        ctx = DiscoveryPhase(ctx).run()       # resolve + codebase + test_discovery
        ctx = HarnessPhase(ctx).run()         # harness_decision + builder + validate
        ctx = BaselinePhase(ctx).run()        # baseline + profile + metrics
        ctx = ExplorePhase(ctx).run()           # commandment + validate + kernel_analysis
        return ctx
```

### What each phase looks like (one example)

```python
# src/minisweagent/preprocess/phases/harness.py

class HarnessPhase:
    """Produce a validated test harness conforming to the universal contract.

    Calls 2 sub-steps + 1 validator. 1 LLM subagent (HarnessBuilder).
    """

    def __init__(self, ctx: PreprocessContext):
        self.ctx = ctx
        self.language = ctx.language

    def run(self) -> PreprocessContext:
        ctx = self.ctx

        # B.1 — decide between existing user harness, UTA generation, or builder adaptation
        if not ctx.discovery.tests:
            ctx = self._run_unit_test_agent(ctx)      # LLM subagent (conditional)

        # B.2 — run HarnessBuilder to adapt user tests into standard contract
        ctx = self._run_harness_builder(ctx)          # LLM subagent (always)

        # B.3 — validate the produced harness against the contract
        validate_harness(ctx.harness_path, self.language)  # raises ContractViolation
        return ctx

    def _run_unit_test_agent(self, ctx): ...
    def _run_harness_builder(self, ctx): ...
```

**Read this flow**: 4 mandatory phases + 1 conditional (Translation). 3 one-shot LLM subagents (+ 1 for translation); 2 contract validators; everything else deterministic. No `if language.name == "triton"` anywhere. Adding a new phase = adding a new class in `phases/`.

---

## §10 — Deterministic vs LLM subagent table

| Component | Today | After | Why |
|---|---|---|---|
| Language detection | String-matching in 3 places | `registry.detect_best()` (pure data) | Pattern match; LLM overkill |
| Codebase context | Deterministic walk | Same | Mechanical |
| Test discovery | MCP call | Same | Already exists |
| **Harness production** | UTA when no file; else copy | **HarnessBuilder LLM subagent** | Needs to reason about test file shape |
| Harness validation | None | `validate_harness` (deterministic) | Regex + subprocess |
| Baseline execution | Subprocess | Same | Mechanical |
| Profiling | Metrix MCP | Same | Already exists |
| Metrics assembly | JSON | Same | Mechanical |
| **Commandment** | 530-LoC Python branching | **Pure Jinja render** | Data substitution |
| Commandment validation | None | `validate_commandment` | Deterministic |
| Kernel analysis rubric | Missing | **KernelAnalysisAgent LLM subagent** | Needs reasoning |
| Unit test generation (no tests) | UTA LLM | Same | Needs reasoning |
| Shape fixing | Shape-fixer LLM | Same | Needs reasoning |
| **Kernel translation** | — | **TranslationLoop multi-round subagent** (composes OptimizationAgent; different verification + exit condition) | Needs semantic preservation reasoning + golden-match loop |
| Patch verification | `save_and_test` subprocess | Same | Already deterministic |
| Golden-match (translation) | — | Deterministic tensor allclose | Pure comparison |
| **KB top-k synthesis** | **Deterministic regex + 40 KB unstructured raw injection (formatter.py)** | **CrossSessionMemoryAnalysisAgent** (runtime subagent; single path per round when `GEAK_USE_CROSS_SESSION_MEMORY=1`; writes structured `cross_session_memory_insights.md` with analysis + ranked recs + FULL reference KB entries) | Deciding which retrieved experience applies, ranking recommendations, and flagging dead-ends is reasoning — not pattern-matching. Structured output preserves raw source material (diffs, baseline codes, profiler traces) that the main agent needs to cross-reference; subagent's value is curation + ranking + rationale, not hiding data. Today's `_PARAM_PATTERNS` regex is Triton-hardcoded (language-leak). |
| KB retrieval (top-k ranking) | Deterministic (code similarity + scaled success) | Same | Ranking is pure math; stays in `retriever.py` |

**Total LLM subagents in the whole system**: 6
- 4 one-shot preprocess: `UnitTestAgent`, `HarnessBuilder`, `KernelAnalysisAgent`, `ShapeFixer`
- 1 standalone multi-round loop: `TranslationLoop` (only for `geak translate`; composes `OptimizationAgent` but has distinct verify + exit)
- 1 per-round runtime: `CrossSessionMemoryAnalysisAgent` (always the path when `GEAK_USE_CROSS_SESSION_MEMORY=1`)

Everything else is deterministic. All 6 live in `src/minisweagent/subagents/`.

---

## §10.5 — `CrossSessionMemoryAnalysisAgent` design deep-dive

### Why this is a subagent instead of deterministic injection

Today's `memory/cross_session/formatter.py` does three things that should not be deterministic:

1. **Relevance judgment** — deciding which of top-k retrieved experiences applies to the current kernel, and in what priority. Done today implicitly by the main optimization agent while it's also supposed to be generating a patch.
2. **Conflict resolution** — when KB entry 1 says "fuse X and Y" and KB entry 2 says "avoid fusion on shape M", the formatter dumps both raw. The main agent has to disentangle.
3. **Language-specific extraction** — `_PARAM_PATTERNS` regex hardcodes `num_warps`, `tl.constexpr_dtype`, `hipLaunchKernelGGL`. This is a language leak living in core memory code that CI should reject once `check_language_leaks.py` gets strict.

And the output volume is enormous: `_MAX_CONTEXT_FULL = 40_000` characters of raw KB data pushed into every task body.

A focused subagent converts this to a structured `cross_session_memory_insights.md` (~5-25 KB) — analysis + ranked recommendations + FULL reference KB entries (baseline code, winning diffs, profiler traces, dead-ends). It moves the language-specific regex extraction into the subagent's prompt (legal — inside `subagents/`, not in `memory/` core), preserves the raw source material the main agent benefits from (diffs, baseline codes), and adds curation + ranking + rationale that the deterministic regex cannot produce.

### Single memory path (NOT a router)

Earlier drafts of this plan had a "smart router" that fast-pathed high-confidence matches through a deterministic formatter and slow-pathed everything else through the subagent. The user rejected that design: it added complexity, introduced a PR-70→PR-71 regression window, and made the subagent's reliability untestable in production (because most runs would bypass it). The final design is simple:

- `GEAK_USE_CROSS_SESSION_MEMORY=1` → the subagent runs every round, writes `artifacts/cross_session_memory_insights.md`, that file is injected. No deterministic fast-path. No router. No `format_single_high_confidence`.
- `GEAK_USE_CROSS_SESSION_MEMORY=0` → no retrieval, no subagent call, no context. Agent explores from scratch.

```python
# src/minisweagent/run/unified.py — call site in run_pipeline (per round, per task)

def _maybe_inject_memory_context(ctx, round_idx: int) -> str:
    """One path. Flag on → subagent; flag off → empty."""
    if os.environ.get("GEAK_USE_CROSS_SESSION_MEMORY", "1") != "1":
        return ""
    retrieved = cross_session.retrieve(ctx)          # top-k by code similarity
    if not retrieved:
        return ""                                     # nothing in KB to synthesize
    analysis_md_path = CrossSessionMemoryAnalysisAgent(ctx.language).run(
        target_code=ctx.target_kernel_code,
        target_profile=ctx.profiler_summary,
        retrieved=retrieved,
        # Subagent reasons about staleness from content (profiler patterns, stored
        round_number=round_idx,
        out_path=ctx.artifacts_dir / "cross_session_memory_insights.md",
    )
    return analysis_md_path.read_text()              # ~5-25 KB structured markdown
```

### Subagent I/O contract

**Inputs** (passed as prompt body; no file-system tools needed in v1):
- `target_code`: full source of the kernel being optimized
- `target_profiler`: Metrix JSON summary (top-5 hotspots, arithmetic intensity, memory roofline ratio, shape regime)
- `language`: `KernelLanguage` name + `system_prompt` + `idioms` (so the subagent knows the target language's patterns)
- `retrieved`: list of up to 3 `ExperienceRecord`s, each containing:
  - `baseline_code` (full source of the KB kernel)
  - `winning_diff` (unified diff)
  - `strategy_name` + `strategies_tried` + `dead_ends` (with reasons)
  - `profiler_before` / `profiler_after`
  - `speedup`
  - (no version-hash flag; subagent reads stored profiler/code patterns directly to detect staleness)
  - `code_sim` (0.0 – 1.0)

**Process** (one LLM call, no tool loop, typical latency 5-15s):
1. Compute applicability score for each retrieved experience given current kernel + profiler.
2. Identify conflicts or redundancies between entries.
3. Emit structured recommendation.

**Output** (`cross_session_memory_insights.md` — structured markdown, NOT just a JSON-rendered summary). The subagent produces a file with this structure (reviewer-raised: include FULL reference info from retrieved experiences, not just distilled recommendations):

```markdown
# Cross-Session Memory Insights for <current_kernel_name>

Generated by CrossSessionMemoryAnalysisAgent at <timestamp>
Retrieved top-k = 3 experiences.

---

## 1. Analysis Summary

- **Overall assessment**: Entry 2 is high-confidence (prioritize its strategy).
  Entry 1 is a known dead-end for this shape regime.
  Entry 3 is tangential (different op family — ignore).
- **Staleness scan (from content)**: Entry 2's profiler_before matches the current
  profile shape (both compute-bound, similar hotspots) — strategy likely still
  applies. Entry 1's stored diff references `num_stages=4` pipelining which caused
  regression in the KB run; avoid regardless of env. Entry 3 references API
  patterns that are not present in current kernel — likely tangential.
- **none_applicable**: false

## 2. Top Recommended Strategies

### 2.1 Strategy: Cast tl.arange to int32; set XBLOCK=128
- **Source**: KB entry 2 — `fused_rms_fp8` (speedup 2.23x)
- **Confidence**: high
- **Why applicable**: Current kernel has identical `tl.arange + offset` pattern;
  shape regime (M=512, D=1024) matches KB entry within 2x.
- **Concrete pattern**:
  ```python
  offsets = tl.arange(0, BLOCK).to(tl.int32)
  ```

### 2.2 ...

## 3. Avoid / Known Dead-Ends

### 3.1 num_stages=4 pipelining
- **Source**: KB entry 1 — `gemm_a16wfp4` (regressed to 0.83x)
- **Why**: Caused regression on similar shape; memory pattern here does not
  benefit from pipelining.

## 4. Reference: Full KB Entries

### Entry 1: `gemm_a16wfp4` (best_speedup 1.43x, this entry was a regression)
- **Code similarity to current**: 0.42
- **Baseline fingerprint match**: no (aiter commit differs)
- **Baseline code** (full source of the KB kernel):
  ```python
  @triton.jit
  def gemm_a16wfp4_kernel(...):
      ...
  ```
- **Winning diff** (unified patch):
  ```diff
  --- a/kernel.py
  +++ b/kernel.py
  @@ ... @@
  -    BLOCK_M = 64
  +    BLOCK_M = 128
  ...
  ```
- **Profiler BEFORE**: latency=45ms, roofline=0.3 compute-bound; hotspots: tl.dot (72%).
- **Profiler AFTER**: latency=31ms, roofline=0.5; hotspots: tl.dot (60%), shared mem (15%).
- **Strategies tried**:
  - BLOCK_M=128: +1.43x  ← winning
  - num_stages=4: 0.83x  ← REGRESSION
  - swizzle reorder: 1.07x
- **Dead-ends**: {num_stages=4: "register pressure overflow"}

### Entry 2: `fused_rms_fp8` (best_speedup 2.23x)
  (same structure as Entry 1)

### Entry 3: `fused_qkv_rope` (best_speedup 1.13x)
  (same structure as Entry 1)
```

**Size expectation**: ~5-25 KB depending on diff and baseline-code sizes (still much less than today's 40 KB raw dump, AND includes active analysis + raw source material so the agent can cross-reference the recommendation with the actual diff).

The full markdown contents are read by `compose_task_body()` and prepended to the main agent's task body. The agent sees ALL the KB material (not a summary); the subagent's value is **curation + ranking + rationale**, not data-hiding.

### Lifecycle — where in the flow it runs

```
preprocess (once) ──►  round 1  ──►  round 2  ──►  round 3  ──►  …
                        │             │             │
                        ▼             ▼             ▼
                  compose_task_body()
                        │
                        └─► retriever.retrieve(top-k)
                              │
                              └─► IF GEAK_USE_CROSS_SESSION_MEMORY=1:
                                        └─► CrossSessionMemoryAnalysisAgent.run(
                                                target_code, target_profile,
                                                retrieved)
                                             └─► writes artifacts/cross_session_memory_insights.md (~5-25 KB structured)
                                             └─► contents injected into task_body
                                     ELSE:
                                        └─► no injection; agent explores from scratch
```

Called fresh at the start of every round (the retrieved set may change as the agent makes progress; the subagent re-evaluates with the latest state).

### Rollout — behind a flag + ablation

`GEAK_USE_CROSS_SESSION_MEMORY=1` env flag (renamed from the earlier `GEAK_USE_CROSS_SESSION_MEMORY` — shorter, matches the user-facing semantic "enable cross-session memory?"). Default `0` on PR-70; flipped to `1` in PR-73 after the ablation gate.

**Ablation experiment (gate for flipping default to `1` — binding spec; PR-72 in §8 matches these numbers exactly)**:

- **Gate run (required, green before PR-73)**:
  - **6-kernel representative subset**: `gemm_a16w16_atomic`, `fast_rms_layernorm`, `fused_rms_fp8`, `topk`, `fused_qkv_rope`, `fused_qk_rope_cache_mla` (2 each from §0.2's high/mid/low speedup tiers)
  - **3 seeds × 2 arms × 5 rounds** = 180 runs
  - Arms: (a) `GEAK_USE_CROSS_SESSION_MEMORY=0` (no memory), (b) `GEAK_USE_CROSS_SESSION_MEMORY=1` (subagent — the single enabled path)
  - **Scheduling**: ~15 min avg per run × 180 / 4 GPUs ≈ **~11 wall-clock hours (~0.5 GPU-day on 4 GPUs)**
- **Confirmation run (optional, post-PR-73)**: full 18 kernels × 4 seeds × 2 arms × 5 rounds = 720 runs (~4 GPU-days on 4 GPUs). Runs on a dedicated weekend slot; not gating.
- **Metrics**: best speedup, rounds-to-first-≥1.10x, context tokens used, % of valid patches.
- **Pass criteria to flip default** (simpler now — only 2 arms):
  - arm (b) `subagent_kb` speedup ≥ 1.05× arm (a) `no_kb` geomean-across-6-kernels (memory must beat no-memory), AND
  - arm (b) strictly better on ≥ 4/6 kernels in `rounds-to-first-speedup` (memory should accelerate first-speedup discovery), AND
  - arm (b) `cross_session_memory_insights.md` file size ≤ 25 KB on average (on-disk budget honored — includes ranked recs AND full reference KB entries with diffs/baseline code/profiler; compare vs today's unstructured 40 KB dump).
- **Evidence artifact**: `docs/refactor/PR72_ABLATION_REPORT.md` attached to PR-73 with per-kernel CSV.

**Rollback**: 1-line env var change (`GEAK_USE_CROSS_SESSION_MEMORY=0`) reverts to no-memory mode. The subagent path is the only memory path; disabling it removes memory injection entirely.

### Language-leak cleanup bonus

The current `formatter.py` has 10 Triton/HIP-specific regex patterns. After PR-71:
- `_PARAM_PATTERNS` moves into `kernel_languages/triton/memory_hints.md` and `kernel_languages/hip/memory_hints.md` as natural-language hints fed to the CrossSessionMemoryAnalysisAgent prompt.
- A new `KernelLanguage.memory_hints: str` field holds these.
- `formatter.py` shrinks by ~150 LoC.
- `check_language_leaks.py` goes from WARN to FAIL on the memory subsystem.

---

## §10.6 — What else could be a subagent? (discussion; NOT added to plan yet)

Reviewing every pipeline stage against the decision rule **"use a subagent only when judgment/reasoning beats deterministic logic; keep determinism everywhere else to avoid hallucination"**:

### Real candidates already implicitly subagents — should formally become subagents later

These two already make LLM calls, just aren't labeled as subagents and aren't in `subagents/`. Flagging for future review (not adding to the refactor plan now):

#### Candidate A — `TaskGenerator` (the per-round planner)

- **Today**: `agents/heterogeneous/task_generator.py` (997 LoC, 38 LLM-call references). Takes strategy hints + round history + profile + KB context, asks an LLM to produce N task specs per round. Already the largest LLM subagent in the system by call volume.
- **After Phase 3 (PR-31)**: Lives in `run/task_generator.py`, still makes LLM calls.
- **Worth promoting?** Yes — for consistency with the "one folder, one pattern" principle. It's an LLM agent that produces structured output; it should extend `SubagentBase` and live at `subagents/planning/task_generator.py`.
- **Risk of leaving it in `run/`**: drift. Two patterns ("`SubagentBase` subagents" and "LLM-calling code in `run/`") will diverge; next person adding a subagent-like thing won't know which to follow.
- **Why not add to current plan**: would expand Phase 3 scope. The current plan treats it as a library module in `run/task_generator.py`. Moving it to `subagents/planning/` can be a follow-up refactor (PR-80+) once the base pattern has proven itself in Phases 2, 6, 7.

#### Candidate B — `SelectPatchAgent` (post-round patch selector)

- **Today**: `agents/select_patch_agent.py` (224 LoC). Explicitly subclasses `DefaultAgent`. Reads N candidate patches, asks LLM which is best. Already an LLM subagent in all but name.
- **Worth promoting?** Yes. Belongs at `subagents/selection/patch_selector.py` after PR-50 collapses the agent hierarchy.
- **Why not add to current plan**: PR-50 already deletes the 4-layer inheritance chain (DefaultAgent etc.). Moving `SelectPatchAgent` to the new folder + making it extend `SubagentBase` is a natural follow-up to PR-50, but it doesn't change behavior — it's cosmetic. Defer to PR-80.

### Candidates worth discussing but probably NO (keep deterministic)

| Candidate | Why "no" |
|---|---|
| **Codebase context summarizer** (`preprocess/codebase_context.py`, 499 LoC) | Walks repo + assembles snippet. Deterministic is correct; LLM would add nondeterminism + cost without clear signal gain. Only "maybe" if codebases exceed ~100KB and need summarization — separate feature, not unification. |
| **Commandment rendering** | Jinja + validator is the right abstraction. Pure data substitution. LLM would risk hallucinating test commands. |
| **Profile interpreter** | Already covered by `KernelAnalysisAgent` (it reads profile and produces [A]-[D] rubric). Don't duplicate. |
| **Strategy manager** (`tools/strategy_manager.py`, 742 LoC, 0 LLM refs) | Pure JSON state machine. The agent writes to it via tool calls; reasoning happens at call time, not inside the manager. Keep as tool. |
| **Benchmark validator** | `validate_harness` contract is deterministic regex + subprocess. Rejecting malformed harnesses should not use an LLM — that's exactly the case where hallucination is dangerous. |
| **Error triager** (interpreting save_and_test stderr) | Already inside `OptimizationAgent`'s natural loop (the agent sees stderr and reasons). Spawning a subagent here adds latency for a task the main loop already handles. |
| **Config auto-tuner** (`max_rounds`, `num_parallel`) | Overengineering. User-set. |
| **KB retrieval / ranking** | Pure code-similarity + scaled-success math. Deterministic is correct. `CrossSessionMemoryAnalysisAgent` reasons over the RETRIEVED results, not the retrieval itself. |
| **KB memory-context injection control** (the flag decision) | `GEAK_USE_CROSS_SESSION_MEMORY` env check. Pure boolean. |
| **Golden-match** (translation verification) | Tensor allclose. Pure numeric comparison. |
| **Language detection** | Pattern match against file content/extension. Deterministic is correct; ambiguity is rare and falls through to user-specified flag. |

### Marginal — could go either way, flag for future observation

| Candidate | Pro | Con |
|---|---|---|
| **`result_scanning.py`** (heterogeneous round-to-round summarizer, 210 LoC, 2 LLM refs → essentially deterministic) | Could upgrade to LLM subagent: "given round N results, what signals should inform round N+1's planner?" | The planner (`TaskGenerator`) already consumes `result_scanning`'s output and reasons over it. Adding another LLM layer between them is likely redundant. Keep deterministic for now; revisit only if planner is demonstrably miss-reasoning about round history. |

### Bottom line

- **Current 6 subagents are the right set.** The pipeline has reached a clean "LLM subagent where reasoning is needed; deterministic everywhere else" equilibrium.
- **2 existing LLM-calling modules** (`TaskGenerator`, `SelectPatchAgent`) **should be formally reclassified as subagents later** (PR-80+ follow-up) — they already behave like subagents, they just don't live in the folder. This is cosmetic / consistency, not a behavior change.
- **Nothing else in the pipeline benefits from being an LLM subagent.** The deterministic components (validators, Jinja renderers, retrievers, routers, contract checkers, profile collectors) are deterministic for a reason: that's where hallucination would silently corrupt results, and the logic is regular enough that LLMs don't add value.
- **The principle holds**: "LLM subagents where judgment is needed; deterministic everywhere else." This is the design rule future contributors inherit.

---

## §10.7 — RAG vs cross-session memory (where does RAG fit?)

Reviewer-raised: with the new `CrossSessionMemoryAnalysisAgent` + `cross_session_memory_insights.md` design, where does RAG (from PR #90) fit in? Is it replaced? Kept? Gated? The answer: **kept and complementary — they serve different retrieval tasks.**

### The two retrieval systems side-by-side

| Dimension | Cross-session memory (our KB) | RAG (PR #90 MCP) |
|---|---|---|
| **Corpus** | `memory/cross_session/knowledge_base.json` — past GEAK runs: kernel code, winning diffs, profiler traces, strategies, dead-ends, speedups | Generic documentation — Triton docs, HIP programming guide, ROCm reference, kernel optimization patterns, papers |
| **Unit of retrieval** | `ExperienceRecord` (a specific past kernel optimization run) | Document chunks (a paragraph from Triton's `tl.dot` docs, a GEMM tiling pattern, etc.) |
| **Access pattern** | **Pre-round subagent call** — `CrossSessionMemoryAnalysisAgent` retrieves top-k once, synthesizes `cross_session_memory_insights.md`, agent reads the file | **Per-turn agent tool call** — main agent calls `query(...)` or `optimize(...)` during its tool-using loop whenever it needs a doc lookup |
| **Who reads** | Subagent (raw read) → main agent (via synthesized markdown file) | Main agent directly (via MCP tool) |
| **Typical query** | "What did GEAK try on similar fp8 RMS-norm kernels before? What worked / failed?" | "What's the signature of `tl.dot_scaled`?" / "How do I use HIP cooperative groups?" |
| **Result shape** | Structured markdown: ranked recs + full KB entries | Text snippets from docs |
| **Size** | `cross_session_memory_insights.md` ~5-25 KB | Per-tool-call: typically ~1-3 KB (RAG postprocessor wraps) |
| **Cadence** | Once per round | Whenever the agent decides to look something up during a round |
| **Gated by flag** | `GEAK_USE_CROSS_SESSION_MEMORY=1` (default after PR-73) | `rag: true/false` in config (unchanged from PR #90) |
| **Language-specific?** | Yes — filtered by `ctx.language.kb_namespace` | No — RAG corpus is cross-language (it has Triton AND HIP AND general GPU content) |
| **Subagent mediates?** | Yes — `CrossSessionMemoryAnalysisAgent` | No — agent calls RAG tool directly |

### Why both

Cross-session memory answers **"what worked before on kernels like this one?"** — kernel-specific, past-experience-grounded. When the KB has high-similarity matches, it accelerates optimization dramatically (see §0.2 speedups).

RAG answers **"what does the language/hardware documentation say?"** — generic, reference-grounded. It's how the agent figures out API signatures, flag meanings, and hardware constraints without a human having to include those in the prompt.

Neither subsumes the other:
- Cross-session memory cannot tell you what `tl.dot_scaled`'s arguments are (docs task — RAG).
- RAG cannot tell you that GEAK tried `num_stages=4` on a similar kernel and it regressed to 0.83x (experience task — cross-session).

### How they interact in the round loop

```
Round start:
  1. Pre-round: CrossSessionMemoryAnalysisAgent.run() → cross_session_memory_insights.md
     (IF GEAK_USE_CROSS_SESSION_MEMORY=1)
  2. compose_task_body(language, mode, memory_ctx=read(insights_file)) → task_body
  3. OptimizationAgent(tools=[..., query, optimize, ...]).run(task_body)
     ▸ Agent reads the insights section for context (cross-session memory content)
     ▸ Agent issues query("tl.dot memory layout") mid-round (RAG tool call)
     ▸ Agent issues optimize("matmul tiling") mid-round (RAG tool call)
     ▸ Agent applies patch, runs save_and_test, ...
Round end:
  4. evaluate_round_best → round_N_evaluation.json
  5. if win: record_optimization_outcome → writes to cross-session KB (not RAG corpus)
```

Both retrievals happen independently with no routing logic between them. If the user disables cross-session memory (`GEAK_USE_CROSS_SESSION_MEMORY=0`), RAG still works. If RAG is disabled (`rag: false`), cross-session memory still works.

### PR #90 integration preserved

PR #90's `mcp_tools/rag-mcp/` + `scripts/build_index.py` + `query` / `optimize` MCP tools are **kept as-is**. No refactor PR touches the RAG subsystem internals. The only interaction points:

- `KernelLanguage.tool_set` (PR-11 extends) includes `query` and `optimize` if the language uses RAG. For Triton and HIP today: yes.
- `tools/rag_postprocessor.py` (PR #90) stays — it wraps RAG results with per-query filtering.
- `scripts/build_index.py --force` remains the one-time setup step for RAG index building (documented in AKA's setup instructions).

§13 "What this refactor explicitly does NOT touch" already lists RAG. Re-stated here for emphasis.

---

## §11 — Smoke tests (all run on every PR; grounded in §0 evidence)

All smoke tests assert on the exact log markers captured from live runs (see §0.1). No guessing — these markers were verified.

### 11.1 `tests/smoke/test_triton_hetero_invariants.py`

Runs 1 round of Triton optimization on a fixture kernel. Asserts these **12 markers** appear in order (counts verified against §0.1):

```python
EXPECTED_MARKERS_TRITON = [
    r"Normalized kernel_type from task content: triton",
    r"--- Step 1/7: Resolve kernel URL ---",
    r"--- Step 2/7: Codebase Context ---",
    r"--- Step 3/7: Test Discovery ---",
    r"--- Step 4/7: Harness Validation ---",
    r"--- Step 5/7: Kernel Profiling ---",
    r"--- Step 6/7: Baseline Metrics ---",
    r"--- Step 7/7: Commandment ---",
    r"Using heterogeneous mode based on discovery",
    r"run_orchestrator:.*heterogeneous=True",
    r"Cross-session memory",
    r"Exploration Phase",
]
```

PR-42's default flip replaces markers 2-8 with phase-based equivalents (`Phase: Discovery`, `Phase: Harness`, `Phase: Baseline`, `Phase: Explore`). PR-42 updates the expected list AND emits **both** old and new markers for at least the 2-week observation window (PR-42 to Phase 5). AKA is updated in parallel (PR-09b). After observation window, old markers may be dropped in PR-54.

### 11.2 `tests/smoke/test_hip_homo_invariants.py`

Runs 1 round of HIP optimization on a fixture HIP wrapper. Asserts these **11 markers**:

```python
EXPECTED_MARKERS_HIP = [
    r"Normalized kernel_type from task content: hip",
    r"--- Step 1/7: Resolve kernel URL ---",
    r"--- Step 5/7: Kernel Profiling ---",    # steps 2-4 legitimately skipped today
    r"--- Step 6/7: Baseline Metrics ---",
    r"--- Step 7/7: Commandment ---",
    r"Using homogeneous mode based on discovery",
    r"Retriever: category=\w+ language=hip",
    r"Cross-session memory injected into homogeneous task \(\d+ chars\)",
    r"Homogeneous Agent \(\d+ agents, GPUs \[[^]]+\]\)",
    r"Sub-agent \d+ \(task_\d+\) started on GPU \d+",
    r"Sub-agent \d+ \(task_\d+\) started on GPU \d+",   # >=2 sub-agents in parallel
]
```

**Important**: today HIP skips steps 2-4 silently (bug #1 in §0.3). Smoke test asserts the skip explicitly. After PR-24 (commandment Jinja + universal enforcement), HIP runs ALL phases (including discovery), so PR-24 updates the expected marker list to match the full Triton set adjusted for HIP.

### 11.3 `tests/smoke/test_add_language_scaffolder.py`

Scaffold a "noop" language, validate contract, run 1-round smoke, cleanup. Fails if any file outside `kernel_languages/noop/` was modified.

### 11.4 `tests/smoke/test_translate_triton_to_hip.py`

50-LoC add kernel in Triton. Runs `geak translate --source tiny_add.py --target-language hip --output /tmp/tiny_add.hip --test test.py`. Asserts:
- `/tmp/tiny_add.hip` compiles
- Output tensors allclose(source, target) within 1e-5
- `translation_report.json` exists with `"golden_match": true`
- Translation completed in ≤ 3 attempts

### 11.5 `tests/smoke/test_cross_session_memory_analysis.py`

Unit test (no GPU). Fixture set:
1. **Well-formed-output fixture**: 3 synthetic `ExperienceRecord`s + 1 synthetic target kernel. Asserts `CrossSessionMemoryAnalysisAgent.run()` output:
   - `top_recommendations` non-empty OR `none_applicable=true`
   - No unknown keys in output
   - Output text ≤ 3,000 characters on this synthetic fixture (smoke-test size target; production insights file on disk is 5-25 KB — see §10.5 "Output" spec).

2. **Staleness-detection fixture** (reviewer Concern 3.2 fix): 1 synthetic `ExperienceRecord` where `profiler_before.bottleneck="compute"` and `winning_diff` references compute-bound patterns (e.g., `num_stages=4` pipelining); the current target kernel profile has `bottleneck="memory"`. Asserts the subagent output EITHER:
   - lists this entry in the `avoid` / "known dead-end" section with a reason mentioning the profile mismatch, OR
   - flags it as "likely stale" in `overall_assessment`, OR
   - marks `none_applicable=true` if no other entries are good.

   This deterministic test catches the case the removed `baseline_fingerprint` was originally meant to flag: if the subagent silently recommends a stale entry, the test fails and PR-3 is blocked until the prompt is hardened. Runs on every PR — ~5-second unit test, no GPU needed.

### 11.6 `tests/regression/test_kb_speedups.py` (nightly, not per-PR)

Runs each kernel in §0.2 for 5 rounds × 1 seed and asserts the best-of-5 speedup meets the PR gate threshold. Does NOT run on every PR (too expensive: ~8 hours). Runs nightly on `refactor-test` branch; PR-42's default flip is gated on a green run of this suite.

### 11.7 CI gates (static, run on every PR in <30s)

| Script | Purpose | Fails when |
|---|---|---|
| `scripts/check_language_leaks.py` | No `kernel_type == "triton"\|"hip"` outside `kernel_languages/` | After PR-10; WARN-only before |
| `scripts/check_one_cli_file.py` | Only `cli.py` has `typer.Typer()` or `@app.command` | From PR-06 onward |
| `scripts/check_no_agent_inheritance.py` | No class subclasses `DefaultAgent\|InteractiveAgent\|StrategyAgent` | From PR-50 onward |
| `scripts/check_subagent_location.py` | `SubagentBase` only used under `src/minisweagent/subagents/` | From PR-21 onward |
| `scripts/check_baseline_speedups.py` | Any kernel in §0.2 below its regression threshold in last nightly | From PR-00; WARN until nightly suite stable; then FAIL |

Any PR that breaks 11.1-11.5 or any CI gate above fails automatically.

---

## §12 — Risk matrix (grounded in historical incidents)

Each row references the historical bug (§0.3) or verified-but-untested concern that motivates the mitigation.

| Risk | Prob | Historical precedent | Mitigation |
|---|---|---|---|
| PR-42 default flip breaks HIP homogeneous runs | medium | Bug #1 (HIP commandment silent no-op): HIP path has a different contract today; flipping to unified exposes that | 2-week observation window after PR-40; nightly regression suite (§11.6) gates flip; env flag `GEAK_UNIFIED=0` rollback in <1 min |
| PR-42 default flip breaks Triton hetero runs | low | Triton hetero path is well-tested; KB regression data protects it | Nightly `test_kb_speedups.py`; per-kernel regression thresholds (§0.2) |
| PR-50 agent collapse loses subtle inheritance behavior | medium | 4-layer chain has implicit super().method() dispatches; audit §7 | Full unit suite + 5 Triton + 5 HIP regression run; PR-02 alias keeps imports working for staged rollout |
| PR-22 HarnessBuilder produces malformed harnesses | medium-high | No precedent; new LLM subagent | Contract validator PR-23 rejects fast; flag-gated default off until validator has **≥ 29/30 pass rate on fixture corpus** — see `tests/fixtures/harness_corpus/` spec below |
| PR-24 Jinja commandment diverges from old commandment.py byte output | medium | No precedent; new template system | PR-24 includes byte-level diff check; PR-24 lands behind `GEAK_USE_JINJA_COMMANDMENT=0` default; flip in PR-42 |
| PR-61 TranslationLoop produces wrong-semantics output | low-medium | PR #153's PyTorch→FlyDSL runs showed 15/15 pass rate with golden-match validation | Golden-match tensor allclose; performance-regression gate (0.5x/0.8x from PR #153); optional self-review |
| PR-70 `CrossSessionMemoryAnalysisAgent` produces poor recommendations for high-conf KB matches | low | Today's deterministic formatter also has this failure mode (bug #4); single-path design means the subagent must handle all cases, which is tested by the PR-72 ablation gate |
| PR-70 subagent recommends a known dead-end | low-medium | Bug #4 precedent — retriever ranking errors happened before | Ablation harness PR-72 gates PR-73 default flip; subagent prompt includes "flag dead-ends" instruction |
| PR-73 default flip regresses on some kernels | medium | Bug #4 retriever regression caused `fused_rms_fp8` to drop from 2.23x→1.11x | Full §0.2 regression table required green before flip; rollback via env var |
| `target_code=0B` regression resurfaces | low | Bug #2 happened before; fixed in PR #166 commit 8578d62b | PR-10 carries fix forward; `test_task_file_absolute_paths.py` unit test gates regression |
| Patch apply failures resurface | low | Bug #3 happened before; fixed in PR #161 via `git apply --3way` | PR-00 carries fix forward; `test_save_and_test_patch_apply.py` gates regression |
| `_PARAM_PATTERNS` language leak recurs | low | Bug #7; will be rewritten in PR-71 | `check_language_leaks.py` CI gate goes from WARN to FAIL after PR-71 |
| Kernel not in KB gets no memory context (exploration mode) | low | Expected behavior; happens for all new kernels today | `CrossSessionMemoryAnalysisAgent` returns "none applicable" and agent explores from scratch — verified behavior preserved |
| Adding new language requires core edits | low | PR #155 FlyDSL attempt touched ~9 core files | `check_language_leaks.py` CI gate after PR-10; `geak add-language` scaffolder prevents by construction |
| Second Typer entry regresses | low | Bug analog: PR #153 flip-flopped 3 times between separate CLI and flag | `check_one_cli_file.py` CI gate from PR-06 |
| Agent subclassing `DefaultAgent` regresses | low | Today we have exactly this 4-layer chain | `check_no_agent_inheritance.py` CI gate from PR-50 |
| Subagent placed outside `subagents/` | low | Today `select_patch_agent.py` is in `agents/` | `check_subagent_location.py` CI gate from PR-21 |
| Docker image drift breaks baselines | medium | KB staleness is bug #5 precedent | Pin `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` in `.github/workflows/nightly.yml`; log image hash in every kernel run; subagent flags entries whose stored profiler patterns differ from current run's |
| GPU contention during nightly regression runs | medium | Verified: live runs on slots 0-3, slots 4-7 unavailable | `monitor.sh` kill-rogue-processes (paused per user request); nightly suite uses exclusive slot allocation |

---

## §13 — What this refactor explicitly does NOT touch

- Model selection / LLM provider code
- Cross-session memory schema (no schema changes; existing fields preserved)
- RAG subsystem (PR #90) — kept intact as generic docs lookup; complementary to cross-session memory (see §10.7)
- `adaptive_ratio_v1` controller for `mode=auto` (planned as a separate follow-up)

### §13.1 — Explicit preservation-of-functionality audit (reviewer-raised)

User-raised: "make sure we are not loosing any functionality from base pipeline". Going through every observed capability in §0.5(a) current diagram, below is what the refactor does with each. Nothing is silently dropped.

| Base-pipeline capability | Where it lives today | Where it lives after refactor | Lost? |
|---|---|---|---|
| `geak -t "..."` NL-first invocation | `run/mini.py:app` | `cli.py:optimize` (same NL parser) | No |
| LLM call #1 `parse_pipeline_params` (mode, max_rounds) | `mini.py:230-450` | `cli.py::optimize` calls same function (moved, not rewritten) | No |
| LLM call #2 `parse_task_info` (kernel, repo, gpu_ids, target_language) | `mini.py:230-450` | Same | No |
| Language auto-detection | `_normalize_kernel_type` + `_infer_kernel_type` × 3 sites | `kernel_languages/__init__.py:registry.detect_best()` — one site | No, collapsed |
| 7 preprocess steps | `preprocessor.py:run_preprocessor` (1386 LoC) | 4 phases (`preprocess/phases/{discovery,harness,baseline,explore}.py`) + 1 conditional (`translation.py`) | No, reorganized |
| `CODEBASE_CONTEXT.md` generation | `preprocess/codebase_context.py` (499 LoC) | Same file called from `DiscoveryPhase` | No |
| ATD test discovery (MCP) | `run/preprocess/harness_utils.py` | Same; called from `HarnessPhase` | No |
| `UnitTestAgent` harness skeleton when no tests | `agents/unit_test_agent.py` + `run/preprocess/unit_test_agent.py` | Merged → `subagents/preprocess/unit_test_agent.py` | No, deduplicated |
| Harness validation | `harness_utils.execute_harness_validation` + validators | `HarnessPhase` calls `HarnessBuilder` subagent + `validate_harness()` from `kernel_languages/contract.py` | No, enhanced (contract formalized) |
| Metrix profiling (MCP) | `kernel_profile.py` | Same file called from `BaselinePhase` | No |
| Baseline metrics collection | `baseline.py::build_baseline_metrics` | Same file called from `BaselinePhase` | No |
| `COMMANDMENT.md` generation | `commandment.py` (530 LoC Python branching) | `ExplorePhase` renders `kernel_languages/<lang>/commandment.j2` (Jinja) + `validate_commandment()` | No, but per-round enforcement newly uniform (bug #1 fixed) |
| Heterogeneous path: exploration phase (LLM reads kernel) | `run_heterogeneous_orchestrator` | `run_pipeline(mode="planned"\|"auto")` — same LLM behavior, cleaner code | No |
| Heterogeneous: `generate_tasks` / `dispatch_tasks` / `collect_results` | `heterogeneous/task_generator.py` + `tools.py` | `run/task_generator.py` (language-agnostic, A/B-validated in PR-31) + `tools/orchestrator_tools.py` | Functionality preserved; Triton prompts moved to `kernel_languages/triton/planner_strategy_hints.md`; A/B gate prevents regression |
| Heterogeneous: `post_round_evaluate` with FULL_BENCHMARK | `heterogeneous/orchestrator.py` + `postprocess/evaluation.py` | `run_pipeline` calls `evaluate_round_best` in every mode | No, made uniform |
| Heterogeneous: `record_final_outcome` → memory.db | `postprocess/results.py` + `memory/integration.py` | `run_pipeline` calls it for ALL modes now (bug #6 fix) | No, extended |
| Homogeneous path: N copies of same task | `ParallelAgent.run_parallel` (609 LoC) | `run_pipeline(mode="fixed")` uses `compose_task_body(mode="fixed")` + `pool_run` | No |
| Homogeneous: ThreadPoolExecutor parallel execution | `parallel_agent.py:414-609` | `run/pool_runner.py` — single path for both modes | No, deduplicated |
| Homogeneous: `_select_best_from_parallel_runs` via `SelectPatchAgent` | `homogeneous_agent.py` | Same `SelectPatchAgent`, called from `run_pipeline` | No |
| `StrategyInteractiveAgent` step loop (the agent) | `agents/strategy_interactive.py` → 4-layer chain (998 LoC) | `agents/optimization_agent.py` (merged, ~500 LoC) | No, collapsed |
| Tools: bash, editor, save_and_test, submit | `tools/*.py` | Same | No |
| Tool: strategy_manager (strategy_list.json) | `tools/strategy_manager.py` | Same | No |
| Tool: profile_kernel (Metrix MCP) | `tools/profiling_tools.py` | Same | No |
| Tool: query / optimize (RAG MCP, PR #90) | `tools/mcp_bridge.py` + external `mcp_tools/rag-mcp/` | Same | No (see §10.7) |
| Tool: sub_agent | `tools/sub_agent_tool.py` | Same | No |
| Tool: baseline_metrics | `tools/baseline_metrics_tool.py` | Same | No |
| Tool: resolve_kernel_url (adapter) | `tools/resolve_kernel_url.py` (reviewer-caught: it's a tool, NOT a shim) | Kept as-is | No |
| Tool: check_kernel_compatibility | `tools/check_compat.py` | Same | No |
| Cross-session memory retrieve | `memory/cross_session/retriever.py` called via `assemble_memory_context` | Same retriever, called by `CrossSessionMemoryAnalysisAgent` pre-round | No, wrapped by subagent |
| Cross-session memory injection into task body | `formatter.py` — deterministic regex + 40 KB raw dump | `cross_session_memory_insights.md` file — structured, subagent-curated, ~5-25 KB | No data lost (diffs, baseline codes, profiler info all preserved in the insights file); presentation improved |
| `classify_kernel_category` (kernel-family classifier) | `memory/cross_session_memory.py:6` (3 production callsites) | Moved to `memory/cross_session/__init__.py` in PR-19; 3 callsites updated | No, safely relocated |
| KB schema (`ExperienceRecord` fields) | `memory/cross_session/schemas.py` | Unchanged | No |
| KB write on successful round | `record_optimization_outcome` (hetero path only today) | Called uniformly by `run_pipeline` for both modes | No, extended to homo |
| `final_report.json` / `round_N_evaluation.json` artifacts | `postprocess/evaluation.py` | Same filenames, same schema — AKA consumes them | No |
| `geak-orchestrate` CLI | `run/orchestrator.py` (redundant, homo raises NotImplementedError) | Deleted in PR-1; resume-from-preprocess via `geak resume --preprocess-dir` subcommand added in PR-2 | No — redundancy removed, capability preserved |
| `geak-preprocess` CLI (`run/preprocess/preprocessor.py:main`) | 2nd aux entry point in pyproject.toml | **Deleted in PR-1** per user direction. Preprocessing runs as part of `geak -t "..."` via `PreprocessOrchestrator`. If anyone needs standalone preprocessing, call `from minisweagent.preprocess.orchestrator import PreprocessOrchestrator; PreprocessOrchestrator(ctx).run()` directly in Python. | No (replaced by orchestrator call) |
| `kernel-profile` CLI (`preprocess/kernel_profile.py:main`) | aux entry point | **Deleted in PR-1**. The BaselinePhase calls `preprocess/kernel_profile.py` internally. | No (now called from BaselinePhase) |
| `commandment` CLI (`preprocess/commandment.py:main`) | aux entry point | **Deleted in PR-1**. Jinja rendering in ExplorePhase replaces it (`commandment.py` the module is also deleted in PR-2). | No (functionality moves into Jinja templates) |
| `validate-commandment` CLI (`tools/validate_commandment.py:main`) | aux entry point | **Deleted in PR-1** (the stub module it points to is also deleted in PR-1). `kernel_languages/contract.py::validate_commandment()` replaces it, called from ExplorePhase. | No (function moved to contract module) |
| `run-harness` CLI (`preprocess/run_harness.py:main`) | aux entry point | **Deleted in PR-1**. The Python function `run_harness()` stays callable from inside the pipeline (BaselinePhase + TranslationPhase use it). | No (function still callable from Python) |
| `baseline-metrics` CLI (`preprocess/baseline.py:main`) | aux entry point | **Deleted in PR-1**. BaselinePhase calls `preprocess/baseline.py` internally. | No |
| `codebase-context` CLI (`preprocess/codebase_context.py:main`) | aux entry point | **Deleted in PR-1**. DiscoveryPhase calls `preprocess/codebase_context.py` internally. | No |
| `resolve-kernel-url` CLI (`preprocess/resolve_kernel_url.py:main`) | aux entry point | **Deleted in PR-1**. The `ResolveKernelUrlTool` adapter in `tools/resolve_kernel_url.py` stays (agent tool); the CLI entry is dropped. | No (tool form preserved) |
| `task-generator` CLI (`agents/heterogeneous/task_generator.py:main`) | aux entry point | **Deleted in PR-1** (module itself deleted in PR-3 along with `agents/heterogeneous/`). The new `run/task_generator.py` replaces it, invoked only from `run_pipeline` — not a standalone CLI. | No (functionality preserved in `run/task_generator.py`) |
| `--yolo`, `--exit-immediately`, `-o logs_dir` flags (AKA uses) | `mini.py` Typer options | Same flags on `cli.py::optimize` | No |
| NL prompt-first or CLI-flag invocation (both work) | `mini.py` supports both | `cli.py::optimize` supports both | No |
| Worktree isolation per sub-agent task | `run/task_file.py` + dispatch | Same | No |
| Docker exec layer | AKA's `launch_agent.py` invokes `geak` inside container | Unchanged | No |

**Net capability delta**:
- **Added**: `geak add-language`, `geak translate`, `CrossSessionMemoryAnalysisAgent` + structured `cross_session_memory_insights.md`, universal per-round FULL_BENCHMARK enforcement (fixes bug #1), uniform KB write across modes (fixes bug #6), contract validators for harness + commandment, 8+ CI gates
- **Removed**: 4-layer agent inheritance chain, `geak-orchestrate` second CLI + **9 other aux CLI entries** (per user direction: only `geak -t "..."` survives), duplicated `run_pool` / ThreadPoolExecutor split, 40 KB raw unstructured KB dump, 16+ language-branching sites in core, ~9,000 LoC of dead code / duplicates
- **Preserved**: every capability in the table above (37 items checked)

A PR that removes any capability not listed above must be rejected. CI gate `check_no_capability_loss.py` (added in PR-00 WARN, PR-42 FAIL) runs a subset of the table as assertions against `geak --help`, `grep "tool.*register"`, `cat pyproject.toml [project.scripts]`, and the nightly kernel suite (§0.2). Any capability that disappears trips the gate.

---

## §14 — Summary (revised after user direction to consolidate)

**3 PRs (was 38; collapsed per user direction on 2026-04-22 after all reviewer fixes integrated), ~6-10 weeks, each rollback-safe via `git revert`, every claim grounded in §0 evidence and verified by reviewer + user.**

### Timeline: 14-week vs 18-20-week reality

Reviewer-raised Gap 2 correctly pointed out the original 14-week estimate assumed perfect parallelism. The plan now acknowledges:
- **14 weeks** is achievable ONLY with 2-3 developers on independent phases (Phase 0/1/2 can parallelize; Phase 3+ is mostly sequential).
- **18-20 weeks** is realistic for a single developer because of sequential gates: Phase 0 → Phase 1 (PR-09b AKA audit + PR-19 safe move) → Phase 2 (preprocess split, gated on PR-20 dependencies) → PR-31 A/B run (~1 wall-clock day on 4 GPUs) → Phase 4 flip → 2-week observation → Phase 5 → Phase 7 gate (~0.5 GPU-day on 4 GPUs) → flip → demolition.

### Flag matrix (after 3-PR consolidation)

With 3 large PRs instead of 38 small ones, most staged-rollout flags become unnecessary — each PR lands with the feature on by default because it's the whole-feature unit. Only 2 flags remain:

| Flag | Introduced | Default | Purpose | Removal |
|---|---|---|---|---|
| `GEAK_USE_CROSS_SESSION_MEMORY` | PR-3 | `1` (on) | Opt-out for ablation and low-budget LLM accounts (e.g., regression CI that doesn't need memory). `=0` = no retrieval, no subagent call, no context injected; agent explores from scratch. | **Kept as permanent opt-out** (see rationale in §10.5) |
| `GEAK_TRANSLATION_SELF_REVIEW` | PR-2 | `0` (off) | Experimental self-review for `geak translate` (single LLM query post-golden-match to flag residual quality issues). | Reviewed after 3 months of use data |

**Deleted vs the old 38-PR plan** (no longer needed because each 3-PR lands whole-feature):
- `GEAK_USE_HARNESS_BUILDER` — HarnessBuilder lands with PR-2 and is default-on (gated by the ≥29/30 fixture corpus)
- `GEAK_USE_JINJA_COMMANDMENT` — Jinja commandment lands with PR-2 and is default-on (gated by byte-diff check against old `commandment.py` output)
- `GEAK_UNIFIED` — `run_pipeline(ctx, mode)` lands with PR-3 and is default-on (gated by PR-3 A/B equivalence gate + §0.2 regression)

### Target state after PR-3

- **1 CLI file** (`src/minisweagent/cli.py`) — enforced by CI
- **1 agent class** (`OptimizationAgent`) — no inheritance; enforced by CI
- **1 tool-resolution site** (`run/unified.py:_resolve_tools`) — enforced by CI
- **1 pipeline** (`run_pipeline(ctx, mode)`) — unified across Triton + HIP + future languages
- **1 subagent folder** (`src/minisweagent/subagents/`) with 3 purpose-grouped subfolders — enforced by CI
- **4 preprocess phases** (Discovery, Harness, Baseline, Explore) + 1 conditional (Translation)
- **1 way to add a language**: `geak add-language X` + fill 7 files in `kernel_languages/X/` — zero core edits, enforced by CI
- **1 way to translate**: `geak translate --source ... --target-language ...` (standalone) OR `geak -t "..." --target-language Y` (integrated)
- **0 language-branching sites in core** (down from 16+ today per audit §9 inventory) — enforced by `check_language_leaks.py`
- **~25,000 LoC** (down from 34,301 — reviewer-corrected; −9,000 net)
- **5 smoke tests** per PR + **1 nightly regression suite** (kernel speedups) + **7 static CI gates** (includes `check_agent_tool_set.py`)

### The 6 LLM subagents

All 6 subagents live in `src/minisweagent/subagents/` and extend `SubagentBase`. **`SubagentBase` is a peer class of `OptimizationAgent`, not a subclass of it.** Subagents that need a full LLM tool loop instantiate `OptimizationAgent` via composition (the `_make_optimization_agent()` helper; see §16.2). This keeps the two class families orthogonal: `OptimizationAgent` owns per-call tool-loop mechanics; `SubagentBase` owns narrow-task lifecycle (config, prompt, optional retry).

Reviewer-raised point: not all subagents use the same method — some are one-shot, one is multi-round. `SubagentBase` exposes **two execution methods** (subclasses override EXACTLY ONE):

```python
# subagents/base.py
class SubagentBase:
    """Shared concerns: language binding, config loading, model resolution,
    prompt composition via compose_task_body(). Subclass overrides ONE of:
    - run(**inputs) -> str | dict        (one-shot: composes ONE OptimizationAgent call)
    - loop(**inputs, max_attempts: int, verify_fn) -> Result   (multi-round: composes N OptimizationAgent calls)
    """
    def __init__(self, language: KernelLanguage, config_path: Path): ...

    def run(self, **inputs) -> str | dict:
        """One-shot: compose task body, call _make_optimization_agent().run(...), return submission."""
        raise NotImplementedError("Subagent must override run() or loop()")

    def loop(self, *, max_attempts: int, verify_fn, **inputs) -> "Result":
        """Multi-round: compose task body then compose an OptimizationAgent PER ATTEMPT until
        verify_fn returns ok OR max_attempts hit. Each attempt gets feedback from prior attempts."""
        raise NotImplementedError("Subagent must override run() or loop()")

    def _make_optimization_agent(self, tools: list) -> "OptimizationAgent":
        """COMPOSITION boundary. Instantiate a short-lived OptimizationAgent."""
        ...
```

| Subagent | File | Method | Lifecycle | Trigger | Purpose |
|---|---|---|---|---|---|
| `UnitTestAgent` | `subagents/preprocess/unit_test_agent.py` | `run()` | Harness phase (conditional) | no user test file | Generate test harness skeleton |
| `HarnessBuilder` | `subagents/preprocess/harness_builder.py` | `run()` | Harness phase (one-shot) | always | Adapt user test file to universal contract |
| `KernelAnalysisAgent` | `subagents/preprocess/kernel_analysis.py` | `run()` | Explore phase (one-shot) | always | Produce analysis rubric markdown |
| `ShapeFixerAgent` | `subagents/preprocess/shape_fixer.py` | `run()` | Harness phase (conditional) | shape mismatch | Fix kernel shape errors |
| `CrossSessionMemoryAnalysisAgent` | `subagents/memory/cross_session_memory_analysis.py` | `run()` | per-round runtime | `GEAK_USE_CROSS_SESSION_MEMORY=1` | Writes structured `cross_session_memory_insights.md` (analysis + ranked recs + FULL reference KB entries) |
| `TranslationLoop` | `subagents/translation/translator.py` | **`loop()`** | Translation phase (conditional) multi-round | `--target-language` set | Rewrite in target language, golden-match verified |

CI gate `check_subagent_base_contract.py` (added in PR-21) verifies every subagent implements exactly one of `run()` or `loop()` (not both, not neither). This keeps the base-class claim honest.

**Future subagents (PR-80+, not in this plan)** — reviewer correctly noted these are subagents-in-disguise today but deferred to a follow-up to keep scope tight:
- `TaskGeneratorAgent` — already LLM-based at `agents/heterogeneous/task_generator.py`; formal reclassification to `subagents/planning/` in PR-80
- `SelectPatchAgent` — already `DefaultAgent` subclass; formal reclassification to `subagents/selection/` in PR-81

### What the refactor preserves (non-negotiable)

From §0.1 — all 12 Triton + 11 HIP invariant log markers preserved in old form during observation window; replaced with phase-based equivalents in PR-42 (emit-both) and AKA/dashboards updated in parallel via PR-09b.
From §0.2 — all 13 kernel speedups at or above their regression thresholds; nightly suite + PR-31 A/B + PR-42 full regression all required green.
From §0.3 — all 10 historical bugs fixed and guarded by unit/CI tests.

### CLI file size reality check (reviewer-raised Nit 5)

Reviewer correctly pointed out the earlier "~250 LoC cli.py" target was optimistic. After PR-53 moves all of `mini.py` (629 LoC) + adds `add-language`, `translate`, `resume`, `test-language` subcommands + default callback + config merge logic, realistic target is **400-600 LoC**. Plan updated.

### Next action

**Land PR-1 on `refactor-test`.** It's foundation + cleanup — CI gates + 5 smoke tests + `KernelLanguage` + `SubagentBase` + ~8,000 LoC of dead-code deletions + **all 9 auxiliary CLI entries removed** (only `geak -t "..."` survives per user direction). Sets the foundation PR-2 and PR-3 depend on. Full details in §8 PR-1.

Revised commit structure for PR-00 (reviewer-approved):
```
docs: add docs/refactor/EXECUTION_PLAN.md + CODEBASE_AUDIT.md + INVARIANTS.md
test: add tests/smoke/ with 3 smoke test files (Triton 12 markers, HIP 11 markers, memory-analysis unit)
test: add tests/regression/baseline_speedups.yaml with §0.2 thresholds
ci: add scripts/check_one_cli_file.py, check_language_leaks.py (WARN), check_subagent_location.py (WARN)
ci: add nightly workflow that runs tests/regression/test_kb_speedups.py on 4-GPU slot
chore: delete 5 verified-safe files (not 7):
  - debug_runtime.py (root; byte-identical duplicate)
  - tools/discovery_types.py (7-LoC re-export stub)
  - tools/validate_commandment.py (7-LoC re-export stub)
  - agents/shape_fixer_agent.py (7-LoC re-export stub)
  - run/hello_world.py (mini-swe-agent leftover)
  NOT deleted: memory/cross_session_memory.py (reviewer-caught: 3 production imports; deferred to PR-19)
  NOT deleted: tools/resolve_kernel_url.py (reviewer-caught: it's a tool adapter, not a shim)
```

### Rollback posture

Every PR has exactly one of these rollback strategies:
- **Revert** — pure additive PRs (Phases 0-3, 6, 7)
- **Env flag flip** — behind-flag PRs (PR-22, PR-24, PR-40, PR-70) or default flips (PR-42, PR-73)
- **Cherry-pick revert** — Phase 5 demolitions after 2-week observation window

No PR requires simultaneous revert of multiple commits.

### Version-freeze policy during refactor (reviewer-raised Gap 7)

For the 14-week (or 18-20-week) refactor duration:
- `refactor-test` is the only production-path branch.
- Bug-fix-only PRs from `main` are cherry-picked weekly (max 1-2 hrs merge per week).
- The 2-week observation window between PR-42 and Phase 5 is **freeze** on `refactor-test` except for docs/CI/smoke-test PRs. Parallel Phase 5 prep work happens on `refactor-test-phase5-prep` branch (isolated).

---

## §15 — Reviewer response log (verified-evidence audit)

A critical reviewer flagged 3 Tier-0 (plan-breaking), 7 Tier-1 (major design), 7 Tier-2 (gaps), and 8 Tier-3 (nitpicks) issues. Each was verified with hard evidence (`rg`, `Read`, `wc -l`, `diff -q`). This section records the verification result and plan response for every issue, so future readers and reviewers can trace each decision.

### Tier 0 — Plan-breaking (all CONFIRMED; fixed)

| # | Reviewer claim | Verification method | Result | Fix location |
|---|---|---|---|---|
| A | `memory/cross_session_memory.py` has 3 production imports — deleting in PR-00 will `ImportError` every hetero run | `rg classify_kernel_category src/` → 3 hits at `agents/heterogeneous/orchestrator.py:291`, `run/postprocess/results.py:175,537` | **CONFIRMED** | PR-00 no longer deletes; new **PR-19** does safe move-then-delete |
| B | `tools/resolve_kernel_url.py` is a `ResolveKernelUrlTool` adapter, not a 56-LoC shim | `Read` confirmed class wrapping `{output, returncode}`; `rg ResolveKernelUrlTool` → registered at `tools_runtime.py:147` | **CONFIRMED** | PR-00 no longer deletes; file STAYS |
| C | Actual LoC is 34,301 not ~30,000 | `wc -l src/minisweagent/**/*.py` → 34,301 across 143 files | **CONFIRMED (off by 14%)** | §0.4 and §14 target adjusted: 34,301 → ~26,000 (−8,000 preserved) |

### Tier 1 — Major design issues

| # | Issue | Verification | Response |
|---|---|---|---|
| 1 | Step N/7 → Phase X rename breaks AKA / dashboards | AKA: `rg 'Step \d+/\d+' /data/sapmajum/AgentKernelArena/` → 0 hits; AKA reads `final_report.json`/`task_result.yaml` (JSON/YAML) only | **PARTIAL** — AKA not affected (verified). User directive: "AKA is not a problem, we can change AKA accordingly." External dashboards: **PR-09b explicit audit** required. PR-42 emits **both** old and new markers for 2-week observation window |
| 2 | PR-31 rewrites 997-LoC `task_generator.py` (agent-based reasoning, 3-5x speedup driver) in 400 LoC without A/B | `Read` confirmed agent-based priority scheme (0/5/10/15), Triton-centric `TASKGEN_SYSTEM_PROMPT`, `result_scanning` integration | **CONFIRMED CRITICAL** — PR-31 scope revised to ~700 LoC; hard A/B gate added before PR-42: 13 kernels × 3 seeds × 2 arms × 5 rounds = 390 runs × 15 min / 4 GPUs ≈ **~1 wall-clock day on 4 GPUs** (arithmetic reviewer-corrected from original 6-day claim) |
| 3 | 5 env flags = "band-aid forest"; no named removal PRs | Plan listed 5 flags without removal PRs | **CONFIRMED** — Principle #12 added; flag matrix in §14 lists every flag's removal PR or explicit rationale for permanent opt-out |
| 4 | `baseline_fingerprint` is vaporware (load-bearing in §10.5/§12 but doesn't exist) | `rg baseline_fingerprint src/minisweagent/memory/` → 0 hits | **CONFIRMED — and resolved by REMOVAL** (subsequent user audit). The fingerprint was originally introduced as a gate condition for the smart-router fast path. After the user removed the fast-path entirely (single memory path now), the fingerprint had no remaining load-bearing use case; the subagent reasons about staleness from KB content (profiler patterns, code patterns, stored diffs). PR-09 was deleted. Bug #5 mitigation now relies on: (a) `save_and_test --3way` (PR #161, already merged), (b) structured insights file giving the subagent full content to detect staleness, (c) docker pinning in nightly CI. Reviewer's concern ("vaporware") is satisfied: the mechanism no longer exists, so it's no longer load-bearing-but-missing. |
| 5 | "ONE agent class" breaks tool registration — who decides the tool set? | `Read tools/tools_runtime.py:120-162` — `allowed: set[str]` parameter, decider = 4-layer chain | **CONFIRMED** — §4 now specifies: tool resolution happens in `run/unified.py:_resolve_tools` from `ctx.language.tool_set + mode-specific + CLI overrides`. New CI gate `check_tool_resolve_single_site.py` prevents regressions |
| 6 | TranslationAgent is a different loop (different verify + exit), not "same class, 3 differences" | Verification criterion = tensor allclose (not Submitted/speedup); exit = golden match within max_attempts (not after N rounds) | **CONFIRMED** — §6 renamed to `TranslationLoop` which **composes `OptimizationAgent`** rather than claiming to be it; §10 table and §14 table updated to reflect multi-round vs one-shot distinction |
| 7 | PR-70 → PR-71 ordering creates regression window where fast-path loses `_PARAM_PATTERNS` signal | PR-71's own risk note admitted this | **CONFIRMED** — Phase 7 reordered: PR-70 → PR-72 (ablation) → PR-73 (default flip) → **PR-71 (language-leak cleanup LAST)**. PR-71 depends on PR-73 green |

### Tier 2 — Gaps / uncertainties

| # | Gap | Response |
|---|---|---|
| 1 | Ablation harness cost (1,080 runs) unbudgeted | §10.5 and PR-3 revised: with the single-path design (no deterministic_kb arm) the ablation is **6-kernel subset × 3 seeds × 2 arms × 5 rounds = 180 runs ≈ 0.5 GPU-day on 4 GPUs**. Runs as post-merge validation to confirm `GEAK_USE_CROSS_SESSION_MEMORY=1` default, not a merge gate. Optional full 18-kernel confirmation on a dedicated weekend. |
| 2 | 14-week timeline assumes perfect parallelism | §14 now says **"14 weeks with 2-3 devs on independent phases; 18-20 weeks realistic for single dev"**; sequential gates (PR-09b → PR-11 → PR-20 → PR-31 A/B → PR-42 → observation → Phase 5 → PR-72 gate → PR-73) explicitly listed |
| 3 | Nightly regression suite capacity conflicts with ablation | §11.6 now says `scripts/schedule_gate.py` alternates slot allocation between nightly regression and PR-72 gate run |
| 4 | AKA audit for `--heterogeneous` usage | `rg --heterogeneous /data/sapmajum/AgentKernelArena/` → 0 CLI-flag hits (uses `GEAK_CONFIG_NAME` env var). Non-blocking; PR-09b documents the migration |
| 5 | PR-50 test coverage — existing tests depend on 4-layer inheritance | **PR-49 added** (before PR-50) to rewrite `tests/agents/test_*.py` against the single `OptimizationAgent` contract |
| 6 | Scaffolder contract hidden in `harness.j2` | §5 updated: `kernel_languages/_templates/harness.j2.j2` ships with contract-required blocks pre-wired; user fills `{% block device_launch %}...{% endblock %}` only |
| 7 | 2-week observation window blocks vs contaminates | §14 version-freeze policy: observation window is **freeze on `refactor-test`** except docs/CI; parallel Phase 5 prep on isolated `refactor-test-phase5-prep` branch |

### Tier 3 — Nitpicks (all addressed)

| # | Nit | Fix |
|---|---|---|
| 1 | Marker count mismatch (§0.1 has 12 Triton, 11 HIP; §11 claimed 9/7) | §11.1/§11.2 counts **corrected to 12/11** with verified regex list |
| 2 | PR-02 alias booby-trapped after PR-50 | PR-50 description now explicitly says it **overwrites** the PR-02 alias file with the merged class definition (intentional staged rollout) |
| 3 | "16+ language-branching sites" — narrow grep finds 3 | §1 note added: audit counts **semantic sites** (prompt templates, dict-based dispatch, `_LANGUAGE_GUIDANCE` maps, strategy YAML) not just `==` comparisons. Exact audit incantation cited in §9 |
| 4 | §14 LoC numbers wrong (30,000→22,000) | **Fixed to 34,301→26,000** |
| 5 | `cli.py` ~250 LoC unrealistic after consolidating `mini.py` + subcommands | **Fixed to 400-600 LoC** target |
| 6 | §10 subagent count inconsistent (5 vs 6 vs 7-8) | §10 / §14 / §10.6 now consistent: **6 now, +2 (TaskGenerator, SelectPatchAgent) in PR-80+ follow-up**, explicitly deferred out of scope |
| 7 | Flag name inconsistency `GEAK_USE_CROSS_SESSION_ANALYSIS` vs class `CrossSessionMemoryAnalysisAgent` | **Standardized to `GEAK_USE_CROSS_SESSION_MEMORY`** (matches class name) |
| 8 | PR-54 flag removal list incomplete | PR-54 description and §14 flag matrix now explicit: `GEAK_USE_CROSS_SESSION_MEMORY` is **kept as permanent opt-out** (intentional exception to Principle #12, documented rationale) |

### v3 reviewer audit (second review pass — 2026-04-22, verified in codebase)

After the v2 fixes landed, a second review pass caught additional issues — all verified in the actual codebase and addressed:

| # | v3 reviewer finding | Verification | Resolution |
|---|---|---|---|
| T0-new | `pyproject.toml` ships **10** CLI entry points, not 2 as §0.5(a) claimed | Read `pyproject.toml:94-106` — confirmed 10 scripts | §0.5(a) now enumerates all 10; user directed "only `geak -t` needed" — all 9 auxiliary entries **deleted in PR-1**. §13.1 audit has per-entry disposition. |
| T0-new | PR-00 deleting `tools/validate_commandment.py` breaks `validate-commandment` entry | pyproject.toml entry points to exactly that module | PR-1 removes both the stub AND the pyproject entry in the same commit — no stale pointer |
| T1.1 | PR-73 pass criteria reference non-existent `deterministic_kb` arm | Single-path design has 2 arms (`no_kb`, `subagent_kb`) | PR-3 A/B gate criteria updated to 2-arm spec in §8 PR-3 section |
| T1.2 | 2-arm ablation can't detect regression vs today's 40 KB raw dump | Agreed — scope clarified | Plan explicitly says ablation's purpose is to justify `=1` default, not to prove improvement over today's formatter; today's regression is caught by §0.2 speedup-floor suite + invariant smoke tests |
| T1.3 | Context budget given 3 KB (smoke), 3 KB (gate), 25 KB (spec), 5-25 KB (reality) | Three different numbers in different places | Reconciled: 3 KB is the **smoke-test synthetic-fixture target** (§11.5); 5-25 KB is the **production file-on-disk size** (§10.5); PR-3 gate uses `file_size_bytes ≤ 25 KB avg` criterion. §11.5 + PR-3 criteria cross-link. |
| T1.4 | §15 Gap-1 response stale (said "3 arms × 270 runs") | Contradicted §10.5's 2-arm × 180 runs | Gap-1 text updated to 2 arms × 180 runs |
| T1.5 | §10 table row "KB router (fast vs slow)" stale — router was removed | Verified removed from §10.5 | Row deleted from §10 table |
| T1.6 | `target_language: "KernelLanguage" = None` runtime-error risk | Non-Optional type + None default = type liar; if `cli.py` forgets to set it, orchestrator AttributeError | Fix: `Optional[KernelLanguage] = None` + `__post_init__` that defaults to `language` — downstream never sees None |
| T2.1 | §1 header LoC still 30,000 / 22,000 | Should be 34,301 / 25,000 | §1 updated |
| T2.2 | §1 "Dead code" header says "11 files / ~1,500 LoC" but table sums 16 / 2,262 | Header wrong | §1 header updated to "16 files / ~2,262 LoC" |
| T2.3 | §9 promises "audit incantation" for 16+ sites but doesn't include one | Readers have to take the claim on faith | §9.1 added with exact `rg` incantations (narrow finds 3; broad finds 20-file list) + full 16-site enumeration from `GEAK_codebase_audit.md` |
| T2.4 | §16.6 CI-gate table omits `check_agent_tool_set.py` (which §4 introduces) | Inconsistency | §16.6 table now lists 12 CI gates including `check_agent_tool_set.py` and `check_no_capability_loss.py` |
| T2.5 | §16.8 "day-1 walkthrough" runs 6 scripts but 3 don't exist on day 1 of PR-1 | Section title misleading | §16.8 retitled "end-state walkthrough"; each command annotated with landing PR marker |
| C3.2 | Staleness detection has no deterministic fixture — could silently regress to bug #5 | `baseline_fingerprint` was removed, staleness is now "subagent reasoning only" | §11.5 now ships a **staleness-detection fixture**: 1 synthetic KB entry with `profile.bottleneck=compute` + current target `bottleneck=memory`, asserts the subagent flags it in `avoid` / `overall_assessment` / `none_applicable` |
| C3.3 | 3 seeds × 2 arms may be under-powered for ±5% noise | Gate decisions could be noise-driven | Noted as a future tuning knob; if the first gate run shows flaky pass/fail, bump to 5 seeds (300 runs ≈ 0.8 wall-clock day). Not changing now — the §0.2 regression suite + invariant smoke tests provide a safety net |

### USER-direction follow-up (third pass — 2026-04-22)

After v3 reviewer fixes integrated, the user directed two structural simplifications:

| # | User direction | Resolution |
|---|---|---|
| U1 | "only `geak -t "..."` works — we don't need all those CLI entrypoints like harness/commandment" | All 9 auxiliary CLI entry points **deleted in PR-1** (pyproject.toml cleanup). The underlying Python modules that matter (preprocess steps, contract validators) stay callable from inside the pipeline; they just aren't exposed as standalone CLIs. |
| U2 | "don't create so many PRs, only 2-3 PRs targeting preprocessing, orchestration/optimization, and other things" | 38 PRs → **3 PRs**: PR-1 (Foundation + Cleanup + CLI), PR-2 (Preprocessing), PR-3 (Orchestration + Optimization). Staged-rollout flags mostly deleted (each PR lands whole-feature). Only `GEAK_USE_CROSS_SESSION_MEMORY` and `GEAK_TRANSLATION_SELF_REVIEW` remain (see flag matrix in §14). Tradeoff: larger PRs per review, but faster calendar time and tighter internal consistency within each PR. |

### What the reviewer got right that we kept (do not lose these)

- **Invariant-first design** (§0.1): grounding in live log markers + KB speedup thresholds + bug-to-PR mapping — **preserved and strengthened** with verified marker counts
- **Per-PR rollback posture** (§14): every PR has one named rollback strategy — **preserved**
- **CI-as-guardrail** (5 gates): mechanically enforces invariants — **strengthened** (added `check_tool_resolve_single_site.py`)
- **Separation of deterministic vs LLM subagent** (§10): "LLM where judgment is needed" — **preserved**
- **Additive-first / delete-last** (Principle #1): flag-gated rollout — **preserved + Principle #12 added** to prevent flag proliferation
- **§10.6 candid "what else could be a subagent"** discussion: defers TaskGenerator + SelectPatchAgent to PR-80+ — **preserved**

### What the reviewer questioned that we pushed back on

None. Every reviewer claim was verified with hard evidence; every verified claim resulted in a plan change. The user's direction "AKA is not a problem, we can change AKA accordingly" scoped Issue 1 / Gap 4 down but did not reject them — the coordination work moved to PR-09b.

### Review questions answered

1. **Developers?** Plan now accommodates both single-dev (18-20 weeks) and multi-dev (14 weeks with 2-3 on independent phases) cadences; §14 has explicit sequential-gate chain.
2. **AKA audit?** PR-09b audits; `rg` findings cited above. Non-blocking per user; AKA updated in parallel.
3. **Observation-window cost?** §14 freeze policy + `refactor-test-phase5-prep` isolation branch prevents contamination. PR-72 gate budget cut to 1 GPU-day on the 4-GPU slot.
4. **Version-freeze policy?** Documented in §14 (weekly cherry-pick from main for bug fixes; freeze on refactor-test during observation).
5. **Why include translation?** Phase 6 is additive and isolated. Remains in scope; can be split to a follow-up project if scope pressure mounts, with minor plan adjustment.

---

## §16 — Implementation blueprint (concrete classes, signatures, end-to-end flow)

This section is the reference implementation spec. Every class, method signature, dataclass field, and call pattern the refactor introduces. If this section disagrees with §8 PR descriptions, §16 is authoritative for shape and §8 is authoritative for ordering.

### §16.1 — Core data objects

#### `KernelLanguage` (frozen dataclass — PR-01, extended in PR-11, PR-60, PR-70 prep)

```python
# src/minisweagent/kernel_languages/base.py
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class KernelLanguage:
    # ─── identity ───
    name: str                                    # "triton", "hip", "flydsl", ...
    file_extensions: frozenset[str]              # {".py"} for Triton, {".cpp", ".hip", ".cu"} for HIP
    detect_hints: tuple[str, ...] = ()           # regex patterns that boost detection confidence
                                                 #   Triton: (r"@triton\.jit", r"^import triton")
                                                 #   HIP: (r"__global__\s+void", r"hipLaunchKernelGGL")

    # ─── LLM prompts (markdown, loaded lazily) ───
    system_prompt_path: Path                     # kernel_languages/triton/system_prompt.md
    optimization_prompt_path: Path               # kernel_languages/triton/optimization_prompt.md
    planner_strategy_hints_path: Path            # kernel_languages/triton/planner_strategy_hints.md

    # ─── preprocess templates (Jinja) ───
    harness_template_path: Path                  # kernel_languages/triton/harness.j2
    commandment_template_path: Path              # kernel_languages/triton/commandment.j2 — SINGLE
                                                  # SOURCE OF TRUTH for Setup/Correctness/Benchmark/
                                                  # FullBenchmark/Profile commands. Per-language
                                                  # quirks (HIP's `make`, `rocprof`, Triton's `python3`,
                                                  # Metrix profiler call) live ONLY here.
    builder_hints_path: Path                     # kernel_languages/triton/builder_hints.md

    # ─── execution environment (setup-time, NOT commands) ───
    # NOTE: earlier drafts had test_runner_command + profiler_command fields. Removed —
    # they duplicated what commandment.j2 already encodes. Harness CLI is UNIVERSAL
    # (python3 {harness} --{correctness|benchmark|full-benchmark|profile}) across all
    # languages, enforced by HarnessBuilder + validate_harness() contract. Code that
    # needs a command reads the rendered COMMANDMENT.md by section anchor.
    eval_env: dict = field(default_factory=dict) # {"PIP": [...], "DOCKERFILE_HINTS": "..."}

    # ─── tools (PR-11 adds this) ───
    tool_set: frozenset[str] = field(default_factory=frozenset)  # {bash, save_and_test, ...}

    # ─── cross-session memory ───
    kb_namespace: str                            # "triton" / "hip" — separates KB entries per language
    memory_hints_path: Path | None = None        # PR-71: kernel_languages/triton/memory_hints.md

    # ─── translation (PR-60) ───
    idioms_path: Path | None = None              # kernel_languages/triton/idioms.md (natural-language)
    builtin_types: frozenset[str] = field(default_factory=frozenset)
    translation_hints: dict = field(default_factory=dict)  # {"hip": {"thresholds": (0.5, 0.8), ...}}
```

**Loading pattern** (lazy): the `_path` fields are `Path` objects; the actual markdown is loaded on first access via `KernelLanguage.get_system_prompt()` helper, which reads the file and caches the result.

#### `PreprocessContext` (dataclass — PR-11 expands; PR-60 adds translation fields)

```python
# src/minisweagent/run/context_types.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class DiscoveryResult:
    kernel_name: str
    kernel_type: str
    tests: list[Path] = field(default_factory=list)
    shape_regime: str | None = None

@dataclass
class TranslationResult:
    ok: bool
    translated_kernel_path: Path | None = None
    attempts: int = 0
    golden_matched: bool = False
    perf_ratio: float | None = None
    reason: str | None = None

@dataclass
class PreprocessContext:
    # ─── core identity ───
    kernel_path: Path                    # gets updated to TRANSLATED path after Translation phase
    task_text: str
    artifacts_dir: Path
    language: "KernelLanguage"           # populated by cli.py after detection (PR-11)

    # ─── translation mode (PR-2) ───
    # target_language defaults to the SAME object as `language` when the user doesn't
    # explicitly pass --target-language. Translation phase is skipped unless target != language.
    # Runtime safety: Optional typing + __post_init__ guarantees target_language is
    # always non-None by the time the orchestrator sees ctx — downstream can
    # safely compare `if ctx.target_language != ctx.language:` without a None check.
    target_language: Optional["KernelLanguage"] = None
    translate_only: bool = False
    translation_result: Optional[TranslationResult] = None

    def __post_init__(self):
        # Defaults target_language to language when omitted so the orchestrator
        # never receives None. This closes the reviewer-flagged footgun where
        # a caller forgetting to wire target_language would AttributeError inside
        # TranslationPhase accessing target_language.system_prompt_path.
        if self.target_language is None:
            self.target_language = self.language

    # ─── mode + parallel control ───
    mode: str = "auto"                   # "auto" is default; also: "fixed" | "planned" | "translate"
    num_parallel: int = 1
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    max_rounds: int = 5
    cli_overrides: dict = field(default_factory=dict)

    # ─── preprocess outputs (filled by phases) ───
    discovery: Optional[DiscoveryResult] = None
    codebase_context_path: Path | None = None
    harness_path: Path | None = None
    baseline_latency_ms: float | None = None
    profile_path: Path | None = None
    baseline_metrics_path: Path | None = None
    commandment_path: Path | None = None
    kernel_analysis_md: str | None = None

    # ─── round tracking ───
    round: int = 0
```

### §16.2 — Agent + subagent class hierarchy (final shape after PR-73)

Two **peer** class families. They are NOT in a parent-child relationship. `OptimizationAgent` is standalone; `SubagentBase` is standalone. Subagents compose (have-a) `OptimizationAgent`; they do NOT inherit (is-a) from it.

```
agents/                                       ← the main agent family
  agent_spec.py        — AgentConfig + exceptions (Submitted, LimitsExceeded, …)
                         [SHARED primitives used by BOTH families]
  optimization_agent.py — OptimizationAgent   ← the ONE main agent class
                         — no inheritance chain (PR-50 collapses 4 layers into this)
                         — NOT a subclass of SubagentBase
                         — INSTANTIATED by `run_pipeline` to run round-level optimization
                         — CAN also be instantiated internally by subagents (via composition,
                           see SubagentBase._make_optimization_agent)
  select_patch_agent.py — SelectPatchAgent    ← LLM patch picker (deferred to PR-80+ for move
                                                 to subagents/selection/)

subagents/                                    ← the subagent family
  base.py               — SubagentBase        ← base for ALL subagents
                         — NOT a subclass of OptimizationAgent
                         — exposes two execution modes: run() one-shot, loop() multi-round
                         — each subclass overrides EXACTLY ONE of {run, loop}
                         — helper _make_optimization_agent(tools) composes an
                           OptimizationAgent when a subagent needs a full tool loop
  preprocess/
    unit_test_agent.py   — UnitTestAgent(SubagentBase)            [run()]
    harness_builder.py   — HarnessBuilder(SubagentBase)           [run()]
    kernel_analysis.py   — KernelAnalysisAgent(SubagentBase)      [run()]
    shape_fixer.py       — ShapeFixerAgent(SubagentBase)          [run()]
  memory/
    cross_session_memory_analysis.py
                         — CrossSessionMemoryAnalysisAgent(SubagentBase) [run()]
  translation/
    translator.py        — TranslationLoop(SubagentBase)          [loop()]
```

**Why peers and not parent/child?**

| Attribute | `OptimizationAgent` | `SubagentBase` (and subclasses) |
|---|---|---|
| Lifetime | Long (full round loop; tens of LLM turns) | Short (one focused task, typically 1-5 LLM turns) |
| Tool set | Large (bash, editor, save_and_test, profile, strategy_manager, sub_agent, …) | Narrow (e.g., HarnessBuilder: editor + bash + submit) |
| Exit condition | `Submitted` exception OR step/cost limit OR round budget | For `run()`: first Submitted; for `loop()`: verify_fn OK OR max_attempts |
| Owns control flow? | Yes (round-level orchestration feeds it) | No (subagent owns its own retry logic, if any) |
| Configured by | `run/unified.py:_resolve_tools` + `AgentConfig` | Per-subagent YAML in `subagents/<area>/configs/<name>.yaml` |
| Who instantiates it? | `run/unified.py` (main pipeline) OR `SubagentBase._make_optimization_agent` (subagent internals) | `preprocess/orchestrator.py` phases OR `cross_session/router.py` OR `cli.py translate` |

Forcing them to share a common base class would either:
(a) leak long-loop concerns (step counters, strategy_manager) into narrow subagents, OR
(b) leak subagent concerns (structured JSON submission, `max_attempts` loop) into the main optimization path.

Keeping them as peers that share only `AgentConfig` + exception primitives (in `agents/agent_spec.py`) is the clean separation. CI gate `check_no_agent_inheritance.py` enforces: no class subclasses both; no class subclasses the old `DefaultAgent`/`InteractiveAgent`/`StrategyAgent` chain; `SubagentBase` subclasses don't subclass `OptimizationAgent` either.

#### `OptimizationAgent` (PR-50 merges 4 classes into this)

```python
# src/minisweagent/agents/optimization_agent.py
from __future__ import annotations
from typing import Callable, Optional
from minisweagent.agents.agent_spec import (
    AgentConfig, Submitted, NonTerminatingException, LimitsExceeded, TerminatingException,
)

class OptimizationAgent:
    """THE agent class. No inheritance chain. Language-agnostic.

    Invariants:
    - Never imports KernelLanguage
    - Never branches on kernel_type
    - Tool list is resolved before __init__ (see run/unified.py:_resolve_tools)
    - Notifies strategy_changed via callback (single-purpose hook; was a whole class before)
    """

    def __init__(
        self,
        config: AgentConfig,
        model,
        env,
        tools: list,                                       # already resolved by caller
        on_strategy_changed: Optional[Callable] = None,    # optional; rich-console printer in CLI
    ):
        self.config = config
        self.model = model
        self.env = env
        self.tools = tools
        self.on_strategy_changed = on_strategy_changed
        self._messages: list = []
        self._step_count = 0

    def run(self, task_body: str) -> dict:
        """Execute until Submitted or step/cost limit. Returns FinalReport dict."""
        self._messages = [
            {"role": "system", "content": self.config.system_template.format(**self._prompt_vars())},
            {"role": "user", "content": task_body},
        ]
        try:
            while True:
                self._step()
        except Submitted as s:
            return self._build_report(submission=s.output)
        except (LimitsExceeded, TerminatingException) as e:
            return self._build_report(terminated=True, reason=str(e))

    def _step(self): ...   # LLM call → parse tool call → execute tool → observation
    def _prompt_vars(self) -> dict: ...
    def _build_report(self, **kw) -> dict: ...
```

#### `SubagentBase` (PR-07 scaffolds; PR-21 fills in)

```python
# src/minisweagent/subagents/base.py
from pathlib import Path
from abc import ABC
from minisweagent.kernel_languages.base import KernelLanguage
# SubagentBase does NOT import/subclass OptimizationAgent; it COMPOSES one when needed.

class SubagentBase(ABC):
    """Base for narrow-task LLM subagents. Peer class to OptimizationAgent (NOT a subclass).

    Subclasses override EXACTLY ONE of run() or loop() (CI-enforced via
    scripts/check_subagent_base_contract.py).

    To execute an LLM turn, a subagent usually calls self._make_optimization_agent(tools)
    — this INSTANTIATES (composes) an OptimizationAgent for a short lifetime, and that
    agent is the one doing the actual tool loop. SubagentBase owns the subagent lifecycle
    (config, prompts, retry); OptimizationAgent owns the per-call tool loop.
    """

    def __init__(self, language: KernelLanguage, config_path: Path):
        self.language = language
        self.config = self._load_config(config_path)
        self.model = self._resolve_model(self.config)

    def run(self, **inputs) -> str | dict:
        """One-shot: single OptimizationAgent composition. Returns submission."""
        raise NotImplementedError("Override run() OR loop() — not both, not neither.")

    def loop(self, *, max_attempts: int, verify_fn, **inputs):
        """Multi-round: repeatedly compose OptimizationAgent until verify_fn OK or max_attempts."""
        raise NotImplementedError("Override run() OR loop() — not both, not neither.")

    # ── shared helpers (NO inheritance from OptimizationAgent; composition only) ──
    def _load_config(self, path: Path) -> dict: ...
    def _resolve_model(self, config: dict): ...
    def _compose_prompt(self, **inputs) -> str:
        """Uses compose_task_body() with the subagent's language + mode."""
        ...

    def _make_optimization_agent(self, tools: list) -> "OptimizationAgent":
        """COMPOSITION boundary. Instantiate a short-lived OptimizationAgent for one LLM call.

        This is the ONLY place a subagent touches OptimizationAgent. Subagents never
        subclass it — they borrow its tool-loop implementation by instantiation.
        """
        from minisweagent.agents.optimization_agent import OptimizationAgent  # local import
        return OptimizationAgent(
            config=self.config,                  # subagent's own config, not shared state
            model=self.model,
            env=self._default_env(),
            tools=tools,
            on_strategy_changed=None,            # subagents don't emit strategy events
        )
```

**Visual of the composition boundary:**

```
preprocess/phases/harness.py
  └─ HarnessBuilder(SubagentBase)          ← is-a SubagentBase
       .run(user_test_files=..., out_path=...)
         └─ self._make_optimization_agent(tools=[editor, bash, submit])
              └─ OptimizationAgent(...)     ← has-a, NOT is-a
                   .run(task_body)           ← does the actual LLM tool loop
                                             ← returns submission
         ← returns path to generated harness.py

run/unified.py::run_pipeline (main path — NOT via SubagentBase)
  └─ for r in 1..max_rounds:
       └─ OptimizationAgent(config, model, env, tools).run(task_body)
                                             ← same class, different caller context,
                                               long-lived (full round)
```

CI gate `scripts/check_subagent_base_contract.py` verifies (by AST walk) that each subclass of `SubagentBase` overrides exactly one of `run` / `loop`.

#### Subagent examples (two patterns)

```python
# subagents/preprocess/harness_builder.py — ONE-SHOT pattern
class HarnessBuilder(SubagentBase):
    def run(self, *, user_test_files: list[Path], out_path: Path) -> Path:
        task_body = self._compose_prompt(
            user_files=user_test_files,
            harness_template=self.language.harness_template_path.read_text(),
            builder_hints=self.language.builder_hints_path.read_text(),
        )
        agent = self._make_optimization_agent(tools=[editor, bash, submit])
        result = agent.run(task_body)
        out_path.write_text(result["submission"])
        return out_path


# subagents/translation/translator.py — MULTI-ROUND pattern
class TranslationLoop(SubagentBase):
    def __init__(self, source_lang, target_lang, config_path):
        super().__init__(target_lang, config_path)
        self.source_lang = source_lang

    def loop(self, *, src_kernel_path: Path, test_harness_path: Path,
             max_attempts: int = 3, verify_fn=None) -> TranslationResult:
        golden = run_harness(src_kernel_path, test_harness_path, mode="correctness")
        for attempt in range(max_attempts):
            task_body = self._compose_prompt(
                src_lang=self.source_lang,
                tgt_lang=self.language,
                src_code=src_kernel_path.read_text(),
                golden_ref_path=golden.outputs_path,
                attempt_feedback=self._feedback_from_prev_attempts(attempt),
            )
            agent = self._make_optimization_agent(tools=[editor, bash, run_harness])
            draft = agent.run(task_body)
            draft_path = write_draft(draft)
            target_result = run_harness(draft_path, test_harness_path, mode="correctness")
            if tensors_allclose(golden.outputs, target_result.outputs, atol=1e-5):
                perf = validate_translation_performance(
                    golden.latency, target_result.latency,
                    thresholds=self.language.translation_hints.get(self.source_lang.name, (0.5, 0.8)),
                )
                return TranslationResult(ok=True, translated_kernel_path=draft_path,
                                          attempts=attempt + 1, golden_matched=True,
                                          perf_ratio=perf.ratio)
        return TranslationResult(ok=False, attempts=max_attempts, reason="all attempts failed golden match")
```

### §16.3 — Pipeline pseudocode (final shape after PR-73)

```python
# src/minisweagent/cli.py (the ONE CLI file)
@app.command()
def optimize(task: str, mode: str = "auto", target_language: Optional[str] = None, ...):
    """geak -t "..." [--mode fixed|planned|auto (default)] [--target-language Y]

    target_language defaults to the detected kernel_language (no translation).
    Pass --target-language only when you want translation before optimization.
    """
    ctx = build_preprocess_context(task=task, mode=mode, target_language=target_language, ...)
    # build_preprocess_context enforces: if target_language_name is None,
    # ctx.target_language = ctx.language (the detected one); else registry.detect_best_by_name().
    ctx = PreprocessOrchestrator(ctx).run()
    if ctx.translate_only and ctx.translation_result:
        write_final_report(ctx, translation_only=True)
        return
    run_pipeline(ctx, mode=ctx.mode)


# src/minisweagent/preprocess/orchestrator.py
class PreprocessOrchestrator:
    def __init__(self, ctx: PreprocessContext):
        self.ctx = ctx

    def run(self) -> PreprocessContext:
        c = self.ctx
        if c.target_language != c.language:          # default ==, so Translation phase skipped unless user set target
            c = TranslationPhase(c).run()                  # optional
            if c.translate_only:
                return c
        c = DiscoveryPhase(c).run()                        # resolve + codebase + tests
        c = HarnessPhase(c).run()                          # [+HarnessBuilder] + validate
        c = BaselinePhase(c).run()                         # baseline + profile + metrics
        c = ExplorePhase(c).run()                            # commandment + validate + kernel_analysis
        return c


# src/minisweagent/run/unified.py
def run_pipeline(ctx: PreprocessContext, mode: str) -> dict:
    """Replaces run_homogeneous_agent + run_heterogeneous_orchestrator."""
    tools = _resolve_tools(ctx, mode)                        # §4 single-site resolution
    best_result = None
    for r in range(1, ctx.max_rounds + 1):
        ctx.round = r
        retrieved = cross_session_retrieve(ctx)
        memory_ctx = _maybe_inject_memory_context(ctx, r)    # single path: subagent when flag=1, else empty
        tasks = build_tasks_for_round(ctx, memory_ctx, mode) # uses compose_task_body
        results = pool_run(ctx, tasks, tools)                # replaces run_pool + ThreadPoolExecutor
        eval_ = evaluate_round(ctx, results)                 # FULL_BENCHMARK per round (HIP too now)
        if eval_.best_speedup > getattr(best_result, "speedup", 0):
            best_result = eval_
    if best_result and best_result.verified:
        record_optimization_outcome(ctx, best_result)        # KB write for ALL modes now
    return finalize_report(ctx, best_result)


def _resolve_tools(ctx: PreprocessContext, mode: str) -> list:
    """Single tool-resolution site. CI-enforced (check_tool_resolve_single_site.py)."""
    tool_names = set(ctx.language.tool_set)
    if mode == "planned":
        tool_names |= ORCHESTRATOR_TOOLS
    elif mode == "translate":
        tool_names |= TRANSLATION_TOOLS
    tool_names = _apply_cli_overrides(tool_names, ctx.cli_overrides)
    return build_tool_table(tool_names)  # calls tools_runtime registration
```

### §16.4 — End-to-end sequence diagrams

#### (a) `geak -t "optimize foo.py ..." --mode auto`

```
cli.py::optimize
 ├─ build_preprocess_context()              → PreprocessContext(language=Triton, mode="auto", ...)
 ├─ PreprocessOrchestrator.run()
 │   ├─ Phase0 skipped (no target_language)
 │   ├─ DiscoveryPhase.run()              → ctx.discovery, codebase_context_path
 │   ├─ HarnessPhase.run()
 │   │   ├─ harness_decision()              → reuse user harness OR spawn UnitTestAgent.run()
 │   │   ├─ HarnessBuilder.run()            → ctx.harness_path
 │   │   └─ validate_harness(harness_path)  → ContractViolation or OK
 │   ├─ BaselinePhase.run()               → ctx.baseline_latency_ms, profile_path
 │   └─ ExplorePhase.run()
 │       ├─ render_commandment (Jinja)      → ctx.commandment_path
 │       ├─ validate_commandment()          → OK
 │       └─ KernelAnalysisAgent.run()       → ctx.kernel_analysis_md
 └─ run_pipeline(ctx, mode="auto")
     ├─ _resolve_tools(ctx, "auto")         → [bash, save_and_test, profile_kernel, ...]
     └─ for r in 1..5:
         ├─ cross_session_retrieve(ctx)     → top-k ExperienceRecords
         ├─ _maybe_inject_memory_context()  → CrossSessionMemoryAnalysisAgent.run() → cross_session_memory_insights.md (~5-25 KB) when flag=1; else empty
         ├─ build_tasks_for_round()         → compose_task_body(language, mode, memory_ctx, ...)
         ├─ pool_run(ctx, tasks, tools)     → OptimizationAgent.run(task_body) per task_body
         ├─ evaluate_round()                → FULL_BENCHMARK → verified speedup
         └─ if win: record_optimization_outcome()  # KB write (now in both modes)
```

#### (b) `geak translate --source foo.py --target-language hip --output foo.hip`

```
cli.py::translate
 ├─ build_preprocess_context(translate_only=True, target_language=hip)
 ├─ PreprocessOrchestrator.run()
 │   └─ TranslationPhase.run()
 │       ├─ run_harness(src=foo.py, mode="correctness") → golden outputs saved
 │       ├─ TranslationLoop(source_lang=triton, target_lang=hip).loop(
 │       │     src_kernel_path=foo.py,
 │       │     test_harness_path=test.py,
 │       │     max_attempts=3,
 │       │   )
 │       │   ├─ attempt 1: OptimizationAgent.run(compose_task_body(mode="translate", ...))
 │       │   │             → draft_1.hip → run_harness(draft_1, correctness)
 │       │   │             → tensors_allclose? YES → return TranslationResult(ok=True, ...)
 │       │   OR (if failure) retry with feedback up to 3 times
 │       ├─ validate_translation_performance()
 │       ├─ ctx.translation_result = TranslationResult(...)
 │       ├─ ctx.kernel_path = draft_1.hip       # switch to translated file
 │       ├─ ctx.language = target_language (hip)
 │       └─ return ctx  (Discovery/Harness/Baseline/Explore skipped because translate_only=True)
 └─ write_final_report(translation_only=True, output=foo.hip)
```

#### (c) `geak add-language flydsl --extensions '.fly,.flydsl'`

```
cli.py::add_language
 └─ scaffold_language("flydsl", extensions=[".fly", ".flydsl"])
     ├─ mkdir kernel_languages/flydsl/
     ├─ render 7 files from kernel_languages/_templates/*.j2 (skeleton with TODOs for user)
     ├─ append `from .flydsl.kernel_language import FlyDSLKernelLanguage` to kernel_languages/__init__.py
     ├─ register in _Registry
     └─ print instructions: "Fill in kernel_languages/flydsl/*.md, then: geak test-language flydsl"

geak test-language flydsl
 └─ smoke_test_language("flydsl")
     ├─ import kernel_languages.flydsl.kernel_language  → FlyDSLKernelLanguage
     ├─ validate_harness(flydsl.harness_template_path.render(), flydsl)  → OK
     ├─ validate_commandment(flydsl.commandment_template_path.render(), flydsl) → OK
     ├─ registry.detect(Path("test_fixture.fly"))  → should return flydsl with score > 0.9
     └─ geak -t "Optimize test_fixture.fly ..." --max-rounds 1  → 1-round smoke (~5 min)
```

### §16.5 — Step-by-step implementation order (3 PRs, ~6-10 weeks)

Restructured from 20 weekly slots to 3 PR-sized milestones per user direction.

| Milestone | PR | Weeks | Files touched | Build artifact |
|---|---|---|---|---|
| **M1** | **PR-1** · Foundation + Cleanup + CLI | W1-W3 | +`cli.py`, +`kernel_languages/{base.py,contract.py,__init__.py,scaffolder.py,_templates/*,triton/*,hip/*}`, +`subagents/{base.py,__init__.py,preprocess/__init__.py,memory/__init__.py,translation/__init__.py}`, +`tests/smoke/*` (5 tests), +`tests/regression/baseline_speedups.yaml`, +`scripts/check_*.py` (7 CI gates), +`docs/refactor/*` (3 docs); −16 mini-swe-agent-heritage files (~2,262 LoC), −9 aux CLI entries in pyproject.toml, −3 duplicate stubs, moved `classify_kernel_category`; renamed `pipeline_types.py` → `context_types.py` | CI green; all 5 smoke tests + §0.2 regression + §13.1 capability audit pass; only `geak -t "..."` CLI entry |
| **M2** | **PR-2** · Preprocessing | W4-W6 | +`preprocess/orchestrator.py`, +`preprocess/phases/{discovery,harness,baseline,explore,translation}.py` (5 files), +`subagents/preprocess/{harness_builder,kernel_analysis,unit_test_agent,shape_fixer}.py` + configs (4 subagents), +`subagents/translation/translator.py`, +`kernel_languages/_translation/{triton_to_hip,hip_to_triton,_fallback}.md`, +`tests/fixtures/harness_corpus/` (30 fixtures), +`scripts/harness_corpus_gate.py`, +`geak translate` + `geak resume` subcommands in `cli.py`, +byte-diff check `scripts/pr2_commandment_diff.py`; −`preprocessor.py` 1386 LoC, −`commandment.py` 530 LoC, −`harness_utils.py` template-gen parts (1,100+ LoC of 1,660), −`run/preprocess/unit_test_agent.py` + `agents/unit_test_agent.py` (duplicates merged), −`run/preprocess/shape_fixer_agent.py` (moved), split `pipeline_helpers.py` 915 LoC into 4 files | Phase-based preprocess; HarnessBuilder ≥29/30 fixture corpus green; `geak translate` end-to-end works; 12-marker Triton + 11-marker HIP smoke tests still green |
| **M3** | **PR-3** · Orchestration + Optimization | W7-W9 + optional W10 ablation | +`agents/optimization_agent.py` (collapsed), +`run/unified.py` (run_pipeline), +`run/compose.py` (compose_task_body), +`run/task_generator.py` (language-agnostic, replaces heterogeneous/task_generator.py 997 LoC), +`run/pool_runner.py` + `run/pool_logger.py` (split from parallel_helpers 844 LoC), +`subagents/memory/cross_session_memory_analysis.py`, +`kernel_languages/{triton,hip}/memory_hints.md` (regex patterns moved from formatter), +`scripts/ab_task_generator.py`, +`tests/agents/test_optimization_agent.py` (replaces 3 old tests), +`--mode {fixed,planned,auto}` flag on `cli.py`; **A/B run** (390 runs × 15min / 4 GPUs ≈ 1 wall-clock day on 4 GPUs — gates PR-3 merge); −`agents/{default,interactive,strategy_agent,strategy_interactive}.py` (4-layer chain), −`agents/{heterogeneous,homogeneous}/` (entire dirs), −ThreadPoolExecutor homo branch of `parallel_agent.py`, −`memory/cross_session_memory.py` deprecation shim finally removed, −`run/utils/prompts.py`, −old agent tests; moved `heterogeneous/tools.py` → `tools/orchestrator_tools.py` | Single `OptimizationAgent`; unified `run_pipeline`; CI gates `check_no_agent_inheritance`, `check_tool_resolve_single_site`, `check_agent_tool_set` all FAIL-strict; PR-3 A/B green; §0.2 regression green |
| **M4** | **Post-merge validation** (optional) | W10 | Run `scripts/memory_ablation.py` (180 runs × 15min / 4 GPUs ≈ 0.5 GPU-day; 2 arms `no_kb` vs `subagent_kb`) | `docs/refactor/PR3_ABLATION_REPORT.md`; confirms `GEAK_USE_CROSS_SESSION_MEMORY=1` default; if fails, flip default to `0` (1-line env var change) |

### §16.6 — CI gates summary (all active by end of refactor)

| Gate | Purpose | Landed | Activation |
|---|---|---|---|
| `scripts/check_one_cli_file.py` | Only `cli.py` has Typer | PR-1 | FAIL from PR-1 (cli.py lands in PR-1) |
| `scripts/check_language_leaks.py` | No `kernel_type == "X"` outside `kernel_languages/` | PR-1 WARN | PR-2 FAIL (language detection + phases moved); FAIL-strict on `memory/` after PR-3's regex move |
| `scripts/check_subagent_location.py` | `SubagentBase` only in `subagents/` | PR-1 | FAIL from PR-1 |
| `scripts/check_subagent_base_contract.py` | Each subclass overrides exactly one of run/loop | PR-1 | FAIL from PR-1 |
| `scripts/check_no_agent_inheritance.py` | No class subclasses DefaultAgent/InteractiveAgent/StrategyAgent | PR-1 WARN | PR-3 FAIL (4-layer collapse happens in PR-3) |
| `scripts/check_tool_resolve_single_site.py` | `ToolRuntime(allowed=...)` only called in `run/unified.py:_resolve_tools` | PR-1 WARN | PR-3 FAIL |
| `scripts/check_agent_tool_set.py` | No tool in resolved tool list may import from `memory.cross_session.retriever` or `.knowledge_base` (agent cannot directly read KB; only via subagent-written insights file) | PR-1 WARN | PR-3 FAIL |
| `scripts/check_no_capability_loss.py` | §13.1 audit: every listed capability is reachable in `geak --help` / agent tool set / pyproject | PR-1 WARN | PR-3 FAIL |
| `scripts/harness_corpus_gate.py` | HarnessBuilder ≥ 29/30 on fixture corpus | PR-2 | Required green before PR-2 merges |
| `scripts/ab_task_generator.py` | PR-3 A/B equivalence gate (old vs new `task_generator`) | PR-3 | Required green before PR-3 merges |
| `scripts/check_baseline_speedups.py` | Nightly §0.2 kernel regression | PR-1 WARN | PR-3 FAIL |
| `scripts/pr2_commandment_diff.py` | Byte-level diff of Jinja vs old Python commandment output | PR-2 | Required green before PR-2 merges |

### §16.7 — Fixture corpus concrete file list (PR-22)

For the ≥ 29/30 gate (addressing reviewer #6), the corpus ships with exactly these fixtures:

```
tests/fixtures/harness_corpus/
├── SPEC.md                                      # contract the corpus enforces
├── KNOWN_FAILURES.md                            # the ≤ 1 allowed failure documented here
├── triton_direct/                               # ≥ 10
│   ├── 01_fast_rms_layernorm.py
│   ├── 02_topk.py
│   ├── 03_llama_ff.py
│   ├── 04_fused_rms_fp8.py
│   ├── 05_moe_routing_sigmoid_top1.py
│   ├── 06_lean_atten_paged.py
│   ├── 07_rope.py
│   ├── 08_fused_qkv_rope.py
│   ├── 09_gemm_a16w16_atomic.py
│   ├── 10_fused_append_shared_experts.py
│   └── ...
├── triton_wrapper/                              # ≥ 6 (aiter-wrapped)
│   ├── 01_gemm_aiter.py
│   ├── 02_mla_decode_aiter.py
│   ├── 03_fused_mxfp4_quant_moe_sort_aiter.py
│   └── ...
├── hip_raw/                                     # ≥ 8
│   ├── 01_silu.cpp
│   ├── 02_ball_query.cu
│   ├── 03_knn.hip
│   ├── 04_three_nn.cu
│   └── ...
├── hip_pybind/                                  # ≥ 4
│   ├── 01_assign_score_withk.cpp
│   ├── 02_gather_points.cpp
│   └── ...
└── edge/                                        # ≥ 2
    ├── 01_dynamic_shape.py
    └── 02_multi_return.py
```

Each fixture directory contains:
- `kernel.{py,cpp,cu,hip}` — the kernel source
- `user_test.py` — deliberately unformatted user test file (diverse style)
- `expected_harness.py` — hand-written reference harness (what `HarnessBuilder` should produce)
- `expected_validation.yaml` — `{valid: true, emits_geak_result_latency_ms: true, emits_geak_result_speedup: true}`

Gate script `scripts/harness_corpus_gate.py` runs HarnessBuilder on all 30+ fixtures, passes each output through `validate_harness()`, and counts valid vs invalid. Required: ≥ 29/30 pass.

### §16.8 — Build + smoke-test walkthrough (end-state; what any new dev runs AFTER all 3 PRs land)

**Note** (reviewer-raised Issue 2.5): This walkthrough describes the **end state**. Not all scripts/tests exist on day 1 of PR-1. Below each command, a marker indicates which PR lands it — a dev at PR-1 only runs those marked `[PR-1]`; PR-2 enables `[PR-2]` checks; PR-3 enables `[PR-3]` checks.

```bash
# 1. Clone + install
git clone -b refactor-test https://github.com/AMD-AGI/GEAK.git
cd GEAK
pip install -e .

# 2. Run static CI gates (< 30 sec)
python scripts/check_one_cli_file.py                # [PR-1]  — FAIL-strict from PR-1
python scripts/check_language_leaks.py              # [PR-1]  — WARN in PR-1, FAIL in PR-2
python scripts/check_subagent_location.py           # [PR-1]  — FAIL-strict from PR-1
python scripts/check_subagent_base_contract.py      # [PR-1]  — FAIL-strict from PR-1
python scripts/check_no_agent_inheritance.py        # [PR-1 WARN → PR-3 FAIL]
python scripts/check_tool_resolve_single_site.py    # [PR-1 WARN → PR-3 FAIL]
python scripts/check_agent_tool_set.py              # [PR-1 WARN → PR-3 FAIL]
python scripts/check_no_capability_loss.py          # [PR-1 WARN → PR-3 FAIL]
python scripts/harness_corpus_gate.py               # [PR-2]
python scripts/ab_task_generator.py --check-report  # [PR-3]  — reads docs/refactor/PR3_AB_REPORT.md

# 3. Run smoke tests (~15 min with GPU)
pytest tests/smoke/test_triton_hetero_invariants.py         # [PR-1] 12 Triton markers
pytest tests/smoke/test_hip_homo_invariants.py              # [PR-1] 11 HIP markers
pytest tests/smoke/test_cross_session_memory_analysis.py    # [PR-1] unit (stub); [PR-3] real subagent
pytest tests/smoke/test_add_language_scaffolder.py          # [PR-1] scaffolder
pytest tests/smoke/test_translate_triton_to_hip.py          # [PR-2] translation

# 4. Run a real kernel (full pipeline, ~20 min) — needs PR-3 for unified pipeline
export AMD_LLM_API_KEY=<key>
geak -t "Optimize /path/to/triton_kernel.py. Tests: python3 tests/harness.py ..."    # [any PR-X]
# After PR-3: watch for phase markers (Phase: Discovery, Phase: Harness, ...)
# Check logs_dir/final_report.json for verified_speedup

# 5. Try translation (needs PR-2)
geak translate --source kernel.py --target-language hip --output kernel.hip --test test.py   # [PR-2]

# 6. Add a new language
geak add-language mylang --extensions '.mylang'
# edit kernel_languages/mylang/*
geak test-language mylang
```

---
