# How cross-session memory + RAG enhance (not replace) the agent

Two knowledge paths, non-overlapping roles, both optional. Neither issues
directives — each supplies evidence the agent consults while forming its
own plan from the actual kernel source and profile.

## The two paths

| | Cross-session memory KB | rag-mcp (PR #90) |
|---|---|---|
| What | Per-kernel verified experiences from past GEAK runs | Generic GPU/ROCm/HIP documentation |
| Content | Real code diffs, full `original_kernel_code`, profiler_metrics, per-round strategies (winners + dead-ends), trajectories, `kernel_structure` | 4207 markdown chunks: AMD aiter customer-case reports, optimization guides, kernel-type best practices |
| Similarity | By exact `original_kernel_code` (byte-identical = verbatim-applicable) OR code-term overlap + category + bottleneck | FAISS (BGE-large 1024-dim) + BM25 hybrid + BGE reranker |
| Delivery | Injected into task context at run start (~20KB cap) | MCP tools `query` / `optimize` the agent calls on demand |
| Toggle | `GEAK_USE_KNOWLEDGE_BASE=1` (read), `GEAK_SAVE_TO_KNOWLEDGE_BASE=1` (write) | `tools.rag: true` in `geak.yaml` |

## Full lifecycle of a single kernel optimization run

### 1. Preflight (before R1)

```
orchestrator.run_heterogeneous()
  ├─ reads profile.json, baseline_metrics.json, kernel.py
  ├─ KB retrieval: memory.integration.assemble_memory_context()
  │   └─ retriever.retrieve_context()
  │       ├─ Stage 1: broad fetch of N candidates from SQLite backend
  │       ├─ Stage 2: text-similarity scoring
  │       │   ├─ keyword overlap (kernel name, category terms)
  │       │   ├─ name-stem boost (shared tokens, e.g. "rms", "rope")
  │       │   ├─ category-match boost
  │       │   └─ scaled success boost (≥2.5x=+0.20, ≥1.5x=+0.10, ≥1.10x=+0.05)
  │       ├─ Stage 3: relevance gate — only inject if
  │       │           text_sim >= 0.05 OR category matches
  │       ├─ Stage 3b: diversity re-rank (top-5)
  │       └─ Stage 4: formatter.format_landscape_context()
  │           ├─ top-hit note (ONLY if original_kernel_code is
  │           │                byte-identical to current kernel.py)
  │           ├─ framing: "added context from similar past runs;
  │           │           examine, cross-reference, decide"
  │           └─ per entry: kernel_structure, strategies with speedups,
  │                         per-round winners + regressions with diffs,
  │                         profiling_insight, baseline_benchmark,
  │                         round_insights (trajectory)
  │
  └─ RAG tool registration (tools_runtime._register_rag_mcp)
      └─ query / optimize added to orchestrator's tool list,
        prompted into agent system message via rag_tools_desc
```

Result: the agent enters task generation with
- KB context (~20KB) describing past runs as evidence
- `query` and `optimize` tools it may call at will
- Its own profiler output, baseline metrics, kernel source

### 2. Task generation

`task_generator` produces 4 strategies per round. For each candidate:

- Agent reads the KB block and asks itself:
    - Is any past run's `original_kernel_code` ≈ my current kernel.py?
      If yes, its winning diff is a candidate starting point.
    - Are any past dead-ends structurally similar to what I was about
      to try? Then deprioritize.
    - Does any `key_insight` or `kernel_structure` suggest a pattern
      relevant to my profile's bottleneck?

- Agent optionally calls `query("MoE L2 cache ordering")` or
  `optimize(kernel_type="fused_moe", gpu_model="MI355X")` when it
  needs background on a specific technique — NOT for whole-kernel
  recipes.

- Agent outputs 4 strategy names + per-strategy instructions to
  sub-agents.

### 3. Sub-agent dispatch (per round, 4 parallel sub-agents on 4 GPUs)

- Each sub-agent gets the SAME KB context + RAG tools + its
  assigned strategy name.
- Sub-agent iterates: read kernel.py → propose diff → `save_and_test`
  with `--benchmark` (noisy but fast) → iterate → submit best patch.
- Sub-agent may also call `query`/`optimize` on demand.

### 4. Round evaluation

- Best patch across 4 sub-agents is verified with `--full-benchmark`
  (25 shapes × 30 iters geomean, stable).
- If verified_speedup ≥ 1.10x AND `GEAK_SAVE_TO_KNOWLEDGE_BASE=1`:
    - `extractor.extract_experience()` builds an `ExperienceRecord`
      with all 29 fields populated:
        - `patch_content` (real git diff)
        - `original_kernel_code` (snapshot of kernel.py as the
          agent saw it at the start of the run — this is the
          identity key for future matching)
        - `strategies` (every per-task best_results.json with diff +
          measured speedup, tagged `is_regression` / `is_marginal`)
        - `round_insights`, `what_worked`, `what_failed`, `dead_ends`
        - `profiling_metrics` (numeric), `profiling_insight` (summary)
        - `baseline_benchmark` (embedded baseline_metrics.json)
        - `kernel_structure` (auto-inspected: `@triton.jit` count,
          `@triton.autotune`, class count, HIP/cpp_extension presence)
        - `agent_reasoning_samples` (top-3 from working notebook)
    - Stored to SQLite DB; becomes retrievable on next run of any
      kernel.
    - Below-threshold runs are SKIPPED — KB stays above 1.10x noise floor.

### 5. Next kernel in queue

At run start, DB is seeded from `knowledge_base.json` if empty,
otherwise used as-is. Our most recent successful runs are therefore
visible to the next kernel; if any of them match by
`original_kernel_code` OR code terms / category / bottleneck, they
contribute to the context block.

## Why this enhances rather than constrains

1. **Code-based identity, not name/URL.** `_build_top_hint` only emits
   a "verbatim-applicable" note when `original_kernel_code` is
   byte-identical. Cross-kernel entries appear only as REFERENCE diffs
   the agent may or may not adopt after comparing against its current
   kernel.

2. **No directive framing.** Former "DECIDE", "FIRST MOVE", "make it
   your first priority" language was removed. Current framing: "added
   context from past runs; examine, cross-reference with YOUR kernel
   + YOUR profile + what would actually apply, make an informed
   decision. The KB does not prescribe — it informs."

3. **Relevance gate.** KB context is NOT injected when no entry
   scores above the similarity threshold (no category match AND
   text_sim < 0.05). No irrelevant past runs ever reach the agent.

4. **Context budget cap (20KB).** Prevents the KB from dominating the
   prompt and crowding out the agent's reasoning on the actual kernel.

5. **RAG is opt-in.** `query` / `optimize` are tools the agent chooses
   to call, not pre-injected content. Documentation is fetched only
   when the agent decides it needs background.

6. **Threshold discipline.** Only ≥1.10x verified entries are stored
   (single source of truth: `GEAK_MEMORY_MIN_SPEEDUP`). Dead-ends are
   captured as AVOID hints inside each entry's `strategies` array, not
   as top-level entries.

## Turn each path on/off

| Scenario | Flags |
|---|---|
| Full stack (default for `mem=on`) | `GEAK_USE_KNOWLEDGE_BASE=1`, `GEAK_SAVE_TO_KNOWLEDGE_BASE=1`, `tools.rag: true` |
| KB only, no generic RAG | same as above + `tools.rag: false` |
| RAG only, no KB | `GEAK_MEMORY_DISABLE=1`, `tools.rag: true` |
| Pure baseline (neither) | `GEAK_MEMORY_DISABLE=1`, `tools.rag: false` |

## Module layout (post-simplification)

```
src/minisweagent/memory/cross_session/
  schemas.py         # ExperienceRecord + StrategySkill dataclasses
  config.py          # CrossSessionConfig (env-var driven)
  integration.py     # Public API: assemble_memory_context, record_optimization_outcome
  backends/
    base.py          # MemoryBackend protocol
    local.py         # SQLite backend (default)
    remote.py        # HTTP backend (opt-in via GEAK_CROSS_SESSION_MEMORY_URL)
  retriever.py       # Multi-stage retrieval funnel
  formatter.py       # Context formatting (top-hit note, framing, per-entry)
  extractor.py       # ExperienceRecord from run artifacts
  consolidation.py   # Aggregate experiences → StrategySkill (admin tool)
  cli.py             # Admin CLI (consolidate, dump, list)
  knowledge_base.json # Seed data

mcp_tools/rag-mcp/   # Official PR #90 RAG server (generic docs)
```

No `rag_hook.py` (redundant with rag-mcp). No `_build_rag_block`
in formatter (RAG is served as MCP tool, not pre-injected).
