# GEAK-v3

GEAK is an AI-powered framework for automated GPU kernel optimization, built on top of mini-SWE-agent.

It enables systematic, profiling-driven, and scalable optimization of GPU kernels — evolving from single-kernel tuning (v1/v2) to full repository-level autonomous optimization (v3).

## Table of Contents

- [Evolution: From Kernel-Level to Repo-Level Automation](#evolution-from-kernel-level-to-repo-level-automation)
- [Core Architecture](#core-architecture)
  - [End-to-End Optimization Engine](#end-to-end-optimization-engine)
  - [Tool-Augmented Intelligence Layer](#tool-augmented-intelligence-layer)
  - [Parallel Exploration & Scaling](#parallel-exploration--scaling)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Output & Artifacts](#output--artifacts)
- [Features](#features)
  - [Unit Test Discovery](#unit-test-discovery)
  - [System Tools (Built-in)](#system-tools-built-in)
  - [Best Patch Selection](#best-patch-selection)
  - [Knowledge Base Retrieval](#knowledge-base-retrieval)
- [Summary](#summary)

---

## Evolution: From Foundation to Platform

### GEAK v1 — Foundation (Triton)

GEAK v1 established the foundation with Triton-based kernel generation.

- Reflexion-based kernel generation
- Instruction → Triton kernels
- TritonBench / ROCmBench improvements

**Outcome:** AI viability proven — LLM-based agents can generate and improve GPU kernels.

### GEAK v2 — Expansion (Agent Family)

GEAK v2 expanded into a multi-agent system for HIP kernel optimization.

- **OptimAgent:** profiling-driven optimization with multi-offspring exploration
- **OpenEvolve:** genetic optimization for kernel evolution
- support HIP → HIP kernel optimization

**Outcome:** Scalable multi-agent system
### GEAK v3 — Platform (L1 → L3)

GEAK v3 evolves into a unified platform supporting the full optimization stack.

- Support **L3** kernel optimization (repository-level, full lifecycle)
- Reduce human intervention via closed-loop automation
- Unified kernel optimization (test discovery, baselines, profiling, strategy execution, validation)

**Outcome:** Anyone can optimize kernels — from single-kernel tuning to autonomous repo-level optimization.

---

## Core Architecture

### End-to-End Optimization Engine

At its core, GEAK runs a fully autonomous optimization loop:

**Test Detection → Baseline → Profiling → Strategy Planning → (Patch Generation → Validation) × N → Best-performing Kernel**

Each optimization step is:

- Correctness-verified
- Performance-measured
- Version-tracked

### Tool-Augmented Intelligence Layer

GEAK v3 introduces a structured tool ecosystem that enhances agent reasoning and execution quality.

The system integrates:

- **Profiling** for quantitative bottleneck identification
  (memory bandwidth, occupancy, register pressure, execution stalls)

- **Optimization Strategy Management** for tracking explored techniques, marking successful/failed strategies, and prioritizing high-impact directions

- **Version & Patch Management** for automatic diff tracking, benchmarking history, regression detection, and best-patch selection

- **Knowledge Retrieval** for on-demand AMD/NVIDIA GPU knowledge retrieval during optimization

### Parallel Exploration & Scaling

GEAK v3 supports parallel optimization agents. This parallel scaling:

- Raises the optimization ceiling
- Increases robustness of exploration
- Reduces dependence on single optimization trajectories

---

## Getting Started

### Installation

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK
pip install -e .

# To use the Knowledge feature, also install the langchain dependencies
pip install -e '.[langchain]'

# Set model name and key

# Option 1: set a LiteLLM model + provider API key
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"

# Anthropic example
export MSWEA_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="YOUR_KEY"

# Option 2: If you use AMD LLM Gateway (model_class: amd_llm)
export AMD_LLM_API_KEY="YOUR_KEY"
```

### Usage

#### Basic (single-agent) GPU kernel optimization

```bash
# Interactive REPL
mini

# Run with a specific task
mini -t "fix the bug in main.py"

# Auto-execute mode (no confirmation needed)
mini --yolo

# Enable RAG knowledge retrieval
mini --rag

# Use specific config.yaml
mini --config geak.yaml \
  --task "Optimize the kernel in src/kernel.cpp" \
  --yolo
```

#### Parallel optimization (multiple agents + best patch selection)

- Each agent works in an isolated git workspace
- Patches and test results are saved separately
- After all runs finish, GEAK automatically selects the best patch based on the specified metric

```bash
mini --config geak.yaml \
  --num-parallel 4 \
  --repo /path/to/kernel/repo \
  --task "Optimize block_reduce kernel" \
  --gpu-ids 0,1,2,3 \
  --metric "Extract Bandwidth in GB/s (higher is better)" \
  --yolo
```

**Notes:**

- `--num-parallel`: number of optimization agents
- `--repo`: required when `--num-parallel > 1` (each agent uses an isolated git worktree)
- `--gpu-ids`: comma-separated GPU IDs for agents
- `--metric`: natural-language instruction for extracting/comparing metrics from test logs
- `--yolo`: run end-to-end without interactive confirmation

### Configuration

#### Loading Configurations
`mini` loads configs in layers:

1. base config: `mini.yaml`
2. template: `mini_kernel_strategy_list.yaml` (default)
3. user override: `--config geak.yaml` (**final override**)

This means you can configure tools and parallel defaults directly in `geak.yaml`.

#### RAG Configuration

File: `rag_config.yaml` — controls the RAG retrieval pipeline:

| Parameter | Description |
|-----------|-------------|
| `retrieval.embed_top_k` / `bm25_top_k` | Number of candidates from Embedding / BM25 retrieval |
| `retrieval.enable_bm25` | Whether to enable BM25 dual-path recall |
| `retrieval.mcp_top_k` | Number of final results returned |
| `reranker.enable_reranker` | Whether to enable re-ranking |
| `fusion.semantic_weight` / `bm25_weight` | Fusion weights for Embedding and BM25 |
| `summary.enable_rag_subagent` | Whether to enable LLM summarization |
| `debug.verbose` | Whether to print verbose RAG tool logs |

### Output & Artifacts

GEAK saves patches + test logs so results are reproducible.

- **Default output base**: `optimization_logs/`
- **Auto-generated run directory**: `optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`
- **Parallel runs**: subfolders `parallel_0/`, `parallel_1/`, ...

Typical structure (parallel run):

```bash
optimization_logs/<kernel>_<timestamp>/
├── parallel_0/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   └── agent_0.log
├── parallel_1/
│   └── ...
├── best_results.json
└── select_agent.log
```

---

## Features

### Unit test discovery

If you pass `--create-test`, or you **do not** provide `--test-command`, GEAK will run a **UnitTestAgent** that tries to discover or create tests:

```bash
mini --config geak.yaml \
  --repo /path/to/kernel/repo \
  --create-test \
  --task "Optimize device_batch_memcpy kernel"
```

### System Tools (built-in)

| Tool | Purpose | Key outputs |
| --- | --- | --- |
| `profiling` | Profile workload to identify bottlenecks | rocprofiler-compute summary |
| `strategy_manager` | Track optimization strategies | `.optimization_strategies.md` |
| `test_perf` | Save patch and run test_command | `patch_N.patch`, `patch_N_test.txt` |

### Best patch selection

After parallel runs finish, GEAK runs a selection agent that reads all test logs, extracts metrics, and writes `best_results.json` + `select_agent.log`.

### Knowledge Base Retrieval

Integrates AMD AI DevTool for hybrid knowledge base retrieval (BGE Embedding + BM25 + Reranking), with built-in AMD GPU and NVIDIA GPU knowledge bases.

**Knowledge Base Structure**
```
knowledge-base/
├── amd-knowledge-base/
│   ├── layer-1-hardware/         # Hardware architecture
│   ├── layer-2-compute-stack/    # HIP, ROCm
│   ├── layer-3-libraries/        # rocBLAS, MIOpen, etc.
│   ├── layer-4-frameworks/       # PyTorch, TensorFlow
│   ├── layer-5-llm/              # LLM related
│   ├── layer-6-extended/         # Optimization guides
│   └── best-practices/
├── nvidia-knowledge-base/
├── comparisons/
└── INDEX.md
```

**Knowledge Retrieval Architecture**

```
Semantic + BM25 → RRF Fusion → BGE Reranker → Top K
```

- **Embedding**: BAAI/bge-large-en-v1.5 (semantic recall)
- **BM25**: Keyword-based recall
- **Fusion**: RRF (Reciprocal Rank Fusion)
- **Reranker**: BAAI/bge-reranker-large

**Usage**

**1. Pre-download ROCm Library Source (Recommended)**

```bash
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
```

**2. Build Semantic Index (Required for First Use)**

```bash
# Build index for all documents under knowledge-base/
python scripts/build_index.py --force
```

Index saved to `~/.cache/amd-ai-devtool/semantic-index/` by default:
- `index.faiss` + `index.pkl` — FAISS semantic search index
- `bm25_index.pkl` — BM25 keyword search index

**Rebuild** when: knowledge base documents are added/modified, or indexing logic changes.

**Adding New Documents**

1. **Location**: Place the file under the appropriate subdirectory (e.g., `layer-6-extended/optimize-guides/*.md`)
2. **Format**: Every `.md` file must include a YAML frontmatter:
   ```yaml
   ---
   tags: ["category1", "category2"]   # Required
   priority: "L1-important"           # Required
   source_url: "https://..."          # Required
   rocm_version: "6.0+"              # Required
   last_updated: 2026-01-14           # Required
   ---
   ```
3. **Filename**: Use English, make it descriptive (e.g., `bf16-vector-load-store.md`)
4. **Quality**: 800–1200 words, with at least 2 syntactically correct code examples
5. **Rebuild index after adding**: `python scripts/build_index.py --force`

**3. Test Retrieval**

```bash
python scripts/test_embedding_search.py      # Test FAISS semantic search
python scripts/test_hybrid_retrieval.py      # Test hybrid retrieval (Embedding + BM25 + Reranker)
python scripts/test_rrf_fusion.py            # Test RRF fusion algorithm
```

**4. Enable Knowledge Retrieval**

```bash
mini --rag        # Enable RAG knowledge retrieval
mini --rag -d     # Enable RAG with debug output
```

Inside the agent, use `@amd:your query` to invoke retrieval.

---

## Summary

GEAK v3 enables reproducible, measurable, and scalable GPU kernel optimization at repository scale. It integrates:

- **Profiling** + **Strategy Management** + **Parallel Exploration** for autonomous optimization
- **Knowledge Retrieval** with AMD/NVIDIA knowledge bases for informed decision-making

Contributions, experiments, and feedback are welcome.
