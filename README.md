# GEAK-v3

English | [中文](README_zh.md)

GEAK is an AI-powered framework for automated GPU kernel optimization, built on top of mini-SWE-agent.

It enables systematic, profiling-driven, and scalable optimization of GPU kernels — evolving from single-kernel tuning (v1/v2) to full repository-level autonomous optimization (v3).

**v3 also integrates AMD AI DevTool (MCP) for hybrid knowledge base retrieval**, bringing built-in AMD/NVIDIA GPU knowledge directly into the agent's context.

## Table of Contents

- [Evolution: From Kernel-Level to Repo-Level Automation](#evolution-from-kernel-level-to-repo-level-automation)
- [Core Architecture](#core-architecture)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Output & Artifacts](#output--artifacts)
- [MCP Integration (AMD AI DevTool)](#mcp-integration-amd-ai-devtool)
- [Features](#features)
  - [Unit test discovery](#unit-test-discovery)
  - [System Tools (built-in)](#system-tools-built-in)
  - [Best patch selection](#best-patch-selection)
- [Knowledge Base](#knowledge-base)
- [Project Structure](#project-structure)
- [Summary](#summary)

---

## Evolution: From Kernel-Level to Repo-Level Automation

### GEAK v1 / v2 — Single Kernel Optimization

Earlier versions focused on optimizing individual GPU kernels through iterative patch generation and performance validation.

They demonstrated that LLM-based agents can:

- Analyze kernel structure
- Propose optimization strategies
- Generate performance-improving patches

### GEAK v3 — Autonomous Repo-Level Optimization

GEAK v3 upgrades the system into a full lifecycle GPU optimization framework.

It operates at repository scale and automates:

- 🔍 Test discovery and generation
- 📊 Baseline performance measurement
- 🧠 Profiling-guided bottleneck diagnosis
- 🎯 Strategy planning and execution
- ✅ Patch validation and regression testing
- 🔁 Multi-round iterative improvement

The system forms a closed-loop optimization engine capable of continuous performance evolution with minimal human intervention.

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

- **MCP RAG Retrieval** for on-demand AMD/NVIDIA GPU knowledge retrieval during optimization

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
git switch -c dev origin/dev
pip install -e .

# To use the MCP RAG feature, also install the langchain dependencies
pip install -e '.[langchain]'

# Set LLM API key
export AMD_LLM_API_KEY="YOUR_KEY"
```

### Usage

#### Interactive REPL (mini-swe-agent mode)

```bash
# Interactive REPL
mini

# Run with a specific task
mini -t "fix the bug in main.py"

# Auto-execute mode (no confirmation needed)
mini --yolo

# Enable MCP knowledge retrieval
mini --mcp
```

#### Basic (single-agent) GPU kernel optimization

Add `--yolo` to run end-to-end without interactive confirmation.

```bash
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

`mini` loads configs in layers:

1. base config: `mini.yaml`
2. template: `mini_kernel_strategy_list.yaml` (default)
3. user override: `--config geak.yaml` (**final override**)

This means you can configure tools and parallel defaults directly in `geak.yaml`.

All config files are located in `src/minisweagent/config/`. Use `mini -c <config_name>` to select one.

| File | Purpose | Mode |
|------|---------|------|
| `mini.yaml` | Default config for `mini` | yolo |
| `default.yaml` | DefaultAgent base config | confirm |
| `github_issue.yaml` | Auto-solve GitHub Issues | — |
| `rag_config.yaml` | RAG retrieval pipeline config | — |

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

## MCP Integration (AMD AI DevTool)

Integrates AMD AI DevTool for hybrid knowledge base retrieval (BGE Embedding + BM25 + Reranking), with built-in AMD GPU and NVIDIA GPU knowledge bases.

### 1. Pre-download ROCm Library Source (Recommended)

```bash
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
```

### 2. Build Semantic Index (Required for First Use)

```bash
# Build index for all documents under knowledge-base/
python scripts/build_index.py --force
```

Index saved to `~/.cache/amd-ai-devtool/semantic-index/` by default:
- `index.faiss` + `index.pkl` — FAISS semantic search index
- `bm25_index.pkl` — BM25 keyword search index

Rebuild when: knowledge base documents are added/modified, or indexing logic changes.

### 3. Test Retrieval

```bash
python scripts/test_embedding_search.py      # Test FAISS semantic search
python scripts/test_hybrid_retrieval.py      # Test hybrid retrieval (Embedding + BM25 + Reranker)
python scripts/test_rrf_fusion.py            # Test RRF fusion algorithm
```

### 4. Enable MCP

```bash
mini --mcp        # Enable MCP
mini --mcp -d     # Enable MCP with debug output
```

Inside the agent, use `@amd:your query` to invoke retrieval.

### 5. RAG Retrieval Architecture

```
Semantic + BM25 → RRF Fusion → BGE Reranker → Top K
```

- **Embedding**: BAAI/bge-large-en-v1.5 (semantic recall)
- **BM25**: Keyword-based recall
- **Fusion**: RRF (Reciprocal Rank Fusion)
- **Reranker**: BAAI/bge-reranker-large

Config: `src/minisweagent/config/rag_config.yaml`

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

---

## Knowledge Base

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

To add new documents, place `.md` files with required YAML frontmatter (`tags`, `priority`, `source_url`, `rocm_version`, `last_updated`) and rebuild the index.

---

## Project Structure

```
src/minisweagent/
├── agents/                    # Agent implementations
│   ├── default.py             #   Core agent
│   ├── interactive.py         #   Human-in-the-loop agent
│   ├── parallel_agent.py      #   Parallel multi-agent
│   ├── strategy_interactive.py#   Strategy-guided agent
│   └── unit_test_agent.py     #   Unit test discovery agent
├── models/                    # LLM model interfaces
│   ├── amd_llm.py             #   AMD LLM Gateway (router)
│   ├── amd_base.py            #   AMD base model
│   ├── amd_claude.py          #   Claude via AMD gateway
│   └── litellm_model.py       #   LiteLLM (multi-provider)
├── mcp_integration/           # MCP (AMD AI DevTool) integration
│   ├── mcp_environment.py     #   MCP environment wrapper
│   ├── langchain_retrieval.py #   Hybrid retrieval
│   └── prompts.py             #   MCP-specific prompts
├── tools/                     # Tool implementations
│   ├── tools.json             #   Tool schema definitions
│   ├── tools_runtime.py       #   Tool runtime
│   ├── editor_tool.py         #   File editor
│   ├── profiling_tools.py     #   GPU profiling
│   └── strategy_manager.py    #   Strategy tracker
├── config/                    # YAML config files
│   ├── mini.yaml
│   ├── default.yaml
│   ├── github_issue.yaml
│   └── rag_config.yaml
└── run/                       # Entry points
    ├── mini.py                #   Main CLI (`mini` command)
    └── utils/
```

Other top-level directories:
- `scripts/` — Index building and retrieval test scripts
- `knowledge-base/` — RAG knowledge base (AMD / NVIDIA)
- `examples/` — HIP kernel examples and subagent examples

---

## Summary

GEAK v3 enables reproducible, measurable, and scalable GPU kernel optimization at repository scale. It integrates:

- **Profiling** + **Strategy Management** + **Parallel Exploration** for autonomous optimization
- **MCP RAG Retrieval** with AMD/NVIDIA knowledge bases for informed decision-making

Contributions, experiments, and feedback are welcome.
