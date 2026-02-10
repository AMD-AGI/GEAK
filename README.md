English | [中文](README_zh.md)

# Mini SWE Agent

A minimal AI coding agent powered by LLM-generated Bash commands. The core agent is ~100 lines of code.

## Installation

```bash
pip install -e .

# To use the MCP RAG feature, also install the langchain dependencies
pip install -e '.[langchain]'
```

## Usage

```bash
# Interactive REPL
mini

# Run with a specific task
mini -t "fix the bug in main.py"

# Auto-execute mode (no confirmation needed)
mini --yolo

# Enable MCP
mini --mcp
```

## MCP Integration

Integrates AMD AI DevTool for hybrid knowledge base retrieval (BGE Embedding + BM25 + Reranking), with built-in AMD GPU and NVIDIA GPU knowledge bases.

### 1. Pre-download ROCm Library Source (Recommended)

The agent may need to reference ROCm library source code at runtime. Pre-cloning is recommended to avoid timeouts when downloading a large repo on the fly:

```bash
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
```

### 2. Build Semantic Index (Required for First Use)

The knowledge base index must be built before MCP retrieval can work:

```bash
# Build index for all documents under knowledge-base/
# --force overwrites any existing index
python scripts/build_index.py --force
```

The index is saved to `~/.cache/amd-ai-devtool/semantic-index/` by default. Output files:

- `index.faiss` + `index.pkl` — FAISS semantic search index
- `bm25_index.pkl` — BM25 keyword search index

You need to rebuild the index when:

1. Knowledge base documents are added or modified
2. Chunking or indexing logic is changed
3. Metadata parsing bugs are fixed

### 3. Test Retrieval

After building the index, verify it with the test scripts:

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
- **Fusion**: RRF (Reciprocal Rank Fusion) for deduplication and merging
- **Reranker**: BAAI/bge-reranker-large (re-ranking)

Config file: `src/minisweagent/config/rag_config.yaml` — tune retrieval parameters, toggle BM25 dual-path recall, reranking, LLM summarization, etc.

## Project Structure

```
src/minisweagent/
├── __init__.py                # Version, protocols, global config
├── agents/                    # Agent implementations
│   ├── default.py             #   Core agent (~100 lines)
│   ├── interactive.py         #   Human-in-the-loop agent
│   └── interactive_textual.py #   Textual TUI agent
├── models/                    # LLM model interfaces
│   ├── litellm_model.py       #   LiteLLM (supports most providers)
│   ├── anthropic_model.py     #   Anthropic
│   ├── amd_llm.py             #   AMD LLM Gateway
│   ├── openrouter_model.py    #   OpenRouter
│   └── portkey_model.py       #   Portkey
├── environments/              # Execution environments
│   ├── local.py               #   Local subprocess
│   ├── docker.py              #   Docker/Podman
│   └── singularity.py         #   Singularity/Apptainer
├── config/                    # YAML config files (see "Configuration" below)
│   ├── mini.yaml              #   Default config for `mini` command
│   ├── default.yaml           #   DefaultAgent base config
│   ├── github_issue.yaml      #   GitHub issue solving config
│   └── rag_config.yaml        #   RAG retrieval config
├── run/                       # Entry points
│   ├── mini.py                #   Main CLI (`mini` command)
│   ├── hello_world.py         #   Simple example
│   ├── github_issue.py        #   GitHub issue solver
│   └── inspector.py           #   Trajectory browser
├── mcp_integration/           # MCP (AMD AI DevTool) integration
│   ├── mcp_environment.py     #   MCP environment wrapper
│   ├── langchain_retrieval.py #   Hybrid retrieval (Embedding + BM25)
│   └── prompts.py             #   MCP-specific prompts
└── utils/                     # Utilities
    ├── log.py                 #   Logging
    └── subagent.py            #   Sub-agent utilities
```

Other top-level directories:

- `scripts/` — Utility scripts
- `knowledge-base/` — RAG knowledge base (AMD / NVIDIA)

## Configuration

All config files are located in `src/minisweagent/config/`. Use `mini -c <config_name>` to select one.

### Agent Configs

| File | Purpose | Model | Mode | Notes |
|------|---------|-------|------|-------|
| `mini.yaml` | Default config for `mini` | AMD LLM Gateway claude-opus-4.5 | yolo | Primary config for daily use. temperature=0.0, output truncation at 20000 chars, timeout 3600s |
| `default.yaml` | DefaultAgent base config | Not bound to a specific model | confirm | Generic base config. temperature=0.0, output truncation at 10000 chars (5000 head + 5000 tail) |
| `mini_no_temp.yaml` | No-temperature variant | Not bound to a specific model | confirm | Nearly identical to default.yaml but without temperature setting. cost_limit=3 |
| `mini_reverse_kl.yaml` | GPU kernel optimization analysis | AMD LLM Gateway claude-opus-4.5 | confirm | Analyzes kernel optimization history in a repo and generates reports. Long prompt |
| `github_issue.yaml` | Auto-solve GitHub Issues | Not bound to a specific model | — | Runs inside Docker (python:3.11, working dir /testbed) |

### RAG Config

File: `rag_config.yaml` — controls the RAG retrieval pipeline:

| Parameter | Description |
|-----------|-------------|
| `retrieval.embed_top_k` / `bm25_top_k` | Number of candidates from Embedding / BM25 retrieval |
| `retrieval.enable_bm25` | Whether to enable BM25 dual-path recall |
| `retrieval.mcp_top_k` | Number of final results returned |
| `reranker.enable_reranker` | Whether to enable re-ranking |
| `fusion.semantic_weight` / `bm25_weight` | Fusion weights for Embedding and BM25 |
| `summary.enable_rag_subagent` | Whether to enable LLM summarization |
| `debug.verbose` | Whether to print verbose MCP tool logs |

## Knowledge Base

### Directory Structure

```
knowledge-base/
├── amd-knowledge-base/
│   ├── layer-1-hardware/         # Hardware architecture
│   ├── layer-2-compute-stack/    # Compute stack (HIP, ROCm)
│   ├── layer-3-libraries/        # Libraries (rocBLAS, MIOpen, etc.)
│   ├── layer-4-frameworks/       # Frameworks (PyTorch, TensorFlow)
│   ├── layer-5-llm/              # LLM related
│   ├── layer-6-extended/         # Extended knowledge
│   └── best-practices/           # Best practices
├── nvidia-knowledge-base/        # Same layer structure
├── comparisons/                  # Cross-platform comparison docs
└── INDEX.md
```

### Adding New Documents

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
