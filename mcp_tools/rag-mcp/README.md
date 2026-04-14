# RAG MCP Server

RAG (Retrieval-Augmented Generation) knowledge base server for GEAK. It provides GPU/ROCm/HIP optimization knowledge to the agent via two MCP tools:

- **`query`** — Search the knowledge base by topic (e.g., "shared memory optimization").
- **`optimize`** — Retrieve optimization suggestions for a specific kernel type and GPU model.

The server uses hybrid retrieval: FAISS embedding search + BM25 keyword search + RRF fusion + BGE reranker.

## Setup

All commands below should be run from the **mini-swe-agent root directory**.

### Step 1: Install dependencies

```bash
pip install -e mcp_tools/rag-mcp
```

### Step 2: Build the knowledge base index

```bash
python scripts/build_index.py --force
```

This builds FAISS and BM25 indexes from the documents in `knowledge-base/`. The index is saved to `~/.cache/amd-ai-devtool/semantic-index/`. First run will also download the embedding model (~1.3 GB).

### Step 3: Enable RAG in config

Edit `src/minisweagent/config/geak.yaml`:

```yaml
tools:
  rag: true                      # Enable RAG tools (query, optimize)
  rag_enable_postprocessor: true  # Optional: LLM post-processing of RAG results
```

RAG is **disabled by default**. If you skip this step, the `query` and `optimize` tools will not be available to the agent.

### Step 4 (Optional): Add custom knowledge documents

To add your own optimization documents to the knowledge base, place `.md` files under `knowledge-base/amd-knowledge-base/layer-6-extended/optimize-guides/`, then rebuild the index:

```bash
python scripts/build_index.py --force
```

Use Markdown files with clear heading hierarchy (`#`, `##`, `###`). The indexer splits documents by heading structure, which yields better retrieval results.

### Startup checks

When `rag: true`, GEAK performs two checks before running the pipeline:

1. **Dependency check** — Verifies that `rag-mcp` package is installed. If not, an error is shown with the install command.
2. **Index check** — Verifies that the semantic index exists at `~/.cache/amd-ai-devtool/semantic-index/`. If not, an error is shown with the build command.

## Postprocessor

When `rag_enable_postprocessor: true`, RAG retrieval results are post-processed by an LLM before being returned to the agent. The postprocessor (defined in `src/minisweagent/tools/rag_postprocessor.py`):

- Filters out irrelevant chunks
- Removes duplicates
- Reorganizes content into a structured format

This is optional. Set to `false` to return raw retrieval results directly.

## Directory Structure

```
mcp_tools/rag-mcp/
├── knowledge-base/           # Knowledge base documents (the RAG data source)
│   ├── amd-knowledge-base/   # AMD GPU/ROCm/HIP optimization knowledge
│   ├── nvidia-knowledge-base/
│   └── comparisons/
├── src/rag_mcp/
│   ├── server.py             # MCP server entry point (query + optimize tools)
│   ├── retrieval.py          # HybridRetriever (FAISS + BM25 + RRF + reranker)
│   └── config/
│       └── rag_config.yaml   # Server-level retrieval config (top_k, weights, etc.)
├── pyproject.toml            # Package definition and dependencies
└── tests/
```

## Configuration

The retrieval behavior can be tuned via `src/rag_mcp/config/rag_config.yaml`:

```yaml
retrieval:
  embed_top_k: 25       # Candidates from FAISS embedding search
  bm25_top_k: 25        # Candidates from BM25 keyword search
  mcp_top_k: 3          # Final results returned to agent

reranker:
  enable_reranker: true  # BGE reranker for final ranking

fusion:
  semantic_weight: 0.7   # Weight for embedding results in RRF
  bm25_weight: 0.3       # Weight for BM25 results in RRF
```
