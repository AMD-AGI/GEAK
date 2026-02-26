# mini-swe-agent Index Building and Test Scripts

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Build Index

```bash
# Build or rebuild index
python build_index.py --force

# Index output location: ~/.cache/amd-ai-devtool/semantic-index/
```

## Test Search

```bash
# Test FAISS semantic search
python test_embedding_search.py

# Test hybrid retrieval (Embedding + BM25 + Reranker)
python test_hybrid_retrieval.py
```

## File Descriptions

| File | Purpose |
|------|---------|
| `build_index.py` | Build FAISS + BM25 index |
| `test_embedding_search.py` | Test semantic search |
| `test_hybrid_retrieval.py` | Test hybrid retrieval |
| `requirements.txt` | Python dependencies |
