# mini-swe-agent 索引构建和测试脚本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 构建索引

```bash
# 首次构建或重建索引
python build_index.py --force

# 索引输出位置: ~/.cache/amd-ai-devtool/semantic-index/
```

## 测试搜索

```bash
# 测试 FAISS 语义搜索
python test_embedding_search.py

# 测试混合检索（Embedding + BM25 + Reranker）
python test_hybrid_retrieval.py
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `build_index.py` | 构建 FAISS + BM25 索引 |
| `test_embedding_search.py` | 测试纯语义搜索 |
| `test_hybrid_retrieval.py` | 测试混合检索 |
| `requirements.txt` | Python 依赖 |
