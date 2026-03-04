"""
Hybrid retrieval for GPU knowledge base.

Provides semantic search using:
1. Embedding-based retrieval (FAISS + BGE)
2. BM25 keyword retrieval
3. RRF (Reciprocal Rank Fusion) for score merging
4. BGE reranker for final ranking

Standalone module — no dependency on minisweagent.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + word-boundary tokenization for BM25."""
    return re.findall(r'\b\w+\b', text.lower())


def _parse_tags(raw: Any) -> list[str]:
    """Normalize tags from metadata — may be a list, JSON string, or CSV string."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in raw.split(",") if t.strip()]


class HybridRetriever:
    """
    Hybrid retrieval with embedding + BM25 + BGE reranker.

    Pipeline:
    1. Embedding retriever (FAISS) -> top-k candidates
    2. BM25 retriever -> top-k candidates
    3. RRF fusion and deduplication
    4. BGE reranker -> final ranked results
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        embed_top_k: int = 25,
        bm25_top_k: int = 25,
        rrf_k: int = 60,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        enable_reranker: bool = True,
    ):
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.embed_top_k = embed_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.enable_reranker = enable_reranker

        self._embeddings = None
        self._vectorstore = None
        self._bm25 = None
        self._bm25_docs = None
        self._reranker = None

    # ------------------------------------------------------------------
    # Lazy-loaded components
    # ------------------------------------------------------------------

    @property
    def embeddings(self):
        """Lazy load HuggingFace embeddings."""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def vectorstore(self):
        """Lazy load FAISS vectorstore. Supports both LangChain and original formats."""
        if self._vectorstore is None:
            from langchain_community.vectorstores import FAISS
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index not found at {self.index_path}")

            lc_faiss = self.index_path / "index.faiss"
            lc_pkl = self.index_path / "index.pkl"
            orig_faiss = self.index_path / "faiss.index"
            orig_chunks = self.index_path / "chunks.pkl"

            if lc_faiss.exists() and lc_pkl.exists():
                self._vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            elif orig_faiss.exists() and orig_chunks.exists():
                import faiss
                from langchain_community.docstore.in_memory import InMemoryDocstore
                from langchain_core.documents import Document

                index = faiss.read_index(str(orig_faiss))
                with open(orig_chunks, 'rb') as f:
                    chunks = pickle.load(f)

                documents = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        content = chunk.get('content', '')
                        metadata = chunk.get('metadata', {})
                        if 'source_file' in chunk:
                            metadata['source'] = str(chunk['source_file'])
                    else:
                        content = getattr(chunk, 'content', str(chunk))
                        metadata = getattr(chunk, 'metadata', {})
                    documents.append(Document(page_content=content, metadata=metadata))

                docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
                index_to_docstore_id = {i: str(i) for i in range(len(documents))}

                self._vectorstore = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id,
                )
            else:
                raise FileNotFoundError(
                    f"No valid index found at {self.index_path}. "
                    f"Expected either (index.faiss, index.pkl) or (faiss.index, chunks.pkl)"
                )
        return self._vectorstore

    @property
    def bm25(self):
        """Lazy load BM25 index."""
        if self._bm25 is None:
            bm25_path = self.index_path / "bm25_index.pkl"
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    self._bm25 = pickle.load(f)
        return self._bm25

    @property
    def bm25_docs(self):
        """Lazy load BM25 documents."""
        if self._bm25_docs is None:
            docs_path = self.index_path / "bm25_documents.pkl"
            if docs_path.exists():
                with open(docs_path, 'rb') as f:
                    self._bm25_docs = pickle.load(f)
        return self._bm25_docs

    @property
    def reranker(self):
        """Lazy load BGE reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    # ------------------------------------------------------------------
    # Search stages
    # ------------------------------------------------------------------

    def _search_embedding(self, query: str, k: int) -> list[tuple[Any, float, str]]:
        if k <= 0:
            return []
        search_k = min(k, self.vectorstore.index.ntotal)
        results = self.vectorstore.similarity_search_with_score(query, k=search_k)
        return [(doc, score, "embedding") for doc, score in results]

    def _search_bm25(self, query: str, k: int) -> list[tuple[Any, float, str]]:
        if k <= 0 or self.bm25 is None or self.bm25_docs is None:
            return []
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.bm25_docs[idx], float(scores[idx]), "bm25")
                for idx in top_indices if scores[idx] > 0]

    def _rrf_merge(
        self,
        embed_results: list[tuple[Any, float, str]],
        bm25_results: list[tuple[Any, float, str]],
    ) -> list[tuple[Any, str, float, float]]:
        """Reciprocal Rank Fusion merge.

        Returns list of (document, source, original_score, rrf_score).
        """
        rrf_scores: dict[int, tuple[Any, str, float, float]] = {}

        for rank, (doc, orig_score, source) in enumerate(embed_results):
            doc_hash = hash(doc.page_content[:500])
            rrf_score = self.semantic_weight / (self.rrf_k + rank + 1)
            rrf_scores[doc_hash] = (doc, source, orig_score, rrf_score)

        for rank, (doc, orig_score, source) in enumerate(bm25_results):
            doc_hash = hash(doc.page_content[:500])
            bm25_score = self.bm25_weight / (self.rrf_k + rank + 1)
            if doc_hash in rrf_scores:
                existing_doc, existing_source, existing_orig, existing_rrf = rrf_scores[doc_hash]
                rrf_scores[doc_hash] = (existing_doc, existing_source + "+bm25", existing_orig, existing_rrf + bm25_score)
            else:
                rrf_scores[doc_hash] = (doc, source, orig_score, bm25_score)

        return sorted(rrf_scores.values(), key=lambda x: x[3], reverse=True)

    def _rerank(self, query: str, docs_with_source: list[tuple[Any, str, float]], k: int) -> list[tuple[Any, float, str, float]]:
        if not docs_with_source:
            return []
        pairs = [(query, doc.page_content) for doc, _, _ in docs_with_source]
        scores = self.reranker.predict(pairs)
        doc_scores = sorted(
            [(doc, float(score), source, orig_score)
             for (doc, source, orig_score), score in zip(docs_with_source, scores)],
            key=lambda x: x[1],
            reverse=True,
        )
        return doc_scores[:k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        min_content_length: int = 200,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float, str, float]]:
        """
        Hybrid search with reranking.

        Returns list of (document, score, source, original_score).
        """
        embed_results = self._search_embedding(query, self.embed_top_k)
        bm25_results = self._search_bm25(query, self.bm25_top_k)

        merged_with_rrf = self._rrf_merge(embed_results, bm25_results)

        if merged_with_rrf:
            rrf_scores = [s for _, _, _, s in merged_with_rrf]
            top3 = rrf_scores[:3] if len(rrf_scores) >= 3 else rrf_scores
            logger.info("[RRF] Merged: %d unique candidates", len(merged_with_rrf))
            logger.info("[RRF] Top-3 scores: %s", [f"{s:.4f}" for s in top3])
            logger.info("[RRF] Score range: %.4f - %.4f", min(rrf_scores), max(rrf_scores))
            excellent = sum(1 for s in rrf_scores if s >= 0.014)
            good = sum(1 for s in rrf_scores if 0.010 <= s < 0.014)
            fair = sum(1 for s in rrf_scores if 0.005 <= s < 0.010)
            weak = sum(1 for s in rrf_scores if s < 0.005)
            logger.info("[RRF] Quality: Excellent=%d, Good=%d, Fair=%d, Weak=%d", excellent, good, fair, weak)

        if not merged_with_rrf:
            return []

        if self.enable_reranker:
            merged = [(doc, source, orig_score) for doc, source, orig_score, _ in merged_with_rrf]
            reranked = self._rerank(query, merged, len(merged))
            logger.info("[Rerank] Enabled, reranked %d candidates", len(reranked))
        else:
            reranked = [(doc, rrf_score, source, orig_score)
                        for doc, source, orig_score, rrf_score in merged_with_rrf]
            logger.info("[Rerank] DISABLED, using RRF scores for %d candidates", len(reranked))

        filtered = []
        for doc, score, source, orig_score in reranked:
            if len(doc.page_content) < min_content_length:
                continue
            if filters:
                if 'layers' in filters and doc.metadata.get('layer') not in filters['layers']:
                    continue
                if 'category' in filters and doc.metadata.get('category') != filters['category']:
                    continue
            filtered.append((doc, score, source, orig_score))
            if len(filtered) >= k:
                break

        return filtered

    def query(
        self,
        topic: str,
        layer: str | None = None,
        vendor: str = "all",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the knowledge base. Returns list of result dicts."""
        filters = {}
        if layer:
            filters['layers'] = [layer]

        results = self.search(topic, k=top_k, filters=filters if filters else None)

        return [
            {
                "id": f"chunk_{i}",
                "title": doc.metadata.get("section", doc.metadata.get("source", "Unknown")[:50]),
                "content": doc.page_content,
                "content_length": len(doc.page_content),
                "score": round(score, 3),
                "original_score": round(orig_score, 4),
                "layer": doc.metadata.get("layer", "unknown"),
                "category": doc.metadata.get("category", "unknown"),
                "tags": _parse_tags(doc.metadata.get("tags", [])),
                "retrieval_method": source,
            }
            for i, (doc, score, source, orig_score) in enumerate(results)
        ]
