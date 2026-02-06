"""
Hybrid LangChain-based retrieval for GPU knowledge base.

This module provides semantic search using:
1. Embedding-based retrieval (FAISS + BGE)
2. BM25 keyword retrieval
3. BGE reranker for final ranking

No dependency on amd_ai_devtool.
"""

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


# Default paths
DEFAULT_INDEX_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25."""
    return re.findall(r'\b\w+\b', text.lower())


class HybridRetriever:
    """
    Hybrid retrieval with embedding + BM25 + BGE reranker.
    
    Pipeline:
    1. Embedding retriever (FAISS) -> top-k candidates
    2. BM25 retriever -> top-k candidates  
    3. Merge and deduplicate
    4. BGE reranker -> final ranked results
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        embed_top_k: int = 25,
        bm25_top_k: int = 25,
        # RRF fusion parameters
        rrf_k: int = 60,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        # Reranker switch
        enable_reranker: bool = True,
    ):
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.embed_top_k = embed_top_k
        self.bm25_top_k = bm25_top_k
        # RRF configuration
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        # Reranker switch
        self.enable_reranker = enable_reranker
        
        # Lazy-loaded components
        self._embeddings = None
        self._vectorstore = None
        self._bm25 = None
        self._bm25_docs = None
        self._reranker = None

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
            
            # Check for LangChain format (index.faiss, index.pkl)
            lc_faiss = self.index_path / "index.faiss"
            lc_pkl = self.index_path / "index.pkl"
            
            # Check for original format (faiss.index, chunks.pkl)
            orig_faiss = self.index_path / "faiss.index"
            orig_chunks = self.index_path / "chunks.pkl"
            
            if lc_faiss.exists() and lc_pkl.exists():
                # LangChain format - use standard loader
                self._vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            elif orig_faiss.exists() and orig_chunks.exists():
                # Original format - load manually and create FAISS vectorstore
                import faiss
                from langchain_core.documents import Document
                
                # Load FAISS index
                index = faiss.read_index(str(orig_faiss))
                
                # Load chunks
                with open(orig_chunks, 'rb') as f:
                    chunks = pickle.load(f)
                
                # Convert chunks to LangChain Documents
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
                
                # Create docstore and index_to_docstore_id mapping
                from langchain_community.docstore.in_memory import InMemoryDocstore
                docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
                index_to_docstore_id = {i: str(i) for i in range(len(documents))}
                
                # Create FAISS vectorstore
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

    def _search_embedding(self, query: str, k: int) -> list[tuple[Any, float, str]]:
        """Search using embedding similarity.
        
        Returns:
            List of (document, score, source) tuples where source='embedding'
        """
        if k <= 0:
            return []
        search_k = min(k, self.vectorstore.index.ntotal)
        results = self.vectorstore.similarity_search_with_score(query, k=search_k)
        return [(doc, score, "embedding") for doc, score in results]

    def _search_bm25(self, query: str, k: int) -> list[tuple[Any, float, str]]:
        """Search using BM25.
        
        Returns:
            List of (document, score, source) tuples where source='bm25'
        """
        if k <= 0 or self.bm25 is None or self.bm25_docs is None:
            return []
        
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [(self.bm25_docs[idx], float(scores[idx]), "bm25") 
                for idx in top_indices if scores[idx] > 0]

    def _merge_results(
        self, 
        embed_results: list[tuple[Any, float, str]], 
        bm25_results: list[tuple[Any, float, str]]
    ) -> list[tuple[Any, str, float]]:
        """Merge and deduplicate results, tracking source.
        
        Returns:
            List of (document, source, original_score) tuples
        """
        seen = set()
        merged = []
        
        for doc, score, source in embed_results:
            content_hash = hash(doc.page_content[:500])
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append((doc, source, score))
        
        for doc, score, source in bm25_results:
            content_hash = hash(doc.page_content[:500])
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append((doc, source, score))
        
        return merged

    def _rrf_merge(
        self,
        embed_results: list[tuple[Any, float, str]],
        bm25_results: list[tuple[Any, float, str]],
    ) -> list[tuple[Any, str, float, float]]:
        """
        Merge and deduplicate results using RRF (Reciprocal Rank Fusion).
        
        RRF formula: score = semantic_weight / (k + rank + 1) + bm25_weight / (k + rank + 1)
        
        Args:
            embed_results: Embedding search results (doc, score, source)
            bm25_results: BM25 search results (doc, score, source)
        
        Returns:
            List of (document, source, original_score, rrf_score) tuples
        """
        rrf_scores = {}  # doc_hash -> (doc, source, orig_score, rrf_score)
        
        # Process embedding results
        for rank, (doc, orig_score, source) in enumerate(embed_results):
            doc_hash = hash(doc.page_content[:500])
            rrf_score = self.semantic_weight / (self.rrf_k + rank + 1)
            rrf_scores[doc_hash] = (doc, source, orig_score, rrf_score)
        
        # Process BM25 results
        for rank, (doc, orig_score, source) in enumerate(bm25_results):
            doc_hash = hash(doc.page_content[:500])
            bm25_score = self.bm25_weight / (self.rrf_k + rank + 1)
            
            if doc_hash in rrf_scores:
                # Document appeared in both retrievers - accumulate RRF score
                existing_doc, existing_source, existing_orig, existing_rrf = rrf_scores[doc_hash]
                # Keep first source, use first original score, accumulate RRF score
                rrf_scores[doc_hash] = (existing_doc, existing_source + "+bm25", existing_orig, existing_rrf + bm25_score)
            else:
                # New document from BM25 only
                rrf_scores[doc_hash] = (doc, source, orig_score, bm25_score)
        
        # Sort by RRF score (descending)
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x[3],  # Sort by rrf_score
            reverse=True
        )
        
        return sorted_results

    def _rerank(self, query: str, docs_with_source: list[tuple[Any, str, float]], k: int) -> list[tuple[Any, float, str, float]]:
        """Rerank documents using BGE reranker.
        
        Returns:
            List of (document, rerank_score, source, original_score) tuples
        """
        if not docs_with_source:
            return []
        
        pairs = [(query, doc.page_content) for doc, _, _ in docs_with_source]
        scores = self.reranker.predict(pairs)
        
        doc_scores = sorted(
            [(doc, float(score), source, orig_score) 
             for (doc, source, orig_score), score in zip(docs_with_source, scores)],
            key=lambda x: x[1], 
            reverse=True
        )
        return doc_scores[:k]

    def search(
        self,
        query: str,
        k: int = 10,
        min_content_length: int = 200,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float, str, float]]:
        """
        Hybrid search with reranking.

        Args:
            query: Search query string
            k: Number of final results to return
            min_content_length: Minimum content length filter
            filters: Optional metadata filters

        Returns:
            List of (document, rerank_score, source, original_score) tuples
            where source is 'embedding' or 'bm25'
        """
        # Stage 1: Get candidates from each retriever
        embed_results = self._search_embedding(query, self.embed_top_k)
        bm25_results = self._search_bm25(query, self.bm25_top_k)
        
        # Stage 2: RRF fusion and deduplication
        merged_with_rrf = self._rrf_merge(embed_results, bm25_results)
        
        # Log RRF score statistics
        if merged_with_rrf:
            rrf_scores = [rrf_score for _, _, _, rrf_score in merged_with_rrf]
            import sys
            top3_scores = rrf_scores[:3] if len(rrf_scores) >= 3 else rrf_scores
            print(f"  [RRF] Merged: {len(merged_with_rrf)} unique candidates", file=sys.stderr)
            print(f"  [RRF] Top-3 scores: {[f'{s:.4f}' for s in top3_scores]}", file=sys.stderr)
            print(f"  [RRF] Score range: {min(rrf_scores):.4f} - {max(rrf_scores):.4f}", file=sys.stderr)
            
            # Quality distribution
            excellent = sum(1 for s in rrf_scores if s >= 0.014)
            good = sum(1 for s in rrf_scores if 0.010 <= s < 0.014)
            fair = sum(1 for s in rrf_scores if 0.005 <= s < 0.010)
            weak = sum(1 for s in rrf_scores if s < 0.005)
            print(f"  [RRF] Quality: Excellent: {excellent}, Good: {good}, Fair: {fair}, Weak: {weak}", file=sys.stderr)
        
        if not merged_with_rrf:
            return []
        
        # Stage 3: Rerank (可选)
        if self.enable_reranker:
            # Convert to format expected by _rerank: (doc, source, orig_score)
            merged = [(doc, source, orig_score) for doc, source, orig_score, _ in merged_with_rrf]
            reranked = self._rerank(query, merged, len(merged))
            print(f"  [Rerank] Enabled, reranked {len(reranked)} candidates", file=sys.stderr)
        else:
            # 跳过精排，直接使用 RRF 分数排序
            reranked = [(doc, rrf_score, source, orig_score) 
                        for doc, source, orig_score, rrf_score in merged_with_rrf]
            print(f"  [Rerank] DISABLED, using RRF scores for {len(reranked)} candidates", file=sys.stderr)
        
        # Stage 4: Apply filters
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
        """
        Query the knowledge base (main interface).

        Args:
            topic: Search topic/query
            layer: Optional layer filter
            vendor: Optional vendor filter
            top_k: Number of results

        Returns:
            List of result dictionaries with source indicator
        """
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
                "tags": doc.metadata.get("tags", []),
                "retrieval_method": source,  # "embedding" or "bm25"
            }
            for i, (doc, score, source, orig_score) in enumerate(results)
        ]


# Backward compatibility alias
LangChainRetriever = HybridRetriever


class KnowledgeTools:
    """Tool implementations for MCP-style queries using HybridRetriever."""

    def __init__(self, retriever: HybridRetriever | None = None):
        self._retriever = retriever

    @property
    def retriever(self) -> HybridRetriever:
        """Lazy load retriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    async def query(self, args: dict[str, Any]) -> str:
        """Execute query tool."""
        topic = args.get("topic", "")
        layer = args.get("layer")
        vendor = args.get("vendor", "all")
        top_k = args.get("top_k", 10)

        results = self.retriever.query(topic, layer=layer, vendor=vendor, top_k=top_k)

        if not results:
            return f"No results found for topic: {topic}"

        output = f"Found {len(results)} results for '{topic}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"## Result {i}: {r['title']}\n\n"
            output += f"**Layer**: {r['layer']} | **Category**: {r['category']}\n"
            if r.get("tags"):
                output += f"**Tags**: {', '.join(r['tags'])}\n"
            output += f"\n{r['content']}\n\n"
            if i < len(results):
                output += "---\n\n"

        return output

    async def optimize(self, args: dict[str, Any]) -> str:
        """Execute optimization suggestion tool."""
        code_type = args.get("code_type", "")
        context = args.get("context", "")
        gpu_model = args.get("gpu_model", "")
        top_k = args.get("top_k", 10)

        query = f"{gpu_model} {code_type} optimization {context}".strip()
        if not query or query == "optimization":
            query = "GPU kernel optimization best practices"

        results = self.retriever.query(query, top_k=top_k)

        if not results:
            return f"No optimization suggestions found for: {code_type}"

        output = f"Optimization suggestions for {code_type}:\n\n"
        for i, r in enumerate(results, 1):
            output += f"## Suggestion {i}: {r['title']}\n\n"
            output += f"{r['content']}\n\n"
            if i < len(results):
                output += "---\n\n"

        return output

    async def example(self, args: dict[str, Any]) -> str:
        """Execute code example tool."""
        category = args.get("category", "")
        use_case = args.get("use_case", "")
        top_k = args.get("top_k", 10)

        query = f"{category} {use_case} code example".strip()
        results = self.retriever.query(query, top_k=top_k)

        if not results:
            return f"No code examples found for: {category} {use_case}"

        output = f"Code examples for {category} - {use_case}:\n\n"
        for i, r in enumerate(results, 1):
            output += f"## Example {i}: {r['title']}\n\n"
            output += f"{r['content']}\n\n"
            if i < len(results):
                output += "---\n\n"

        return output

    async def troubleshoot(self, args: dict[str, Any]) -> str:
        """Execute troubleshooting tool."""
        error_message = args.get("error_message", "")
        context = args.get("context", "")
        top_k = args.get("top_k", 10)

        query = f"troubleshoot {error_message} {context}".strip()
        results = self.retriever.query(query, top_k=top_k)

        if not results:
            return f"No troubleshooting information found for: {error_message}"

        output = f"Troubleshooting for '{error_message}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"## Solution {i}: {r['title']}\n\n"
            output += f"{r['content']}\n\n"
            if i < len(results):
                output += "---\n\n"

        return output

    async def compat(self, args: dict[str, Any]) -> str:
        """Execute compatibility check tool."""
        rocm_version = args.get("rocm_version", "")
        components = args.get("components", [])

        query = f"ROCm {rocm_version} compatibility"
        if components:
            comp_names = [c.get("name", "") for c in components]
            query += " " + " ".join(comp_names)

        results = self.retriever.query(query, top_k=5)

        if not results:
            return f"No compatibility information found for ROCm {rocm_version}"

        output = f"Compatibility information for ROCm {rocm_version}:\n\n"
        for i, r in enumerate(results, 1):
            output += f"## {r['title']}\n\n"
            output += f"{r['content']}\n\n"

        return output

    async def docs(self, args: dict[str, Any]) -> str:
        """Return documentation URLs."""
        category = args.get("category", "all")

        docs_urls = {
            "all": [
                "https://rocm.docs.amd.com/",
                "https://rocm.docs.amd.com/projects/HIP/",
                "https://rocm.docs.amd.com/projects/rocBLAS/",
            ],
            "hip": [
                "https://rocm.docs.amd.com/projects/HIP/en/latest/",
                "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html",
            ],
            "libraries": [
                "https://rocm.docs.amd.com/projects/rocBLAS/en/latest/",
                "https://rocm.docs.amd.com/projects/rocFFT/en/latest/",
                "https://rocm.docs.amd.com/projects/MIOpen/en/latest/",
            ],
            "ai_frameworks": [
                "https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html",
                "https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/",
            ],
            "performance": [
                "https://rocm.docs.amd.com/projects/rocprofiler/en/latest/",
                "https://rocm.docs.amd.com/projects/omniperf/en/latest/",
            ],
            "installation": [
                "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/",
            ],
        }

        urls = docs_urls.get(category, docs_urls["all"])
        return f"Documentation URLs for '{category}':\n\n" + "\n".join(f"- {url}" for url in urls)

