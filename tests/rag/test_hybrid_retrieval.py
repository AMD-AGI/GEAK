# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""
Hybrid Retrieval Pipeline with Embedding + BM25 + BGE Reranker

This module provides a hybrid retrieval system that:
1. Retrieves candidates using embedding-based semantic search (FAISS)
2. Retrieves candidates using BM25 keyword search
3. Merges and deduplicates all candidates
4. Reranks using BGE-reranker-large for final ranking
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Default models
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"

# Default index path
DEFAULT_INDEX_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Get LangChain HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


class EmbeddingRetriever:
    """Embedding-based retriever using FAISS."""
    
    def __init__(self, index_path: Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.index_path = index_path
        self.embeddings = get_embeddings(embedding_model)
        self.vectorstore: Optional[FAISS] = None
        self._load_index()
    
    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        faiss_index = self.index_path / "index.faiss"
        pkl_file = self.index_path / "index.pkl"
        
        if faiss_index.exists() and pkl_file.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✓ Loaded FAISS index ({self.vectorstore.index.ntotal} vectors)")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vectorstore = None
        else:
            logger.warning(f"FAISS index not found at {self.index_path}")
    
    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Search using embedding similarity."""
        if self.vectorstore is None:
            return []
        
        search_k = min(k, self.vectorstore.index.ntotal)
        results = self.vectorstore.similarity_search_with_score(query, k=search_k)
        
        # Return (document, score) tuples
        return [(doc, float(score)) for doc, score in results]


class BM25Retriever:
    """BM25 keyword-based retriever."""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self._load_index()
    
    def _load_index(self) -> None:
        """Load BM25 index from disk."""
        bm25_file = self.index_path / "bm25_index.pkl"
        docs_file = self.index_path / "bm25_documents.pkl"
        
        if bm25_file.exists() and docs_file.exists():
            try:
                with open(bm25_file, 'rb') as f:
                    self.bm25 = pickle.load(f)
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"✓ Loaded BM25 index ({len(self.documents)} documents)")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.bm25 = None
                self.documents = []
        else:
            logger.warning(f"BM25 index not found at {self.index_path}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Search using BM25 scoring."""
        if self.bm25 is None or not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append((self.documents[idx], float(scores[idx])))
        
        return results


class BGEReranker:
    """BGE Cross-Encoder reranker."""
    
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load reranker model."""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"✓ Loaded BGE reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder."""
        if self.model is None or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get reranker scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with scores
        return [(doc, float(score)) for doc, score in doc_scores[:top_k]]


class HybridRetriever:
    """
    Hybrid retrieval pipeline combining:
    1. Embedding-based retrieval (FAISS + BGE)
    2. BM25 keyword retrieval
    3. BGE reranker for final ranking
    """
    
    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        embed_top_k: int = 10,
        bm25_top_k: int = 0,  # Disabled by default
    ):
        self.index_path = Path(index_path)
        self.embed_top_k = embed_top_k
        self.bm25_top_k = bm25_top_k
        
        # Initialize retrievers
        logger.info("Initializing hybrid retriever...")
        self.embedding_retriever = EmbeddingRetriever(self.index_path, embedding_model)
        self.bm25_retriever = BM25Retriever(self.index_path)
        self.reranker = BGEReranker(reranker_model)
        
        logger.info("✓ Hybrid retriever initialized")
    
    def _merge_results(
        self,
        embed_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Document]:
        """Merge and deduplicate results from multiple retrievers."""
        seen_contents = set()
        merged = []
        
        # Add embedding results first (usually more relevant)
        for doc, score in embed_results:
            content_hash = hash(doc.page_content[:500])  # Hash first 500 chars
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(doc)
        
        # Add BM25 results
        for doc, score in bm25_results:
            content_hash = hash(doc.page_content[:500])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(doc)
        
        return merged
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_content_length: int = 200,
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid retrieval with reranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            filters: Optional metadata filters (applied post-reranking)
            min_content_length: Minimum content length filter
        
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        logger.info(f"Hybrid search: '{query[:50]}...' (top_k={top_k})")
        
        # Stage 1: Get candidates from each retriever
        embed_results = self.embedding_retriever.search(query, k=self.embed_top_k)
        logger.info(f"  Embedding retriever: {len(embed_results)} candidates")
        
        bm25_results = self.bm25_retriever.search(query, k=self.bm25_top_k)
        logger.info(f"  BM25 retriever: {len(bm25_results)} candidates")
        
        # Stage 2: Merge and deduplicate
        merged_docs = self._merge_results(embed_results, bm25_results)
        logger.info(f"  Merged: {len(merged_docs)} unique candidates")
        
        if not merged_docs:
            return []
        
        # Stage 3: Rerank all candidates
        reranked = self.reranker.rerank(query, merged_docs, top_k=len(merged_docs))
        logger.info(f"  Reranked: {len(reranked)} candidates")
        
        # Stage 4: Apply filters
        filtered = []
        for doc, score in reranked:
            # Content length filter
            if len(doc.page_content) < min_content_length:
                continue
            
            # Metadata filters
            if filters:
                if 'layers' in filters:
                    if doc.metadata.get('layer') not in filters['layers']:
                        continue
                if 'category' in filters:
                    if doc.metadata.get('category') != filters['category']:
                        continue
                if 'tags' in filters:
                    doc_tags = doc.metadata.get('tags', [])
                    if not any(tag in doc_tags for tag in filters['tags']):
                        continue
            
            filtered.append((doc, score))
            if len(filtered) >= top_k:
                break
        
        logger.info(f"  Final: {len(filtered)} results")
        return filtered
    
    def search(self, query: str, k: int = 10, **kwargs) -> List[Tuple[Document, float]]:
        """Alias for retrieve() for compatibility."""
        return self.retrieve(query, top_k=k, **kwargs)


def build_bm25_index(documents: List[Document], save_path: Path) -> BM25Okapi:
    """
    Build and save BM25 index from documents.
    
    Args:
        documents: List of LangChain Documents
        save_path: Directory to save the index
    
    Returns:
        BM25Okapi index
    """
    import re
    
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    # Tokenize all documents
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save index and documents
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_path / "bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25, f)
    
    with open(save_path / "bm25_documents.pkl", 'wb') as f:
        pickle.dump(documents, f)
    
    logger.info(f"✓ Saved BM25 index ({len(documents)} documents) to {save_path}")
    
    return bm25


# Hybrid retrieval test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Hardcoded test cases
    test_queries = [
        "HIP kernel shared memory optimization",
        "How to install ROCm?",
        "vLLM serving on AMD GPU",
        "BF16 vector load store operations",
        "Triton kernel for AMD GPU",
    ]
    
    retriever = HybridRetriever()
    
    print("=" * 60)
    print("Hybrid Retrieval Test (Embedding + BM25 + Reranker)")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        results = retriever.retrieve(query, top_k=5)
        
        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results:")
            for i, (doc, score) in enumerate(results, 1):
                layer = doc.metadata.get('layer', 'unknown')
                print(f"  [{i}] Score={score:.4f} Layer={layer}")
                print(f"      {doc.page_content[:100]}...")
        print()



