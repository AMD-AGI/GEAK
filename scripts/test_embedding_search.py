#!/usr/bin/env python3
"""
Test semantic search with the built LangChain FAISS index.

Usage:
    python test_search.py "your search query"
    python test_search.py --interactive
"""

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_INDEX_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def load_index(
    index_path: Path = DEFAULT_INDEX_PATH,
    model_name: str = DEFAULT_MODEL,
) -> FAISS:
    """Load FAISS index with embeddings."""
    print(f"Loading index from {index_path}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    print(f"✓ Index loaded ({vectorstore.index.ntotal} vectors)\n")
    return vectorstore


def search(vectorstore: FAISS, query: str, k: int = 5):
    """Search the index and display results."""
    print(f"🔍 Query: {query}\n")
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        layer = doc.metadata.get("layer", "?")
        category = doc.metadata.get("category", "?")
        vendor = doc.metadata.get("vendor", "?")
        
        print(f"--- Result {i} (score: {score:.3f}) ---")
        print(f"Layer: {layer} | Category: {category} | Vendor: {vendor}")
        print(f"Content ({len(doc.page_content)} chars):")
        print(doc.page_content[:500])
        if len(doc.page_content) > 500:
            print("...")
        print()


def interactive_mode(vectorstore: FAISS):
    """Interactive search mode."""
    print("Interactive search mode. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Enter query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            search(vectorstore, query)
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def main():
    """Run embedding search tests with hardcoded queries."""
    vectorstore = load_index()
    
    # 硬编码测试用例
    test_queries = [
        "HIP kernel shared memory optimization",
        "How to install ROCm?",
        "vLLM serving on AMD GPU",
        "Matrix multiplication with rocBLAS",
        "PyTorch training on MI250X",
    ]
    
    print("=" * 60)
    print("Embedding Search Test (FAISS only)")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        search(vectorstore, query, k=3)
        print("-" * 40)


if __name__ == "__main__":
    main()



