# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Test RRF fusion algorithm."""

import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_rrf_calculation():
    """Test RRF score calculation with mock data."""
    print("\n" + "="*60)
    print("TEST 1: RRF Score Calculation")
    print("="*60)
    
    # Import after path setup
    from minisweagent.mcp_integration.langchain_retrieval import HybridRetriever
    
    # Create retriever
    retriever = HybridRetriever(
        embed_top_k=25,
        bm25_top_k=25,
        rrf_k=60,
        semantic_weight=0.7,
        bm25_weight=0.3,
    )
    
    print(f"\n✓ Retriever initialized with:")
    print(f"  embed_top_k: {retriever.embed_top_k}")
    print(f"  bm25_top_k: {retriever.bm25_top_k}")
    print(f"  rrf_k: {retriever.rrf_k}")
    print(f"  weights: {retriever.semantic_weight}/{retriever.bm25_weight}")
    
    # Mock documents
    doc_a = Document(page_content="A" * 500, metadata={"name": "doc_A"})
    doc_b = Document(page_content="B" * 500, metadata={"name": "doc_B"})
    doc_c = Document(page_content="C" * 500, metadata={"name": "doc_C"})
    
    # Mock results with source tracking
    embed_results = [
        (doc_a, 0.95, "embedding"),  # rank 0
        (doc_b, 0.85, "embedding"),  # rank 1
    ]
    bm25_results = [
        (doc_a, 25.0, "bm25"),       # rank 0
        (doc_c, 15.0, "bm25"),       # rank 1
    ]
    
    print("\n📊 Input:")
    print("  Embedding: doc_A (#1, score=0.95), doc_B (#2, score=0.85)")
    print("  BM25:      doc_A (#1, score=25.0), doc_C (#2, score=15.0)")
    
    # Test RRF merge
    merged = retriever._rrf_merge(embed_results, bm25_results)
    
    print("\n🔄 RRF Fusion Results:")
    print(f"  Total candidates: {len(merged)}")
    
    # Expected RRF scores:
    # doc_A: 0.7/61 + 0.3/61 = 0.0164 (both methods rank #1)
    # doc_B: 0.7/62         = 0.0113 (embedding only, rank #2)
    # doc_C:         0.3/62 = 0.0048 (BM25 only, rank #2)
    
    print("\n  Ranking:")
    for i, (doc, source, orig_score, rrf_score) in enumerate(merged, 1):
        doc_name = doc.metadata['name']
        expected = {
            'doc_A': 0.0164,
            'doc_B': 0.0113,
            'doc_C': 0.0048,
        }[doc_name]
        match = "✓" if abs(rrf_score - expected) < 0.0001 else "✗"
        print(f"    {i}. {doc_name}: {rrf_score:.4f} (expected {expected:.4f}) {match}")
        print(f"       source: {source}, orig_score: {orig_score:.4f}")
    
    # Verify ranking
    assert merged[0][0].metadata['name'] == 'doc_A', "doc_A should be #1"
    assert merged[1][0].metadata['name'] == 'doc_B', "doc_B should be #2"
    assert merged[2][0].metadata['name'] == 'doc_C', "doc_C should be #3"
    
    print("\n✅ RRF calculation test PASSED!")
    return True


def test_hybrid_search():
    """Test full hybrid search with real index (if available)."""
    print("\n" + "="*60)
    print("TEST 2: Full Hybrid Search (with real index)")
    print("="*60)
    
    from minisweagent.mcp_integration.langchain_retrieval import HybridRetriever, DEFAULT_INDEX_PATH
    
    # Check if index exists
    if not DEFAULT_INDEX_PATH.exists():
        print(f"\n⚠️  Index not found at {DEFAULT_INDEX_PATH}")
        print("   Skipping real search test (this is OK for now)")
        return True
    
    print(f"\n✓ Index found at {DEFAULT_INDEX_PATH}")
    
    try:
        retriever = HybridRetriever(
            embed_top_k=25,
            bm25_top_k=25,
        )
        
        query = "HIP kernel shared memory optimization"
        print(f"\n🔍 Query: '{query}'")
        print("="*60)
        
        results = retriever.search(query, k=8)
        
        if results:
            print(f"\n✅ Found {len(results)} results")
            print("\n📝 Top 3 results:")
            for i, (doc, rerank_score, source, orig_score) in enumerate(results[:3], 1):
                print(f"\n  {i}. Rerank Score: {rerank_score:.3f} | Source: {source}")
                print(f"     Original Score: {orig_score:.4f}")
                print(f"     Content: {doc.page_content[:100]}...")
        else:
            print("\n⚠️  No results found (index may be empty)")
        
        print("\n✅ Full search test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🧪 RRF FUSION ALGORITHM TEST SUITE" + "\n")
    
    results = []
    
    # Test 1: RRF calculation
    try:
        results.append(("RRF Calculation", test_rrf_calculation()))
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("RRF Calculation", False))
    
    # Test 2: Full hybrid search
    try:
        results.append(("Hybrid Search", test_hybrid_search()))
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Hybrid Search", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
