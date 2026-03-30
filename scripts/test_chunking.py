#!/usr/bin/env python3
"""Test chunking with header prefix optimization."""

from pathlib import Path
from langchain_core.documents import Document
from build_index import get_markdown_splitter, SplitterConfig

# Test document path (container path, using actual knowledge base file)
TEST_DOC = Path("/workspace/mini-swe-agent/knowledge-base/amd-knowledge-base/layer-6-extended/optimize-guides/Report_flash_attention.md")

def main():
    # 1. Initialize splitter
    config = SplitterConfig(chunk_size=1000, chunk_overlap=200, min_chunk_size=200)
    splitter = get_markdown_splitter(config)
    
    # 2. Read test document
    content = TEST_DOC.read_text(encoding="utf-8")
    doc = Document(page_content=content, metadata={"source": str(TEST_DOC)})
    
    # 3. Execute chunking
    chunks = splitter(doc)
    
    print(f"Generated {len(chunks)} chunks in total\n")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}]")
        print(f"Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"Sub-chunk: {chunk.metadata.get('sub_chunk', 'N/A')}")
        print(f"Length: {len(chunk.page_content)} chars")
        print("-" * 40)
        # Print first 8 lines
        lines = chunk.page_content.split('\n')[:8]
        for line in lines:
            print(line[:80] + ("..." if len(line) > 80 else ""))
        print("=" * 60)

if __name__ == "__main__":
    main()
