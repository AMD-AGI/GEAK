#!/usr/bin/env python3
"""Test chunking with header prefix optimization."""

from pathlib import Path
from langchain_core.documents import Document
from build_index import get_markdown_splitter, SplitterConfig

# 测试文档路径（容器内路径，使用知识库中的实际文件）
TEST_DOC = Path("/workspace/mini-swe-agent/knowledge-base/amd-knowledge-base/layer-6-extended/optimize-guides/Report_flash_attention.md")

def main():
    # 1. 初始化 splitter
    config = SplitterConfig(chunk_size=1000, chunk_overlap=200, min_chunk_size=200)
    splitter = get_markdown_splitter(config)
    
    # 2. 读取测试文档
    content = TEST_DOC.read_text(encoding="utf-8")
    doc = Document(page_content=content, metadata={"source": str(TEST_DOC)})
    
    # 3. 执行 chunking
    chunks = splitter(doc)
    
    # 4. 打印结果
    print(f"总共生成 {len(chunks)} 个 chunks\n")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\n【Chunk {i+1}】")
        print(f"Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"Sub-chunk: {chunk.metadata.get('sub_chunk', 'N/A')}")
        print(f"长度: {len(chunk.page_content)} 字符")
        print("-" * 40)
        # 打印前 8 行
        lines = chunk.page_content.split('\n')[:8]
        for line in lines:
            print(line[:80] + ("..." if len(line) > 80 else ""))
        print("=" * 60)

if __name__ == "__main__":
    main()
