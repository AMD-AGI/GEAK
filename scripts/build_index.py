# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Build semantic search index using LangChain with hybrid retrieval support.

This script reads files from the knowledge base, chunks them using
document-type-specific splitters, and builds both:
1. FAISS index with BGE embeddings (semantic search)
2. BM25 index (keyword search)

Supported document types:
- Markdown (.md) - MarkdownHeaderTextSplitter for structure-aware splitting
- Code (.cpp, .hpp, .c, .h, .py, .hip) - Language-aware code splitter
- Logs (.log, .txt) - Line-based splitter
- Config (.yaml, .yml, .json) - Structure-preserving splitter

Usage:
    python build_index.py [--kb-path PATH] [--output-path PATH] [--force]
"""

import argparse
import pickle
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi


# Default paths
DEFAULT_KB_PATH = Path(__file__).parent.parent / "knowledge-base"
DEFAULT_OUTPUT_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def _resolve_embedding_device(device: str) -> str:
    """Resolve embedding device: 'auto' -> cuda if available else cpu; otherwise use as-is."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# =============================================================================
# Document Type Detection & Splitter Configuration
# =============================================================================

class DocType(Enum):
    """Document types with specialized splitting strategies."""
    MARKDOWN = "markdown"
    CODE_CPP = "code_cpp"
    CODE_PYTHON = "code_python"
    CODE_HIP = "code_hip"
    LOG = "log"
    CONFIG = "config"
    TEXT = "text"


@dataclass
class SplitterConfig:
    """Configuration for document splitting."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 200


# File extension to document type mapping
EXTENSION_TO_DOCTYPE: dict[str, DocType] = {
    # Markdown
    ".md": DocType.MARKDOWN,
    ".markdown": DocType.MARKDOWN,
    # C/C++ code
    ".cpp": DocType.CODE_CPP,
    ".c": DocType.CODE_CPP,
    ".hpp": DocType.CODE_CPP,
    ".h": DocType.CODE_CPP,
    ".cc": DocType.CODE_CPP,
    ".cxx": DocType.CODE_CPP,
    # HIP/CUDA code
    ".hip": DocType.CODE_HIP,
    ".cu": DocType.CODE_HIP,
    ".cuh": DocType.CODE_HIP,
    # Python code
    ".py": DocType.CODE_PYTHON,
    ".pyx": DocType.CODE_PYTHON,
    # Logs
    ".log": DocType.LOG,
    # Config files
    ".yaml": DocType.CONFIG,
    ".yml": DocType.CONFIG,
    ".json": DocType.CONFIG,
    ".toml": DocType.CONFIG,
    # Plain text
    ".txt": DocType.TEXT,
    ".rst": DocType.TEXT,
}


def detect_doc_type(file_path: Path) -> DocType:
    """Detect document type from file extension."""
    suffix = file_path.suffix.lower()
    return EXTENSION_TO_DOCTYPE.get(suffix, DocType.TEXT)


def get_markdown_splitter(config: SplitterConfig) -> Callable[[Document], list[Document]]:
    """
    Get markdown-aware splitter that preserves header hierarchy.
    
    Uses MarkdownHeaderTextSplitter to split on headers first,
    then RecursiveCharacterTextSplitter for large sections.
    """
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,  # Strip headers; full header chain is prepended later
    )
    
    # Secondary splitter for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n```", "\n\n", "\n", " ", ""],
    )
    
    def split_markdown(doc: Document) -> list[Document]:
        """Split markdown document preserving structure."""
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        chunks = []
        for split in header_splits:
            chunk_metadata = {**doc.metadata}
            
            # Build full header hierarchy chain
            header_prefix = ""
            for level in ["header_1", "header_2", "header_3", "header_4"]:
                if level in split.metadata:
                    prefix_marker = "#" * int(level[-1])
                    header_prefix += f"{prefix_marker} {split.metadata[level]}\n"
                    chunk_metadata["section"] = split.metadata[level]
            
            # Check if secondary splitting is needed
            if len(split.page_content) > config.chunk_size:
                sub_chunks = text_splitter.split_text(split.page_content)
                for i, sub_chunk in enumerate(sub_chunks):
                    if len(sub_chunk) >= config.min_chunk_size:
                        # Prepend full header prefix
                        content = header_prefix + sub_chunk
                        chunks.append(Document(
                            page_content=content,
                            metadata={**chunk_metadata, "sub_chunk": i}
                        ))
            elif len(split.page_content) >= config.min_chunk_size:
                # Prepend full header prefix
                content = header_prefix + split.page_content
                chunks.append(Document(
                    page_content=content,
                    metadata=chunk_metadata
                ))
        
        return chunks
    
    return split_markdown


def get_code_splitter(language: Language, config: SplitterConfig) -> Callable[[Document], list[Document]]:
    """
    Get language-aware code splitter.
    
    Splits code at natural boundaries (functions, classes) while
    preserving context.
    """
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    
    def split_code(doc: Document) -> list[Document]:
        """Split code document at natural boundaries."""
        chunks = code_splitter.split_text(doc.page_content)
        
        result = []
        for i, chunk in enumerate(chunks):
            if len(chunk) >= config.min_chunk_size:
                # Try to extract function/class name for metadata
                func_match = re.search(r'(?:void|int|float|double|auto|def|class)\s+(\w+)', chunk)
                section = func_match.group(1) if func_match else f"code_block_{i}"
                
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "section": section, "chunk_idx": i}
                ))
        
        return result
    
    return split_code


def get_log_splitter(config: SplitterConfig) -> Callable[[Document], list[Document]]:
    """
    Get log-aware splitter.
    
    Splits logs at natural boundaries (timestamps, blank lines)
    while keeping related entries together.
    """
    # Log-specific separators
    log_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\n",                    # Blank lines
            "\n[",                     # Common log prefix
            "\n20",                    # Timestamps starting with year
            "\nERROR",                 # Error markers
            "\nWARNING",               # Warning markers
            "\nINFO",                  # Info markers
            "\n",
            " ",
        ],
    )
    
    def split_log(doc: Document) -> list[Document]:
        """Split log document preserving entry boundaries."""
        chunks = log_splitter.split_text(doc.page_content)
        
        result = []
        for i, chunk in enumerate(chunks):
            if len(chunk) >= config.min_chunk_size:
                # Try to detect log level
                log_level = "unknown"
                if "ERROR" in chunk.upper():
                    log_level = "error"
                elif "WARNING" in chunk.upper() or "WARN" in chunk.upper():
                    log_level = "warning"
                elif "INFO" in chunk.upper():
                    log_level = "info"
                elif "DEBUG" in chunk.upper():
                    log_level = "debug"
                
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "log_level": log_level, "chunk_idx": i}
                ))
        
        return result
    
    return split_log


def get_config_splitter(config: SplitterConfig) -> Callable[[Document], list[Document]]:
    """
    Get config-file-aware splitter.
    
    Tries to preserve YAML/JSON structure while splitting large files.
    """
    config_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n---",                   # YAML document separator
            "\n\n",                    # Blank lines
            "\n  ",                    # Indented blocks (2 space)
            "\n    ",                  # Indented blocks (4 space)
            "\n",
            " ",
        ],
    )
    
    def split_config(doc: Document) -> list[Document]:
        """Split config document preserving structure."""
        chunks = config_splitter.split_text(doc.page_content)
        
        result = []
        for i, chunk in enumerate(chunks):
            if len(chunk) >= config.min_chunk_size:
                # Try to extract top-level key
                key_match = re.match(r'^(\w+):', chunk.strip())
                section = key_match.group(1) if key_match else f"config_block_{i}"
                
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "section": section, "chunk_idx": i}
                ))
        
        return result
    
    return split_config


def get_text_splitter(config: SplitterConfig) -> Callable[[Document], list[Document]]:
    """Get generic text splitter as fallback."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    def split_text(doc: Document) -> list[Document]:
        """Split generic text document."""
        chunks = text_splitter.split_text(doc.page_content)
        
        return [
            Document(page_content=chunk, metadata={**doc.metadata, "chunk_idx": i})
            for i, chunk in enumerate(chunks)
            if len(chunk) >= config.min_chunk_size
        ]
    
    return split_text


def get_splitter_for_doc_type(doc_type: DocType, config: SplitterConfig) -> Callable[[Document], list[Document]]:
    """Get appropriate splitter for document type."""
    splitter_map = {
        DocType.MARKDOWN: get_markdown_splitter(config),
        DocType.CODE_CPP: get_code_splitter(Language.CPP, config),
        DocType.CODE_HIP: get_code_splitter(Language.CPP, config),  # HIP uses C++ syntax
        DocType.CODE_PYTHON: get_code_splitter(Language.PYTHON, config),
        DocType.LOG: get_log_splitter(config),
        DocType.CONFIG: get_config_splitter(config),
        DocType.TEXT: get_text_splitter(config),
    }
    return splitter_map.get(doc_type, get_text_splitter(config))


# =============================================================================
# Document Loading & Processing
# =============================================================================

def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content."""
    metadata = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 2:
            yaml_content = parts[1]
            for line in yaml_content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip("\"'")
    return metadata


def remove_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content."""
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)


def infer_metadata_from_path(file_path: Path) -> dict:
    """Infer metadata from file path structure."""
    path_str = str(file_path)
    metadata = {}
    
    # Infer vendor
    if "amd-knowledge-base" in path_str:
        metadata["vendor"] = "amd"
    elif "nvidia-knowledge-base" in path_str:
        metadata["vendor"] = "nvidia"
    elif "comparisons" in path_str:
        metadata["vendor"] = "comparison"
    
    # Infer layer
    for part in file_path.parts:
        if part.startswith("layer-"):
            metadata["layer"] = part.replace("layer-", "")
            break
        if part == "best-practices":
            metadata["layer"] = "best-practices"
            break
    
    # Category from parent directory
    metadata["category"] = file_path.parent.name
    
    return metadata


def load_all_files(
    kb_path: Path,
    exclude_patterns: list[str] | None = None,
) -> list[tuple[Document, DocType]]:
    """
    Load all supported files from knowledge base as LangChain Documents.
    
    Args:
        kb_path: Path to knowledge base
        exclude_patterns: List of patterns to exclude (e.g., ["nvidia-knowledge-base", "test"])
    
    Returns list of (Document, DocType) tuples for type-specific splitting.
    """
    documents = []
    exclude_patterns = exclude_patterns or []
    
    # Find all supported files
    supported_extensions = set(EXTENSION_TO_DOCTYPE.keys())
    all_files = []
    for ext in supported_extensions:
        all_files.extend(kb_path.rglob(f"*{ext}"))
    all_files = sorted(set(all_files))
    
    # Filter out excluded patterns
    if exclude_patterns:
        filtered_files = []
        for f in all_files:
            path_str = str(f)
            if not any(pattern in path_str for pattern in exclude_patterns):
                filtered_files.append(f)
        excluded_count = len(all_files) - len(filtered_files)
        all_files = filtered_files
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} files matching patterns: {exclude_patterns}")
    
    # Count by type
    type_counts: dict[DocType, int] = {}
    
    print(f"Found {len(all_files)} files")
    
    for file_path in all_files:
        # Skip certain files
        if file_path.name in ["README.md", "INDEX.md", "CONTRIBUTING.md"]:
            continue
        
        try:
            content = file_path.read_text(encoding="utf-8")
            doc_type = detect_doc_type(file_path)
            
            # Parse frontmatter for markdown files
            if doc_type == DocType.MARKDOWN:
                frontmatter = parse_frontmatter(content)
                content = remove_frontmatter(content)
            else:
                frontmatter = {}
            
            if not content.strip():
                continue
            
            # Build metadata
            metadata = infer_metadata_from_path(file_path)
            metadata.update(frontmatter)
            metadata["source"] = str(file_path)
            metadata["doc_type"] = doc_type.value
            
            documents.append((Document(page_content=content, metadata=metadata), doc_type))
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    # Print summary
    print("\n  Files by type:")
    for doc_type in sorted(type_counts.keys(), key=lambda x: x.value):
        print(f"    {doc_type.value}: {type_counts[doc_type]} files")
    
    return documents


def chunk_documents_by_type(
    documents: list[tuple[Document, DocType]],
    config: SplitterConfig,
) -> list[Document]:
    """
    Chunk documents using type-specific splitters.
    
    Each document type uses an appropriate splitting strategy:
    - Markdown: Header-aware splitting
    - Code: Language-aware function/class boundaries
    - Logs: Entry-aware splitting
    - Config: Structure-preserving splitting
    """
    all_chunks = []
    type_chunk_counts: dict[DocType, int] = {}
    
    for doc, doc_type in documents:
        # Get appropriate splitter
        splitter = get_splitter_for_doc_type(doc_type, config)
        
        # Split document
        chunks = splitter(doc)
        all_chunks.extend(chunks)
        
        type_chunk_counts[doc_type] = type_chunk_counts.get(doc_type, 0) + len(chunks)
    
    # Print chunk distribution by type
    print("\n  Chunks by document type:")
    for doc_type in sorted(type_chunk_counts.keys(), key=lambda x: x.value):
        print(f"    {doc_type.value}: {type_chunk_counts[doc_type]} chunks")
    
    return all_chunks


# Legacy function for backward compatibility
def load_markdown_files(kb_path: Path) -> list[Document]:
    """Load all markdown files from knowledge base as LangChain Documents."""
    docs_with_types = load_all_files(kb_path)
    return [doc for doc, doc_type in docs_with_types if doc_type == DocType.MARKDOWN]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk documents using LangChain text splitter (legacy, uses markdown splitter)."""
    config = SplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitter = get_markdown_splitter(config)
    
    all_chunks = []
    for doc in documents:
        all_chunks.extend(splitter(doc))
    
    return all_chunks


def tokenize_text(text: str) -> list[str]:
    """Simple tokenization for BM25."""
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25_index(
    chunks: list[Document],
    output_path: Path,
) -> tuple[BM25Okapi, list[Document]]:
    """Build and save BM25 index from document chunks."""
    
    # Tokenize all document contents
    tokenized_corpus = [tokenize_text(chunk.page_content) for chunk in chunks]
    
    # Build BM25 index
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index
    bm25_index_path = output_path / "bm25_index.pkl"
    with open(bm25_index_path, 'wb') as f:
        pickle.dump(bm25_index, f)
    
    # Save documents for BM25 retrieval
    bm25_docs_path = output_path / "bm25_documents.pkl"
    with open(bm25_docs_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    return bm25_index, chunks


def build_index(
    kb_path: Path = DEFAULT_KB_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 200,
    exclude_patterns: list[str] | None = None,
    device: str = "auto",
):
    """Build semantic search index from knowledge base using LangChain."""
    embedding_device = _resolve_embedding_device(device)

    print("\n🔍 LangChain Semantic Index Builder\n")
    print(f"Knowledge base: {kb_path}")
    print(f"Output path: {output_path}")
    print(f"Embedding model: {model_name}")
    print(f"Embedding device: {embedding_device}")
    print(f"Chunk config: size={chunk_size}, overlap={chunk_overlap}, min={min_chunk_size}")
    if exclude_patterns:
        print(f"Exclude patterns: {exclude_patterns}")
    print()
    
    # Check if index exists
    if output_path.exists() and not force:
        print(f"⚠️  Index already exists at {output_path}")
        print("   Use --force to rebuild")
        return
    
    # Splitter configuration
    splitter_config = SplitterConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    
    # Step 1: Load all files with type detection
    print("Step 1: Loading files with type detection...")
    documents_with_types = load_all_files(kb_path, exclude_patterns=exclude_patterns)
    print(f"  ✓ Loaded {len(documents_with_types)} documents")
    
    # Step 2: Chunk documents using type-specific splitters
    print("\nStep 2: Chunking with type-specific splitters...")
    chunks = chunk_documents_by_type(documents_with_types, splitter_config)
    print(f"  ✓ Created {len(chunks)} chunks")
    
    # Show distribution
    layer_counts = {}
    for chunk in chunks:
        layer = chunk.metadata.get("layer", "unknown")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    print("\n  Chunks by layer:")
    for layer in sorted(layer_counts.keys()):
        print(f"    Layer {layer}: {layer_counts[layer]} chunks")
    
    # Step 3: Initialize embeddings
    print(f"\nStep 3: Initializing embeddings ({model_name}) on {embedding_device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"  ✓ Embeddings model loaded on {embedding_device}")
    
    # Step 4: Build FAISS index
    print("\nStep 4: Building FAISS index...")
    print("  This may take a few minutes...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  ✓ Index built with {len(chunks)} vectors")
    
    # Step 5: Save FAISS index
    print("\nStep 5: Saving FAISS index...")
    output_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_path))
    
    # Calculate FAISS size
    faiss_size = sum(f.stat().st_size for f in output_path.glob("index.*"))
    print(f"  ✓ Saved FAISS index to {output_path}")
    print(f"  FAISS index size: {faiss_size / 1024:.1f} KB")
    
    # Step 6: Build BM25 index
    print("\nStep 6: Building BM25 index...")
    bm25_index, bm25_docs = build_bm25_index(chunks, output_path)
    print(f"  ✓ Built BM25 index with {len(bm25_docs)} documents")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.glob("*"))
    print(f"  Total index size: {total_size / 1024:.1f} KB")
    
    # Step 7: Test search
    print("\nStep 7: Testing semantic search...")
    test_queries = [
        "How to install ROCm?",
        "HIP kernel shared memory optimization",
        "Matrix multiplication with rocBLAS",
    ]
    
    for query in test_queries:
        results = vectorstore.similarity_search_with_score(query, k=1)
        if results:
            doc, score = results[0]
            layer = doc.metadata.get("layer", "?")
            category = doc.metadata.get("category", "?")
            print(f"  ✓ Embedding: '{query}' → Layer {layer}, {category} (score: {score:.3f})")
    
    # Step 8: Test BM25 search
    print("\nStep 8: Testing BM25 search...")
    for query in test_queries:
        query_tokens = tokenize_text(query)
        scores = bm25_index.get_scores(query_tokens)
        top_idx = scores.argmax()
        if scores[top_idx] > 0:
            doc = bm25_docs[top_idx]
            layer = doc.metadata.get("layer", "?")
            category = doc.metadata.get("category", "?")
            print(f"  ✓ BM25: '{query}' → Layer {layer}, {category} (score: {scores[top_idx]:.3f})")
    
    print("\n✓ Hybrid index (FAISS + BM25) built successfully!")
    print(f"\nIndex Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  FAISS dimension: 1024 (BGE-large)")
    print(f"  BM25 documents: {len(bm25_docs)}")
    print(f"  Total index size: {total_size / 1024:.1f} KB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build semantic search index using LangChain")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=DEFAULT_KB_PATH,
        help=f"Path to knowledge base directory (default: {DEFAULT_KB_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save index (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=200,
        help="Minimum chunk size to keep (default: 200)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to exclude from indexing (e.g., --exclude nvidia-knowledge-base test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device for embedding model: auto (use GPU if available), cuda, or cpu (default: auto)",
    )
    
    args = parser.parse_args()
    
    build_index(
        kb_path=args.kb_path,
        output_path=args.output_path,
        model_name=args.model,
        force=args.force,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        exclude_patterns=args.exclude,
        device=args.device,
    )


if __name__ == "__main__":
    main()

