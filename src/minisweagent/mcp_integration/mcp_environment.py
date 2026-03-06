# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""
MCPEnabledEnvironment - Extended LocalEnvironment with RAG tool support.

This module provides an Environment class that can execute both:
1. Regular bash commands (passed to subprocess)
2. RAG tool calls (routed to LangChain retrieval)

LangChain Retrieval Configuration:
- Chunks source: aig_eval_6.4.3_1007_ai_dev_knowledge_bge_subagent (2077 chunks)
- Embedding search: top_k=10 candidates (BAAI/bge-large-en-v1.5)
- BM25 keyword search: DISABLED (top_k=0)
- BGE reranker: Final ranking

RAG tool syntax: @amd:<tool_name> <json_args>
Example: @amd:query {"topic": "HIP optimization"}
"""

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.mcp_integration.langchain_retrieval import HybridRetriever, KnowledgeTools
from minisweagent.mcp_integration.subagent import RAGFilterSubAgent, SubAgentConfig


logger = logging.getLogger(__name__)


def load_rag_config(config_path: Path | None = None) -> dict:
    """Load RAG config with default fallback."""
    default_config = Path(__file__).parent.parent / "config" / "rag_config.yaml"
    config_path = config_path or default_config
    
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            print(f"[RAG Config] Loaded from {config_path}", file=sys.stderr)
            return cfg
    print(f"[RAG Config] Not found at {config_path}, using defaults", file=sys.stderr)
    return {}


# Semantic index location (supports both LangChain and original formats)
# Available indexes:
#   - subagent-chunks: Original format (faiss.index, chunks.pkl) - using this
#   - subagent-langchain-index: LangChain format (index.faiss, index.pkl)
#   - semantic-index: LangChain format for langchain-split chunks
SEMANTIC_INDEX_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
@dataclass
class MCPEnvironmentConfig(LocalEnvironmentConfig):
    """Configuration for RAG-enabled environment."""
    mcp_prefix: str = "@amd:"
    rag_subagent_api_key: str | None = None  # Read from AMD_LLM_API_KEY env var



class MCPEnabledEnvironment(LocalEnvironment):
    """
    Extended LocalEnvironment with RAG tool support.
    
    Intercepts commands starting with '@amd:' and routes them to
    LangChain hybrid retrieval, while passing other commands to
    the standard bash execution.
    """
    
    def __init__(
        self, 
        *, 
        config_class: type = MCPEnvironmentConfig,
        auto_build_index: bool = True,
        rag_config_path: Path | None = None,
        rag_subagent_api_key: str | None = None,  # Read from AMD_LLM_API_KEY env var
        **kwargs
    ):
        # Load RAG config
        cfg = load_rag_config(rag_config_path)
        
        # Read from config, keep original defaults as fallback
        retrieval = cfg.get("retrieval", {})
        reranker = cfg.get("reranker", {})
        fusion = cfg.get("fusion", {})
        summary = cfg.get("summary", {})
        debug = cfg.get("debug", {})
        
        self._embed_top_k = retrieval.get("embed_top_k", 25)
        enable_bm25 = retrieval.get("enable_bm25", True)
        self._bm25_top_k = retrieval.get("bm25_top_k", 25) if enable_bm25 else 0
        self._mcp_top_k = retrieval.get("mcp_top_k", 8)
        
        self._enable_reranker = reranker.get("enable_reranker", True)
        
        self._rrf_k = fusion.get("rrf_k", 60)
        self._semantic_weight = fusion.get("semantic_weight", 0.7)
        self._bm25_weight = fusion.get("bm25_weight", 0.3)
        
        self._enable_rag_subagent = summary.get("enable_rag_subagent", True)
        self._rag_subagent_model = summary.get("rag_subagent_model", "claude-opus-4.5")
        
        self._verbose = debug.get("verbose", False)
        
        # Print config to stderr (consistent with other logs)
        print(f"[RAG Config] Retrieval: embed_top_k={self._embed_top_k}, bm25_top_k={self._bm25_top_k} (enable_bm25={enable_bm25}), rag_top_k={self._mcp_top_k}", file=sys.stderr)
        print(f"[RAG Config] Reranker: enable_reranker={self._enable_reranker}", file=sys.stderr)
        print(f"[RAG Config] Fusion: rrf_k={self._rrf_k}, semantic_weight={self._semantic_weight}, bm25_weight={self._bm25_weight}", file=sys.stderr)
        print(f"[RAG Config] Summary: enable_rag_subagent={self._enable_rag_subagent}, model={self._rag_subagent_model}", file=sys.stderr)
        print(f"[RAG Config] Debug: verbose={self._verbose}", file=sys.stderr)
        
        # Inject rag_subagent_api_key into kwargs for MCPEnvironmentConfig
        kwargs = {
            **kwargs,
            "rag_subagent_api_key": rag_subagent_api_key,
        }
        super().__init__(config_class=config_class, **kwargs)
        self._knowledge_tools: Optional[KnowledgeTools] = None
        self._retriever: Optional[HybridRetriever] = None
        self._tool_map = None
        self._rag_subagent: Optional[RAGFilterSubAgent] = None
        
        # Check semantic index exists
        if auto_build_index:
            self._ensure_semantic_index()
    
    def _ensure_semantic_index(self) -> None:
        """Check if semantic index exists (supports both formats)."""
        # LangChain format (index.faiss, index.pkl)
        lc_faiss = SEMANTIC_INDEX_PATH / "index.faiss"
        lc_pkl = SEMANTIC_INDEX_PATH / "index.pkl"
        
        # Original format (faiss.index, chunks.pkl)
        orig_faiss = SEMANTIC_INDEX_PATH / "faiss.index"
        orig_chunks = SEMANTIC_INDEX_PATH / "chunks.pkl"
        
        if lc_faiss.exists() and lc_pkl.exists():
            print("✅ LangChain semantic index found (LangChain format)")
            return
        
        if orig_faiss.exists() and orig_chunks.exists():
            print("✅ Semantic index found (original format)")
            return
        
        # Also check for BM25 files
        bm25_index = SEMANTIC_INDEX_PATH / "bm25_index.pkl"
        if not bm25_index.exists():
            print("⚠️  BM25 index not found, hybrid retrieval will use embedding only")
        
        print("⚠️  LangChain index not found at:", SEMANTIC_INDEX_PATH)
        print("   Build index with: python langchain/build_index.py --force")
    
    @property
    def retriever(self) -> HybridRetriever:
        """Lazy initialization of LangChain hybrid retriever."""
        if self._retriever is None:
            # Hybrid: Embedding + BM25 + RRF Fusion + Reranker
            self._retriever = HybridRetriever(
                index_path=SEMANTIC_INDEX_PATH,
                embed_top_k=self._embed_top_k,
                bm25_top_k=self._bm25_top_k,
                rrf_k=self._rrf_k,
                semantic_weight=self._semantic_weight,
                bm25_weight=self._bm25_weight,
                enable_reranker=self._enable_reranker,
            )
        return self._retriever
    
    @property
    def knowledge_tools(self) -> KnowledgeTools:
        """Lazy initialization of knowledge tools."""
        if self._knowledge_tools is None:
            self._knowledge_tools = KnowledgeTools(self.retriever)
            self._init_tool_map()
        return self._knowledge_tools
    
    @property
    def rag_subagent(self) -> RAGFilterSubAgent | None:
        """Lazy initialization of RAG filter sub-agent."""
        if not self._enable_rag_subagent:
            return None
        
        if self._rag_subagent is None:
            self._rag_subagent = RAGFilterSubAgent(SubAgentConfig(
                model_name=self._rag_subagent_model,
                api_key=self.config.rag_subagent_api_key,
                enabled=True,
            ))
        return self._rag_subagent
    
    def _init_tool_map(self):
        """Initialize mapping from tool names to functions."""
        self._tool_map = {
            # Short names
            "query": self._knowledge_tools.query,
            "example": self._knowledge_tools.example,
            "optimize": self._knowledge_tools.optimize,
            "compat": self._knowledge_tools.compat,
            "troubleshoot": self._knowledge_tools.troubleshoot,
            "docs": self._knowledge_tools.docs,
            # Full names
            "query_knowledge": self._knowledge_tools.query,
            "get_code_example": self._knowledge_tools.example,
            "suggest_optimization": self._knowledge_tools.optimize,
            "check_compatibility": self._knowledge_tools.compat,
            "get_documentation_urls": self._knowledge_tools.docs,
        }
    
    def execute(
        self, 
        command: str, 
        cwd: str = "", 
        *, 
        timeout: int | None = None
    ) -> Dict[str, Any]:
        """
        Execute a command - either RAG tool or bash command.
        
        Args:
            command: Command to execute. If starts with '@amd:', routes to RAG.
            cwd: Working directory for bash commands.
            timeout: Timeout for bash commands.
        
        Returns:
            Dict with 'output' and 'returncode' keys.
        """
        command = command.strip()
        
        # Verbose: print command before execution
        if self._verbose:
            print(f"\n{'='*60}")
            print(f"🔧 [ENV] Executing command:")
            print(f"{'='*60}")
            print(command)
            print(f"{'='*60}")
        
        # Check if this is a RAG tool call
        prefix = self.config.mcp_prefix
        if command.startswith(prefix):
            if self._verbose:
                print(f"✅ [ENV] This is a RAG command! Will route to RAG retrieval.")
            result = self._execute_mcp(command[len(prefix):])
        else:
            if self._verbose:
                print(f"⚠️  [ENV] This is a BASH command, not RAG.")
            result = super().execute(command, cwd, timeout=timeout)
        
        # Verbose: print result after execution
        if self._verbose:
            print(f"\n📤 [ENV] Command output (first 500 chars):")
            print(f"{'-'*60}")
            print(result.get("output", "")[:500])
            print(f"{'-'*60}")
            print(f"📤 [ENV] Return code: {result.get('returncode')}")
        
        return result
    
    def _execute_mcp(self, command: str) -> Dict[str, Any]:
        """
        Execute a RAG tool call.
        
        Args:
            command: Tool call without prefix, e.g., 'query {"topic": "HIP"}'
        
        Returns:
            Dict with 'output' and 'returncode' keys.
        """
        try:
            # Parse tool name and arguments
            tool_name, args = self._parse_mcp_command(command)
            
            if self._verbose:
                print(f"[RAG] Calling tool: {tool_name}")
                print(f"[RAG] Arguments: {args}")
            
            # Output stats to stderr for RAG tools
            self._print_mcp_stats(tool_name, args)
            
            # Execute the tool
            result = asyncio.run(self._async_call_tool(tool_name, args))
            
            return {
                "output": result,
                "returncode": 0
            }
            
        except json.JSONDecodeError as e:
            return {
                "output": f"RAG Error: Invalid JSON arguments - {e}",
                "returncode": 1
            }
        except KeyError as e:
            available = ", ".join(sorted(self._tool_map.keys()))
            return {
                "output": f"RAG Error: Unknown tool {e}. Available: {available}",
                "returncode": 1
            }
        except Exception as e:
            return {
                "output": f"RAG Error: {type(e).__name__}: {str(e)}",
                "returncode": 1
            }
    
    def _format_content_preview(self, content: str, max_len: int = 200) -> str:
        """Format content preview for logging - single line, truncated."""
        preview = content.replace('\n', ' ').replace('\r', ' ')
        preview = ' '.join(preview.split())  # Collapse multiple spaces
        if len(preview) > max_len:
            preview = preview[:max_len] + "..."
        return preview

    def _print_mcp_stats(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Output RAG tool stats to stderr for logging with query results, retrieval method, and content."""
        print(f"[RAG-STATS] Calling {tool_name} with LangChain retrieval (embed_top_k={self._embed_top_k}, bm25_top_k={self._bm25_top_k})", file=sys.stderr)
        print(f"[RAG-STATS] Args: {args}", file=sys.stderr)
        
        top_k = self._mcp_top_k
        
        # Get query results for stats
        try:
            if tool_name in ("query", "query_knowledge"):
                topic = args.get("topic", "")
                layer = args.get("layer")
                
                # Perform actual retrieval
                filters = {}
                if layer:
                    filters['layers'] = [layer]
                
                results = self.retriever.search(topic, k=top_k, filters=filters if filters else None)
                
                # Count by retrieval method
                embed_count = sum(1 for r in results if r[2] == "embedding")
                bm25_count = sum(1 for r in results if r[2] == "bm25")
                
                print(f"[RAG-STATS] query returned {len(results)} results (top_k={top_k}) | embedding: {embed_count}, bm25: {bm25_count}", file=sys.stderr)
                for i, (doc, score, source, orig_score) in enumerate(results, 1):
                    title = doc.metadata.get('section', doc.metadata.get('title', 'Unknown'))[:40]
                    content_len = len(doc.page_content)
                    layer_info = doc.metadata.get('layer', 'unknown')
                    method_tag = "[EMB]" if source == "embedding" else "[BM25]"
                    content_preview = self._format_content_preview(doc.page_content)
                    print(f"  [{i}] {method_tag} Score={score:.4f} (orig={orig_score:.4f}) Layer={layer_info} Len={content_len} Title={title}", file=sys.stderr)
                    print(f"      Content: {content_preview}", file=sys.stderr)
            
            elif tool_name in ("example", "get_code_example"):
                category = args.get("category", "")
                use_case = args.get("use_case", "")
                query = f"{category} {use_case} code example".strip()
                
                results = self.retriever.search(query, k=top_k)
                
                embed_count = sum(1 for r in results if r[2] == "embedding")
                bm25_count = sum(1 for r in results if r[2] == "bm25")
                
                print(f"[RAG-STATS] example returned {len(results)} results (top_k={top_k}) | embedding: {embed_count}, bm25: {bm25_count}", file=sys.stderr)
                for i, (doc, score, source, orig_score) in enumerate(results, 1):
                    title = doc.metadata.get('section', doc.metadata.get('title', 'Unknown'))[:40]
                    content_len = len(doc.page_content)
                    method_tag = "[EMB]" if source == "embedding" else "[BM25]"
                    content_preview = self._format_content_preview(doc.page_content)
                    print(f"  [{i}] {method_tag} Score={score:.4f} (orig={orig_score:.4f}) Title={title} Len={content_len}", file=sys.stderr)
                    print(f"      Content: {content_preview}", file=sys.stderr)
            
            elif tool_name in ("optimize", "suggest_optimization"):
                code_type = args.get("code_type", "")
                context = args.get("context", "")
                gpu_model = args.get("gpu_model", "")
                query = f"{gpu_model} {code_type} optimization {context}".strip()
                
                results = self.retriever.search(query, k=top_k)
                
                embed_count = sum(1 for r in results if r[2] == "embedding")
                bm25_count = sum(1 for r in results if r[2] == "bm25")
                
                print(f"[RAG-STATS] optimize returned {len(results)} results (top_k={top_k}) | embedding: {embed_count}, bm25: {bm25_count}", file=sys.stderr)
                for i, (doc, score, source, orig_score) in enumerate(results, 1):
                    title = doc.metadata.get('section', doc.metadata.get('title', 'Unknown'))[:40]
                    content_len = len(doc.page_content)
                    method_tag = "[EMB]" if source == "embedding" else "[BM25]"
                    content_preview = self._format_content_preview(doc.page_content)
                    print(f"  [{i}] {method_tag} Score={score:.4f} (orig={orig_score:.4f}) Title={title} Len={content_len}", file=sys.stderr)
                    print(f"      Content: {content_preview}", file=sys.stderr)
            
            elif tool_name == "troubleshoot":
                error_message = args.get("error_message", "")
                context = args.get("context", "")
                query = f"troubleshoot {error_message} {context}".strip()
                
                results = self.retriever.search(query, k=top_k)
                
                embed_count = sum(1 for r in results if r[2] == "embedding")
                bm25_count = sum(1 for r in results if r[2] == "bm25")
                
                print(f"[RAG-STATS] troubleshoot returned {len(results)} results (top_k={top_k}) | embedding: {embed_count}, bm25: {bm25_count}", file=sys.stderr)
                for i, (doc, score, source, orig_score) in enumerate(results, 1):
                    content_len = len(doc.page_content)
                    method_tag = "[EMB]" if source == "embedding" else "[BM25]"
                    content_preview = self._format_content_preview(doc.page_content)
                    print(f"  [{i}] {method_tag} Score={score:.4f} (orig={orig_score:.4f}) Len={content_len}", file=sys.stderr)
                    print(f"      Content: {content_preview}", file=sys.stderr)
            
            elif tool_name in ("compat", "check_compatibility"):
                # For compatibility check, just log the call
                print(f"[RAG-STATS] compat check for ROCm {args.get('rocm_version', 'unknown')}", file=sys.stderr)
            
            elif tool_name in ("docs", "get_documentation_urls"):
                print(f"[RAG-STATS] docs returned document URL list", file=sys.stderr)
        
        except Exception as e:
            print(f"[RAG-STATS] Error getting stats: {e}", file=sys.stderr)

    def _parse_mcp_command(self, command: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse RAG command into tool name and arguments.
        
        Args:
            command: e.g., 'query {"topic": "HIP"}' or 'query_knowledge {...}'
        
        Returns:
            Tuple of (tool_name, args_dict)
        """
        # Split on first space or first '{'
        match = re.match(r'(\w+)\s*(.*)', command.strip())
        
        if not match:
            raise ValueError(f"Invalid RAG command format: {command}")
        
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Parse JSON arguments
        if args_str:
            args = json.loads(args_str)
        else:
            args = {}
        
        return tool_name, args
    
    async def _async_call_tool(
        self, 
        tool_name: str, 
        args: Dict[str, Any]
    ) -> str:
        """
        Asynchronously call a RAG tool using LangChain retrieval.
        
        Args:
            tool_name: Name of the tool to call.
            args: Arguments for the tool.
        
        Returns:
            Tool execution result as string.
        """
        # Ensure knowledge tools are initialized
        _ = self.knowledge_tools
        
        if tool_name not in self._tool_map:
            raise KeyError(tool_name)
        
        # Inject top_k into args for tools that support it
        if tool_name in ("query", "query_knowledge", "example", "get_code_example", 
                         "optimize", "suggest_optimization", "troubleshoot"):
            args = {**args, "top_k": self._mcp_top_k}
        
        tool_func = self._tool_map[tool_name]
        result = await tool_func(args)
        
        # Handle None result
        if result is None:
            logger.warning(f"RAG tool {tool_name} returned None")
            result = ""
        
        # Process result with RAG filter sub-agent for query/example/optimize tools
        if self.rag_subagent and tool_name in (
            "query", "query_knowledge", "example", "get_code_example",
            "optimize", "suggest_optimization", "troubleshoot"
        ):
            logger.info(f"Processing {tool_name} result with RAG filter sub-agent")
            original_query = args.get("topic") or args.get("category") or args.get("code_type") or args.get("error_message", "")
            result = self.rag_subagent.process(result, query=original_query)
        
        return result
    
    def get_template_vars(self) -> Dict[str, Any]:
        """Get template variables including RAG info."""
        base_vars = super().get_template_vars()
        base_vars["mcp_available"] = True
        base_vars["mcp_prefix"] = self.config.mcp_prefix
        return base_vars


# Convenience function for quick setup
def create_mcp_environment(**kwargs) -> MCPEnabledEnvironment:
    """Create a RAG-enabled environment with sensible defaults."""
    return MCPEnabledEnvironment(**kwargs)