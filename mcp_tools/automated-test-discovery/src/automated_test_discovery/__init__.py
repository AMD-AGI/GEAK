"""
Automated Test Discovery MCP Server

Single-tool MCP for discovering tests and benchmarks for GPU kernels.
No configuration needed - uses content-based detection.

Tool:
- discover: Find tests and benchmarks for a kernel file
"""

from .server import mcp

__all__ = ["mcp"]
