"""
MCP Tools for mini-kernel Agent

Core MCP Tools:
- metrix-mcp: Detailed hardware metrics via metrix

Each tool is invoked on demand, based on agent state and optimization progress.
"""

from .metrix import MetrixTool

__all__ = [
    "MetrixTool",
]
