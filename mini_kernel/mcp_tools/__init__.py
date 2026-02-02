"""
MCP Tools for mini-kernel Agent

Core MCP Tools:
- metrix-mcp: Detailed hardware metrics via metrix
- discovery: Automated test and benchmark discovery

Each tool is invoked on demand, based on agent state and optimization progress.
"""

from .metrix import MetrixTool
from .discovery import DiscoveryPipeline, discover

__all__ = [
    "MetrixTool",
    "DiscoveryPipeline",
    "discover",
]
