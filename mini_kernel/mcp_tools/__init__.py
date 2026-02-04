"""
MCP Tools for mini-kernel Agent

Core MCP Tools:
- metrix-mcp: Detailed hardware metrics via metrix (now at mcp_tools/metrix-mcp/)
- discovery: Automated test and benchmark discovery

Each tool is invoked on demand, based on agent state and optimization progress.
"""

# Import from new centralized location for backward compatibility
try:
    import sys
    from pathlib import Path
    # Add mcp_tools/metrix-mcp to path
    metrix_mcp_path = Path(__file__).parent.parent.parent / "mcp_tools" / "metrix-mcp" / "src"
    if str(metrix_mcp_path) not in sys.path:
        sys.path.insert(0, str(metrix_mcp_path))
    from metrix_mcp import MetrixTool
except ImportError:
    # Fallback to local copy if new location not available
    from .metrix import MetrixTool

from .discovery import DiscoveryPipeline, discover

__all__ = [
    "MetrixTool",
    "DiscoveryPipeline",
    "discover",
]
