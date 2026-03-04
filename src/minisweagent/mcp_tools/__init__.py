# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""MCP tools package.

MetrixTool is canonical in mcp_tools/metrix-mcp/src/metrix_mcp/core.py.
This module re-exports it for backward compatibility.
"""

try:
    from metrix_mcp.core import MetrixTool
except ImportError:
    MetrixTool = None  # metrix-mcp not installed

__all__ = ["MetrixTool"]
