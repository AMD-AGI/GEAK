# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""
MCP Protocol Client for GEAK Agent.

Provides proper JSON-RPC 2.0 communication with MCP servers.
"""

from .client import MCPClient
from .config import MCP_SERVERS, get_server_config

__version__ = "0.1.0"
__all__ = ["MCPClient", "MCP_SERVERS", "get_server_config"]
