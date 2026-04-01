# GEAK MCP Server
"""
MCP (Model Context Protocol) server implementation for GEAK.

This module provides MCP tools for AI agents to:
- Manage user model configuration
- Create optimization tasks
- Submit tasks for execution
- Check task status
- Download optimization results

Usage:
    # Start MCP server with HTTP transport
    python -m server.mcp.server
"""

from server.mcp.server import create_mcp_server

__all__ = ["create_mcp_server"]
