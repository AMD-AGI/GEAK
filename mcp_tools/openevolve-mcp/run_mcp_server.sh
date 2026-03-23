#!/bin/bash
# Start OpenEvolve MCP Server
cd "$(dirname "$0")"
python3 -m openevolve_mcp.server
