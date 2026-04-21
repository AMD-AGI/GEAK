#!/bin/bash
# Start RAG Knowledge Base MCP Server
cd "$(dirname "$0")"
python3 -m rag_mcp.server
