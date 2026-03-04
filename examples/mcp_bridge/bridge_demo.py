# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

#!/usr/bin/env python3
"""Example: Use MCPToolBridge to call an MCP server tool programmatically.

Usage:
    python examples/mcp_bridge/bridge_demo.py
"""

from minisweagent.tools.mcp_bridge import MCPToolBridge


def main():
    # Create a bridge to the profiler MCP server
    bridge = MCPToolBridge("profiler-mcp", timeout=60)

    # Get a bound tool
    profile = bridge.tool("profile_kernel")
    print(f"Created tool: {profile}")

    # Call it (will start the MCP server on first call)
    print("\nCalling profile_kernel with a simple command...")
    result = profile(command="echo hello", backend="metrix")
    print(f"Return code: {result['returncode']}")
    print(f"Output: {result['output'][:200]}")


if __name__ == "__main__":
    main()
