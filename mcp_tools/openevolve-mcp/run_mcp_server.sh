# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

#!/bin/bash
# Start OpenEvolve MCP Server
cd "$(dirname "$0")"
python3 -m openevolve_mcp.server
