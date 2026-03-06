# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Automated Test Discovery MCP Server

Single-tool MCP for discovering tests and benchmarks for GPU kernels.
No configuration needed - uses content-based detection.

Tool:
- discover: Find tests and benchmarks for a kernel file
"""

from .server import mcp

__all__ = ["mcp"]
