# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Unified GPU kernel profiling MCP server.

Supports two backends:
- metrix: AMD Metrix API (structured JSON, bottleneck classification)
- rocprof-compute: rocprof-compute CLI (deep roofline + instruction mix analysis)

Both backends are exposed through a single `profile_kernel` tool with a `backend` parameter.
"""
