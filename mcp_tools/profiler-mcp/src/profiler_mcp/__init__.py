"""Unified GPU kernel profiling MCP server.

Supports two backends:
- metrix: AMD Metrix API (structured JSON, bottleneck classification)
- rocprof-compute: rocprof-compute CLI (deep roofline + instruction mix analysis)

Both backends are exposed through a single `profile_kernel` tool with a `backend` parameter.

``MetrixTool`` is defined in ``profiler_mcp.core``. The ``metrix`` PyPI package is a
required dependency of ``profiler-mcp``.
"""
