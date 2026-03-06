"""Unified GPU kernel profiling MCP server.

Supports two backends:
- metrix: AMD Metrix API (structured JSON, bottleneck classification)
- rocprof-compute: rocprof-compute CLI (deep roofline + instruction mix analysis)

Both backends are exposed through a single `profile_kernel` tool with a `backend` parameter.
"""
