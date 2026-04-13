# MCP servers in GEAK

This directory is where you add **Model Context Protocol** servers that the GEAK agent can call as ordinary tools. Each subdirectory here is one server package. The main application discovers those packages automatically, starts them as subprocesses, and merges their tool definitions into the agent’s tool list.

## Installation

`fastmcp` and `mcp[cli]` are core dependencies — installed automatically with `pip install -e .`.

From the GEAK repository root, install the main package with:

```bash
pip install -e .          # runtime install
# or
pip install -e '.[full]'  # runtime + dev/langchain extras
```

GEAK launches the shipped MCP servers in this directory directly from the repository by setting each server's `src/` directory on `PYTHONPATH`, so there is no separate `.[mcp]` extra to install.

If you want to work on one server package in isolation, install that package directly, for example:

```bash
pip install -e mcp_tools/profiler-mcp
```

## Directory and module naming

Use a **hyphenated** folder name for the server (for example, `profiler-mcp`). GEAK maps that name to a **Python package** by replacing hyphens with underscores (`profiler_mcp`).

Expected layout:

```text
<server-folder>/          # e.g. profiler-mcp
  pyproject.toml          # optional but recommended for dependencies
  src/
    <package_name>/       # e.g. profiler_mcp — must match folder name with - → _
      __init__.py         # optional
      server.py           # required entrypoint
```

Discovery only considers a folder if this file exists:

`<server-folder>/src/<package_name>/server.py`

The runtime launches the server with:

`python3 -m <package_name>.server`

using the server folder as the working directory and `src` on `PYTHONPATH`, so imports and `-m` execution must work from that layout.

## Implementing `server.py` with FastMCP

GEAK expects servers built with **FastMCP**. Create a `FastMCP` instance, then expose every capability the model should call with the **`@mcp.tool()`** decorator.

**Recommended patterns** (see the in-tree `profiler-mcp` package for a full example):

- Keep **helpers** (normalization, subprocess calls, backend selection) as plain functions or small modules; keep **`@mcp.tool()`** functions as the **public MCP surface** the protocol advertises.
- Give each tool a clear docstring: it becomes part of what the model sees, alongside the JSON schema derived from the function signature.
- Return structured data (for example `dict` with `success`, `error`, and result fields) so callers can parse outcomes reliably.
- Log important steps at INFO; on failure, log exceptions and return a structured error instead of letting the process exit uncleanly.
- Provide a **`main()`** (or equivalent) that runs `mcp.run()`, and support `python -m <package>.server` so the launcher above works.

Avoid registering duplicate or “internal” operations as separate MCP tools unless the model truly needs them; prefer one well-documented tool that delegates to helpers.

## How GEAK wires your server in

1. **`collect_mcp_tools`** (in `src/minisweagent/tools/mcp_bridge.py`) walks this directory, builds one **`MCPToolBridge`** per valid server, queries each server’s **`tools/list`**, and produces:
   - **`mcp_bridges`**: bridge objects whose `.tool(<name>)` returns a synchronous callable for each MCP tool name.
   - **`mcp_tool_list`**: OpenAI-style tool entries (name, description, parameters) merged into the agent’s schema. Descriptions are tagged with **`[MCP: <server-folder>]`** so tooling can tell which server owns a tool.

2. **`tools_runtime.py`** imports that collector at load time, extends the global tools list with **`mcp_tool_list`**, and **`ToolRuntime`** registers the corresponding callables in its dispatch table so the LLM can invoke them like any other tool.

Environment overrides (for example GPU device variables) set on **`ToolRuntime.set_env`** are forwarded to each bridge so subprocess servers see the same environment policy as shell tools where applicable.

## Controlling which servers the agent uses

Only folders under **`mcp_tools`** that match the layout above are discovered. The practical way to **exclude** a server is to not place it here, or to remove or rename its folder so it no longer matches the discovery rule.

If you need **per-server enable/disable** without moving directories, the right place to implement that is **`ToolRuntime` initialization** in your application (for example an allowlist, a blocklist, or a boolean flag interpreted by your fork). Until such options exist in your tree, discovery follows “everything valid under `mcp_tools`.”

## Checklist for a new server

- [ ] Folder name uses hyphens; under `src/`, the package directory uses underscores and matches the folder name rule.
- [ ] `server.py` lives at `src/<package>/server.py` and runs under `python3 -m <package>.server`.
- [ ] FastMCP is used; every model-facing operation is a **`@mcp.tool()`** function with a helpful docstring and sensible parameters.
- [ ] Dependencies are declared (for example in `pyproject.toml`) so the subprocess can import your code.
- [ ] You validated behavior locally; compare structure and style to **`profiler-mcp`** when in doubt.
