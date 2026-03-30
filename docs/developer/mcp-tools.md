# Adding your own MCP server

This guide is for **contributors or integrators** who want the GEAK agent to call a **Model Context Protocol (MCP)** server you maintain. Once your server matches the layout below, GEAK will **discover it automatically** under `mcp_tools/` and expose its tools to the model the same way it does for `bash`, editors, and other tools.

## What you add

Create a **new immediate subdirectory** of **`mcp_tools/`** (repository root) with this layout:

```text
mcp_tools/<server-folder>/          # use hyphens, e.g. my-tools-mcp
  pyproject.toml                   # recommended: declare dependencies
  src/
    <package_name>/                # folder name with "-" → "_" e.g. my_tools_mcp
      server.py                    # required entrypoint
      __init__.py                  # optional
```

**Naming rule:** `<package_name>` must be `<server-folder>` with every hyphen replaced by an underscore (example: `profiler-mcp` → `profiler_mcp`).

**Discovery:** GEAK only picks up a folder if this file exists:

`mcp_tools/<server-folder>/src/<package_name>/server.py`

**How it is started:** From that server folder, GEAK runs:

`python3 -m <package_name>.server`

with `PYTHONPATH` including that folder’s `src/`. Your package must work when launched that way.

## Implementation (FastMCP)

GEAK expects servers implemented with **FastMCP**:

- Instantiate `FastMCP`, expose model-facing functions with **`@mcp.tool()`**.
- Use clear **docstrings** and typed parameters; they drive what the model sees and the tool JSON schema.
- Provide an entry in **`server.py`** so **`python3 -m <package_name>.server`** runs the server (typically `mcp.run()` in `main()`).

**In-repo examples:** `mcp_tools/profiler-mcp/`, `mcp_tools/metrix-mcp/`, and **`mcp_tools/README.md`** (detailed checklist and patterns — read that file when adding a new server).

**Dependencies:** Put them in your server’s **`pyproject.toml`** (or equivalent) so the subprocess environment can import your code. If the server fails to start or list tools, GEAK will **skip** it for that run rather than break the whole agent.

## After you add the folder

- Restart or rerun the agent. Valid servers under `mcp_tools/` are loaded automatically; you do not register them in YAML by name for discovery.
- In the tool list the model sees, descriptions from your MCP tools are suffixed with **`[MCP: <server-folder>]`** so humans can tell which server owns a tool. If two tools share the same name across servers, GEAK may rename one to stay unique.

To **exclude** a server from GEAK, remove it from `mcp_tools/`, or change the folder so it no longer matches the path rule above.

## Environment

If the run sets device-related environment variables for the agent, those overrides are generally **passed through** to MCP server processes as well. Do not hardcode GPU indices inside the server unless you intend to ignore the host run.

## Verify locally

From the repository root:

```bash
cd mcp_tools/<server-folder>
PYTHONPATH=src python3 -m <package_name>.server
```

Fix import and startup errors before expecting the agent to see your tools. For logging, look for issues under the **`minisweagent.tools.mcp_client`** and **`mcp_bridge`** loggers when a server fails to list tools.

## Native tools (alternative)

If you prefer a tool implemented **in-process** in Python (no MCP subprocess), you add a **native** tool: implement the callable, register it, and add a matching entry in **`src/minisweagent/tools/tools.json`** — see **`src/minisweagent/tools/tools_runtime.py`** for how built-ins are wired. Prefer **MCP** when the feature has heavy or optional dependencies so the main install stays smaller.
