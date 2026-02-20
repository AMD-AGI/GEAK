"""MCPToolBridge -- expose MCP server tools as sync callables for ToolRuntime.

One bridge instance per MCP server. Each bridge can expose multiple tools
via the `.tool(name)` factory. Tools are registered in ToolRuntime._tool_table
and dispatched like native tools.

Usage:
    profiler = MCPToolBridge("profiler-mcp", server_config={...}, timeout=300)
    tool_table["profile_kernel"] = profiler.tool("profile_kernel")

    # Later, ToolRuntime.dispatch calls:
    tool_table["profile_kernel"](command="python3 kernel.py", backend="metrix")
    # Returns: {"output": "...", "returncode": 0}
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MCPToolBridge:
    """Wraps an MCP server so its tools can be called synchronously.

    - Lazy start: subprocess is spawned on the first tool call.
    - Configurable timeout per call (default 300s / 5 min).
    - All exceptions are caught and returned as ``{output, returncode}``.
    - ``.tool(name)`` returns a callable bound to one MCP tool name.

    Internally, each bridge maintains a **persistent background event loop**
    in a daemon thread.  The ``MCPClient`` (subprocess + stdio pipes) lives
    on that loop for its entire lifetime, avoiding the "Future attached to a
    different loop" error that occurs when ``asyncio.run()`` creates and
    destroys a new loop on every call.
    """

    def __init__(
        self,
        server_name: str,
        server_config: dict[str, Any] | None = None,
        timeout: float = 300,
    ):
        self.server_name = server_name
        self.server_config = server_config or self._default_config(server_name)
        self.timeout = timeout
        self._client = None

        # Persistent event loop -- created lazily by _get_loop().
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_env(self, env: dict[str, str]) -> None:
        """Merge extra env vars into server config (must be called before first tool call)."""
        self.server_config.setdefault("env", {}).update(env)

    def tool(self, tool_name: str) -> _BoundTool:
        """Return a callable that invokes *tool_name* on this MCP server."""
        return _BoundTool(bridge=self, tool_name=tool_name)

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call an MCP tool synchronously. Returns ``{output, returncode}``."""
        try:
            raw = self._run_async(self._async_call(tool_name, arguments or {}))
            return self._format_result(raw)
        except Exception as e:
            logger.error(f"MCPToolBridge({self.server_name}).{tool_name} failed: {e}")
            return {"output": f"MCP tool error: {e}", "returncode": 1}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _async_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        client = await self._ensure_client()
        return await asyncio.wait_for(
            client.call_tool(tool_name, arguments),
            timeout=self.timeout,
        )

    async def _ensure_client(self):
        if self._client is None:
            # Import here to avoid hard dependency at module level
            _ensure_mcp_client_importable()
            from mcp_client import MCPClient

            self._client = MCPClient(self.server_name, self.server_config)
            await self._client.start()
            logger.info(f"MCPToolBridge: started {self.server_name}")
        return self._client

    # ------------------------------------------------------------------
    # Persistent background event loop
    # ------------------------------------------------------------------

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return (and lazily create) a persistent event loop on a daemon thread.

        The loop stays alive for the lifetime of this bridge instance so that
        the ``MCPClient`` subprocess and its asyncio pipes remain valid across
        multiple ``call_tool`` invocations.
        """
        if self._loop is not None and not self._loop.is_closed():
            return self._loop

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name=f"mcp-loop-{self.server_name}",
            daemon=True,
        )
        self._loop_thread.start()

        # Best-effort cleanup at interpreter shutdown
        atexit.register(self._shutdown_loop)

        return self._loop

    def _shutdown_loop(self):
        """Stop the background loop (called at exit or manually)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)

    def _run_async(self, coro):
        """Schedule *coro* on the persistent background loop and block until done.

        This replaces the old approach of calling ``asyncio.run()`` (which
        creates and destroys a new loop each time, invalidating cached
        MCPClient subprocess handles).
        """
        loop = self._get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=self.timeout + 10)

    @staticmethod
    def _format_result(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert MCP result to the ``{output, returncode}`` format ToolRuntime expects."""
        if raw.get("isError"):
            content = raw.get("content", [])
            text = content[0].get("text", str(content)) if content else str(raw)
            return {"output": text, "returncode": 1}

        content = raw.get("content", [])
        if content and isinstance(content, list):
            # MCP returns list of content blocks; join text blocks
            parts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            text = "\n".join(parts) if parts else str(content)
        else:
            text = str(raw)
        return {"output": text, "returncode": 0}

    @staticmethod
    def _default_config(server_name: str) -> dict[str, Any]:
        """Build default server config from well-known MCP server locations."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        mcp_dir = repo_root / "mcp_tools" / server_name

        if not mcp_dir.exists():
            raise FileNotFoundError(f"MCP server directory not found: {mcp_dir}. Provide explicit server_config.")

        # Derive the Python module name from the directory name (e.g., profiler-mcp -> profiler_mcp)
        module_name = server_name.replace("-", "_")
        src_dir = mcp_dir / "src"

        return {
            "command": ["python3", "-m", f"{module_name}.server"],
            "cwd": str(mcp_dir),
            "env": {"PYTHONPATH": str(src_dir)},
        }


class _BoundTool:
    """A callable that invokes a specific tool on an MCPToolBridge."""

    def __init__(self, bridge: MCPToolBridge, tool_name: str):
        self._bridge = bridge
        self._tool_name = tool_name

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return self._bridge.call_tool(self._tool_name, kwargs)

    def __repr__(self) -> str:
        return f"MCPTool({self._bridge.server_name}::{self._tool_name})"


def _ensure_mcp_client_importable():
    """Add mcp-client to sys.path if needed."""
    try:
        import mcp_client  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        client_src = str(repo_root / "mcp_tools" / "mcp-client" / "src")
        if client_src not in sys.path:
            sys.path.insert(0, client_src)
