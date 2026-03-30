import json
from pathlib import Path
from typing import Any

from minisweagent.tools.bash_command import BashCommand
from minisweagent.tools.save_and_test import SaveAndTestTool
from minisweagent.tools.str_replace_editor import str_replace_editor
from minisweagent.tools.strategy_manager import StrategyManagerTool
from minisweagent.tools.submit import SubmitTool

json_path = Path(__file__).parent / "tools.json"
with open(json_path, encoding="utf-8") as f:
    _all_tools = json.load(f)

try:
    from minisweagent.tools.mcp_bridge import collect_mcp_tools

    _mcp_bridges, _mcp_tools = collect_mcp_tools()
    _all_tools.extend(_mcp_tools)
except Exception:
    _mcp_bridges, _mcp_tools = [], []

_TOOL_PROFILES: dict[str, set[str] | None] = {
    "full": None,
    "swe": {
        "bash",
        "str_replace_editor",
        "save_and_test",
        "submit",
        "profile_kernel",
        "baseline_metrics",
        "strategy_manager",
    },
}


def get_tools_list(use_strategy_manager: bool = False) -> list:
    """Get filtered tools list based on settings.

    Args:
        use_strategy_manager: If True, include strategy_manager tool. If False, exclude it.
    Returns:
        List of tool definitions for the API.
    """
    excluded = set()
    if not use_strategy_manager:
        excluded.add("strategy_manager")
    return [t for t in _all_tools if t["name"] not in excluded]


# Backward compatibility
tools_list = _all_tools


class ToolRuntime:
    def __init__(
        self,
        use_strategy_manager: bool = False,
        strategy_file: str = ".optimization_strategies.md",
        on_strategy_change=None,
        patch_output_dir: str | None = None,
        tool_profile: str = "full",
    ):
        self._tool_profile = tool_profile
        self._mcp_bridges: list = list(_mcp_bridges)
        allowed = _TOOL_PROFILES.get(tool_profile)

        self._tool_table = {
            "bash": BashCommand(),
            "str_replace_editor": str_replace_editor(),
            "save_and_test": SaveAndTestTool(),
            "submit": SubmitTool(),
        }

        if allowed is not None:
            if "baseline_metrics" in allowed:
                try:
                    from minisweagent.tools.baseline_metrics_tool import BaselineMetricsTool

                    self._tool_table["baseline_metrics"] = BaselineMetricsTool()
                except ImportError:
                    pass
            if use_strategy_manager and "strategy_manager" in allowed:
                self._tool_table["strategy_manager"] = StrategyManagerTool(
                    filepath=strategy_file, on_change_callback=on_strategy_change
                )
            if "profile_kernel" in allowed:
                self._register_profiler_mcp()
            self._sub_agent_tool = None
        else:
            if use_strategy_manager:
                self._tool_table["strategy_manager"] = StrategyManagerTool(
                    filepath=strategy_file, on_change_callback=on_strategy_change
                )

            try:
                from minisweagent.tools.baseline_metrics_tool import BaselineMetricsTool

                self._tool_table["baseline_metrics"] = BaselineMetricsTool()
            except ImportError:
                pass
            try:
                from minisweagent.tools.check_compat import CheckKernelCompatibilityTool

                self._tool_table["check_kernel_compatibility"] = CheckKernelCompatibilityTool()
            except ImportError:
                pass
            try:
                from minisweagent.tools.resolve_kernel_url import ResolveKernelUrlTool

                self._tool_table["resolve_kernel_url"] = ResolveKernelUrlTool()
            except ImportError:
                pass

            try:
                from minisweagent.tools.sub_agent_tool import SubAgentTool

                self._sub_agent_tool = SubAgentTool()
                self._tool_table["sub_agent"] = self._sub_agent_tool
            except ImportError:
                self._sub_agent_tool = None

            self._register_mcp_tools()

        self.use_strategy_manager = use_strategy_manager
        self._codebase_context: str | None = None

    def wrap_rag_tools_with_subagent(self) -> None:
        """Wrap RAG MCP tools with RAGFilterSubAgent for result filtering."""
        from minisweagent.mcp_integration.subagent import RAGFilterSubAgent, SubAgentConfig

        subagent = RAGFilterSubAgent(SubAgentConfig(enabled=True))

        def _wrap(tool_callable):
            def wrapper(**kwargs):
                result = tool_callable(**kwargs)
                output = result.get("output", "")
                if output and result.get("returncode") == 0:
                    query = kwargs.get("topic") or kwargs.get("code_type") or ""
                    result["output"] = subagent.process(output, query=query)
                return result
            return wrapper

        for name in list(self._tool_table):
            if name in ("query", "optimize"):
                self._tool_table[name] = _wrap(self._tool_table[name])

    def _register_profiler_mcp(self):
        """Register only the profiler-mcp tool."""
        for bridge in self._mcp_bridges:
            if bridge.server_name == "profiler-mcp":
                self._tool_table["profile_kernel"] = bridge.tool("profile_kernel")
                return

    def _register_mcp_tools(self):
        """Register all MCP server tools discovered at module level."""
        for bridge in self._mcp_bridges:
            _mcp_server_name = bridge.server_name
            for _mcp_tool in _mcp_tools:
                if f"[MCP: {_mcp_server_name}]" in _mcp_tool.get("description", ""):
                    base_name = (
                        _mcp_tool["name"].split("__")[0]
                        if f"__{_mcp_server_name}" in _mcp_tool["name"]
                        else _mcp_tool["name"]
                    )
                    self._tool_table[_mcp_tool["name"]] = bridge.tool(base_name)

    def set_env(self, env: dict[str, str]) -> None:
        """Propagate environment overrides (e.g. HIP_VISIBLE_DEVICES) to tools."""
        env = dict(env)
        bash = self._tool_table.get("bash")
        if bash is not None:
            bash._env_override = env
        for bridge in self._mcp_bridges:
            bridge.set_env(env)

    def set_cwd(self, cwd: str | None) -> None:
        """Propagate working directory to the bash tool so commands run in the correct worktree."""
        bash = self._tool_table.get("bash")
        if bash is not None:
            bash._cwd = cwd

    def set_codebase_context(self, context: str | None) -> None:
        """Store codebase context and propagate to SubAgentTool if present."""
        self._codebase_context = context
        if self._sub_agent_tool and context:
            self._sub_agent_tool._codebase_context = context

    def get_tools_schema(self) -> list[dict]:
        """Return JSON tool schemas for only the tools registered in _tool_table."""
        return [t for t in _all_tools if t["name"] in self._tool_table]

    def get_tools_list(self) -> list:
        """Get the tools list for API based on current settings."""
        return get_tools_list(self.use_strategy_manager)

    def disable_tools(self, names) -> None:
        """Disable tools by name (removes from schema + dispatch)."""
        if not names:
            return
        for n in list(names):
            self._tool_table.pop(n, None)

    def dispatch(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        tool_call format:
        {
            "name": "bash",
            "arguments": {...}
        }
        """
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if name not in self._tool_table:
            raise ValueError(f"Unknown tool: {name}")

        if name == "bash" and "command" not in args:
            args = {**args, "command": ""}

        return self._tool_table[name](**args)
