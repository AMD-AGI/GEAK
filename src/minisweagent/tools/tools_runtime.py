import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from minisweagent.tools.bash_command import BashCommand
from minisweagent.tools.save_and_test import SaveAndTestTool
from minisweagent.tools.str_replace_editor import str_replace_editor
from minisweagent.tools.strategy_manager import StrategyManagerTool
from minisweagent.tools.submit import SubmitTool

_TOOLS_JSON_PATH = Path(__file__).parent / "tools.json"

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
        "query",
        "optimize",
    },
}


class ToolRuntime:
    @classmethod
    def load_tools_json(cls) -> list[dict[str, Any]]:
        """Load native tool definitions from ``tools.json`` (fresh list each call)."""
        with open(_TOOLS_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _prepare_mcp(cls) -> tuple[list[dict[str, Any]], list]:
        """Return ``(mcp_tool_schemas, dispatch_bridges)`` for one runtime.

        This still performs **two** bridge lifecycles on purpose:

        1. ``collect_mcp_tools`` — spawn bridges, run ``tools/list``, build OpenAI-style
           schemas, then drop those bridge objects (they are not safe to share across
           parallel agents / long-lived dispatch).
        2. ``_populate_mcp_bridges`` — fresh bridges used only for ``call_tool`` on this
           :class:`ToolRuntime` instance (stdio + asyncio isolation).
        """
        schemas: list[dict[str, Any]] = []
        try:
            from minisweagent.tools.mcp_bridge import collect_mcp_tools

            boot_bridges, mcp_tools = collect_mcp_tools()
            del boot_bridges
            schemas = list(mcp_tools)
        except Exception as exc:
            logger.warning("MCP tool schema discovery failed; MCP tools will be unavailable: %s", exc)

        try:
            from minisweagent.tools.mcp_bridge import _populate_mcp_bridges

            bridges = _populate_mcp_bridges()
        except Exception as exc:
            logger.warning("MCP bridge creation failed; MCP dispatch will be unavailable: %s", exc)
            bridges = []

        return schemas, bridges

    @classmethod
    def fetch_tools_list(
        cls,
        use_strategy_manager: bool = True,
        tool_profile: str = "full",
    ) -> list:
        """Return tool API definitions without keeping a runtime (pays full MCP discovery).

        Prefer :meth:`get_tools_list` when you already have an instance.
        """
        return cls(
            use_strategy_manager=use_strategy_manager,
            tool_profile=tool_profile,
        ).get_tools_list()

    def __init__(
        self,
        use_strategy_manager: bool = True,
        strategy_file: str = ".optimization_strategies.md",
        on_strategy_change=None,
        patch_output_dir: str | None = None,
        tool_profile: str = "full",
    ):
        json_tools = copy.deepcopy(self.load_tools_json())
        mcp_schema_tools, self._mcp_bridges = self._prepare_mcp()
        self.tools_list: list[dict[str, Any]] = json_tools + mcp_schema_tools

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
            if "query" in allowed or "optimize" in allowed:
                self._register_rag_mcp(allowed)
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

    def wrap_rag_tools_with_postprocessor(self, api_key: str | None = None) -> None:
        """Wrap RAG MCP tools with RAGPostProcessor for result filtering."""
        from minisweagent.tools.rag_postprocessor import RAGPostProcessor, RAGPostProcessorConfig

        postprocessor = RAGPostProcessor(RAGPostProcessorConfig(enabled=True, api_key=api_key))

        def _wrap(tool_callable):
            def wrapper(**kwargs):
                result = tool_callable(**kwargs)
                output = result.get("output", "")
                if output and result.get("returncode") == 0:
                    query = kwargs.get("topic") or kwargs.get("code_type") or ""
                    result["output"] = postprocessor.process(output, query=query)
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

    def _register_rag_mcp(self, allowed: set[str] | None = None):
        """Register RAG MCP tools (query, optimize) from rag-mcp server."""
        for bridge in self._mcp_bridges:
            if bridge.server_name == "rag-mcp":
                for tool_name in ("query", "optimize"):
                    if allowed is None or tool_name in allowed:
                        self._tool_table[tool_name] = bridge.tool(tool_name)
                return

    def _register_mcp_tools(self):
        """Register all MCP server tools whose schemas are on ``self.tools_list``."""
        for bridge in self._mcp_bridges:
            server_name = bridge.server_name
            tag = f"[MCP: {server_name}]"
            for mcp_tool in self.tools_list:
                if tag not in (mcp_tool.get("description") or ""):
                    continue
                raw_name = mcp_tool.get("name")
                if not isinstance(raw_name, str):
                    continue
                base_name = raw_name.split("__")[0] if f"__{server_name}" in raw_name else raw_name
                self._tool_table[raw_name] = bridge.tool(base_name)

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

    def get_tools_list(self) -> list[dict]:
        """OpenAI-style tool definitions for this runtime.

        Includes only tools present in ``_tool_table`` (so names match
        :meth:`dispatch`), and omits ``strategy_manager`` when
        ``use_strategy_manager`` is false.
        """
        excluded: set[str] = set()
        if not self.use_strategy_manager:
            excluded.add("strategy_manager")
        table = self._tool_table
        return [t for t in self.tools_list if t["name"] in table and t["name"] not in excluded]

    def disable_tools(self, names) -> None:
        """Disable tools by name (removes from schema + dispatch + tools_list)."""
        if not names:
            return
        for n in list(names):
            self._tool_table.pop(n, None)
        ban = set(names)
        self.tools_list = [t for t in self.tools_list if t.get("name") not in ban]

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
