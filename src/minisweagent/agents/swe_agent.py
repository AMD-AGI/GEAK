"""SWE-agent: codes manually with bash, editor, test, profiler, and strategy manager.

Unlike StrategyInteractiveAgent which has access to LLM-powered MCP tools
(kernel-evolve, kernel-ercs), the SWE agent relies on its own reasoning to
read code, make edits, test correctness, and profile performance.

Tool set: bash, str_replace_editor, save_and_test, submit, profile_kernel,
baseline_metrics, strategy_manager.
"""

from minisweagent import Environment, Model
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.tools.tools_runtime import ToolRuntime


class SweAgent(InteractiveAgent):
    """SWE-agent for manual kernel optimization.

    Operates with a reduced tool set (no kernel-evolve / kernel-ercs MCP
    tools).  Best for targeted edits, autotune configs, and straightforward
    optimizations where the agent should read-think-edit-test-profile on
    its own.
    """

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, **kwargs)

        # Replace the full ToolRuntime created by DefaultAgent with a
        # restricted one that only registers SWE-relevant tools.
        self.toolruntime = ToolRuntime(
            use_strategy_manager=self.config.use_strategy_manager,
            strategy_file=self._get_strategy_file()
            if self.config.use_strategy_manager
            else ".optimization_strategies.md",
            on_strategy_change=self._get_strategy_callback(),
            patch_output_dir=self.config.patch_output_dir,
            tool_profile="swe",
        )
        self._setup_save_and_test_context()

        # Override model tools so the LLM only sees dispatchable tools
        if hasattr(self.model, "set_tools"):
            self.model.set_tools(self.toolruntime.get_tools_schema())
        else:
            model_impl = getattr(self.model, "_impl", self.model)
            model_impl.tools = self.toolruntime.get_tools_schema()
