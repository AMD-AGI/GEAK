"""Agent that maintains a structured optimization strategy list while working.

During a kernel optimization run the agent keeps a strategy list -- a
structured markdown file that tracks what approaches it plans to try,
which ones are in progress, and what results each produced.  The file is
managed through the ``strategy_manager`` tool so the format stays
consistent.

On top of the base InteractiveAgent this class adds:

1. **UI callbacks** -- every time the strategy list changes, the agent
   formats the update and calls ``notify_strategy_changed()``.
   Subclasses (e.g. StrategyInteractiveAgent for the CLI) override that
   method to display the update in their UI.

2. **Configurable tool profile** -- the ``tool_profile`` config field
   (default ``"swe"``) controls which tools are available to the agent.
   ``"swe"`` gives a focused set (bash, editor, save_and_test, submit,
   profiler, baseline_metrics, strategy_manager).  ``"full"`` adds all
   registered MCP tool bridges on top of that.
"""

import sys

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.tools.tools_runtime import ToolRuntime


class StrategyAgent(InteractiveAgent):
    """InteractiveAgent with a strategy list and a selectable tool profile.

    The strategy list is a structured plan the agent maintains as it
    optimizes a kernel.  Each entry has a name, status (planned / in
    progress / done), priority, and result.  The ``strategy_manager``
    tool handles all mutations; this class just wires up the change
    callback so a UI can react.

    The ``tool_profile`` config field controls which tools the agent
    sees.  Pass ``"swe"`` (the default) for the core set, or ``"full"``
    to include additional MCP-backed tools.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Recreate ToolRuntime with the configured tool_profile (default "swe").
        # DefaultAgent.__init__ already created one with profile="full";
        # we replace it here so the LLM only sees the intended tool set.
        self.toolruntime = ToolRuntime(
            use_strategy_manager=self.config.use_strategy_manager,
            strategy_file=self._get_strategy_file()
            if self.config.use_strategy_manager
            else ".optimization_strategies.md",
            on_strategy_change=self._get_strategy_callback(),
            patch_output_dir=self.config.patch_output_dir,
            tool_profile=self.config.tool_profile,
        )
        self._setup_save_and_test_context()

        # Override model tools so the LLM only sees dispatchable tools
        if hasattr(self.model, "set_tools"):
            self.model.set_tools(self.toolruntime.get_tools_schema())
        else:
            model_impl = getattr(self.model, "_impl", self.model)
            model_impl.tools = self.toolruntime.get_tools_schema()

        print(
            f"[DEBUG] StrategyAgent initialized (profile={self.config.tool_profile}, "
            f"strategy_file={self.config.strategy_file_path})",
            file=sys.stderr,
        )

        self._send_initial_strategy_data()

    def _send_initial_strategy_data(self):
        """Send initial strategy data to UI if file exists."""
        from minisweagent.tools.strategy_manager import StrategyManager

        strategy_file = self._get_strategy_file()
        manager = StrategyManager(filepath=strategy_file)

        if manager.exists():
            try:
                strategy_list = manager.load()
                self._on_strategy_changed(strategy_list)
            except Exception as e:
                print(f"[WARNING] Failed to load initial strategy data: {e}", file=sys.stderr)

    def _get_strategy_callback(self):
        """Override to provide callback for strategy tool to notify UI."""
        return self._on_strategy_changed

    def _on_strategy_changed(self, strategy_list):
        """Callback when strategy list changes. Formats and sends to UI."""
        try:
            strategy_file = self._get_strategy_file()

            result = {
                "exists": True,
                "strategies": [],
                "baseline": None,
                "notes": strategy_list.notes,
                "filePath": strategy_file,
            }

            if strategy_list.baseline:
                result["baseline"] = {
                    "metrics": strategy_list.baseline.metrics,
                    "logFile": strategy_list.baseline.log_file,
                }

            for idx, strategy in enumerate(strategy_list.strategies, start=1):
                result["strategies"].append(
                    {
                        "index": idx,
                        "name": strategy.name,
                        "status": strategy.status.value,
                        "description": strategy.description,
                        "priority": strategy.priority,
                        "expected": strategy.expected,
                        "target": strategy.target,
                        "result": strategy.result,
                        "details": strategy.details,
                    }
                )

            self.notify_strategy_changed(result)
            print(f"[DEBUG] Strategy data updated: {len(result['strategies'])} strategies", file=sys.stderr)

        except Exception as e:
            print(f"[ERROR] Failed to process strategy data: {e}", file=sys.stderr)

    def notify_strategy_changed(self, strategy_data: dict):
        """Notify UI that strategy list has changed.

        Args:
            strategy_data: Dictionary containing strategy information

        Default implementation does nothing. Subclasses (VSCodeStrategyAgent,
        StrategyInteractiveAgent) override this to provide UI-specific behavior.
        """
        pass
