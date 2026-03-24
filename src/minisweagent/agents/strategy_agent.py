"""Strategy-based agent with optimization tool support.

This agent provides core strategy management functionality that can be used
with different communication backends (CLI, VS Code, etc.).

Note: Strategy operations are now handled via the `strategy_manager` tool.
This agent only provides the callback mechanism for UI notifications.
"""

import sys

from minisweagent.agents.interactive import InteractiveAgent


class StrategyAgent(InteractiveAgent):
    """Agent with optimization strategy management capabilities.
    
    This agent provides UI notification callbacks for strategy changes.
    Actual strategy operations are handled by the `strategy_manager` tool.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"[DEBUG] StrategyAgent initialized with strategy_file: {self.config.strategy_file_path}", file=sys.stderr)
        
        # Load initial strategy data if file exists
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
                "filePath": strategy_file
            }
            
            if strategy_list.baseline:
                result["baseline"] = {
                    "metrics": strategy_list.baseline.metrics,
                    "logFile": strategy_list.baseline.log_file
                }
            
            for idx, strategy in enumerate(strategy_list.strategies, start=1):
                result["strategies"].append({
                    "index": idx,
                    "name": strategy.name,
                    "status": strategy.status.value,
                    "description": strategy.description,
                    "priority": strategy.priority,
                    "expected": strategy.expected,
                    "target": strategy.target,
                    "result": strategy.result,
                    "details": strategy.details
                })
            
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
