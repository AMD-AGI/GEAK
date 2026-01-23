"""Strategy-based agent with optimization tool support.

This agent provides core strategy management functionality that can be used
with different communication backends (CLI, VS Code, etc.).
"""

import re
import shlex
import sys
import traceback
from pathlib import Path

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig, NonTerminatingException


class StrategyAgent(InteractiveAgent):
    """Agent with optimization strategy management capabilities.
    
    This is an abstract base class that defines the strategy management logic
    while allowing subclasses to implement their own communication mechanisms.
    """

    def __init__(self, *args, strategy_file_path: str = ".optimization_strategies.md", **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_file_path = strategy_file_path
        self._strategy_manager = None  # Lazy initialization
        # patch-related attributes (patch_counter, patch_results, log_file, test_command, etc.) 
        # are now inherited from DefaultAgent
        
        print(f"[DEBUG] StrategyAgent initialized with strategy_file_path: {strategy_file_path}", file=sys.stderr)

    # ============ Abstract Communication Interface ============
    # Subclasses must implement these methods for their specific communication mechanism
    
    def notify_strategy_changed(self, strategy_data: dict):
        """Notify UI that strategy list has changed.
        
        Args:
            strategy_data: Dictionary containing strategy information
        
        Default implementation does nothing (CLI might just log).
        """
        pass
    
    # ============ Core Strategy Management Logic ============
    
    def execute_action(self, action: dict) -> dict:
        """Override to intercept special commands like optool."""
        command = action.get("action", "")
        command_stripped = command.strip()
        
        print(f"[DEBUG] execute_action called", file=sys.stderr)
        print(f"[DEBUG] command: {repr(command[:100])}", file=sys.stderr)
        
        # Check for optimization tool command
        # Match patterns like: "optool ...", "cd x && optool ...", "cmd; optool ...", etc.
        optool_pattern = r'(^|&&|\|\||;)\s*optool(\s|$)'
        has_optool = re.search(optool_pattern, command_stripped)
        
        if has_optool:
            # Extract the optool command part
            match_start = has_optool.start()
            optool_start = command_stripped.find('optool', match_start)
            
            # Extract from optool to the end (or until next &&, ||, ;)
            remainder = command_stripped[optool_start:]
            next_separator = re.search(r'(&&|\|\||;)', remainder)
            if next_separator:
                optool_cmd = remainder[:next_separator.start()].strip()
            else:
                optool_cmd = remainder.strip()
            
            print(f"[DEBUG] Intercepted optool command: {optool_cmd}", file=sys.stderr)
            
            # Ask for confirmation before executing (if needed)
            if self.should_ask_confirmation(command):
                self.ask_confirmation()
            
            try:
                result = self._handle_optool_command(optool_cmd)
                print(f"[DEBUG] optool result: {result[:200] if len(result) > 200 else result}", file=sys.stderr)
                return {"output": result, "returncode": 0}
            except Exception as e:
                error_msg = f"optool command failed: {e}"
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                return {"output": error_msg, "returncode": 1}
        
        # Execute the action normally (save_patch functionality is handled by DefaultAgent)
        return super().execute_action(action)
    
    def _get_strategy_manager(self):
        """Get or create StrategyManager with callback (lazy initialization)."""
        if self._strategy_manager is None:
            from minisweagent.tools.strategy_manager import StrategyManager
            
            # Get working directory from env config or use current directory
            import os
            cwd = self.env.config.cwd or os.getcwd()
            strategy_file = Path(cwd) / self.strategy_file_path
            
            print(f"[DEBUG] Creating StrategyManager with file: {strategy_file}", file=sys.stderr)
            
            self._strategy_manager = StrategyManager(
                filepath=strategy_file,
                on_change_callback=self._on_strategy_changed
            )
            
            # Send initial data if file exists
            if self._strategy_manager.exists():
                try:
                    strategy_list = self._strategy_manager.load()
                    self._on_strategy_changed(strategy_list)
                except Exception as e:
                    print(f"[WARNING] Failed to load initial strategy data: {e}", file=sys.stderr)
        
        return self._strategy_manager
    
    def _on_strategy_changed(self, strategy_list):
        """Callback when strategy list changes."""
        try:
            manager = self._get_strategy_manager()
            
            result = {
                "exists": True,
                "strategies": [],
                "baseline": None,
                "notes": strategy_list.notes,
                "filePath": str(manager.filepath)
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
                    "priority": strategy.priority,  # 添加 priority 字段
                    "expected": strategy.expected,
                    "target": strategy.target,
                    "result": strategy.result,
                    "details": strategy.details
                })
            
            self.notify_strategy_changed(result)
            print(f"[DEBUG] Strategy data updated: {len(result['strategies'])} strategies", file=sys.stderr)
            
        except Exception as e:
            print(f"[ERROR] Failed to process strategy data: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    
    def _handle_optool_command(self, command: str) -> str:
        """Handle optimization tool commands."""
        parts = shlex.split(command)
        if len(parts) < 2:
            return self._optool_help()
        
        subcommand = parts[1]
        manager = self._get_strategy_manager()
        
        if subcommand == "create":
            return self._optool_create(manager, parts[2:])
        elif subcommand == "add":
            return self._optool_add(manager, parts[2:])
        elif subcommand == "mark":
            return self._optool_mark(manager, parts[2:])
        elif subcommand == "update":
            return self._optool_update(manager, parts[2:])
        elif subcommand == "remove":
            return self._optool_remove(manager, parts[2:])
        elif subcommand == "show" or subcommand == "list":
            return self._optool_show(manager, parts[2:])
        elif subcommand == "note":
            return self._optool_note(manager, parts[2:])
        elif subcommand == "priority":
            return self._optool_priority(manager, parts[2:])
        elif subcommand == "path":
            return self._optool_path(manager)
        elif subcommand == "next":
            return self._optool_next(manager)
        elif subcommand == "help" or subcommand == "--help":
            return self._optool_help()
        else:
            return f"Unknown subcommand: {subcommand}\nRun 'optool help' for usage."
    
    def _optool_help(self) -> str:
        """Return help message for optool."""
        return """Optimization Strategy Tool (optool)

Usage: optool <subcommand> [arguments]

Subcommands:
  create --baseline-metrics <metrics> [--strategies <strategies>]
      Create a new strategy list file
      Example: optool create --baseline-metrics "Bandwidth:152.3 GB/s" --strategies "Memory Coalescing|Optimize memory access|+20-30%|Memory-bound"
  
  add <name> <description> <expected> [--target <target>]
      Add a new optimization strategy
      Example: optool add "Loop Unrolling" "Unroll loops" "10-15%" --target "process_data"
  
  mark <index> <status> [--result <result>] [--details <details>]
      Mark strategy status (pending/exploring/successful/failed/partial/skipped)
      Example: optool mark 1 exploring
  
  update <index> [--status <status>] [--result <result>] [--details <details>]
      Update strategy fields
      Example: optool update 1 --result "15% improvement" --details "Works well"
  
  remove <index> [--method skip|delete]
      Remove or skip a strategy (default: skip)
      Example: optool remove 2 --method skip
  
  show [summary|index]
      Show full content (default), summary statistics, or a specific strategy
      Example: optool show           # Show full content
      Example: optool show summary   # Show statistics
      Example: optool show 1         # Show strategy 1
  
  next
      Get the next recommended strategy to explore
      Automatically prioritizes user-marked high-priority strategies
      Example: optool next
  
  note <text>
      Add a note to the strategy list
      Example: optool note "Combined strategies 1 and 3 work well together"
  
  priority <index> <high|normal>
      Set strategy priority (high or normal)
      Example: optool priority 1 high
  
  path
      Show current strategy file path
  
  help
      Show this help message
"""
    
    def _optool_create(self, manager, args: list[str]) -> str:
        """Create a new strategy list file."""
        from minisweagent.tools.strategy_manager import Baseline, Strategy, StrategyStatus
        
        baseline_metrics = []
        baseline_log = None
        strategies_raw = []
        
        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == "--baseline-metrics" and i + 1 < len(args):
                baseline_metrics.append(args[i + 1])
                i += 2
            elif args[i] == "--baseline-log" and i + 1 < len(args):
                baseline_log = args[i + 1]
                i += 2
            elif args[i] == "--strategies" and i + 1 < len(args):
                strategies_raw.append(args[i + 1])
                i += 2
            else:
                i += 1
        
        if not baseline_metrics:
            return "Error: create requires at least one --baseline-metrics\nExample: optool create --baseline-metrics \"Bandwidth:152.3 GB/s\""
        
        # Parse baseline metrics
        metrics = {}
        for metric in baseline_metrics:
            if ":" not in metric:
                return f"Error: Invalid metric format '{metric}', expected 'Key:Value'"
            key, value = metric.split(":", 1)
            metrics[key.strip()] = value.strip()
        
        baseline = Baseline(metrics=metrics, log_file=baseline_log)
        
        # Parse strategies
        strategies = []
        for s in strategies_raw:
            parts = s.split("|")
            if len(parts) < 3:
                return f"Error: Invalid strategy format '{s}', expected 'Name|Description|Expected|Target'"
            
            name = parts[0].strip()
            description = parts[1].strip()
            expected = parts[2].strip()
            target = parts[3].strip() if len(parts) > 3 else None
            
            strategies.append(Strategy(
                name=name,
                status=StrategyStatus.PENDING,
                description=description,
                expected=expected,
                target=target
            ))
        
        # Create the file
        manager.create(baseline, strategies)
        return f"✓ Created strategy list with {len(strategies)} strategies"
    
    def _optool_add(self, manager, args: list[str]) -> str:
        """Add a new strategy."""
        if len(args) < 3:
            return "Error: add requires <name> <description> <expected>\nExample: optool add \"Loop Unrolling\" \"Unroll loops\" \"10-15%\""
        
        name = args[0]
        description = args[1]
        expected = args[2]
        target = None
        
        # Parse optional --target
        if len(args) > 3 and args[3] == "--target" and len(args) > 4:
            target = args[4]
        
        manager.add_strategy(name, description, expected, target=target)
        return f"✓ Added strategy: {name}"
    
    def _optool_mark(self, manager, args: list[str]) -> str:
        """Mark strategy status."""
        if len(args) < 2:
            return "Error: mark requires <index> <status>\nExample: optool mark 1 exploring"
        
        try:
            index = int(args[0])
            status = args[1]
            result = None
            details = None
            
            # Parse optional arguments
            i = 2
            while i < len(args):
                if args[i] == "--result" and i + 1 < len(args):
                    result = args[i + 1]
                    i += 2
                elif args[i] == "--details" and i + 1 < len(args):
                    details = args[i + 1]
                    i += 2
                else:
                    i += 1
            
            manager.mark_status(index, status, result, details)
            return f"✓ Marked strategy {index} as {status}"
        except ValueError:
            return f"Error: Invalid index '{args[0]}', must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def _optool_update(self, manager, args: list[str]) -> str:
        """Update strategy fields."""
        if len(args) < 1:
            return "Error: update requires <index>\nExample: optool update 1 --result \"15% improvement\""
        
        try:
            index = int(args[0])
            status = None
            result = None
            details = None
            
            # Parse optional arguments
            i = 1
            while i < len(args):
                if args[i] == "--status" and i + 1 < len(args):
                    status = args[i + 1]
                    i += 2
                elif args[i] == "--result" and i + 1 < len(args):
                    result = args[i + 1]
                    i += 2
                elif args[i] == "--details" and i + 1 < len(args):
                    details = args[i + 1]
                    i += 2
                else:
                    i += 1
            
            manager.update_strategy(index, status=status, result=result, details=details)
            return f"✓ Updated strategy {index}"
        except ValueError:
            return f"Error: Invalid index '{args[0]}', must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def _optool_remove(self, manager, args: list[str]) -> str:
        """Remove or skip a strategy."""
        if len(args) < 1:
            return "Error: remove requires <index>\nExample: optool remove 2"
        
        try:
            index = int(args[0])
            method = "skip"
            
            # Parse optional --method
            if len(args) > 1 and args[1] == "--method" and len(args) > 2:
                method = args[2]
            
            manager.remove_strategy(index, method=method)
            return f"✓ Removed strategy {index} (method: {method})"
        except ValueError:
            return f"Error: Invalid index '{args[0]}', must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def _optool_show(self, manager, args: list[str]) -> str:
        """Show strategies."""
        try:
            if len(args) == 0:
                # Show full content (like cat .optimization_strategies.md)
                return manager.get_full_content()
            elif args[0] == "summary":
                # Show summary statistics
                summary = manager.get_summary()
                lines = [f"Total strategies: {summary['total']}"]
                if summary['by_status']:
                    lines.append("\nBy status:")
                    for status, count in summary['by_status'].items():
                        lines.append(f"  {status}: {count}")
                return "\n".join(lines)
            else:
                # Show specific strategy
                index = int(args[0])
                strategy = manager.get_strategy(index)
                priority_label = "🔴 HIGH" if strategy.priority >= 100 else "Normal"
                lines = [
                    f"Strategy {index}: {strategy.name}",
                    f"Status: {strategy.status.value}",
                    f"Priority: {priority_label}",
                    f"Description: {strategy.description}",
                    f"Expected: {strategy.expected}",
                ]
                if strategy.target:
                    lines.append(f"Target: {strategy.target}")
                if strategy.result:
                    lines.append(f"Result: {strategy.result}")
                if strategy.details:
                    lines.append(f"Details: {strategy.details}")
                return "\n".join(lines)
        except ValueError:
            return f"Error: Invalid index '{args[0]}', must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def _optool_note(self, manager, args: list[str]) -> str:
        """Add a note."""
        if len(args) < 1:
            return "Error: note requires <text>\nExample: optool note \"Important observation\""
        
        note_text = " ".join(args)
        manager.add_note(note_text)
        return f"✓ Added note: {note_text}"
    
    def _optool_priority(self, manager, args: list[str]) -> str:
        """Set strategy priority."""
        if len(args) < 2:
            return "Error: priority requires <index> <high|normal>\nExample: optool priority 1 high"
        
        try:
            from minisweagent.tools.strategy_manager import Strategy
            
            index = int(args[0])
            priority_input = args[1].lower()
            
            # Support both text labels and numeric values (backward compatibility)
            if priority_input.isdigit():
                priority = int(priority_input)
            else:
                priority = Strategy.priority_from_label(priority_input)
            
            manager.update_priority(index, priority)
            
            # Show human-readable label in output
            label = "high" if priority >= 100 else "normal"
            return f"✓ Set strategy {index} priority to {label}"
        except ValueError as e:
            return f"Error: Invalid index. Must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def _optool_path(self, manager) -> str:
        """Show current strategy file path."""
        return f"""Strategy file path: {str(manager.filepath)}
Configured path: {self.strategy_file_path}"""
    
    def _optool_next(self, manager) -> str:
        """Get the next recommended strategy to explore."""
        if not manager.exists():
            return "No strategy file found. Create one with 'optool create' first."
        
        index, strategy = self.select_next_strategy()
        
        if strategy is None:
            return "No pending strategies available. All strategies have been explored or marked as skipped."
        
        # Format the recommendation
        priority_label = "🔴 HIGH PRIORITY" if strategy.priority >= 100 else f"Priority: {strategy.priority}"
        lines = [
            f"✨ Recommended next strategy: #{index}",
            "",
            f"Name: {strategy.name}",
            f"Status: {strategy.status.value}",
            f"{priority_label}",
            f"Description: {strategy.description}",
            f"Expected improvement: {strategy.expected}",
        ]
        
        if strategy.target:
            lines.append(f"Target: {strategy.target}")
        
        lines.extend([
            "",
            f"To mark as exploring: optool mark {index} exploring",
            f"To view details: optool show {index}"
        ])
        
        return "\n".join(lines)
    
    # ============ Strategy Selection Logic ============
    
    def get_sorted_pending_strategies(self):
        """Get pending strategies sorted by priority."""
        manager = self._get_strategy_manager()
        if not manager.exists():
            return []
        
        from minisweagent.tools.strategy_manager import StrategyStatus
        
        strategy_list = manager.load()
        pending = [s for s in strategy_list.strategies 
                   if s.status == StrategyStatus.PENDING]
        
        # Sort by priority (high first), then by index
        return sorted(enumerate(pending, 1), 
                     key=lambda x: (-x[1].priority, x[0]))
    
    def select_next_strategy(self):
        """Select next strategy to explore based on priority and LLM judgment.
        
        Returns:
            tuple: (index, Strategy) or (None, None) if no strategies available
        """
        sorted_strategies = self.get_sorted_pending_strategies()
        
        if not sorted_strategies:
            return None, None
        
        # Check if there are high priority strategies (>= 100)
        if sorted_strategies[0][1].priority >= 100:
            # User has marked high priority, use first one
            return sorted_strategies[0]
        
        # All are normal/low priority, let LLM choose
        return self._llm_select_strategy(sorted_strategies)
    
    def _llm_select_strategy(self, strategies: list[tuple[int, any]]):
        """Let LLM choose the most promising strategy from normal priority ones.
        
        Args:
            strategies: List of (index, Strategy) tuples
            
        Returns:
            tuple: (index, Strategy) of selected strategy
        """
        # Format strategies for LLM
        strategy_desc = "\n".join([
            f"{idx}. {s.name} - {s.description} (Expected: {s.expected}, Priority: {s.priority})"
            for idx, s in strategies[:10]  # Limit to first 10
        ])
        
        prompt = f"""Available optimization strategies (all Normal/Low priority):

{strategy_desc}

Based on your expertise and the current optimization context, which strategy looks most promising to explore next?
Consider expected impact, implementation complexity, and risk.

Respond with just the strategy number."""
        
        try:
            # Simple query - in practice this would use self.model.query()
            # For now, just return the first strategy with highest expected value
            return strategies[0]
        except Exception as e:
            print(f"[WARNING] LLM selection failed: {e}, using first strategy", file=sys.stderr)
            return strategies[0]

