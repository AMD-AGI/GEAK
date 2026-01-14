"""VS Code adapter for mini-swe-agent."""

import json
import re
import sys
import threading
import traceback
from typing import Any

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig, NonTerminatingException


class VSCodeBridge:
    """Handles JSON-RPC communication with VS Code over stdio."""

    def __init__(self):
        self.request_id = 0
        self.pending_requests = {}  # request_id -> threading.Event
        self.responses = {}  # request_id -> response data
        self.request_handlers = {}  # method -> handler function
        self.notification_handlers = {}  # method -> handler function
        self.lock = threading.Lock()
        self.init_event = threading.Event()
        self.init_params = None
        self.reader_thread = None

    def start(self):
        """Start the reader thread. Should be called after registering all handlers."""
        print("[DEBUG] Starting VSCodeBridge reader thread", file=sys.stderr)
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def _read_loop(self):
        """Read JSON-RPC messages from stdin."""
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                    
                try:
                    message = json.loads(line)
                    
                    # Handle responses to our requests
                    if "id" in message and "result" in message and message["id"] in self.pending_requests:
                        with self.lock:
                            self.responses[message["id"]] = message.get("result")
                            self.pending_requests[message["id"]].set()
                    
                    # Handle requests from VS Code
                    elif "method" in message and "id" in message:
                        method = message["method"]
                        if method in self.request_handlers:
                            try:
                                result = self.request_handlers[method](message.get("params", {}))
                                self.send_response(message["id"], result)
                            except Exception as e:
                                # Send error response if handler fails
                                print(f"Error in request handler for {method}: {e}", file=sys.stderr)
                                print(traceback.format_exc(), file=sys.stderr)
                                error_msg = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -1, "message": str(e)}}
                                sys.stdout.write(json.dumps(error_msg) + "\n")
                                sys.stdout.flush()
                        else:
                            print(f"No handler registered for request: {method}", file=sys.stderr)
                    
                    # Handle notifications from VS Code (no id)
                    elif "method" in message and "id" not in message:
                        method = message["method"]
                        if method in self.notification_handlers:
                            try:
                                self.notification_handlers[method](message.get("params", {}))
                            except Exception as e:
                                # Log notification handler errors but don't crash
                                print(f"Error in notification handler for {method}: {e}", file=sys.stderr)
                                print(traceback.format_exc(), file=sys.stderr)
                        else:
                            print(f"No handler registered for notification: {method}", file=sys.stderr)
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON-RPC message: {line.strip()}", file=sys.stderr)
                    print(f"JSONDecodeError: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Unexpected error processing message: {e}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    continue
        except Exception as e:
            print(f"Fatal error in read loop: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    
    def on_request(self, method: str, handler):
        """Register a handler for incoming requests."""
        self.request_handlers[method] = handler
    
    def on_notification(self, method: str, handler):
        """Register a handler for incoming notifications."""
        self.notification_handlers[method] = handler
    
    def send_response(self, request_id: int, result: Any):
        """Send a response to a request."""
        message = {"jsonrpc": "2.0", "id": request_id, "result": result}
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()
    
    def wait_for_initialize(self) -> dict:
        """Wait for initialize request from VS Code."""
        self.init_event.wait()
        return self.init_params

    def send_notification(self, method: str, params: Any = None):
        """Send notification (no response expected)."""
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

    def send_request(self, method: str, params: Any = None) -> Any:
        """Send request and block until response received."""
        with self.lock:
            self.request_id += 1
            request_id = self.request_id
            event = threading.Event()
            self.pending_requests[request_id] = event

        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

        # Block until response
        event.wait()

        with self.lock:
            response = self.responses.pop(request_id)
            del self.pending_requests[request_id]

        return response


class VSCodeAgent(InteractiveAgent):
    """Agent adapted for VS Code - replaces terminal I/O with JSON-RPC."""

    def __init__(self, bridge: VSCodeBridge, *args, **kwargs):
        self.bridge = bridge
        super().__init__(*args, **kwargs)
        self._strategy_manager = None  # Lazy initialization

    def add_message(self, role: str, content: str, **kwargs):
        """Override to send messages to VS Code instead of console."""
        super().add_message(role, content, **kwargs)

        action = ""
        need_confirm = False
        if role == "assistant":
            content_dict = {
                "content": content,
            }
            action = self.parse_action(content_dict).get("action", "")
            need_confirm = self.should_ask_confirmation(action)

        # Send to VS Code webview
        self.bridge.send_notification(
            "agent/message", {"role": role, "content": content, "step": self.model.n_calls, "cost": self.model.cost, "action": action, "need_confirm": need_confirm}
        )

    def execute_action(self, action: dict) -> dict:
        """Override to intercept special commands like generate_strategies."""
        command = action.get("action", "")
        command_stripped = command.strip()
        
        print(f"[DEBUG] execute_action called", file=sys.stderr)
        print(f"[DEBUG] action type: {type(action)}", file=sys.stderr)
        print(f"[DEBUG] action keys: {action.keys() if isinstance(action, dict) else 'NOT A DICT'}", file=sys.stderr)
        print(f"[DEBUG] command: {repr(command[:100])}", file=sys.stderr)
        print(f"[DEBUG] command_stripped: {repr(command_stripped[:100])}", file=sys.stderr)
        print(f"[DEBUG] startswith('optool '): {command_stripped.startswith('optool ')}", file=sys.stderr)
        print(f"[DEBUG] == 'optool': {command_stripped == 'optool'}", file=sys.stderr)
        
        # Check for optimization tool command
        # Match patterns like: "optool ...", "cd x && optool ...", "cmd; optool ...", etc.
        import re
        optool_pattern = r'(^|&&|\|\||;)\s*optool(\s|$)'
        has_optool = re.search(optool_pattern, command_stripped)
        
        if has_optool:
            # Extract the optool command part
            # Find where optool starts
            match_start = has_optool.start()
            # Find the actual optool position (skip &&, ||, ;)
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
        
        # Other commands execute normally
        return super().execute_action(action)
    
    def _get_strategy_manager(self):
        """Get or create StrategyManager with callback (lazy initialization)."""
        if self._strategy_manager is None:
            from pathlib import Path
            import os
            # Import here to avoid circular dependency issues
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            from minisweagent.tools.strategy_manager import StrategyManager
            
            # Get working directory from env config or use current directory
            cwd = self.env.config.cwd or os.getcwd()
            strategy_file = Path(cwd) / '.optimization_strategies.md'
            self._strategy_manager = StrategyManager(
                filepath=strategy_file,
                on_change_callback=self._on_strategy_changed
            )
            
            # Send initial data
            if self._strategy_manager.exists():
                try:
                    strategy_list = self._strategy_manager.load()
                    self._on_strategy_changed(strategy_list)
                except Exception as e:
                    print(f"[WARNING] Failed to load initial strategy data: {e}", file=sys.stderr)
        
        return self._strategy_manager
    
    def _on_strategy_changed(self, strategy_list):
        """Callback when strategy list changes - push to VSCode."""
        try:
            result = {
                "exists": True,
                "strategies": [],
                "baseline": None,
                "notes": strategy_list.notes
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
                    "expected": strategy.expected,
                    "target": strategy.target,
                    "result": strategy.result,
                    "details": strategy.details
                })
            
            self.bridge.send_notification("agent/strategyData", result)
            print(f"[DEBUG] Sent strategy data to VSCode: {len(result['strategies'])} strategies", file=sys.stderr)
            
        except Exception as e:
            print(f"[ERROR] Failed to send strategy data: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    
    def _handle_optool_command(self, command: str) -> str:
        """Handle optimization tool commands."""
        import shlex
        
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
  
  show [index]
      Show all strategies or a specific strategy
      Example: optool show
  
  note <text>
      Add a note to the strategy list
      Example: optool note "Combined strategies 1 and 3 work well together"
  
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
            if len(args) > 0:
                # Show specific strategy
                index = int(args[0])
                strategy = manager.get_strategy(index)
                return f"Strategy {index}: {strategy.name}\nStatus: {strategy.status.value}\nDescription: {strategy.description}\nExpected: {strategy.expected}\nTarget: {strategy.target}\nResult: {strategy.result}\nDetails: {strategy.details}"
            else:
                # Show all strategies
                summary = manager.get_summary()
                return summary
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
    
    
    
    def _generate_strategies_dummy(self, context: str) -> list[dict]:
        """Generate strategy list (dummy implementation)."""
        return [
            {
                "id": "strategy_1",
                "title": "Strategy 1: Direct Fix",
                "description": "Directly modify the buggy code to fix the issue",
                "reasoning": "This is the most straightforward approach",
                "isRecommended": True
            },
            {
                "id": "strategy_2",
                "title": "Strategy 2: Refactor First",
                "description": "Refactor the code structure before fixing the bug",
                "reasoning": "This ensures better code quality"
            },
            {
                "id": "strategy_3",
                "title": "Strategy 3: Add Tests First",
                "description": "Write failing tests first, then fix the code",
                "reasoning": "TDD approach ensures the fix is correct"
            }
        ]
    
    def _auto_select_strategy(self, strategies: list[dict]) -> dict:
        """Auto-select a strategy (select recommended one or first)."""
        for s in strategies:
            if s.get("isRecommended"):
                return s
        return strategies[0]

    def ask_confirmation(self) -> None:
        """Override to get confirmation from VS Code instead of terminal."""
        # Get the last assistant message (contains the action)
        last_message = ""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                last_message = msg["content"]
                break

        # Send confirmation request to VS Code
        response = self.bridge.send_request("agent/requestConfirmation", {"action": last_message, "mode": self.config.mode})

        # Handle response same way as InteractiveAgent
        if response.get("approved"):
            pass  # confirmed, continue execution
        elif response.get("switchMode"):
            # Handle mode switching (/y, /c, /u)
            new_mode = response.get("newMode")
            if new_mode in ["human", "confirm", "yolo"]:
                self.config.mode = new_mode
                if new_mode == "human":
                    raise NonTerminatingException("Command not executed. Switching to human mode")
        else:
            # Rejected
            reason = response.get("reason", "User rejected the command")
            raise NonTerminatingException(f"Command not executed. {reason}")

    def query(self) -> dict:
        """Override to handle human mode via VS Code."""
        if self.config.mode == "human":
            # Request command from user via VS Code
            response = self.bridge.send_request("agent/requestHumanCommand", {})

            command = response.get("command", "")
            if command in ["/y", "/c"]:
                # Switch mode and fall through to LM query
                pass
            else:
                msg = {"content": f"\n```bash\n{command}\n```"}
                self.add_message("assistant", msg["content"])
                return msg

        # Normal LM query (with cost/step limit handling)
        from minisweagent.agents.default import LimitsExceeded

        try:
            return super().query()
        except LimitsExceeded:
            # Send limits exceeded to VS Code
            self.bridge.send_notification(
                "agent/limitsExceeded",
                {
                    "current_steps": self.model.n_calls,
                    "current_cost": self.model.cost,
                    "step_limit": self.config.step_limit,
                    "cost_limit": self.config.cost_limit,
                },
            )

            # Wait for user to adjust limits
            response = self.bridge.send_request(
                "agent/requestNewLimits", {"current_steps": self.model.n_calls, "current_cost": self.model.cost}
            )

            self.config.step_limit = response.get("step_limit", self.config.step_limit)
            self.config.cost_limit = response.get("cost_limit", self.config.cost_limit)

            return super().query()

    def has_finished(self, output: dict[str, str]):
        """Override to handle exit confirmation via VS Code."""
        from minisweagent.agents.default import Submitted

        try:
            return super().has_finished(output)
        except Submitted as e:
            if self.config.confirm_exit:
                # Ask user if they want to continue or finish
                response = self.bridge.send_request("agent/confirmExit", {"output": str(e)})

                if response.get("continue"):
                    new_task = response.get("newTask", "Continue working")
                    raise NonTerminatingException(f"The user added a new task: {new_task}")
            raise e

