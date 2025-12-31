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

    def add_message(self, role: str, content: str, **kwargs):
        """Override to send messages to VS Code instead of console."""
        super().add_message(role, content, **kwargs)

        # Send to VS Code webview
        self.bridge.send_notification(
            "agent/message", {"role": role, "content": content, "step": self.model.n_calls, "cost": self.model.cost}
        )

    def execute_action(self, action: dict) -> str:
        """Override to intercept special commands like generate_strategies."""
        command = action.get("action", "")
        command_stripped = command.strip()
        
        # Check if this is a strategy generation command
        # Handle both direct calls and bash-wrapped calls
        is_generate_strategies = False
        context = ""
        
        if command_stripped.startswith("generate_strategies"):
            is_generate_strategies = True
            context = command_stripped.replace("generate_strategies", "", 1).strip()
        elif "generate_strategies" in command_stripped:
            # Check for patterns like: bash generate_strategies "..." or bash -c "generate_strategies ..."
            # Match: bash [options] generate_strategies "context"
            match = re.search(r'generate_strategies\s+["\']?([^"\']+)["\']?', command_stripped)
            if match:
                is_generate_strategies = True
                context = match.group(1).strip()
        
        if is_generate_strategies:
            # Remove quotes if present
            context = context.strip('"').strip("'")
            if not context:
                # Use current task as context
                context = self.instance
            
            print(f"[DEBUG] Intercepted generate_strategies command, context: {context[:100]}", file=sys.stderr)
            
            # Handle strategy generation and selection
            result = self._handle_strategy_generation(context)
            
            # Return as dict with 'output' key (matching env.execute return format)
            return {"output": result, "returncode": 0}
        
        # Other commands execute normally
        return super().execute_action(action)
    
    def _handle_strategy_generation(self, context: str) -> str:
        """Handle strategy generation and selection based on mode."""
        # Generate strategy list
        strategies = self._generate_strategies_dummy(context)
        
        # Handle based on mode
        if self.config.mode == "yolo":
            # YOLO mode: auto-select
            selected = self._auto_select_strategy(strategies)
            
            # Send notification (non-blocking)
            self.bridge.send_notification("agent/strategiesGenerated", {
                "strategies": strategies,
                "autoSelected": selected["id"],
                "mode": "auto",
                "step": self.model.n_calls,
                "cost": self.model.cost
            })
            
            # Return selected strategy as command output
            return f"Auto-selected strategy: {selected['title']}\n{selected['description']}"
        else:
            # Confirm mode: request user selection (blocking)
            response = self.bridge.send_request("agent/requestStrategySelection", {
                "strategies": strategies,
                "step": self.model.n_calls,
                "cost": self.model.cost
            })
            
            # Process response
            if response.get("action") == "skip" or not response.get("selectedStrategyId"):
                selected = self._auto_select_strategy(strategies)
                return f"LLM selected strategy: {selected['title']}\n{selected['description']}"
            else:
                selected_id = response["selectedStrategyId"]
                selected = next(s for s in strategies if s["id"] == selected_id)
                return f"User selected strategy: {selected['title']}\n{selected['description']}"
    
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

