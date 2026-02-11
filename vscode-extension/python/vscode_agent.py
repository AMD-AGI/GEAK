"""VS Code adapter for mini-swe-agent."""

import json
import sys
import threading
import traceback
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.strategy_agent import StrategyAgent
from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig, NonTerminatingException, Submitted


def _log_tool_call(tool_calls: dict) -> None:
    """Print tool call to stderr so it appears in extension output."""
    func = tool_calls.get("function", {})
    name = func.get("name", "?")
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    print("\n> Tool Call:", name, file=sys.stderr)
    if args:
        print(json.dumps(args, indent=2, ensure_ascii=False), file=sys.stderr)


def _log_tool_result(tool_name: str, result_content: str) -> None:
    """Print tool result to stderr so it appears in extension output."""
    print(f"\nUser (tool_result: {tool_name}):", file=sys.stderr)
    try:
        data = json.loads(result_content)
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {str(v)[:200]}", file=sys.stderr)
        else:
            print(result_content[:500], file=sys.stderr)
    except (json.JSONDecodeError, TypeError):
        print((result_content[:500] + ("..." if len(result_content) > 500 else "")), file=sys.stderr)


def _log_command(action: str) -> None:
    """Print bash/tool command to stderr so it appears in extension output."""
    if action and not action.startswith("Tool:"):
        print("\n> Command:", action[:300], file=sys.stderr)


def _log_observation(content: str) -> None:
    """Print observation (command result) to stderr so it appears in extension output."""
    if not content:
        return
    print("\nObservation:", file=sys.stderr)
    # Truncate long output
    if len(content) > 800:
        print(content[:800] + "\n... (truncated)", file=sys.stderr)
    else:
        print(content, file=sys.stderr)


def _format_tool_call_for_ui(tool_calls: dict) -> str:
    """Format tool call for UI display (agent message content)."""
    func = tool_calls.get("function", {})
    name = func.get("name", "?")
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    lines = [f">> Tool Call: {name}"]
    if args:
        lines.append(json.dumps(args, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def _format_tool_result_for_ui(tool_name: str, result_content: str) -> str:
    """Format tool result for UI display (observation message)."""
    try:
        data = json.loads(result_content)
        if isinstance(data, dict):
            lines = [f"Observation (tool: {tool_name}):"]
            for k, v in data.items():
                lines.append(f"  {k}: {str(v)}")
            return "\n".join(lines)
    except (json.JSONDecodeError, TypeError):
        pass
    return f"Observation (tool: {tool_name}):\n{result_content}"


class VSCodeBridge:
    """Handles JSON-RPC communication with VS Code over stdio."""

    def __init__(self, stdio_out=None):
        self._out = stdio_out if stdio_out is not None else sys.stdout
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
                                self._out.write(json.dumps(error_msg) + "\n")
                                self._out.flush()
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
        self._out.write(json.dumps(message) + "\n")
        self._out.flush()
    
    def wait_for_initialize(self) -> dict:
        """Wait for initialize request from VS Code."""
        self.init_event.wait()
        return self.init_params

    def send_notification(self, method: str, params: Any = None):
        """Send notification (no response expected)."""
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._out.write(json.dumps(message) + "\n")
        self._out.flush()

    def send_request(self, method: str, params: Any = None) -> Any:
        """Send request and block until response received."""
        with self.lock:
            self.request_id += 1
            request_id = self.request_id
            event = threading.Event()
            self.pending_requests[request_id] = event

        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        self._out.write(json.dumps(message) + "\n")
        self._out.flush()

        # Block until response
        event.wait()

        with self.lock:
            response = self.responses.pop(request_id)
            del self.pending_requests[request_id]

        return response


class VSCodeInteractiveAgent(InteractiveAgent):
    """VSCode Agent without strategy management (基础模式).
    
    This agent provides basic interactive functionality without optool commands
    or strategy management features.
    """

    def __init__(self, bridge: VSCodeBridge, *args, **kwargs):
        self.bridge = bridge
        super().__init__(*args, **kwargs)
        print("[INFO] VSCode Agent: Interactive Mode (no strategy support)", file=sys.stderr)

    def add_message(self, role: str, content: str, **kwargs):
        """Override to send messages to VS Code instead of console."""
        super().add_message(role, content, **kwargs)

        action = ""
        need_confirm = False
        content_to_send = content
        if role == "assistant":
            tool_calls = kwargs.get("tool_calls")
            if tool_calls:
                action = f"Tool: {tool_calls.get('function', {}).get('name', '')}"
                need_confirm = False
                _log_tool_call(tool_calls)
                content_to_send = content + "\n\n" + _format_tool_call_for_ui(tool_calls)
                self._run_tool_and_add_result(content, tool_calls)
            else:
                content_dict = {"content": content}
                action = self.parse_action(content_dict).get("action", "")
                need_confirm = self.should_ask_confirmation(action)
                _log_command(action)
        elif role == "user":
            _log_observation(content)

        # Send to VS Code webview
        self.bridge.send_notification(
            "agent/message", {
                "role": role,
                "content": content_to_send,
                "step": self.model.n_calls,
                "cost": self.model.cost,
                "action": action,
                "need_confirm": need_confirm
            }
        )

    def _run_tool_and_add_result(self, content: str, tool_calls: dict) -> None:
        """Run the tool and add tool result message (so API gets tool_use + tool_result)."""
        tool_use_id = (
            tool_calls.get("id")
            or tool_calls.get("tool_use_id")
            or (self.messages and self.messages[-1].get("tool_calls") or {}).get("id")
            or ""
        )
        if not tool_use_id:
            print("[WARNING] tool_use id missing in tool_calls; tool_result may not match", file=sys.stderr)
        response = {"content": content, "tools": tool_calls}
        err = None
        try:
            output = self.parse_action(response)
        except Exception as e:
            output = {"error": str(e)}
            err = e
        result_content = json.dumps(output) if isinstance(output, dict) else str(output)
        tool_name = tool_calls.get("function", {}).get("name", "")
        DefaultAgent.add_message(
            self, "tool", result_content,
            tool_call_id=tool_use_id,
            name=tool_name,
        )
        _log_tool_result(tool_name, result_content)
        # Notify webview so UI shows tool result as observation
        self.bridge.send_notification(
            "agent/message",
            {
                "role": "user",
                "content": _format_tool_result_for_ui(tool_name, result_content),
                "step": self.model.n_calls,
                "cost": self.model.cost,
                "action": "",
                "need_confirm": False,
            },
        )
        if err is not None:
            raise err

    def get_observation(self, response: dict) -> dict:
        """When tool result was already added in add_message, just return it."""
        last_msg = self.messages[-1] if self.messages else {}
        if last_msg.get("role") == "tool" and response.get("tools"):
            try:
                return json.loads(last_msg.get("content", "{}"))
            except (json.JSONDecodeError, TypeError):
                return {"output": last_msg.get("content", "")}
        return super().get_observation(response)

    def ask_confirmation(self) -> None:
        """Ask user for confirmation via VS Code."""
        last_message = self.messages[-1]["content"] if self.messages else ""
        
        # Send confirmation request to VS Code
        response = self.bridge.send_request(
            "agent/requestConfirmation",
            {"action": last_message, "mode": self.config.mode}
        )

        # Handle response same way as InteractiveAgent
        if response.get("approved") is False:
            # User rejected or modified the command
            # Handle mode switching (/y, /c, /u)
            new_mode = response.get("newMode")
            if new_mode in ["human", "confirm", "yolo"]:
                self.config.mode = new_mode
                if new_mode == "human":
                    raise NonTerminatingException("Command not executed. Switching to human mode")
        else:
            # Approved, continue
            pass

    def get_observation(self, response: dict) -> dict:
        """Override to ensure tool_result is always added after tool_use (API requirement)."""
        last_msg = self.messages[-1] if self.messages else {}
        is_tool_turn = (
            last_msg.get("role") == "assistant"
            and last_msg.get("tool_calls")
            and response.get("tools")
        )
        try:
            return super().get_observation(response)
        except Exception:
            if is_tool_turn:
                tool_info = response["tools"]
                err_content = json.dumps({"error": traceback.format_exc()})
                self.add_message(
                    "tool", err_content,
                    tool_call_id=tool_info.get("id", ""),
                    name=tool_info.get("function", {}).get("name", ""),
                )
            raise

    def query(self) -> dict:
        """Override to handle human mode via VS Code."""
        if self.config.mode == "human":
            # Request command from user via VS Code
            response = self.bridge.send_request("agent/requestHumanCommand", {})
            
            human_command = response.get("command", "")
            if not human_command:
                raise NonTerminatingException("No command provided in human mode")
            
            # Return as observation
            result = self.env.execute({"action": human_command})
            return {"role": "user", "content": result.get("observation", "")}
        
        # Check if we need to increase limits
        if (
            self.config.cost_limit
            and self.model.cost >= self.config.cost_limit
            or self.config.step_limit
            and self.model.n_calls >= self.config.step_limit
        ):
            response = self.bridge.send_request(
                "agent/limitReached",
                {
                    "current_steps": self.model.n_calls,
                    "current_cost": self.model.cost,
                    "step_limit": self.config.step_limit,
                    "cost_limit": self.config.cost_limit,
                },
            )

            if not response.get("continue", False):
                raise NonTerminatingException(
                    "Cost limit reached"
                )

            self.config.step_limit = response.get("step_limit", self.config.step_limit)
            self.config.cost_limit = response.get("cost_limit", self.config.cost_limit)

            return super().query()

        return super().query()

    def has_finished(self, output: dict) -> bool:
        """Override to handle exit confirmation via VS Code."""
        try:
            return super().has_finished(output)
        except Submitted as e:
            if self.config.confirm_exit:
                # Ask user if they want to continue or finish
                response = self.bridge.send_request("agent/confirmExit", {"output": str(e)})
                
                if not response.get("confirmed", True):
                    # User wants to continue
                    return False
            # User confirmed exit or confirm_exit is False
            raise

class VSCodeStrategyAgent(StrategyAgent):
    """VSCode Agent with strategy management (策略模式).
    
    This agent provides full strategy management functionality including optool
    commands and .optimization_strategies.md file management.
    """

    def __init__(self, bridge: VSCodeBridge, *args, **kwargs):
        self.bridge = bridge
        super().__init__(*args, **kwargs)
        
        print(f"[INFO] VSCode Agent: Strategy Mode (file: {self.config.strategy_file_path})", file=sys.stderr)

    # ============ Communication Implementation ============
    
    def add_message(self, role: str, content: str, **kwargs):
        """Override to send messages to VS Code instead of console."""
        super().add_message(role, content, **kwargs)

        action = ""
        need_confirm = False
        content_to_send = content
        if role == "assistant":
            tool_calls = kwargs.get("tool_calls")
            if tool_calls:
                action = f"Tool: {tool_calls.get('function', {}).get('name', '')}"
                need_confirm = False
                _log_tool_call(tool_calls)
                content_to_send = content + "\n\n" + _format_tool_call_for_ui(tool_calls)
                self._run_tool_and_add_result(content, tool_calls)
            else:
                content_dict = {"content": content}
                action = self.parse_action(content_dict).get("action", "")
                need_confirm = self.should_ask_confirmation(action)
                _log_command(action)
        elif role == "user":
            _log_observation(content)

        # Send to VS Code webview
        self.bridge.send_notification(
            "agent/message", {
                "role": role,
                "content": content_to_send,
                "step": self.model.n_calls,
                "cost": self.model.cost,
                "action": action,
                "need_confirm": need_confirm
            }
        )

    def _run_tool_and_add_result(self, content: str, tool_calls: dict) -> None:
        """Run the tool and add tool result message (so API gets tool_use + tool_result)."""
        tool_use_id = (
            tool_calls.get("id")
            or tool_calls.get("tool_use_id")
            or (self.messages and self.messages[-1].get("tool_calls") or {}).get("id")
            or ""
        )
        if not tool_use_id:
            print("[WARNING] tool_use id missing in tool_calls; tool_result may not match", file=sys.stderr)
        response = {"content": content, "tools": tool_calls}
        err = None
        try:
            output = self.parse_action(response)
        except Exception as e:
            output = {"error": str(e)}
            err = e
        result_content = json.dumps(output) if isinstance(output, dict) else str(output)
        tool_name = tool_calls.get("function", {}).get("name", "")
        DefaultAgent.add_message(
            self, "tool", result_content,
            tool_call_id=tool_use_id,
            name=tool_name,
        )
        _log_tool_result(tool_name, result_content)
        self.bridge.send_notification(
            "agent/message",
            {
                "role": "user",
                "content": _format_tool_result_for_ui(tool_name, result_content),
                "step": self.model.n_calls,
                "cost": self.model.cost,
                "action": "",
                "need_confirm": False,
            },
        )
        if err is not None:
            raise err

    def get_observation(self, response: dict) -> dict:
        """When tool result was already added in add_message, just return it."""
        last_msg = self.messages[-1] if self.messages else {}
        if last_msg.get("role") == "tool" and response.get("tools"):
            try:
                return json.loads(last_msg.get("content", "{}"))
            except (json.JSONDecodeError, TypeError):
                return {"output": last_msg.get("content", "")}
        return super().get_observation(response)

    def notify_strategy_changed(self, strategy_data: dict):
        """Notify VS Code that strategy list has changed."""
        self.bridge.send_notification("agent/strategyData", strategy_data)
        print(f"[DEBUG] Sent strategy data to VSCode: {len(strategy_data.get('strategies', []))} strategies", file=sys.stderr)

    def ask_confirmation(self) -> None:
        """Override to get confirmation from VS Code instead of terminal."""
        # Get the last assistant message (contains the action)
        last_message = ""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                last_message = msg["content"]
                break

        # Send confirmation request to VS Code
        response = self.bridge.send_request(
            "agent/requestConfirmation",
            {"action": last_message, "mode": self.config.mode}
        )

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
                "agent/requestNewLimits",
                {"current_steps": self.model.n_calls, "current_cost": self.model.cost}
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

