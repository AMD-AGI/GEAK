"""VS Code adapter for mini-swe-agent."""

import json
import sys
import threading
from typing import Any

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig, NonTerminatingException


class VSCodeBridge:
    """Handles JSON-RPC communication with VS Code over stdio."""

    def __init__(self):
        self.request_id = 0
        self.pending_requests = {}  # request_id -> threading.Event
        self.responses = {}  # request_id -> response data
        self.request_handlers = {}  # method -> handler function
        self.lock = threading.Lock()
        self.init_event = threading.Event()
        self.init_params = None

        # Start thread to read from stdin
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def _read_loop(self):
        """Read JSON-RPC messages from stdin."""
        for line in sys.stdin:
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
                        result = self.request_handlers[method](message.get("params", {}))
                        self.send_response(message["id"], result)
            except json.JSONDecodeError:
                continue
    
    def on_request(self, method: str, handler):
        """Register a handler for incoming requests."""
        self.request_handlers[method] = handler
    
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

