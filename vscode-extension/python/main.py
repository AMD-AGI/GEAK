#!/usr/bin/env python3
"""Entry point for VS Code agent process."""

import sys
import os
import traceback
from pathlib import Path
from io import StringIO

# CRITICAL: Redirect stdout during imports to prevent library output from polluting JSON-RPC
# mini-swe-agent prints version info and config loading messages to stdout
_original_stdout = sys.stdout
sys.stdout = sys.stderr  # Temporarily redirect stdout to stderr during imports

try:
    import yaml
    from minisweagent.config import get_config_path
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model
    from vscode_agent import VSCodeAgent, VSCodeBridge
finally:
    # Restore original stdout for JSON-RPC communication
    sys.stdout = _original_stdout


def main():
    """Run the agent with configuration from VS Code."""
    print(f"[DEBUG] VSCode Agent starting, Python version: {sys.version}", file=sys.stderr)
    print(f"[DEBUG] Working directory: {os.getcwd()}", file=sys.stderr)
    
    bridge = VSCodeBridge()
    print("[DEBUG] VSCodeBridge created", file=sys.stderr)
    
    # Variable to hold agent reference for mode switching
    agent_ref = {"agent": None}

    # Set up handler for initialization request from VS Code
    def handle_initialize(params):
        print(f"[DEBUG] Received initialize request: {params.get('task', 'no task')}", file=sys.stderr)
        bridge.init_params = params
        bridge.init_event.set()
        return {"status": "initialized"}
    
    # Set up handler for mode switching notifications
    def handle_set_mode(params):
        mode = params.get("mode")
        print(f"[DEBUG] Mode change request: {mode}", file=sys.stderr)
        if agent_ref["agent"] and mode in ["human", "confirm", "yolo"]:
            agent_ref["agent"].config.mode = mode
            bridge.send_notification("agent/info", {"message": f"Mode switched to {mode}"})
    
    bridge.on_request("agent/waitInitialize", handle_initialize)
    bridge.on_notification("agent/setMode", handle_set_mode)
    print("[DEBUG] Handlers registered", file=sys.stderr)
    
    # Start the reader thread after all handlers are registered
    bridge.start()
    print("[DEBUG] VSCodeBridge started, waiting for initialize request", file=sys.stderr)

    # Wait for initialization message from VS Code
    init_message = bridge.wait_for_initialize()
    print("[DEBUG] Initialization received, starting agent", file=sys.stderr)

    workspace_path = init_message.get("workspacePath")
    config = init_message.get("config", {})
    task = init_message.get("task")
    config_file_path = init_message.get("configFilePath")

    try:
        # Load config from YAML file if provided
        if config_file_path:
            try:
                print(f"[DEBUG] Loading config from: {config_file_path}", file=sys.stderr)
                config_path = get_config_path(config_file_path)
                yaml_config = yaml.safe_load(config_path.read_text())
                bridge.send_notification("agent/info", {"message": f"Loaded config from {config_path}"})
                
                # Merge YAML config with VS Code config (VS Code config takes precedence)
                # Deep merge for nested dicts
                for section in ["agent", "model", "env"]:
                    if section in yaml_config:
                        if section not in config:
                            config[section] = {}
                        # YAML config as base, VS Code config overrides
                        config[section] = {**yaml_config[section], **config[section]}
            except FileNotFoundError as e:
                print(f"[WARNING] Config file not found: {e}", file=sys.stderr)
                bridge.send_notification("agent/warning", {"message": f"Config file not found: {e}"})
            except Exception as e:
                print(f"[WARNING] Failed to load config file: {e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                bridge.send_notification("agent/warning", {"message": f"Failed to load config file: {e}"})

        # Get model name from config or init message
        model_config = config.get("model", {})
        model_name = init_message.get("modelName") or model_config.get("model_name", "gpt-4")
        print(f"[DEBUG] Using model: {model_name}", file=sys.stderr)

        # Handle API key - if it's at top level, move it to model_kwargs for compatibility
        if "api_key" in model_config and "api_key" not in model_config.get("model_kwargs", {}):
            model_config.setdefault("model_kwargs", {})["api_key"] = model_config.pop("api_key")

        # Get agent configuration (includes strategy_file_path)
        agent_config = config.get("agent", {})
        
        # Create model and environment
        print(f"[DEBUG] Creating model and environment", file=sys.stderr)
        model = get_model(model_name, model_config)
        env = LocalEnvironment(cwd=workspace_path, **config.get("env", {}))

        # Create agent (strategy_file_path is passed via **agent_config)
        print(f"[DEBUG] Creating agent", file=sys.stderr)
        agent = VSCodeAgent(
            bridge=bridge, 
            model=model, 
            env=env,
            **agent_config
        )
        agent_ref["agent"] = agent  # Store reference for mode switching

        # Run agent
        print(f"[DEBUG] Starting agent run with task: {task[:50]}...", file=sys.stderr)
        bridge.send_notification("agent/started", {})
        exit_status, result = agent.run(task)

        # Send completion
        print(f"[DEBUG] Agent finished with exit status: {exit_status}", file=sys.stderr)
        bridge.send_notification("agent/finished", {"exitStatus": exit_status, "result": result})

    except Exception as e:
        print(f"[ERROR] Agent failed with exception: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        bridge.send_notification("agent/error", {"error": str(e), "traceback": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()

