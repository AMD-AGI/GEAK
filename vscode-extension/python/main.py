#!/usr/bin/env python3
"""Entry point for VS Code agent process."""

import sys
import traceback
from pathlib import Path

import yaml

from minisweagent.config import get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

from vscode_agent import VSCodeAgent, VSCodeBridge


def main():
    """Run the agent with configuration from VS Code."""
    bridge = VSCodeBridge()

    # Set up handler for initialization request from VS Code
    def handle_initialize(params):
        bridge.init_params = params
        bridge.init_event.set()
        return {"status": "initialized"}
    
    bridge.on_request("agent/waitInitialize", handle_initialize)

    # Wait for initialization message from VS Code
    init_message = bridge.wait_for_initialize()

    workspace_path = init_message.get("workspacePath")
    config = init_message.get("config", {})
    task = init_message.get("task")
    config_file_path = init_message.get("configFilePath")

    try:
        # Load config from YAML file if provided
        if config_file_path:
            try:
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
                bridge.send_notification("agent/warning", {"message": f"Config file not found: {e}"})
            except Exception as e:
                bridge.send_notification("agent/warning", {"message": f"Failed to load config file: {e}"})

        # Get model name from config or init message
        model_config = config.get("model", {})
        model_name = init_message.get("modelName") or model_config.get("model_name", "gpt-4")

        # Handle API key - if it's at top level, move it to model_kwargs for compatibility
        if "api_key" in model_config and "api_key" not in model_config.get("model_kwargs", {}):
            model_config.setdefault("model_kwargs", {})["api_key"] = model_config.pop("api_key")

        # Create model and environment
        model = get_model(model_name, model_config)
        env = LocalEnvironment(cwd=workspace_path, **config.get("env", {}))

        # Create agent
        agent = VSCodeAgent(bridge=bridge, model=model, env=env, **config.get("agent", {}))

        # Run agent
        bridge.send_notification("agent/started", {})
        exit_status, result = agent.run(task)

        # Send completion
        bridge.send_notification("agent/finished", {"exitStatus": exit_status, "result": result})

    except Exception as e:
        bridge.send_notification("agent/error", {"error": str(e), "traceback": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()

