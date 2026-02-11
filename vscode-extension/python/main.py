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
    from minisweagent.config import get_config_path, builtin_config_dir
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model
    from vscode_agent import VSCodeInteractiveAgent, VSCodeStrategyAgent, VSCodeBridge
finally:
    # Restore original stdout for JSON-RPC communication
    sys.stdout = _original_stdout

# Keep a dedicated handle for JSON-RPC so that print() from agent/tools goes to stderr
_jsonrpc_stdout = sys.stdout


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    """Run the agent with configuration from VS Code."""
    print(f"[DEBUG] VSCode Agent starting, Python version: {sys.version}", file=sys.stderr)
    print(f"[DEBUG] Working directory: {os.getcwd()}", file=sys.stderr)
    
    # Redirect stdout to stderr so only the bridge writes JSON-RPC to the real stdout
    sys.stdout = sys.stderr
    bridge = VSCodeBridge(stdio_out=_jsonrpc_stdout)
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
    template_name = init_message.get("templateName")
    vscode_config = init_message.get("config", {})
    strategy_mode = init_message.get("strategyMode", {})
    task = init_message.get("task")
    
    # VSCode mode always uses geak.yaml as default user config
    user_config_file = "geak.yaml"

    try:
        # 1. Load template configuration
        if not template_name:
            # Fallback for backward compatibility
            template_name = "mini_kernel_strategy_list.yaml"
        
        template_path = builtin_config_dir / template_name
        print(f"[INFO] Loading template: {template_name}", file=sys.stderr)
        template_config = yaml.safe_load(template_path.read_text())
        bridge.send_notification("agent/info", {"message": f"Using template: {template_name}"})
        
        # 2. Load geak.yaml as default user config
        file_config = {}
        try:
            print(f"[INFO] Loading default user config: {user_config_file}", file=sys.stderr)
            user_config_path = get_config_path(user_config_file)
            file_config = yaml.safe_load(user_config_path.read_text())
            bridge.send_notification("agent/info", {"message": f"Loaded user config from {user_config_path}"})
        except FileNotFoundError as e:
            print(f"[WARNING] Default user config not found: {e}", file=sys.stderr)
            bridge.send_notification("agent/warning", {"message": f"Default user config not found: {e}"})
        except Exception as e:
            print(f"[WARNING] Failed to load default user config: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            bridge.send_notification("agent/warning", {"message": f"Failed to load user config: {e}"})
        
        # 3. Merge configs with priority: template → geak.yaml → vscode settings
        config = _deep_merge(template_config, file_config)
        config = _deep_merge(config, vscode_config)
        print(f"[INFO] Config priority: template → geak.yaml → vscode settings", file=sys.stderr)

        # 3b. Apply tool toggles: strategy_manager and profiling (agent + model)
        strategy_enabled = strategy_mode.get("enabled", True)
        if strategy_enabled:
            config.setdefault("agent", {})["use_strategy_manager"] = True
            config.setdefault("model", {})["use_strategy_manager"] = True
            print("[INFO] Strategy mode enabled: use_strategy_manager=True for agent and model", file=sys.stderr)
        tools_cfg = init_message.get("tools", {})
        profiling_cfg = tools_cfg.get("profiling", {})
        if profiling_cfg.get("enabled"):
            ptype = profiling_cfg.get("type", "profiling")
            config.setdefault("agent", {})["profiling_type"] = ptype
            config.setdefault("model", {})["profiling"] = True
            print(f"[INFO] Profiling tool enabled: type={ptype}", file=sys.stderr)
        
        # Debug: Print model config before processing
        print(f"[DEBUG] VSCode config model section: {vscode_config.get('model', {})}", file=sys.stderr)
        print(f"[DEBUG] After merge, model config: {config.get('model', {})}", file=sys.stderr)

        # 4. Get model name from config
        model_config = config.get("model", {})
        model_name = init_message.get("modelName") or model_config.get("model_name", "gpt-4")
        print(f"[INFO] Using model: {model_name}", file=sys.stderr)

        # Handle API key - if it's at top level, move it to model_kwargs for compatibility
        if "api_key" in model_config and "api_key" not in model_config.get("model_kwargs", {}):
            model_config.setdefault("model_kwargs", {})["api_key"] = model_config.pop("api_key")
            print(f"[DEBUG] Moved API key from model_config to model_kwargs", file=sys.stderr)
        
        print(f"[DEBUG] Final model_config before get_model: {model_config}", file=sys.stderr)

        # Get agent configuration
        agent_config = config.get("agent", {})
        
        # 5. Create model and environment
        print(f"[INFO] Creating model and environment", file=sys.stderr)
        model = get_model(model_name, model_config)
        env = LocalEnvironment(cwd=workspace_path, **config.get("env", {}))

        # 6. Create agent based on strategy mode
        strategy_file_path = strategy_mode.get("filePath", ".optimization_strategies.md")
        
        if strategy_enabled:
            print(f"[INFO] Creating VSCodeStrategyAgent (file: {strategy_file_path})", file=sys.stderr)
            agent = VSCodeStrategyAgent(
                bridge=bridge,
                model=model,
                env=env,
                strategy_file_path=strategy_file_path,
                **agent_config
            )
        else:
            print(f"[INFO] Creating VSCodeInteractiveAgent (no strategy support)", file=sys.stderr)
            agent = VSCodeInteractiveAgent(
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

