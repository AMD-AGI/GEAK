#!/usr/bin/env python3
"""Bridge script to interact with StrategyManager from VS Code extension."""

import sys
import json
from pathlib import Path

# Add minisweagent to path
# Path: vscode-extension/python/strategy_bridge.py -> ../../src (mini-swe-agent/src)
bridge_dir = Path(__file__).resolve().parent
mini_swe_agent_root = bridge_dir.parent.parent
sys.path.insert(0, str(mini_swe_agent_root / "src"))

from minisweagent.tools.strategy_manager import StrategyManager

def get_strategy_list(file_path: str = ".optimization_strategies.md") -> dict:
    """Get strategy list from file."""
    manager = StrategyManager(file_path)
    
    if not manager.exists():
        return {"exists": False, "strategies": []}
    
    try:
        strategy_list = manager.load()
        
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
                "priority": strategy.priority,  # Add priority field
                "expected": strategy.expected,
                "target": strategy.target,
                "result": strategy.result,
                "details": strategy.details
            })
        
        return result
        
    except Exception as e:
        return {"exists": True, "error": str(e), "strategies": []}

def mark_strategy(file_path: str, index: int, status: str, result: str = None, details: str = None) -> dict:
    """Mark strategy status."""
    try:
        manager = StrategyManager(file_path)
        manager.mark_status(index, status, result, details)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "get":
        file_path = sys.argv[2] if len(sys.argv) > 2 else ".optimization_strategies.md"
        result = get_strategy_list(file_path)
        print(json.dumps(result))
        
    elif command == "mark":
        if len(sys.argv) < 5:
            print(json.dumps({"error": "Usage: mark <file_path> <index> <status> [result] [details]"}))
            sys.exit(1)
        
        file_path = sys.argv[2]
        index = int(sys.argv[3])
        status = sys.argv[4]
        result_text = sys.argv[5] if len(sys.argv) > 5 else None
        details = sys.argv[6] if len(sys.argv) > 6 else None
        
        result = mark_strategy(file_path, index, status, result_text, details)
        print(json.dumps(result))
        
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()


