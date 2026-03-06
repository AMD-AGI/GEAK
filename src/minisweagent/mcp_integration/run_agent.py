# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""
Run mini-swe-agent with RAG integration.
Reads configuration from mini.yaml automatically.
DEBUG VERSION - prints detailed info at each step.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent, AgentConfig, FormatError, Submitted
from minisweagent.models import get_model

from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
from minisweagent.mcp_integration.prompts import SYSTEM_TEMPLATE, INSTANCE_TEMPLATE


# Default config path
CONFIG_PATH = Path(__file__).parent.parent / "config" / "mini.yaml"


@dataclass
class MCPAgentConfig(AgentConfig):
    """Agent configuration with RAG-aware prompts."""
    system_template: str = SYSTEM_TEMPLATE
    instance_template: str = INSTANCE_TEMPLATE


class DebugMCPEnvironment(MCPEnabledEnvironment):
    """RAG Environment with debug output."""
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
        print(f"\n{'='*60}")
        print(f"🔧 [ENV] Executing command:")
        print(f"{'='*60}")
        print(command)
        print(f"{'='*60}")
        
        # Check if it's RAG or bash
        if command.strip().startswith(self.config.mcp_prefix):
            print(f"✅ [ENV] This is a RAG command! Will route to RAG retrieval.")
        else:
            print(f"⚠️  [ENV] This is a BASH command, not RAG.")
        
        result = super().execute(command, cwd, timeout=timeout)
        
        print(f"\n📤 [ENV] Command output (first 500 chars):")
        print(f"{'-'*60}")
        print(result.get("output", "")[:500])
        print(f"{'-'*60}")
        print(f"📤 [ENV] Return code: {result.get('returncode')}")
        
        return result


class DebugAgent(DefaultAgent):
    """Agent with debug output at each step."""
    
    def run(self, task: str, **kwargs):
        print(f"\n{'#'*60}")
        print(f"# DEBUG: Starting agent run")
        print(f"{'#'*60}")
        print(f"\n📝 [AGENT] Task: {task}")
        return super().run(task, **kwargs)
    
    def query(self):
        print(f"\n{'='*60}")
        print(f"🤖 [AGENT] Querying LLM (call #{self.model.n_calls + 1})...")
        print(f"{'='*60}")
        
        response = super().query()
        
        print(f"\n📥 [AGENT] LLM Response (content):")
        print(f"{'-'*60}")
        content = response.get("content", "")
        print(content[:1000] + ("..." if len(content) > 1000 else ""))
        print(f"{'-'*60}")
        
        return response
    
    def parse_action(self, response):
        content = response.get("content", "")
        actions = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
        
        print(f"\n🔍 [AGENT] Parsing actions from response...")
        print(f"   Found {len(actions)} action(s) in triple backticks")
        
        if actions:
            for i, action in enumerate(actions):
                print(f"   Action {i+1}: {action[:100]}...")
        
        return super().parse_action(response)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration from yaml file."""
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    return {}


def create_agent(
    model_name: str | None = None,
    config_path: Path = CONFIG_PATH,
    debug: bool = False,
) -> DefaultAgent:
    """Create a RAG-enabled agent using mini.yaml configuration."""
    config = load_config(config_path)
    model_config = config.get("model", {})
    
    model = get_model(model_name, model_config)
    
    if debug:
        env = DebugMCPEnvironment()
        agent = DebugAgent(model=model, env=env, config_class=MCPAgentConfig)
    else:
        env = MCPEnabledEnvironment()
        agent = DefaultAgent(model=model, env=env, config_class=MCPAgentConfig)
    
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent with AMD AI DevTool RAG integration"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task for the agent to perform"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Override model name (default: from mini.yaml)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to config file (default: {CONFIG_PATH})"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Show config info
    config = load_config(args.config)
    model_config = config.get("model", {})
    model_name = args.model or model_config.get("model_name", "not set")
    model_class = model_config.get("model_class", "litellm")
    print(f"📋 Config: {args.config}")
    print(f"🤖 Model: {model_name} (class: {model_class})")
    
    # Debug: print loaded prompts
    if args.debug:
        print(f"\n{'#'*60}")
        print("# DEBUG: Loaded Prompts")
        print(f"{'#'*60}")
        print(f"\n📄 SYSTEM_TEMPLATE (first 300 chars):")
        print(f"{'-'*60}")
        print(SYSTEM_TEMPLATE[:300])
        print(f"{'-'*60}")
        print(f"\n📄 INSTANCE_TEMPLATE (first 300 chars):")
        print(f"{'-'*60}")
        print(INSTANCE_TEMPLATE[:300])
        print(f"{'-'*60}")
    
    agent = create_agent(
        model_name=args.model,
        config_path=args.config,
        debug=args.debug,
    )
    
    # Debug: verify agent config
    if args.debug:
        print(f"\n📄 Agent's system_template (first 200 chars):")
        print(f"{'-'*60}")
        print(agent.config.system_template[:200])
        print(f"{'-'*60}")
    
    if args.interactive:
        print("\n🚀 RAG-enabled mini-swe-agent (interactive mode)")
        if args.debug:
            print("🐛 DEBUG MODE ENABLED")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                task = input("Task: ").strip()
                if task.lower() in ["quit", "exit", "q"]:
                    break
                if not task:
                    continue
                    
                status, result = agent.run(task)
                print(f"\n[{status}]\n{result}\n")
                
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        if not args.task:
            parser.print_help()
            return
            
        status, result = agent.run(args.task)
        print(f"[{status}]\n{result}")


if __name__ == "__main__":
    main()
