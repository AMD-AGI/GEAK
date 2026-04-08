"""
Run mini-swe-agent with RAG integration.
Reads configuration from mini.yaml automatically.
DEBUG VERSION - logs detailed info at each step.
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
from minisweagent.mcp_integration.prompts import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE
from minisweagent.models import get_model

logger = logging.getLogger(__name__)

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
        logger.info("\n%s", "=" * 60)
        logger.info("🔧 [ENV] Executing command:")
        logger.info("%s", "=" * 60)
        logger.info("%s", command)
        logger.info("%s", "=" * 60)

        # Check if it's RAG or bash
        if command.strip().startswith(self.config.mcp_prefix):
            logger.info("✅ [ENV] This is a RAG command! Will route to RAG retrieval.")
        else:
            logger.info("⚠️  [ENV] This is a BASH command, not RAG.")

        result = super().execute(command, cwd, timeout=timeout)

        logger.info("\n📤 [ENV] Command output (first 500 chars):")
        logger.info("%s", "-" * 60)
        logger.info("%s", result.get("output", "")[:500])
        logger.info("%s", "-" * 60)
        logger.info("📤 [ENV] Return code: %s", result.get("returncode"))

        return result


class DebugAgent(DefaultAgent):
    """Agent with debug output at each step."""

    def run(self, task: str, **kwargs):
        logger.info("\n%s", "#" * 60)
        logger.info("# DEBUG: Starting agent run")
        logger.info("%s", "#" * 60)
        logger.info("\n📝 [AGENT] Task: %s", task)
        return super().run(task, **kwargs)

    def query(self):
        logger.info("\n%s", "=" * 60)
        logger.info("🤖 [AGENT] Querying LLM (call #%s)...", self.model.n_calls + 1)
        logger.info("%s", "=" * 60)

        response = super().query()

        logger.info("\n📥 [AGENT] LLM Response (content):")
        logger.info("%s", "-" * 60)
        content = response.get("content", "")
        logger.info("%s", content[:1000] + ("..." if len(content) > 1000 else ""))
        logger.info("%s", "-" * 60)

        return response

    def parse_action(self, response):
        content = response.get("content", "")
        actions = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)

        logger.info("\n🔍 [AGENT] Parsing actions from response...")
        logger.info("   Found %s action(s) in triple backticks", len(actions))

        if actions:
            for i, action in enumerate(actions):
                logger.info("   Action %s: %s...", i + 1, action[:100])

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
    parser = argparse.ArgumentParser(description="Run mini-swe-agent with AMD AI DevTool RAG integration")
    parser.add_argument("task", nargs="?", default=None, help="Task for the agent to perform")
    parser.add_argument("--model", "-m", default=None, help="Override model name (default: from mini.yaml)")
    parser.add_argument(
        "--config", "-c", type=Path, default=CONFIG_PATH, help=f"Path to config file (default: {CONFIG_PATH})"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Show config info
    config = load_config(args.config)
    model_config = config.get("model", {})
    model_name = args.model or model_config.get("model_name", "not set")
    model_class = model_config.get("model_class", "litellm")
    logger.info("📋 Config: %s", args.config)
    logger.info("🤖 Model: %s (class: %s)", model_name, model_class)

    # Debug: log loaded prompts
    if args.debug:
        logger.info("\n%s", "#" * 60)
        logger.info("# DEBUG: Loaded Prompts")
        logger.info("%s", "#" * 60)
        logger.info("\n📄 SYSTEM_TEMPLATE (first 300 chars):")
        logger.info("%s", "-" * 60)
        logger.info("%s", SYSTEM_TEMPLATE[:300])
        logger.info("%s", "-" * 60)
        logger.info("\n📄 INSTANCE_TEMPLATE (first 300 chars):")
        logger.info("%s", "-" * 60)
        logger.info("%s", INSTANCE_TEMPLATE[:300])
        logger.info("%s", "-" * 60)

    agent = create_agent(
        model_name=args.model,
        config_path=args.config,
        debug=args.debug,
    )

    # Debug: verify agent config
    if args.debug:
        logger.info("\n📄 Agent's system_template (first 200 chars):")
        logger.info("%s", "-" * 60)
        logger.info("%s", agent.config.system_template[:200])
        logger.info("%s", "-" * 60)

    if args.interactive:
        logger.info("\n🚀 RAG-enabled mini-swe-agent (interactive mode)")
        if args.debug:
            logger.info("🐛 DEBUG MODE ENABLED")
        logger.info("Type 'quit' to exit\n")

        while True:
            try:
                task = input("Task: ").strip()
                if task.lower() in ["quit", "exit", "q"]:
                    break
                if not task:
                    continue

                status, result = agent.run(task)
                logger.info("\n[%s]\n%s\n", status, result)

            except KeyboardInterrupt:
                logger.info("\nBye!")
                break
    else:
        if not args.task:
            parser.print_help()
            return

        status, result = agent.run(args.task)
        logger.info("[%s]\n%s", status, result)


if __name__ == "__main__":
    main()
