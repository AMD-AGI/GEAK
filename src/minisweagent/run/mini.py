#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any


class TeeOutput:
    """捕获 stdout/stderr 到 buffer，同时保持终端输出"""
    def __init__(self, original):
        self.terminal = original
        self.buffer = StringIO()
    
    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)
    
    def flush(self):
        self.terminal.flush()
    
    def getvalue(self):
        return self.buffer.getvalue()

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
from minisweagent.agents.interactive_textual import TextualAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
_logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT = _logs_dir / "last_mini_run.traj.json"
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))
_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
There are two different user interfaces:

[bold green]mini[/bold green] Simple REPL-style interface
[bold green]mini -v[/bold green] Pager-style interface (Textual)

MCP integration (AMD AI DevTool):

[bold green]mini --mcp[/bold green] Enable MCP integration
[bold green]mini --mcp -d[/bold green] Enable MCP integration with debug output

More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    visual: bool = typer.Option(False, "-v", "--visual", help="Toggle (pager-style) UI (Textual) depending on the MSWEA_VISUAL_MODE_DEFAULT environment setting",),
    model_name: str | None = typer.Option( None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    mcp: bool = typer.Option(False, "--mcp", help="Enable MCP integration (AMD AI DevTool)"),
    debug: bool = typer.Option(False, "-d", "--debug", help="Enable debug output (only with --mcp)"),
) -> Any:
    # fmt: on
    # 捕获所有 print 输出到 trajectory
    tee_out, tee_err = TeeOutput(sys.stdout), TeeOutput(sys.stderr)
    sys.stdout, sys.stderr = tee_out, tee_err
    
    configure_if_first_time()
    config_path = get_config_path(config_spec)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())

    if not task:
        console.print("[bold yellow]What do you want to do?")
        task = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
        console.print("[bold green]Got that, thanks![/bold green]")

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    model = get_model(model_name, config.get("model", {}))
    
    # Print model info
    _model_name = getattr(model.config, 'model_name', 'unknown')
    _api_key = getattr(model.config, 'api_key', None)
    if not _api_key:
        # Try to get the actual API key used (for amd_llm model)
        import os as _os
        _api_key = _os.getenv("AMD_LLM_API_KEY") or _os.getenv("LLM_GATEWAY_KEY") or _os.getenv("ANTHROPIC_API_KEY")
    _api_key_display = f"{_api_key[:8]}..." if _api_key and len(_api_key) > 8 else _api_key or "Not set"
    console.print(f"\\[mini-swe-agent] Using model: [bold cyan]{_model_name}[/bold cyan], API key: [bold cyan]{_api_key_display}[/bold cyan]")
    
    # MCP integration
    extra_agent_kwargs = {}
    if mcp:
        try:
            from dataclasses import dataclass
            from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
            from minisweagent.mcp_integration.prompts import SYSTEM_TEMPLATE, INSTANCE_TEMPLATE
            from minisweagent.mcp_integration.run_agent import DebugMCPEnvironment
        except ImportError as e:
            console.print(f"[red]Error: MCP integration requires 'amd-ai-devtool'. Please install it first.[/red]")
            console.print(f"[red]Import error: {e}[/red]")
            raise typer.Exit(1)
        
        # Create MCP-compatible config class for InteractiveAgent
        @dataclass
        class MCPInteractiveAgentConfig(InteractiveAgentConfig):
            system_template: str = SYSTEM_TEMPLATE
            instance_template: str = INSTANCE_TEMPLATE
        
        if debug:
            env = DebugMCPEnvironment(**config.get("env", {}))
            console.print("[bold yellow]🐛 Debug mode enabled[/bold yellow]")
        else:
            env = MCPEnabledEnvironment(**config.get("env", {}))
        
        extra_agent_kwargs["config_class"] = MCPInteractiveAgentConfig
        
        # Remove yaml prompts so MCP prompts take effect
        agent_config = config.get("agent", {}).copy()
        agent_config.pop("system_template", None)
        agent_config.pop("instance_template", None)
        
        console.print("[bold green]🔌 MCP integration enabled[/bold green]")
    else:
        env = LocalEnvironment(**config.get("env", {}))
        agent_config = config.get("agent", {})

    # Both visual flag and the MSWEA_VISUAL_MODE_DEFAULT flip the mode, so it's essentially a XOR
    agent_class = InteractiveAgent
    if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
        agent_class = TextualAgent
        if mcp:
            console.print("[yellow]Warning: MCP integration with -v (Textual UI) is not fully supported yet.[/yellow]")

    agent = agent_class(model, env, **agent_config, **extra_agent_kwargs)
    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(task)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if output:
            # 收集 console 日志
            console_logs = tee_out.getvalue() + tee_err.getvalue()
            if console_logs:
                extra_info = extra_info or {}
                extra_info["console_logs"] = console_logs
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
    return agent


if __name__ == "__main__":
    app()
