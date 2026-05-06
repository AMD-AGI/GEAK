"""First-time global config setup helper for the GEAK CLI.

Exposes a single helper, ``configure_if_first_time``, that ``cli.py``
invokes on every ``geak`` launch.  If the user has not already
configured a model / API key in ``global_config_file`` (and no relevant
API-key environment variables are set), it walks them through an
interactive setup and writes the result.

Previously this module also exposed a standalone Typer app
(``mini-extra config setup|set|unset|edit``), but that CLI was never
registered in ``pyproject.toml`` and had zero production callers.  It
was removed during CLI consolidation.  If an operator needs to edit
the global config file by hand, they can open
``global_config_file`` directly (``$EDITOR`` or any text editor).
"""

import logging
import os

from dotenv import set_key
from prompt_toolkit import prompt
from rich.console import Console
from rich.rule import Rule

from minisweagent import global_config_file

logger = logging.getLogger(__name__)
console = Console(highlight=False)


_SETUP_HELP = """To get started, we need to set up your global config file.

You can edit it manually at the path shown above or re-run this setup by
deleting the ``MSWEA_CONFIGURED`` key from that file.

This setup will ask you for your model and an API key.

Here's a few popular models and the required API keys:

[bold green]anthropic/claude-sonnet-4-5-20250929[/bold green] ([bold green]ANTHROPIC_API_KEY[/bold green])
[bold green]openai/gpt-5[/bold green] or [bold green]openai/gpt-5-mini[/bold green] ([bold green]OPENAI_API_KEY[/bold green])
[bold green]gemini/gemini-2.5-pro[/bold green] ([bold green]GEMINI_API_KEY[/bold green])

[bold]Note: Please always include the provider in the model name.[/bold]

[bold yellow]You can leave any setting blank to skip it.[/bold yellow]

More information at https://mini-swe-agent.com/latest/quickstart/
"""


_API_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "AMD_LLM_API_KEY",
    "LLM_API_KEY",
)


def configure_if_first_time() -> None:
    """Run the interactive setup once per user, on the first geak invocation."""
    if os.getenv("MSWEA_CONFIGURED"):
        return
    if any(os.getenv(k) for k in _API_KEY_NAMES):
        return
    console.print(Rule())
    logger.info("First-time configuration: running setup.")
    _setup()
    console.print(Rule())


def _setup() -> None:
    """Interactive setup for the global config file."""
    console.print(_SETUP_HELP)
    console.print(f"[dim]Writing to: {global_config_file}[/dim]")
    default_model = prompt(
        "Enter your default model (e.g., anthropic/claude-sonnet-4-5-20250929): ",
        default=os.getenv("MSWEA_MODEL_NAME", ""),
    ).strip()
    if default_model:
        set_key(global_config_file, "MSWEA_MODEL_NAME", default_model)
    console.print(
        "[bold yellow]If you already have your API keys set as environment variables, you can skip the next question.[/bold yellow]"
    )
    key_name = prompt("Enter your API key name (e.g., ANTHROPIC_API_KEY): ").strip()
    key_value = None
    if key_name:
        key_value = prompt(
            "Enter your API key value (e.g., sk-1234567890): ",
            default=os.getenv(key_name, ""),
        ).strip()
        if key_value:
            set_key(global_config_file, key_name, key_value)
    if not key_value:
        console.print(
            "[bold red]API key setup not completed.[/bold red] "
            "Totally fine if you have your keys as environment variables."
        )
    set_key(global_config_file, "MSWEA_CONFIGURED", "true")
    console.print("\n[bold yellow]Config finished.[/bold yellow]")


__all__ = ["configure_if_first_time"]
