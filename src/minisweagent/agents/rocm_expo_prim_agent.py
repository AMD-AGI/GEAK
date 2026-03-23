"""ROCm expo prim agent: generates top1/top2/top3 task prompts using mini_reset_goal_prim config.

This agent runs headless (DefaultAgent, no console printing) in a given prompt
folder, reads kernel_task.txt as the task, and is expected to create
*_top1.txt, *_top2.txt, *_top3.txt in that folder for downstream ROCm expo iterations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

if TYPE_CHECKING:
    from rich.console import Console

PRIM_CONFIG_NAME = "mini_reset_goal_prim.yaml"
KERNEL_TASK_FILENAME = "kernel_task.txt"


@dataclass
class RocmExpoPrimAgentConfig(AgentConfig):
    """Config for ROCm expo prim agent; loaded from mini_reset_goal_prim.yaml."""


class RocmExpoPrimAgent(DefaultAgent):
    """Agent that analyzes a task and generates top1/top2/top3 variant prompt files.

    Uses mini_reset_goal_prim.yaml. Runs with cwd set to the prompt folder;
    expects kernel_task.txt there and creates *_top1.txt, *_top2.txt, *_top3.txt.
    """

    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, config_class=RocmExpoPrimAgentConfig, **kwargs)


def run_rocm_expo_prim_agent(
    prompt_folder: Path,
    task_content: str,
    model_name: str | None,
    yolo: bool,
    console: Console,
) -> list[Path]:
    """Run the ROCm expo prim agent to generate top1/top2/top3 prompt files.

    The agent runs with cwd=prompt_folder. The folder must already contain
    kernel_task.txt with the task description. The agent follows the prim yaml
    and is expected to create files ending with top1.txt, top2.txt, top3.txt
    in that folder.

    Verifies that *top1.txt, *top2.txt, *top3.txt exist and returns their paths
    in order (raises FileNotFoundError if any are missing).
    """
    prim_path = builtin_config_dir / PRIM_CONFIG_NAME
    if not prim_path.exists():
        raise FileNotFoundError(f"Prim config not found: {prim_path}")
    prim_config = yaml.safe_load(prim_path.read_text()) or {}
    prim_config.setdefault("agent", {})["mode"] = (
        "yolo" if yolo else prim_config.get("agent", {}).get("mode", "confirm")
    )
    if model_name:
        prim_config.setdefault("model", {})["model_name"] = model_name

    work_dir = Path(prompt_folder).resolve()
    prim_task_content = (
        f"Working directory: {work_dir}\n"
        f"The task description is in {KERNEL_TASK_FILENAME}. Use it as the main task content.\n"
        f"Create three variant prompt files in this directory: "
        f"*_top1.txt, *_top2.txt, *_top3.txt (same base name as the task file, with your "
        f"recommended library enforcement for each of the top 3 strategies).\n\n"
        f"{task_content}"
    )

    console.print("[bold cyan]--- ROCm expo prim: generating top1/top2/top3 task prompts ---[/bold cyan]")
    console.print(f"[dim]Config: {PRIM_CONFIG_NAME}, cwd: {work_dir}[/dim]")

    model = get_model(model_name, prim_config.get("model", {}))
    env_kwargs = prim_config.get("env", {})
    env = LocalEnvironment(**env_kwargs)
    env.config.cwd = str(work_dir)

    agent_config = prim_config.get("agent", {})
    allowed_keys = {f.name for f in dataclasses.fields(AgentConfig)}
    filtered_config = {k: v for k, v in agent_config.items() if k in allowed_keys}
    agent = RocmExpoPrimAgent(model, env, **filtered_config)
    agent.log_file = work_dir / "prim_agent.log"
    agent.run(prim_task_content)

    task_paths: list[Path] = []
    for suffix in ("top1.txt", "top2.txt", "top3.txt"):
        matches = sorted(work_dir.glob(f"*{suffix}"))
        if not matches:
            raise FileNotFoundError(
                f"No file ending with {suffix} found in {work_dir}. "
                "ROCm expo prim should create *top1.txt, *top2.txt, *top3.txt."
            )
        task_paths.append(matches[0])
    return task_paths
