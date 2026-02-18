"""Agent for selecting the best patch from parallel runs."""

import os
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException, Submitted


@dataclass
class SelectPatchAgentConfig(AgentConfig):
    """Config loaded from mini_select_patch.yaml or provided kwargs."""

    task_template: str = ""


class SelectPatchAgent(DefaultAgent):
    """Agent that selects the best patch from parallel runs using multi-turn reasoning."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=SelectPatchAgentConfig, **kwargs)
        self.patch_dir: Path | None = None
        self.all_results: dict = {}

    def add_message(self, role: str, content: str, **kwargs):
        # DefaultAgent already logs assistant messages as:
        # "mini-swe-agent (step N, $COST): ..."
        # Keep select_agent.log concise: log assistant steps + their Observations.
        keep_user = role == "user" and content.lstrip().startswith("Observation:")
        if role != "assistant" and not keep_user and self.log_file:
            log_file = self.log_file
            self.log_file = None
            super().add_message(role, content, **kwargs)
            self.log_file = log_file
            return
        super().add_message(role, content, **kwargs)

    def parse_action(self, response: dict) -> dict:
        if response.get("content"):
            return super().parse_action(response)
        if response.get("tools"):
            from minisweagent.tools.submit import Submitted as ToolSubmitted

            tool_call = response["tools"]["function"]
            prev_cwd = os.getcwd()
            if self.patch_dir:
                os.chdir(self.patch_dir)
            try:
                try:
                    result = self.toolruntime.dispatch(tool_call=tool_call)
                except ToolSubmitted as e:
                    raise Submitted(str(e))
            finally:
                os.chdir(prev_cwd)
            return self._handle_tool_result(result)
        return super().parse_action(response)

    def has_finished(self, output: dict[str, str]):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            if not self.patch_dir or not (self.patch_dir / "best_results.json").exists():
                raise NonTerminatingException(
                    "best_results.json not found. Write best_results.json before echoing COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
                )
        super().has_finished(output)

    def setup_selection_task(self, base_patch_dir: Path, num_parallel: int, metric: str | None) -> str:
        """Setup the task for selecting best patch."""
        base_patch_dir = base_patch_dir.resolve()
        self.patch_dir = base_patch_dir

        # Count actual completed task/parallel directories for accurate hint
        actual_count = num_parallel
        task_dirs = sorted(base_patch_dir.glob("task_*"))
        parallel_dirs = sorted(base_patch_dir.glob("parallel_*"))
        total = len(task_dirs) + len(parallel_dirs)
        if total > 0:
            actual_count = total

        return self.render_template(
            self.config.task_template,
            metric=metric,
            num_parallel=actual_count,
            base_patch_dir=str(base_patch_dir),
        )

    def extract_final_result(self) -> str | None:
        """Extract the final result from best_results.json written by agent."""
        if not self.patch_dir:
            return None

        best_results_file = self.patch_dir / "best_results.json"
        if not best_results_file.exists():
            print("[SelectPatchAgent] best_results.json not found, agent did not complete the task", flush=True)
            return None

        try:
            import json

            best_results = json.loads(best_results_file.read_text())
            best_patch_id = best_results.get("best_patch_id")
            if best_patch_id:
                return best_patch_id
        except json.JSONDecodeError as e:
            print(f"[SelectPatchAgent] Failed to parse best_results.json: {e}", flush=True)

        return None


def run_select_patch(
    patch_dir: Path,
    num_parallel: int,
    metric: str | None,
    model: "Model",
) -> tuple[SelectPatchAgent, str | None]:
    """Create a SelectPatchAgent, run patch selection, and return ``(agent, best_patch_id)``.

    Shared by ``ParallelAgent._select_best_from_parallel_runs`` and the CLI
    ``main()`` below so the setup/run/extract logic lives in one place.
    """
    from minisweagent.config import load_agent_config
    from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

    agent_config, _ = load_agent_config("mini_select_patch")

    env_config = LocalEnvironmentConfig(cwd=str(patch_dir))
    env = LocalEnvironment(**env_config.__dict__)

    agent = SelectPatchAgent(model, env, **agent_config)
    agent.log_file = patch_dir / "select_agent.log"

    task = agent.setup_selection_task(patch_dir, num_parallel, metric)
    if task is None:
        return agent, None

    try:
        agent.run(task)
    except Exception:
        pass

    return agent, agent.extract_final_result()


# ---------------------------------------------------------------------------
# Standalone CLI
#
# Usage:
#   python -m minisweagent.agents.select_patch_agent \
#       --patch-dir ./patches/run7 \
#       --metric "Compare per-kernel latency; lower is better"
# ---------------------------------------------------------------------------


def main():
    import argparse
    import json
    import sys

    from minisweagent.config import load_agent_config
    from minisweagent.models import get_model

    parser = argparse.ArgumentParser(
        description="Select the best patch from parallel optimization runs (standalone)",
    )
    parser.add_argument(
        "--patch-dir",
        required=True,
        help="Base directory containing task_*/parallel_* subdirectories with patches",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric description for the LLM (e.g. 'compare per-kernel latency; lower is better')",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (default: uses mini_select_patch.yaml config)",
    )

    args = parser.parse_args()

    patch_dir = Path(args.patch_dir).resolve()
    if not patch_dir.is_dir():
        print(f"ERROR: patch directory not found: {args.patch_dir}", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted(patch_dir.glob("task_*"))
    parallel_dirs = sorted(patch_dir.glob("parallel_*"))
    num_parallel = len(task_dirs) + len(parallel_dirs)
    if num_parallel == 0:
        print("ERROR: no task_* or parallel_* directories found in patch-dir", file=sys.stderr)
        sys.exit(1)

    _, model_config = load_agent_config("mini_select_patch")
    model = get_model(args.model, model_config)
    metric = args.metric or "Extract performance metrics and calculate the best speedup."

    print(
        f"[SelectPatchAgent CLI] Starting patch selection\n"
        f"  patch_dir:   {patch_dir}\n"
        f"  runs found:  {num_parallel} ({len(task_dirs)} task_*, {len(parallel_dirs)} parallel_*)\n"
        f"  metric:      {metric}",
        flush=True,
    )

    _, best_patch_id = run_select_patch(patch_dir, num_parallel, metric, model)

    if best_patch_id:
        print(f"\nBest patch: {best_patch_id}")
        best_results_file = patch_dir / "best_results.json"
        if best_results_file.exists():
            best_results = json.loads(best_results_file.read_text())
            print(json.dumps(best_results, indent=2))
    else:
        print("WARNING: agent did not produce best_results.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
