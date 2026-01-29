"""Agent for selecting the best patch from parallel runs."""

import os
import re
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

        parallel_dirs: list[Path] = []
        for i in range(num_parallel):
            d = base_patch_dir / f"parallel_{i}"
            if d.exists():
                parallel_dirs.append(d)

        if not parallel_dirs:
            return None

        baseline_parallel = base_patch_dir / "parallel_0"
        baseline_test_log = baseline_parallel / "patch_0_test.txt"
        if not baseline_test_log.exists():
            return None

        all_patches: list[dict] = []
        for parallel_dir in parallel_dirs:
            m = re.match(r"parallel_(\\d+)$", parallel_dir.name)
            parallel_id = int(m.group(1)) if m else -1

            for test_file in sorted(parallel_dir.glob("patch_*_test.txt")):
                patch_name = test_file.name.removesuffix("_test.txt")
                all_patches.append(
                    {
                        "parallel_id": parallel_id,
                        "patch_name": patch_name,
                    }
                )

        if not all_patches:
            return None

        all_patches.sort(key=lambda x: (x["parallel_id"], x["patch_name"]))
        print(f"[SelectPatchAgent] Processing {len(all_patches)} patches (from test logs)...")

        return self.render_template(
            self.config.task_template,
            num_patches=len(all_patches),
            metric=metric,
            baseline={
                "patch_id": "parallel_0/patch_0",
                "test_output_file": str(baseline_test_log),
            },
            all_patches=all_patches,
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

