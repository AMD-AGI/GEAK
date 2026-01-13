"""Agent for selecting the best patch from parallel runs."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


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
        self.log_file: Path | None = None
    
    def add_message(self, role: str, content: str, **kwargs):
        super().add_message(role, content, **kwargs)
        
        # Write to log file if available
        if self.log_file:
            try:
                log_content = f"\n{role.capitalize()}:\n{content}\n"
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_content)
                    f.flush()
            except Exception:
                pass
    
    def setup_selection_task(self, base_patch_dir: Path, num_parallel: int, metric: str | None) -> str:
        """Setup the task for selecting best patch."""
        self.patch_dir = base_patch_dir
        
        # Load all results from parallel runs
        self.all_results = {}
        for i in range(num_parallel):
            parallel_dir = base_patch_dir / f"parallel_{i}"
            results_file = parallel_dir / "results.json"
            if results_file.exists():
                try:
                    results_data = json.loads(results_file.read_text())
                    self.all_results[i] = {
                        "dir": parallel_dir,
                        "results": results_data,
                    }
                except Exception as e:
                    print(f"[SelectPatchAgent] Error reading results from parallel_{i}: {e}")
        
        if not self.all_results:
            return None
        
        # Get unified baseline from parallel 0
        unified_baseline = None
        if 0 in self.all_results:
            unified_baseline = self.all_results[0]["results"].get("patch_0")
        
        if not unified_baseline:
            return None
        
        # Collect all patches with their parallel_id
        all_patches = []
        for parallel_id, data in self.all_results.items():
            results = data["results"]
            
            for patch_name, patch_data in results.items():
                if patch_name.startswith("_"):
                    continue
                all_patches.append({
                    "parallel_id": parallel_id,
                    "patch_name": patch_name,
                    "test_passed": patch_data.get('test_passed', False),
                    "returncode": patch_data.get('returncode', -1),
                    "metric_result": json.dumps(patch_data['metric_result']) if patch_data.get("metric_result") else None,
                })
        
        all_patches.sort(key=lambda x: (x["parallel_id"], x["patch_name"]))
        
        print(f"[SelectPatchAgent] Processing {len(all_patches)} patches...")
        
        # Render task from template
        task = self.render_template(
            self.config.task_template,
            num_patches=len(all_patches),
            metric=metric,
            baseline={
                "test_passed": unified_baseline.get('test_passed', False),
                "returncode": unified_baseline.get('returncode', -1),
                "metric_result": json.dumps(unified_baseline['metric_result'], indent=2) if unified_baseline.get("metric_result") else None,
            },
            all_patches=all_patches,
            base_patch_dir=base_patch_dir,
        )
        
        return task
    
    def parse_action(self, response: dict) -> dict:
        """Parse bash command from response."""
        actions = re.findall(r"```(?:bash)?\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        # Try without code blocks
        lines = response["content"].strip().split("\n")
        for line in lines:
            if line.strip().startswith(("echo", "cat", "grep", "ls")):
                return {"action": line.strip(), **response}
        from minisweagent.agents.default import FormatError
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
    
    def extract_final_result(self) -> str | None:
        """Extract the final result from best_results.json written by agent."""
        if not self.patch_dir:
            return None
        
        best_results_file = self.patch_dir / "best_results.json"
        if not best_results_file.exists():
            print("[SelectPatchAgent] best_results.json not found, agent did not complete the task", flush=True)
            return None
        
        try:
            best_results = json.loads(best_results_file.read_text())
            best_patch_id = best_results.get("best_patch_id")
            if best_patch_id:
                return best_patch_id
        except json.JSONDecodeError as e:
            print(f"[SelectPatchAgent] Failed to parse best_results.json: {e}", flush=True)
        
        return None

