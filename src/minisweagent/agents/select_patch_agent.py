"""Agent for selecting the best patch from parallel runs."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


@dataclass
class SelectPatchAgentConfig(AgentConfig):
    system_template: str = (
        "You are an expert at analyzing kernel patches and their performance results. "
        "Your task is to select the best patch from multiple parallel runs based on speedup metrics and test results."
        "If you meet any issue, please try to fix it and select the best patch again."
    )
    instance_template: str = (
        "{{task}}\n\n"
        "Please analyze the patches and respond with a bash command that outputs your analysis.\n"
        "When you have made your final decision, save your results to best_results.json and output:\n"
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT:parallel_<id>/patch_<name>'\n"
        "For example: echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT:parallel_0/patch_12'\n\n"
        "The best_results.json file should contain the selected patch data with the following structure:\n"
        "{\n"
        '  "best_patch_id": "parallel_<id>/patch_<name>",\n'
        '  "best_patch_speedup": <float>,\n'
        '  "best_patch_file": "<absolute_path_to_patch_file>",\n'
        '  "best_patch_test_output": "<absolute_path_to_test_output_file>",\n'
        '  "llm_selection_analysis": "<your analysis>"\n'
        "}\n"
    )
    step_limit: int = 10
    cost_limit: float = 1.0


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
            parallel_dir = data["dir"]
            
            for patch_name, patch_data in results.items():
                if patch_name.startswith("_"):
                    continue
                all_patches.append({
                    "parallel_id": parallel_id,
                    "patch_name": patch_name,
                    "patch_data": patch_data,
                    "parallel_dir": parallel_dir,
                })
        
        all_patches.sort(key=lambda x: (x["parallel_id"], x["patch_name"]))
        
        print(f"[SelectPatchAgent] Processing {len(all_patches)} patches...")
        
        # Build task description for the agent
        task_parts = [
            f"You need to select the best patch from {len(all_patches)} patches.\n\n"
        ]
        
        if metric:
            task_parts.append(f"Optimization metric: {metric}\n\n")
        
        task_parts.append("## Baseline (parallel_0/patch_0)\n")
        task_parts.append(f"Test passed: {unified_baseline.get('test_passed', False)}\n")
        task_parts.append(f"Return code: {unified_baseline.get('returncode', -1)}\n")
        if "metric_result" in unified_baseline and unified_baseline["metric_result"]:
            task_parts.append(f"Metric result: {json.dumps(unified_baseline['metric_result'], indent=2)}\n")
        task_parts.append("\n")
        
        task_parts.append("## All Patches Summary:\n\n")
        for patch_info in all_patches:
            parallel_id = patch_info["parallel_id"]
            patch_name = patch_info["patch_name"]
            patch_data = patch_info["patch_data"]
            task_parts.append(
                f"- parallel_{parallel_id}/{patch_name}: "
                f"test_passed={patch_data.get('test_passed', False)}, "
                f"returncode={patch_data.get('returncode', -1)}"
            )
            if patch_data.get("metric_result"):
                task_parts.append(f", metric={json.dumps(patch_data['metric_result'])}")
            task_parts.append("\n")
        
        task_parts.append("\n")
        task_parts.append(
            "You can use bash commands to:\n"
            "1. Read patch files: cat {patch_dir}/parallel_*/patch_*.patch\n"
            "2. Read test outputs: cat {patch_dir}/parallel_*/patch_*_test.txt\n"
            "3. Read results: cat {patch_dir}/parallel_*/results.json\n"
            "4. Analyze and compare patches\n\n"
            f"All files are in: {base_patch_dir}\n\n"
            "IMPORTANT: In results.json, each patch object has this structure:\n"
            '{\n'
            '  "patch_X": {\n'
            '    "patch_file": "patch_X.patch",\n'
            '    "test_output_file": "patch_X_test.txt",\n'
            '    "test_passed": true/false,\n'
            '    "returncode": <int>,\n'
            '    "metric_result": {     // NOTE: field name is "metric_result" not "metric"\n'
            '      "values": [...]      // metric values array\n'
            '    }\n'
            '  }\n'
            '}\n\n'
            "When writing Python scripts:\n"
            "1. ALWAYS check if patch value is a dict before accessing: if not isinstance(v, dict): skip\n"
            "2. Use patch_obj['metric_result']['values'] to access metric values\n"
            "3. Use .get() with defaults to handle missing keys safely\n\n"
            "Example of safe iteration:\n"
            "```python\n"
            "for par, k, v, rpath in all_patches:\n"
            "    # Check if v is a valid dict first\n"
            "    if not isinstance(v, dict):\n"
            "        skipped.append((par, k, 'invalid_type'))\n"
            "        continue\n"
            "    # Now safe to use .get()\n"
            "    if not v.get('test_passed', False):\n"
            "        skipped.append((par, k, 'test_failed'))\n"
            "        continue\n"
            "    # Access metric_result safely\n"
            "    metric_result = v.get('metric_result', {})\n"
            "    if not isinstance(metric_result, dict):\n"
            "        skipped.append((par, k, 'no_metric_result'))\n"
            "        continue\n"
            "    values = metric_result.get('values', None)\n"
            "    if not isinstance(values, list):\n"
            "        skipped.append((par, k, 'no_values'))\n"
            "        continue\n"
            "    # Now can safely process values\n"
            "```\n\n"
            "When ready, copy the selected patch data from the corresponding parallel_*/results.json "
            "to best_results.json with the required metadata fields, then output:\n"
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT:parallel_<id>/patch_<name>'\n\n"
            "Example bash commands to save best_results.json:\n"
            "```bash\n"
            "# Extract patch data and build best_results.json\n"
            "cat > best_results.json << 'EOF'\n"
            "{\n"
            '  "best_patch_id": "parallel_0/patch_12",\n'
            '  "best_patch_speedup": <float>,\n'
            f'  "best_patch_file": "{base_patch_dir}/parallel_0/patch_12.patch",\n'
            f'  "best_patch_test_output": "{base_patch_dir}/parallel_0/patch_12_test.txt",\n'
            '  "llm_selection_analysis": "Your analysis here"\n'
            "}\n"
            "EOF\n"
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT:parallel_0/patch_12'\n"
            "```"
        )
        
        return "".join(task_parts)
    
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
        """Extract the final result string (parallel_id/patch_id) from messages."""
        # Look through messages for the final output
        for msg in reversed(self.messages):
            if msg["role"] == "user" and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in msg.get("content", ""):
                content = msg["content"]
                # Extract parallel_id/patch_id pattern
                match = re.search(r'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT:parallel_(\d+)/patch_(\w+)', content)
                if match:
                    parallel_id = int(match.group(1))
                    patch_id = f"patch_{match.group(2)}"
                    return f"parallel_{parallel_id}/{patch_id}"
        
        # Fallback: select baseline (parallel_0/patch_0) and save best_results.json
        if 0 in self.all_results and "patch_0" in self.all_results[0]["results"]:
            self._save_fallback_best_results(
                {"parallel_id": 0, "patch_id": "patch_0"},
                "Fallback: selected baseline patch_0 as no explicit selection was made."
            )
            return "parallel_0/patch_0"
        
        return None
    
    def _save_fallback_best_results(self, best: dict, analysis: str):
        """Save best_results.json in fallback mode."""
        if not self.patch_dir:
            return
        
        parallel_id = best["parallel_id"]
        patch_id = best["patch_id"]
        
        # Load the full patch data
        if parallel_id not in self.all_results:
            return
        
        parallel_data = self.all_results[parallel_id]
        parallel_dir = parallel_data["dir"]
        results = parallel_data["results"]
        
        if patch_id not in results:
            return
        
        patch_data = results[patch_id]
        
        # Build best_results.json
        best_results = {
            "best_patch_id": f"parallel_{parallel_id}/{patch_id}",
            "best_patch_speedup": 1.0,
            "best_patch_file": str(parallel_dir / patch_data.get("patch_file", f"{patch_id}.patch")),
            "best_patch_test_output": str(parallel_dir / patch_data.get("test_output_file", f"{patch_id}_test.txt")),
            "llm_selection_analysis": analysis,
        }
        
        (self.patch_dir / "best_results.json").write_text(
            json.dumps(best_results, indent=2)
        )

