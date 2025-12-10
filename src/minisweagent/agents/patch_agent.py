"""Agent with git patch saving and test execution capability."""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


@dataclass
class PatchAgentConfig(AgentConfig):
    save_patch: bool = True
    test_command: str | None = None
    patch_output_dir: str | None = None
    metric: str | None = None
    mode: str | None = None


class PatchAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=PatchAgentConfig, **kwargs)
        self.patch_results: dict[str, dict] = {}
        self.patch_counter = 0

    def add_message(self, role: str, content: str, **kwargs):
        super().add_message(role, content, **kwargs)
        if role == "assistant":
            print(f"\nmini-swe-agent (step {self.model.n_calls}, ${self.model.cost:.2f}):\n", flush=True)
        else:
            print(f"\n{role.capitalize()}:\n", flush=True)
        print(content, flush=True)

    def execute_action(self, action: dict) -> dict:
        output = super().execute_action(action)
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "SAVE_PATCH_AND_TEST":
            patch_info = self._save_patch_and_test()
            if patch_info:
                output["output"] = output.get("output", "") + "\n" + patch_info
        return output

    def _save_patch_and_test(self) -> str | None:
        patch_name = f"patch_{self.patch_counter}"
        self.patch_counter += 1
        patch_content = ""
        print(f"\n[PatchAgent] Saving patch and running test...", flush=True)
        
        cwd = getattr(self.env, 'working_dir', None)
        if cwd is None:
            cwd = getattr(self.env.config, 'cwd', None) or os.getcwd()
        
        try:
            git_diff_result = subprocess.run(
                ["git", "diff"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            patch_content = git_diff_result.stdout
            if not patch_content.strip():
                # self.patch_counter -= 1
                print(f"[PatchAgent] No changes detected, baseline running.", flush=True)
                # return None
            else:
                print(f"[PatchAgent] Patch {patch_name} captured, running test...", flush=True)
            test_env = os.environ.copy()
            test_env["PYTHONUNBUFFERED"] = "1"
            test_result = subprocess.run(
                self.config.test_command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.env.config.timeout,
                env=test_env,
            )
            test_output = test_result.stdout
            test_passed = test_result.returncode == 0
            self.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": test_passed,
                "returncode": test_result.returncode,
            }
            status = "✓ PASSED" if test_passed else "✗ FAILED"
            print(f"[PatchAgent] Test result for {patch_name}: {status}", flush=True)
            
            if self.config.metric:
                metric_result = self._extract_metric(patch_name, test_output)
                self.patch_results[patch_name]["metric_result"] = metric_result
            
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            
            return self._format_patch_info(patch_name, patch_content, test_output, test_passed, test_result.returncode)
                
        except subprocess.TimeoutExpired:
            test_output = "Test command timed out"
            self.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": False,
                "returncode": -1,
            }
            print(f"[PatchAgent] Test for {patch_name}: ✗ TIMEOUT", flush=True)
            if self.config.metric:
                self.patch_results[patch_name]["metric_result"] = None
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            return self._format_patch_info(patch_name, patch_content, test_output, False, -1)
        except Exception as e:
            test_output = str(e)
            self.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": False,
                "returncode": -1,
            }
            print(f"[PatchAgent] Test for {patch_name}: ✗ ERROR - {e}", flush=True)
            if self.config.metric:
                self.patch_results[patch_name]["metric_result"] = None
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            return self._format_patch_info(patch_name, patch_content, test_output, False, -1)

    def _format_patch_info(self, patch_name: str, patch_content: str, test_output: str, test_passed: bool, returncode: int) -> str:
        status = "PASSED ✓" if test_passed else "FAILED ✗"
        info_parts = [
            f"\n{'='*60}",
            f"Patch saved: {patch_name}",
            f"Test status: {status}",
            f"Return code: {returncode}",
            f"{'='*60}",
            "\n## Patch Content:",
            f"```diff\n{patch_content}\n```",
            "\n## Test Output:",
            f"```\n{test_output}\n```",
            f"{'='*60}\n",
        ]
        return "\n".join(info_parts)
    
    def _extract_metric(self, patch_name: str, test_output: str):
        print(f"[PatchAgent] Extracting metric for {patch_name}...", flush=True)
        prompt = (
            f"Task: {self.config.metric}\n\n"
            f"Test output:\n{test_output}\n\n"
            "Extract the requested metric from the test output above. "
            "Return ONLY a valid JSON object with the extracted data. "
            "If the metric cannot be found, return null. "
            "Examples:\n"
            '- For numeric value: {"value": 123.45}\n'
            '- For array: {"values": [1.2, 3.4, 5.6]}\n'
            '- For multiple metrics: {"bandwidth": 123.45, "latency": 0.5}\n'
        )
        response = self.model.query([{"role": "system", "content": "You are a helpful assistant to analyze kernel test output and extract metrics from test logs."}, {"role": "user", "content": prompt}])
        content = response.get("content", "")
        try:
            result = json.loads(content.strip())
            print(f"[PatchAgent] Metric extracted: {result}", flush=True)
            return result
        except json.JSONDecodeError:
            print(f"[PatchAgent] Failed to parse metric response: {content}", flush=True)
            return None

    def _save_patch_file(self, patch_name: str, patch_content: str):
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_file = output_dir / f"{patch_name}.patch"
        patch_file.write_text(patch_content)
        print(f"[PatchAgent] Patch saved to: {patch_file}", flush=True)
    
    def _save_test_output(self, patch_name: str, test_output: str):
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / f"{patch_name}_test.txt"
        test_file.write_text(test_output)
        print(f"[PatchAgent] Test output saved to: {test_file}", flush=True)
    
    def _update_results_file(self):
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "results.json"
        simplified_results = {}
        for name, data in self.patch_results.items():
            result_entry = {
                "patch_file": data["patch_file"],
                "test_output_file": data["test_output_file"],
                "test_passed": data["test_passed"],
                "returncode": data["returncode"],
            }
            if "metric_result" in data:
                result_entry["metric_result"] = data["metric_result"]
            simplified_results[name] = result_entry
        results_file.write_text(json.dumps(simplified_results, indent=2, ensure_ascii=False))
        print(f"[PatchAgent] Results updated: {results_file}", flush=True)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        print(f"\n{'='*60}", flush=True)
        print(f"[PatchAgent] Starting with patch saving enabled", flush=True)
        print(f"[PatchAgent] Test command: {self.config.test_command}", flush=True)
        print(f"[PatchAgent] Patch output directory: {self.config.patch_output_dir}", flush=True)
        if self.config.metric:
            print(f"[PatchAgent] Metric extraction: {self.config.metric}", flush=True)
        print(f"[PatchAgent] Trigger: Use 'SAVE_PATCH_AND_TEST' in command output", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        exit_status, result = super().run(task, **kwargs)
        
        print(f"\n[PatchAgent] Agent execution completed", flush=True)
        print(f"[PatchAgent] Exit status: {exit_status}", flush=True)
        
        if self.config.patch_output_dir and len(self.patch_results) > 1:
            self._select_best_patch()
        
        self._print_summary()
        print(f"[PatchAgent] Trajectory will be saved by the runner", flush=True)
        return exit_status, result

    def _print_summary(self):
        if not self.patch_results:
            return
        print(f"\n{'='*60}", flush=True)
        print(f"[PatchAgent] Summary:", flush=True)
        print(f"  Total patches: {len(self.patch_results)}", flush=True)
        passed = sum(1 for data in self.patch_results.values() if data["test_passed"])
        failed = len(self.patch_results) - passed
        print(f"  Passed: {passed}", flush=True)
        print(f"  Failed: {failed}", flush=True)
        if self.config.patch_output_dir:
            print(f"  Results saved to: {Path(self.config.patch_output_dir) / 'results.json'}", flush=True)
        print(f"{'='*60}\n", flush=True)

    def _select_best_patch(self) -> str | None:
        if not self.patch_results or len(self.patch_results) <= 1:
            return None
        
        print(f"\n[PatchAgent] Calling LLM to select best patch...", flush=True)
        
        output_dir = Path(self.config.patch_output_dir)
        prompt_parts = ["Analyze the following patches and their test results to select the best patch.\n"]
        
        if self.config.metric:
            prompt_parts.append(f"Metric extraction task: {self.config.metric}\n")
            prompt_parts.append("IMPORTANT: patch_0 is the baseline (no modifications). ")
            prompt_parts.append("Select the patch with the best average improvement compared to the baseline.\n\n")
        else:
            prompt_parts.append("\n")
        
        prompt_parts.append(f"Total patches: {len(self.patch_results)}\n\n")
        
        for patch_name, data in self.patch_results.items():
            is_baseline = patch_name == "patch_0"
            prompt_parts.append(f"## {patch_name}")
            if is_baseline:
                prompt_parts.append(" (BASELINE)\n")
            else:
                prompt_parts.append("\n")
            
            prompt_parts.append(f"Test passed: {data['test_passed']}\n")
            prompt_parts.append(f"Return code: {data['returncode']}\n")
            
            if "metric_result" in data and data["metric_result"] is not None:
                prompt_parts.append(f"Metric result: {json.dumps(data['metric_result'], indent=2)}\n\n")
            else:
                prompt_parts.append("\n")
            
            patch_file = output_dir / data["patch_file"]
            if patch_file.exists():
                patch_content = patch_file.read_text()
                if not is_baseline:
                    prompt_parts.append(f"Patch content:\n```\n{patch_content}\n```\n\n")
        
        if self.config.metric:
            prompt_parts.append(f"The metric is {self.config.metric}\n")
            prompt_parts.append(
                "Based on the metric results, calculate the average improvement of each patch compared to patch_0 (baseline). "
                "Select the patch with the highest average improvement. "
                "If patch_0 has the best metrics, you can select it. "
                "Respond with ONLY the patch name (e.g., 'patch_0', 'patch_1', etc.)."
            )
        else:
            prompt_parts.append(
                "Based on the test results and patch quality, which patch is the best? "
                "Respond with ONLY the patch name (e.g., 'patch_0', 'patch_1', etc.)."
            )
        
        response = self.model.query([{"role": "user", "content": "".join(prompt_parts)}])
        best_patch = response.get("content", "").strip()
        
        if best_patch in self.patch_results:
            print(f"[PatchAgent] LLM selected best patch: {best_patch}", flush=True)
            self.patch_results["_best_patch"] = best_patch
            self._update_results_file()
            return best_patch
        else:
            print(f"[PatchAgent] LLM response invalid: {best_patch}", flush=True)
            return None

        

