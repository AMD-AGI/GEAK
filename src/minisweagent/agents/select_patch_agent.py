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
    )
    instance_template: str = (
        "{{task}}\n\n"
        "Please analyze the patches and respond with a bash command that outputs your analysis.\n"
        "When you have made your final decision, output a command that starts with:\n"
        "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n"
        "followed by your JSON result in the format:\n"
        "{\n"
        '  "best_patch": {"agent_id": <int>, "patch_id": "<string>", "speedup": <float>},\n'
        '  "analysis": "<string>"\n'
        "}"
    )
    step_limit: int = 10
    cost_limit: float = 1.0


class SelectPatchAgent(DefaultAgent):
    """Agent that selects the best patch from parallel runs using multi-turn reasoning."""
    
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=SelectPatchAgentConfig, **kwargs)
        self.patch_dir: Path | None = None
        self.all_results: dict = {}
        self.final_results: list = []
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
        
        # Get unified baseline from agent 0
        unified_baseline = None
        if 0 in self.all_results:
            unified_baseline = self.all_results[0]["results"].get("patch_0")
        
        if not unified_baseline:
            return None
        
        # Collect all patches with their agent_id
        all_patches = []
        for agent_id, data in self.all_results.items():
            results = data["results"]
            parallel_dir = data["dir"]
            
            for patch_name, patch_data in results.items():
                if patch_name.startswith("_"):
                    continue
                all_patches.append({
                    "agent_id": agent_id,
                    "patch_name": patch_name,
                    "patch_data": patch_data,
                    "parallel_dir": parallel_dir,
                })
        
        all_patches.sort(key=lambda x: (x["agent_id"], x["patch_name"]))
        
        # Phase 1: Compute speedup for each patch
        print(f"[SelectPatchAgent] Computing speedup for {len(all_patches)} patches...")
        self.final_results = []
        
        for idx, patch_info in enumerate(all_patches):
            agent_id = patch_info["agent_id"]
            patch_name = patch_info["patch_name"]
            patch_data = patch_info["patch_data"]
            is_baseline = agent_id == 0 and patch_name == "patch_0"
            
            if is_baseline:
                speedup = 1.0
                reasoning = "This is the baseline patch."
            else:
                # For now, use simple heuristic or extract from metric
                # The agent will refine this through multi-turn reasoning
                speedup = self._extract_speedup_heuristic(patch_data, unified_baseline, metric)
                reasoning = "Initial heuristic computation"
            
            self.final_results.append({
                "agent_id": agent_id,
                "patch_id": patch_name,
                "speedup": speedup,
                "reasoning": reasoning,
                "test_passed": patch_data.get("test_passed", False),
                "returncode": patch_data.get("returncode", -1),
                "metric_result": patch_data.get("metric_result"),
            })
        
        # Save initial results
        if self.patch_dir:
            (self.patch_dir / "selection_initial_results.json").write_text(
                json.dumps(self.final_results, indent=2)
            )
        
        # Build task description for the agent
        task_parts = [
            f"You need to select the best patch from {len(self.final_results)} patches.\n\n"
        ]
        
        if metric:
            task_parts.append(f"Optimization metric: {metric}\n\n")
        
        task_parts.append("## Baseline (agent_0/patch_0)\n")
        task_parts.append(f"Test passed: {unified_baseline.get('test_passed', False)}\n")
        task_parts.append(f"Return code: {unified_baseline.get('returncode', -1)}\n")
        if "metric_result" in unified_baseline and unified_baseline["metric_result"]:
            task_parts.append(f"Metric result: {json.dumps(unified_baseline['metric_result'], indent=2)}\n")
        task_parts.append("\n")
        
        task_parts.append("## All Patches Summary:\n\n")
        for result in self.final_results:
            task_parts.append(
                f"- agent_{result['agent_id']}/{result['patch_id']}: "
                f"speedup={result['speedup']:.4f}, test_passed={result['test_passed']}"
            )
            if result.get("metric_result"):
                task_parts.append(f", metric={json.dumps(result['metric_result'])}")
            task_parts.append("\n")
        
        task_parts.append("\n")
        task_parts.append(
            "You can use bash commands to:\n"
            "1. Read patch files: cat {patch_dir}/parallel_*/patch_*.patch\n"
            "2. Read test outputs: cat {patch_dir}/parallel_*/patch_*_test.txt\n"
            "3. Analyze and compare patches\n\n"
            f"All files are in: {base_patch_dir}\n\n"
            "When ready, output your final decision as specified in the instructions."
        )
        
        return "".join(task_parts)
    
    def _extract_speedup_heuristic(self, patch_data: dict, baseline_data: dict, metric: str | None) -> float:
        """Extract speedup using simple heuristics."""
        # If test didn't pass, speedup is 0
        if not patch_data.get("test_passed", False):
            return 0.0
        
        # Try to extract from metric_result if available
        metric_result = patch_data.get("metric_result")
        baseline_metric = baseline_data.get("metric_result")
        
        if metric_result and baseline_metric:
            # Try common metric keys
            for key in ["value", "bandwidth", "throughput", "ops_per_sec"]:
                if key in metric_result and key in baseline_metric:
                    try:
                        patch_val = float(metric_result[key])
                        baseline_val = float(baseline_metric[key])
                        if baseline_val > 0:
                            return patch_val / baseline_val
                    except (ValueError, TypeError):
                        pass
        
        # Default: if test passed, assume neutral speedup
        return 1.0
    
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
    
    def extract_final_result(self) -> dict | None:
        """Extract the final result from messages."""
        # Look through messages for the final output
        for msg in reversed(self.messages):
            if msg["role"] == "user" and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in msg.get("content", ""):
                content = msg["content"]
                # Extract JSON from the output
                try:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Fallback: use highest speedup from final_results
        valid_results = [r for r in self.final_results if r["test_passed"]]
        if valid_results:
            best = max(valid_results, key=lambda x: x["speedup"])
            return {
                "best_patch": {
                    "agent_id": best["agent_id"],
                    "patch_id": best["patch_id"],
                    "speedup": best["speedup"],
                },
                "analysis": f"Fallback selection: highest speedup among passing tests. {best['reasoning']}"
            }
        elif self.final_results:
            first = self.final_results[0]
            return {
                "best_patch": {
                    "agent_id": first["agent_id"],
                    "patch_id": first["patch_id"],
                    "speedup": first["speedup"],
                },
                "analysis": "Fallback: selected first patch as no tests passed."
            }
        
        return None

