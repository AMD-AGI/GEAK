# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Minimal Tutorial Dataloader for GEAK-agent
Uses the self-contained v1 evaluation system with minimal kernel data
"""
import os
import json
import subprocess
from geak_agent.dataloaders.ProblemState import ProblemState
from geak_agent.dataloaders.TB_eval.correctness import test_correctness
from geak_agent.dataloaders.TB_eval.utils import get_temp_file, process_code

# Navigate up from utils/ to tutorial/
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
TUTORIAL_DIR = os.path.dirname(UTILS_DIR)
KERNELS_DIR = os.path.join(TUTORIAL_DIR, "data", "kernels")
GOLDEN_METRICS_DIR = os.path.join(TUTORIAL_DIR, "data", "golden_metrics")
PERF_SCRIPTS_DIR = os.path.join(TUTORIAL_DIR, "data", "perf_scripts")
INSTRUCTIONS_FILE = os.path.join(TUTORIAL_DIR, "data", "instructions.json")


class TutorialDataloader:
    """Minimal dataloader for tutorial with self-contained evaluation"""
    
    def __init__(self, kernel_names=None, corpus_path=None):
        """
        Args:
            kernel_names: List of kernel filenames (e.g., ["add_example.py", "sin_computation.py"])
            corpus_path: Path to train_crawl.json for BM25 retriever
        """
        self.kernels_dir = KERNELS_DIR
        self.golden_metrics_dir = GOLDEN_METRICS_DIR
        self.perf_scripts_dir = PERF_SCRIPTS_DIR
        
        # Load instructions from JSON
        self.instructions_data = self._load_instructions()
        
        # Default to all kernels in instructions.json
        if kernel_names is None:
            kernel_names = [item['file'] for item in self.instructions_data]
        
        self.problem_states = self._load_kernels(kernel_names)
    
    def _load_instructions(self):
        """Load instructions from JSON file"""
        if os.path.exists(INSTRUCTIONS_FILE):
            with open(INSTRUCTIONS_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def _get_instruction_for_kernel(self, kernel_name):
        """Get the actual instruction for a kernel from the dataset"""
        for item in self.instructions_data:
            if item.get('file') == kernel_name:
                return item.get('instruction', ''), item.get('output', ''), item.get('difficulty', 0)
        return None, None, None
    
    def _load_kernels(self, kernel_names):
        """Load kernel files and create problem states"""
        problem_states = []
        
        for kernel_name in kernel_names:
            # Get actual instruction from dataset
            instruction, label, difficulty = self._get_instruction_for_kernel(kernel_name)
            
            if instruction is None:
                print(f"Warning: No instruction found for {kernel_name}, skipping")
                continue
            
            # Load kernel file for test code
            kernel_path = os.path.join(self.kernels_dir, kernel_name)
            if not os.path.exists(kernel_path):
                print(f"Warning: Kernel file {kernel_name} not found, skipping")
                continue
            
            with open(kernel_path, 'r') as f:
                content = f.read()
            
            # Split on the hash separator to get test code
            parts = content.split('#' * 146)
            test_code = parts[1].strip() if len(parts) >= 2 else ""
            
            ps = ProblemState(
                instruction=instruction,
                label=label,
                test_code=test_code,
                filename=kernel_name
            )
            problem_states.append(ps)
        
        return problem_states
    
    def __len__(self):
        return len(self.problem_states)
    
    def write_file(self, file_path):
        """Write results to JSONL file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for ps in self.problem_states:
                output = {
                    "instruction": ps.instruction,
                    "label": ps.label,
                    "filename": ps.filename,
                }
                if ps.test_code:
                    output["test_code"] = ps.test_code
                if ps.solution:
                    output["predict"] = ps.solution
                    output["speedup"] = getattr(ps, 'speedup', 0)
                f.write(json.dumps(output) + "\n")
    
    def test_opt_correctness(self, code, filename, tmp_dir, exe_dir="pass_exe", gpu_id=0):
        """
        Test generated code for correctness and performance.
        Uses the self-contained v1 evaluation system.
        
        Returns: (pass_call, pass_exe, speedup, stdout, stderr)
        """
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(exe_dir, exist_ok=True)
        
        # Get reference kernel path
        ref_file = os.path.join(self.kernels_dir, filename)
        if not os.path.exists(ref_file):
            return False, False, 0, None, f"Reference kernel {filename} not found"
        
        # Create temp file for generated code
        gen_file = os.path.join(tmp_dir, get_temp_file(prefix=f'{filename}_gen'))
        
        # Read test code from reference
        with open(ref_file, 'r') as f:
            ref_content = f.read()
        
        hash_line = "#" * 146
        parts = ref_content.split(hash_line)
        if len(parts) >= 2:
            test_code = parts[1]
        else:
            test_code = ""
        
        # Process and combine generated code with test code
        code = process_code(code)
        full_code = code + '\n\n' + hash_line + '\n' + test_code
        
        with open(gen_file, 'w') as f:
            f.write(full_code)
        
        # Test correctness using v1 system
        try:
            pass_call, pass_exe, stdout, stderr = test_correctness(
                ref_file=ref_file,
                gen_file=gen_file,
                var_name="result_gold",
                atol=1e-3,
                rtol=1e-1,
                verbose=False
            )
        except Exception as e:
            return False, False, 0, None, str(e)
        
        # Calculate speedup if correctness passes
        speedup = 0
        if pass_exe:
            speedup = self._calculate_speedup(code, filename, exe_dir, gpu_id)
            if speedup > 0:
                # Save passing code to exe_dir
                exe_file = os.path.join(exe_dir, filename)
                with open(exe_file, 'w') as f:
                    f.write(full_code)
        
        return pass_call, pass_exe, speedup, stdout, stderr
    
    def _calculate_speedup(self, code, filename, exe_dir, gpu_id=0):
        """
        Calculate speedup by running perf script that benchmarks both generated and reference kernels.
        The perf script outputs JSON with speedup = sum(ref_ms) / sum(gen_ms)
        """
        import tempfile
        import shutil
        import re
        
        op_name = filename.replace('.py', '')
        
        # Check perf script exists
        perf_script_src = os.path.join(self.perf_scripts_dir, f"{op_name}_perf.py")
        if not os.path.exists(perf_script_src):
            print(f"No perf script for {filename}")
            return 0.0
        
        try:
            # Create temp directory for benchmarking
            bench_dir = tempfile.mkdtemp(prefix='geak_bench_')
            
            # Save generated kernel to bench dir (perf script imports from here)
            gen_kernel_file = os.path.join(bench_dir, filename)
            with open(gen_kernel_file, 'w') as f:
                f.write(code)
            
            # Copy performance_utils.py
            perf_utils_src = os.path.join(self.perf_scripts_dir, 'performance_utils.py')
            shutil.copy(perf_utils_src, os.path.join(bench_dir, 'performance_utils.py'))
            
            # Copy perf script and update paths
            with open(perf_script_src, 'r') as f:
                perf_script = f.read()
            
            # Update kernels dir path to absolute path
            perf_script = perf_script.replace(
                "KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'kernels'))",
                f"KERNELS_DIR = '{self.kernels_dir}'"
            )
            
            perf_script_dst = os.path.join(bench_dir, f"{op_name}_perf.py")
            with open(perf_script_dst, 'w') as f:
                f.write(perf_script)
            
            # Run the benchmark
            env = os.environ.copy()
            env["MLU_VISIBLE_DEVICES"] = str(gpu_id)
            
            result = subprocess.run(
                ['python3', perf_script_dst],
                capture_output=True,
                text=True,
                env=env,
                cwd=bench_dir,
                timeout=300  # 5 min timeout
            )
            
            # Extract JSON from stdout (format: ```json\n...\n```)
            stdout = result.stdout
            pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
            match = pattern.search(stdout)
            
            if not match:
                print(f"No JSON output for {filename}: {result.stderr[:300] if result.stderr else stdout[:300]}")
                shutil.rmtree(bench_dir, ignore_errors=True)
                return 0.0
            
            json_str = match.group(1).strip()
            perf_data = json.loads(json_str)
            
            # Get speedup from last entry
            speedup = 0.0
            if perf_data and isinstance(perf_data, list):
                speedup = perf_data[-1].get("speedup", 0.0)
            
            # Cleanup
            shutil.rmtree(bench_dir, ignore_errors=True)
            
            return round(speedup, 4)
            
        except subprocess.TimeoutExpired:
            print(f"Benchmark timed out for {filename}")
            return 0.0
        except Exception as e:
            print(f"Error calculating speedup for {filename}: {e}")
            return 0.0
    
    def test_kernel_profiling(self, code, filename, tmp_dir, save_scripts=True, 
                              exe_dir="pass_exe", target_mlu=None, timeout=20*60):
        """Kernel profiling (placeholder for tutorial)"""
        return False, None, None, None

