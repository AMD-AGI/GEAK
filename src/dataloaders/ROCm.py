import json
import os
import shutil
import subprocess
from loguru import logger
from multiprocessing import Pool, Lock, Value

# Assuming these are in your project structure
from dataloaders.ProblemState import ProblemStateROCm
from tb_eval.evaluators.interface import get_evaluators
from tb_eval.helpers.helper import extract_first_pytest_failure
from tb_eval.perf.efficiency import get_perf_evaluators

class ROCm:
    def __init__(self,
                 statis_path,
                 py_folder,
                 instruction_path,
                 log_root,
                 py_interpreter='python3',
                 result_path=None
                 ):
        self.statis_path = statis_path
        self.py_folder = py_folder
        self.instruction_path = instruction_path
        # This flag is to identify the dataset type in the agent
        self.rocm_tests = True
        self.problem_states = self.load_ps()
        self.log_root = log_root
        
        # Initialize correctness and performance evaluators from tb_eval
        self.evaluator = get_evaluators["rocm"]()
        self.perf_evaluator = get_perf_evaluators["rocm"]()
        logger.info("Custom tests path set to: {}".format(self.py_folder))

    def load_ps(self,):
        problem_states = []
        with open(self.instruction_path, "r", encoding='utf-8') as file:
            instructions = json.load(file)
        statis_data = json.loads(open(self.statis_path, 'r', encoding='utf-8').read())

        for line in instructions:
            instruction = line["instruction"]
            label = line["label"]
            g = label.replace("<|im_end|>", "").replace("<|EOT|>", "")
            tmp = False
            for item in statis_data:
                if g in item["label"]:
                    file = item["file"]
                    tmp = item
                    break
            if tmp: statis_data.remove(tmp)
            
            path = os.path.join(self.py_folder, file)
            assert os.path.exists(path), f"{file} not exist!"
            test_code = open(path, "r", encoding="utf-8").read().split("#"*146)[-1]
            assert "def test_" in  test_code, ""

            problemstate = ProblemStateROCm(
                instruction=instruction,
                label=label, 
                test_code=test_code, 
                filename=file,
                target_kernel_name=line.get("target_kernel_name", "")
            )
            problem_states.append(problemstate)
        return problem_states

    def __len__(self):
        return len(self.problem_states)

    def write_file(self, file_path, start_idx=0, datalen=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_len = datalen if datalen is not None else len(self)
        with open(file_path, 'w') as f:
            for ps in self.problem_states[start_idx:(start_idx + data_len)]:
                output = {
                    "instruction": ps.instruction,
                    "label": ps.label,
                    "file": ps.filename,
                    "target_kernel_name": ps.target_kernel_name,
                    "predict": ps.solution if ps.solution else ""
                }
                f.write(json.dumps(output) + "\n")

    def test_opt_correctness(self, code, filename, tmp_dir="temp", save_scripts=True, exe_dir="pass_exe"):
        tmp_dir = os.path.join(self.log_root, tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        exe_dir = os.path.join(self.log_root, exe_dir)
        # Ensure the final executional script dir exists.
        os.makedirs(exe_dir, exist_ok=True)
        logger.info(f"Testing correctness for {filename} in {tmp_dir}")
        try:
            log_root = os.path.abspath(os.path.join(tmp_dir, "tmp"))
            os.makedirs(log_root, exist_ok=True)

            # This is the directory where the evaluator will save the correct file
            exec_root_eval = os.path.abspath(os.path.join(tmp_dir, "exec_eval"))
            os.makedirs(exec_root_eval, exist_ok=True)
            # import pdb; pdb.set_trace()
            call_status, exec_status, stdout, stderr = self.evaluator(code, log_root, exec_root_eval, filename, atol=1e-2, rtol=1e-2, custom_tests_path=self.py_folder)
            # import ipdb;ipdb.set_trace(context=200)
            if exec_status and save_scripts:
                # The evaluator already saves the file, but we copy it to the agent's expected directory
                src_file = os.path.join(exec_root_eval, filename)
                dst_file = os.path.join(exe_dir, filename)
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
            
            
            return bool(call_status), bool(exec_status), stdout, stderr, stdout, stderr

        except Exception as e:
            logger.error(f"Exception during correctness test for {filename}: {e}")
            return False, False, None, str(e), None, str(e)
        finally:
            if os.path.exists(log_root):
                shutil.rmtree(log_root, ignore_errors=True)
            if os.path.exists(exec_root_eval):
                shutil.rmtree(exec_root_eval, ignore_errors=True)

    def run_perf_evaluation(self, exec_folder, gen_perf_folder):
        """
        Runs the performance evaluation for ROCm using the tb_eval module.

        Args:
            exec_folder (str): The directory containing the correctly executed scripts.
            gen_perf_folder (str): The directory where performance JSON results will be stored.
        
        Returns:
            dict: A dictionary containing performance results, mapping filename to metrics.
        """
        logger.info(f"Starting ROCm performance evaluation for kernels in: {exec_folder}")
        try:
            # The `evaluate` method from PerformanceEvalROCm handles all the steps:
            # 1. Runs pytest for each file in exec_folder.
            # 2. Runs the final efficiency script.
            # 3. Parses and returns the results.
            try:
                perf_results = self.perf_evaluator(exec_folder)
            except Exception as e:
                logger.error(f"Performance evaluation failed: {e}")
                return {}
            logger.success("ROCm performance evaluation completed successfully.")
            return perf_results
        except Exception as e:
            logger.error(f"ROCm performance evaluation failed: {e}")
            return {}
