# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import ast
import subprocess
from random import randint
from tqdm import tqdm
import signal
from multiprocessing import Pool, Lock, Value
from geak_agent.dataloaders.ProblemState import ProblemState
from geak_agent.dataloaders.TB_eval.utils import code_call_exec_success_allclose, code_kernel_profiling
import re
from tb_eval.evaluators.interface import get_evaluators

class TritonBench:
    def __init__(self,
                 statis_path,
                 py_folder,
                 instruction_path,
                 golden_metrics,
                 py_interpreter,
                 perf_ref_folder,
                 perf_G_path,
                 result_path=None
                 ):
        self.statis_path = statis_path
        self.py_folder = py_folder
        self.instruction_path = instruction_path
        self.golden_metrics_folder = golden_metrics
        self.py_interpreter = py_interpreter
        self.perf_ref_folder = perf_ref_folder
        self.perf_G_path = perf_G_path
        self.result_path = result_path

        self.problem_states = self.load_ps(result_path)
        self.evaluator = get_evaluators['tbg']()
    
    def load_ps(self, path):
        problem_states = []
        if path is None:
            with open(self.instruction_path, "r", encoding='utf-8') as file:
                instructions = json.load(file)
            statis_data = json.loads(open(self.statis_path, 'r', encoding='utf-8').read())

            for line in instructions:
                instruction = line["instruction"]
                label = line["output"]

                # get test code
                g = label.replace("<|im_end|>", "").replace("<|EOT|>", "")
                tmp = False
                for item in statis_data:
                    if g in item["output"]:
                        file = item["file"]
                        tmp = item
                        break
                if tmp:
                    statis_data.remove(tmp)
                elif g[50:220] == 'as tl\n\nif triton.__version__ >= "2.1.0":\n    @triton.jit\n    def _fwd_kernel(\n        Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录':
                        file = "context_attn_nopad.py"
                path = os.path.join(self.py_folder, file)
                assert os.path.exists(path), f"{file} not exist!"
                test_code = open(path, "r", encoding="utf-8").read().split("#"*146)[-1]
                assert "def test_" in  test_code, ""

                problemstate = ProblemState(instruction=instruction,
                                            label=label, 
                                            test_code=test_code, 
                                            filename=file, 
                                            )
                
                problem_states.append(
                    problemstate
                )
        else:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    content = json.loads(line)
                    problem_state = ProblemState(instruction=content["instruction"], 
                                                 label=content["label"], 
                                                 filename=content["filename"],
                                                )
                    if "test_code" in content:
                        problem_state.test_code = content["test_code"]
                    if "predict" in content:
                        problem_state.solution = content["predict"] 
                    problem_states.append(problem_state)
        return problem_states

    def __len__(self):
        return len(self.problem_states)
    
    def write_file(self, file_path):
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
                    output["speedup"] = ps.speedup
                    
                f.write(json.dumps(output) + "\n")
    
    def test_opt_correctness(self, code, filename, tmp_dir, exe_dir="pass_exe",gpu_id=0):
        """
        Runs a given Python script on a specified GPU.
        """
        pass_call, pass_exe, speedup, call_stdout, call_stderr = self.evaluator(code, tmp_dir, exe_dir, filename, atol=1e-3, rtol=1e-3, custom_tests_path=None)

        return pass_call, pass_exe, speedup, call_stdout, call_stderr
    
    def test_kernel_profiling(self, code, filename, tmp_dir, save_scripts=True, exe_dir="pass_exe", target_mlu=None, timeout=20*60):
        os.makedirs(exe_dir, exist_ok=True)
        profile_status, stdout_profile, stderr_profile, stdout_analyze = code_kernel_profiling(code=code, fname=filename, temp_root=tmp_dir, py_folder=self.py_folder, target_mlu=target_mlu, timeout=timeout)
        pass_prfiler = False
        if "True" in str(profile_status):
            pass_prfiler=True
        
        return pass_prfiler, stdout_profile, stderr_profile, stdout_analyze
    
    