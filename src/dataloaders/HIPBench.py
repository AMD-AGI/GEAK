# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import ast
import subprocess
from random import randint
from tqdm import tqdm
import signal
from multiprocessing import Pool, Lock, Value
from dataloaders.ProblemState import ProblemState
from dataloaders.HB_eval.utils import code_call_exec_success_allclose

pwd_path = os.path.dirname(__file__)
Dataset_hip_path = os.path.join(pwd_path, "../../example_hip")

class HIPBench:
    def __init__(self,
                #  statis_path,
                #  py_folder,
                 instruction_path,
                #  golden_metrics,
                #  py_interpreter,
                #  perf_ref_folder,
                #  perf_G_path,
                 result_path=None
                 ):
        # self.statis_path = statis_path
        # self.py_folder = py_folder
        self.instruction_path = os.path.join(Dataset_hip_path, instruction_path)
        # self.golden_metrics_folder = golden_metrics
        # self.py_interpreter = py_interpreter
        # self.perf_ref_folder = perf_ref_folder
        # self.perf_G_path = perf_G_path
        # self.result_path = result_path

        self.problem_states = self.load_ps(result_path)
    

    
    # load_ps: load instruction and GT
    def load_ps(self, path):
        problem_states = []
        if path is None:
            with open(self.instruction_path, "r", encoding='utf-8') as file:
                instructions = json.load(file)
            
            for item in instructions:
                file_path = os.path.join(Dataset_hip_path, item["file_path"], item["file_name"])
                test_code = None
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    # instruction = item["instruction"].replace("ori_file", file_content)
                    instruction = item["instruction"]
                    test_code = file_content
                    label = item["task"]

                problemstate = ProblemState(instruction=instruction,
                                            label=label, 
                                            test_code=test_code, 
                                            filename=file_path, 
                                            build_dir=os.path.join(Dataset_hip_path, item["file_path"])
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
    
    # agent dump solution
    def write_file(self, file_path):
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
                f.write(json.dumps(output) + "\n")
        with open(file_path+".hip", "w") as f:
            f.write(ps.solution)

    

    # 
    @classmethod
    def run_single_call(cls, ps, tmp_dir="temp", gpu_id=0):
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, ps.filename)
        script_content = ps.solution
        try:
            with open(temp_path, "w") as temp_file:
                temp_file.write(script_content + "\n" + "#" * 146 + "\n" + ps.test_code)

            env = os.environ.copy()
            env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
            # Run the temporary Python file
            result = subprocess.run(
                ["python", temp_path], 
                capture_output=True, 
                text=True,
                env=env
            )

            success = result.returncode == 0  # Determine if execution was successful

            if success:
                ps.pass_call = True
                return True, None
            else:
                return False, result.stderr

        except Exception as e:
            return False, str(e)
    
    
    def write_perf_file(self, input_folder_path, results_path):
        """
        input_folder_path: the folder path where codes that pass call and exe tests are stored
        results_path: the folder where perf results (json files) are stored
        """
        if os.path.exists('./tmp'):
            os.system('rm -rf ./tmp')
        os.mkdir('./tmp')
        if os.path.exists('./logs'):
            os.system('rm -rf ./logs')
        os.mkdir('./logs')
        if os.path.exists(results_path):
            os.system(f'rm -rf {results_path}')
        os.mkdir(results_path)

        tab = ' ' * 4
        input_file_list = os.listdir(input_folder_path)
       
                
    def _run_perf_script(self, args):
        timeout_sec = 600  # 10 mins
        progress_lock = Lock()
        progress = Value('i', 0)

        gpu_id, script, total_scripts, log_dir = args

        script_name = os.path.basename(script)
        log_file = os.path.join(log_dir, f"{script_name}.log")
        err_file = os.path.join(log_dir, f"{script_name}.err")

        cmd = f"HIP_VISIBLE_DEVICES={gpu_id} python {script}"

        with open(log_file, "w") as log, open(err_file, "w") as err:
            process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=err)
        
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # kill the process group
            err.write(f"\n⏱️ Script timed out after {timeout_sec} seconds\n")

        with progress_lock:
            progress.value += 1
            tqdm.write(f"✅ finished {progress.value}/{total_scripts}: {script_name}")
    
   

    def run_perf_scripts_multithread(self, gpu_count, script_dir = "./tmp", log_dir = "./logs"):
        os.makedirs(log_dir, exist_ok=True)

        scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
        scripts = [os.path.join(script_dir, script) for script in scripts]
        total_scripts = len(scripts)  
        
        with Pool(processes=gpu_count) as pool, tqdm(total=total_scripts, desc="Process", ncols=80) as pbar:
            args_list = [(i % gpu_count, scripts[i], total_scripts, log_dir) for i in range(total_scripts)]

            for _ in pool.imap(self._run_perf_script, args_list):
                pbar.update(1)

            pool.close()
            pool.join()
    
    def run_perf_scripts(self, script_dir = "./tmp", log_dir = "./logs", gpu_id=0):
        """
        Runs a given Python script on a specified GPU.
        """
        os.makedirs(log_dir, exist_ok=True)

        scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
        scripts = [os.path.join(script_dir, script) for script in scripts]
        total_scripts = len(scripts)
        timeout_sec = 600  # 10 mins
        with tqdm(total=total_scripts) as pbar:
            for idx, script in enumerate(scripts):
                script_name = os.path.basename(script)
                log_file = os.path.join(log_dir, f"{script_name}.log")
                err_file = os.path.join(log_dir, f"{script_name}.err")

                cmd = f"HIP_VISIBLE_DEVICES={gpu_id} python {script}"
                # print(f"Running: {cmd}")

                with open(log_file, "w") as log, open(err_file, "w") as err:
                    try:
                        process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=err)
                    
                
                # try:
                        process.wait(timeout=timeout_sec)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # kill the process group
                        err.write(f"\n⏱️ Script timed out after {timeout_sec} seconds\n")
                    except Exception as e:
                        err.write(f"\n❌ Script crashed: {str(e)}\n")

                tqdm.write(f"✅ finished {idx+1}/{total_scripts}: {script_name}")
                pbar.update(1)


    # calculate latency efficiency
    def calculate(self, path_gen, path_ref):
        get_ms = lambda data: [item["ms"] for item in data]
        get_gbs = lambda data: [item["GB/s"] for item in data]
        get_tflops = lambda data: [item["TFLOPS"] for item in data]
        avg = lambda mss: round(sum(mss[0]) / sum(mss[1]), 4)

        data_gen = json.loads(open(path_gen, 'r', encoding='utf-8').read())
        data_ref = json.loads(open(path_ref, 'r', encoding='utf-8').read())
        assert len(data_gen) == len(data_ref), ""
        
        ms_ref, ms_gen = get_ms(data_ref), get_ms(data_gen)
        spdup = avg((ms_ref, ms_gen))


        efficiency = max(round(max(get_gbs(data_gen)) * 100 / 2039, 4), round(max(get_tflops(data_gen)) * 100 / 312, 4))
        efficiency1 = max(round(max(get_gbs(data_ref)) * 100 / 2039, 4), round(max(get_tflops(data_ref)) * 100 / 312, 4))
        if efficiency >= 100 or spdup >= 10:
            assert False, f"{path_gen.split('/')[-1]} test failed!"
        if efficiency1 > efficiency:
            print(f"金标好啊好11111: {efficiency} < {efficiency1}")
        else:
            print(f"生成棒棒棒！！！: {efficiency} > {efficiency1}")
        return spdup, efficiency, round(sum(ms_gen)/len(ms_gen), 4)

    # call + validation
    def test_opt_correctness(self, code, filename, tmp_dir, task_name="", compile_path="", save_scripts=True, exe_dir="pass_exe"):
        """
        Runs a given Python script on a specified GPU.
        """
        # import pdb; pdb.set_trace()
        os.makedirs(exe_dir, exist_ok=True)
        call_status, exec_status, call_stdout, call_stderr, exe_stdout, exe_stderr, perf, efficiency = code_call_exec_success_allclose(code=code, fname=filename, temp_root=tmp_dir, py_folder=None, task_name=task_name, compile_path=compile_path)
        pass_call = False
        pass_exe = False
        if "True" in str(call_status):
            pass_call=True
        if "True" in str(exec_status):
            pass_exe=True
            if False and save_scripts:
                file_exec = os.path.join(exe_dir, filename)
                with open(file_exec, 'w') as f:
                    f.write(code)

        return pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr, perf, efficiency
    
    
    