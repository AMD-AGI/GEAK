import shutil
import os
import subprocess
import re
import numpy as np
import shlex

N = 0

# === Steps to customize for a new task ===
# Two functions need to be change
# 1. check the notation in "def extract_perf_validation" for stdout perf extraction custimization 
# 2. check the notation in "code_call_exec_success_allclose"


def extract_perf_validation(task_name, stdout):
    perf = None
    efficiency = None
    validation_result = None

    # === Customize extract_perf_validation for a new task ===
    # 1. Update the task condition (to the "task" field in your instructions.json):
    #    if task_name == "your_task_name":
    #
    # 2. Adjust the regex pattern to match the latency/performance matric format in stdout:
    #    time_match = re.search(r"Avg latency\s*:\s*([\d.]+)\s*ms", stdout) 
    #    or for multiple perf output:
    #    time_match = re.findall(r"Avg latency\s*:\s*([\d.]+)\s*ms", stdout)
    #
    # 3. Set a meaningful baseline performance number or a list of number for comparison, this can be your original performance of the code:
    #    base_perf = 1500.0  # replace with actual micro/milliseconds from target env
    #    or for multiple perf output, use a list:
    #    base_perf = [1500, 1400] # for multiple perf numbers.
    #
    # 4. Adapt the validation check to match success criteria in stdout:
    #    validation_result = ("SUCCESS" in stdout) and ("FAILED" not in stdout)
    #
    # 5. Compute efficiency - optimized latency vs. the original impl latency (keep consistent ratio):
    #    efficiency = perf / base_perf
    #    or for multiple perf output:
    #    efficiency = []
    #     for i, bp in enumerate(base_perf):
    #         efficiency.append(perf[i] / bp)
    #
    #    Key idea: Only swap out (task name, regex, baseline, success keyword)
    #    and you can reuse the same structure for different tasks.


    if task_name == "rms":
        time_match = re.search(r"Mean\s*:\s*([\d.]+)\s*us", stdout)
        perf = float(time_match.group(1)) if time_match else None

        validation_result = "PASS" in stdout
        base_perf = float('inf') # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf
    
    elif task_name == "render_forward":
        time_match = re.search(r" has been ([\d.]+)ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf') # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf

    elif task_name == "silu":
        time_match = re.search(r"Perf:\s*([\d.]+)\s*us/launch", stdout)
        perf = float(time_match.group(1)) if time_match else None

        validation_result = "PASS" in stdout
        base_perf =  float('inf') # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf

    elif task_name == "bitonic_sort":
        time_match = re.search(r"GPU bitonic sort took ([\d.]+) milliseconds", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = "Validation passed" in stdout and "Validation failed" not in stdout
        base_perf = float('inf')  # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf
        
    elif task_name == "convolution":
        time_match = re.search(r" has been ([\d.]+)ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf')  # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf
        # pass

    elif task_name == "floyd_warshall":
        # import pdb; pdb.set_trace()
        time_match = re.search(r" has been ([\d.]+)ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf')   # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf
        # pass

    elif task_name == "histogram":
        # import pdb; pdb.set_trace()
        time_match = re.search(r"Kernel took ([\d.]+) milliseconds", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = "Validation passed" in stdout and "Validation failed" not in stdout
        base_perf = float('inf')    # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf
        # pass

    elif task_name == "monte_carlo_pi":
        # import pdb; pdb.set_trace()
        times = re.findall(r"which took ([\d.]+) ms", stdout)
        perf = [float(t) for t in times]

        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf')]   # FAKE number, test your unittest in target env and put the number here.
        efficiency = [perf[0]/base_perf[0], perf[1]/base_perf[1]]
        # pass

    elif task_name == "prefix_sum":
        # import pdb; pdb.set_trace()
        time_match = re.search(r" has been ([\d.]+)ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf =float('inf')  # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf
        # pass
    
    elif task_name == "point_to_voxelidx":
        # import pdb; pdb.set_trace()
        time_match = re.search(r" has been ([\d.]+)ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf')  # FAKE number, test your unittest in target env and put the number here.
        efficiency = perf / base_perf

    elif task_name == "gemm_multiply_multiply_xdl_fp8_ab_scale":
        time_match = re.findall(r"Perf: ([\d.]+) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("error" not in stdout and "Error" not in stdout)
        base_perf = [float('inf')] * 8  # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)

    elif task_name == "assign_score_withk":
        time_match = re.findall(r"Perf: ([\d.]+) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf')]    # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    
    elif task_name == "ball_query":
        time_match = re.findall(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf')]    # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    
    elif task_name == "furthest_point_sample":
        time_match = re.findall(r"Perf: ([\d.]+) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf')]   # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    
    elif task_name == "gather_points":
        time_match = re.findall(r"Perf: ([\d.]+) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf')]  # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    
    elif task_name == "knn":
        time_match = re.findall(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf'), float('inf')]  # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)

    elif task_name == "points_in_boxes":
        time_match = re.findall(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf'), float('inf'), float('inf')]  # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)

    elif task_name == "roiaware_pool3d":
        time_match = re.findall(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = [float('inf'), float('inf')]   # FAKE number, test your unittest in target env and put the number here.

        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    
    elif task_name == "roipoint_pool3d":
        time_match = re.search(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf')  # FAKE number, test your unittest in target env and put the number here.

        efficiency = perf / base_perf

    
    elif task_name == "three_interpolate":
        # import pdb; pdb.set_trace()
        time_match = re.search(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf')  # FAKE number, test your unittest in target env and put the number here.

        efficiency = perf / base_perf

    elif task_name == "three_nn":
        # import pdb; pdb.set_trace()
        time_match = re.search(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = float(time_match.group(1)) if time_match else None
        validation_result = ("Validation failed" not in stdout)
        base_perf = float('inf')  # FAKE number, test your unittest in target env and put the number here.

        efficiency = perf / base_perf
    
    elif task_name == "mqa":
        # import pdb; pdb.set_trace()

        time_match = re.findall(r"Perf: ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) ms", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("Validation failed" not in stdout)

        base_perf = [float('inf')] * 96   # FAKE number, test your unittest in target env and put the number here.
        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    
    elif task_name == "gsplat":
        time_match = re.findall(r".*Backward time \(rasterize_to_pixels\): (\d+\.\d+)s", stdout)
        perf = [float(t) for t in time_match] if time_match else None
        validation_result = ("FAILED" not in stdout) and ("PASSED" in stdout)
        base_perf = [float('inf')] * 12   # FAKE number, test your unittest in target env and put the number here.
        efficiency = []
        for i, bp in enumerate(base_perf):
            efficiency.append(perf[i] / bp)
    else:
        assert False
    
    return perf, validation_result, efficiency

def code_call_exec_success_allclose(code, fname, temp_root, py_folder, task_name="", compile_path=""):

    call_status, call_stdout, call_stderr = False, '', ''
    exec_status, exe_stdout, exe_stderr = False, '', ''
    perf = None
    efficiency = None

    # === Customize code_call_exec_success_allclose for a new task ===
    #
    # This function handles multiple pipelines depending on task_name.
    # The general flow for each task is:
    #   1. Run a "compile" command (if needed) to build/setup the code.
    #      - If compile fails → call_status = False
    #   2. Run an "execution" command (test script, bash, binary, pytest, etc).
    #      - Collect stdout/stderr.
    #      - Parse performance & validation using extract_perf_validation().
    #      - If validation fails → exec_status = False
    #   3. Return:
    #      - call_status  → whether compile succeeded
    #      - exec_status  → whether execution validated successfully
    #      - call_stdout/call_stderr → compile logs
    #      - exe_stdout/exe_stderr   → run logs
    #      - perf         → measured performance (latency/throughput)
    #      - efficiency   → perf / baseline
    #
    # === To add a new task: ===
    # Use the same pattern as existing tasks below.
    # For a new task placed in example_hip, and instructions within:
    #
    # if task_name == "my_new_task":
    #     # compile command in text string, this command should compile the kernel code.
    #     compile_cmd = "make"
    #     # perfmance command in text string, this command should run the target program and output the latency on screen.
    #     perf_cmd = "bash perf_eval_rms.sh"
    #     # compile path in text string, above commands should be able to run/execute in cwd path.
    #     cwd = compile_path
    #
    # The above modifications are the only things to do if you setup a similar example in example_hip 
    # 
    # Should one wants to modify the compile/execution pipeline, one can follow the code below.
    #     try:
    #         # Step 1: Define the compile command
    #         run_command = ["make"]   # or ["python3", "setup.py", "build"] etc.
    #         result = subprocess.run(
    #             run_command,
    #             cwd=compile_path,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True,
    #             check=True
    #         )
    #         call_stdout += result.stdout
    #         call_stderr += result.stderr
    #         call_status = True
    #     except subprocess.CalledProcessError as e:
    #         call_stdout += e.stdout
    #         call_stderr += e.stderr
    #         call_status = False
    #
    #     if call_status:
    #         try:
    #             # Step 2: Define the execution command that eventually 
    #             run_command = ["bash", "perf_eval_my_new_task.sh"]
    #             result = subprocess.run(
    #                 run_command,
    #                 cwd=compile_path,
    #                 stdout=subprocess.PIPE,
    #                 stderr=subprocess.PIPE,
    #                 text=True,
    #                 check=True,
    #                 env=os.environ.copy()
    #             )
    #             exe_stdout += result.stdout
    #             exe_stderr += result.stderr
    #
    #             # Step 3: Extract perf & validation
    #             perf, validation, efficiency = extract_perf_validation(task_name, result.stdout)
    #             exec_status = True if validation else False
    #         except subprocess.CalledProcessError as e:
    #             exe_stdout += e.stdout
    #             exe_stderr += e.stderr
    #             call_status = False
    #
    # === Key notes ===
    # - Always separate compile (call_status) and run (exec_status).
    # - Decide compile/run commands per task.
    # - Keep perf extraction consistent via extract_perf_validation().
    # - Validation should fail safely (exec_status = False).
    # - Logs (stdout/stderr) are preserved for debugging.
    # - You can try to comment out the writing of the original file and see if the perf can be parsed correctly in test run. 

    # The following lines write the generated code to replace original kernel file (fname). 
    # For first-time or debug purpose, one can comment the following 2 lines and see if same perf shows up in prompt. 
    with open(fname, "w") as f:
        f.write(code)

    if task_name in ['rms']:
        compile_cmd = "make"
        perf_cmd = "bash perf_eval_rms.sh"
        cwd = compile_path
    elif task_name in ["assign_score_withk", "ball_query", "furthest_point_sample", "gather_points", "knn", "points_in_boxes", "roiaware_pool3d", "roipoint_pool3d", "three_interpolate", "three_nn", "mqa"]:
        compile_cmd = "python3 " + task_name + "/test_"+ task_name +".py"
        perf_cmd = "python3 " + task_name + "/test_"+ task_name +".py"
        cwd = os.path.dirname(compile_path)
    elif task_name == "mqa":
        compile_cmd = "timeout 200s python3 test_"+ task_name +".py"
        perf_cmd = "timeout 200s python3 test_"+ task_name +".py"
        cwd = compile_path
    elif task_name == "gsplat":
        compile_cmd = "python setup.py develop"
        perf_cmd = "timeout 60s pytest -v -s tests/test_basic.py::test_rasterize_to_pixels"
        cwd = compile_path
    elif task_name == "gemm_multiply_multiply_xdl_fp8_ab_scale":
        compile_cmd = "make example_gemm_multiply_multiply_xdl_fp8_ab_scale"
        perf_cmd = "./bin/example_gemm_multiply_multiply_xdl_fp8_ab_scale"
        cwd = os.path.join(re.search(r".*?/composable_kernel", compile_path).group(0), "build")
    else:
        # this is the default pipeline that works with rocm-example/applications/tasks. 
        # if you would like a quick run, you could put your kernels exactly like the one in rocm-example
        compile_cmd = "make"
        perf_cmd = "./applications_" + task_name
        cwd = compile_path

    compile_cmd_list = shlex.split(compile_cmd)
    perf_cmd_list = shlex.split(perf_cmd)
    
    try:
        result = subprocess.run(
            compile_cmd_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        stdout = result.stdout
        stderr = result.stderr
        call_stdout += stdout
        call_stderr += stderr
        call_status = True
    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr
        call_stdout += stdout
        call_stderr += stderr
        call_status = False
    
    if call_status:
        try:
            env = os.environ.copy()
            result = subprocess.run(
                perf_cmd_list,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                env=env
            )
            stdout = result.stdout
            stderr = result.stderr
            exe_stdout += stdout
            exe_stderr += stderr

            if task_name == "gsplat":
                if ("FAILED" not in result.stdout) and ("PASSED" not in result.stdout):
                    stderr += "\n TIMEOUT"

            perf, validation, efficiency = extract_perf_validation(task_name, stdout)
            exec_status = True if validation else False

        except subprocess.CalledProcessError as e:
            stdout = e.stdout
            stderr = e.stderr
            call_stdout += stdout
            call_stderr += stderr
            call_status = False
    
    # Should it need uninstall, follow the gsplat example to execute additional command
    if task_name == "gsplat":
        run_command = ["pip", "uninstall", "-y", "gsplat"]
        result = subprocess.run(
                    run_command,
                    cwd=compile_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

    return call_status, exec_status, call_stdout, call_stderr, exe_stdout, exe_stderr, perf, efficiency
