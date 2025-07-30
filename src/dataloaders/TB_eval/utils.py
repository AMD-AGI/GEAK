import ast
import os
import subprocess
from random import randint
from tqdm import tqdm
from shutil import copyfile
import datetime
import json
from parse_llm_code import extract_code_blocks
import numpy as np

## Implementation from https://arxiv.org/pdf/2107.03374
def passk(n, c, k):
    if n -c < k: return 1.0
    return 1 - np.prod(
        1 - k/ np.arange(
            n-c+1, n+1
        )
    )

def get_time():
    # Get the current time in the format YYYY-MM-DD_HH-MM-SS
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_temp_file(prefix='temp_code'):
    # Generate a unique temporary file name
    temp_file_name = f'{prefix}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name.replace('.py', f'_{randint(999, 999999)}.py')
    return temp_file_name

def code_call_exec_success_stdout(code, fname, temp_root="tmp2", tolerance=2, verbose=False):
    # Save the code to a temporary file
    tmp_triton_folder = os.path.join(temp_root, "triton") #f"{temp_root}_triton"
    tmp_gen_folder = os.path.join(temp_root, "gen") #f"{temp_root}_gen"
    os.makedirs(tmp_triton_folder, exist_ok=True)
    os.makedirs(tmp_gen_folder, exist_ok=True)
    

    triton_root = "dataloaders/TB_eval/TritonBench/data/TritonBench_G_v1"
    RAND_FILE = os.path.join(triton_root, "rand_utils.py")

    copyfile(RAND_FILE, os.path.join(tmp_triton_folder, "rand_utils.py"))
    copyfile(RAND_FILE, os.path.join(tmp_gen_folder, "rand_utils.py"))

    gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
    triton_file = os.path.join(triton_root, fname)
    temp_triton_file = get_temp_file(prefix=f'{fname}_temp_triton')

    gen_file = os.path.join(tmp_gen_folder, gen_file)
    temp_triton_file = os.path.join(tmp_triton_folder, temp_triton_file)

    IMPORT_STATEMENT = f"""
from rand_utils import torch_rand, torch_randint, torch_randn
import torch
torch.set_printoptions(precision={tolerance},profile='full',sci_mode=False)
"""

    hash_line = "#"*146
    ## from triton_file copy everything after the hash_line into gen_file
    with open(triton_file, 'r') as f:
        lines = f.readlines()
        # lines.append(
        #     '\nprint(result_gold)'
        # )
        for iL, line in enumerate(lines):
            if line.strip() == hash_line:
                break
        test_code_lines = lines[iL+1:]
        test_code_lines = IMPORT_STATEMENT.split('\n') + test_code_lines
        test_code_lines_procs = []
        for line in test_code_lines:
            if "torch.rand" in line:
                line = line.replace("torch.rand", "torch_rand")
            test_code_lines_procs.append(line)

    with open(temp_triton_file, 'w') as f:
        triton_lines = lines[:iL] +  [hash_line] + test_code_lines_procs
        for line in triton_lines:
            f.write(line + "\n")

    code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)
    with open(gen_file, 'w') as f:
        f.write(code)

    ## Execute two codes gen_file and triton_file using subprocess. 
    ## 1. If gen_file return error then return status as False, and stdout and stderr from gen file
    ## 2. If triton_file return error then return status as True and stdout and stderr as None
    ## 3. If gen_file and triton_file both return success then compare stdout from gen_file and triton_file. If stdout matches then return status as True, and stdout and stderr as None else return status as False and stdout and stderr as test cases mismatched.

    try:
        # Execute the generated code
        result_gen = subprocess.run(['python3', gen_file], capture_output=True, text=True, timeout=2*60)
        stdout_gen = result_gen.stdout
        stderr_gen = result_gen.stderr

        # Check if the generated code executed successfully
        if result_gen.returncode != 0:
            if verbose:
                print(f"Error in generated code: {stderr_gen}")
            return False, False, stdout_gen, stderr_gen

        # Execute the Triton code
        result_triton = subprocess.run(['python3', temp_triton_file], capture_output=True, text=True, timeout=2*60)
        stdout_triton = result_triton.stdout
        stderr_triton = result_triton.stderr

        # Check if the Triton code executed successfully
        if result_triton.returncode != 0:
            if verbose:
                print(f"Error in Triton code: {stderr_triton}")
            return None, None, None, None

        with open(gen_file+".out", 'w') as f:
            f.write(stdout_gen)
        with open(temp_triton_file+".out", 'w') as f:
            f.write(stdout_triton)

        with open(gen_file+".err", 'w') as f:
            f.write(stderr_gen)
        with open(temp_triton_file+".err", 'w') as f:
            f.write(stderr_triton)

        # Compare the outputs
        if stdout_gen == stdout_triton:
            return True, True, None, None
        else:
            return True, False, stdout_gen, "Error: not all test cases passed. The generated code and ground truth code produced different outputs."
    except Exception as e:
        if verbose:
            print(f"File: {fname}, Execution error: {e}")
        return False, False, None, str(e)
    # Clean up the temporary file
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"File: {fname} timed out!")
        return None, None, None, "Time out"
    finally:
        pass
        # print(f"temp file for File: {fname} removed!")
        # if os.path.exists(gen_file):
        #     os.remove(gen_file)
    return False, False, None, None

def extract_code_from_llm_output(response):
    # Extract code blocks from the LLM response
    code = None
    if "```" not in response:
        return response
    code_blocks = extract_code_blocks(response)
    for _code in code_blocks.code_dict_list:
        code += _code['context'] + "\n"
    return code

def get_fname_difficulty_from_label(label):
    triton_root = "dataloaders/TB_eval/TritonBench/data/TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json"
    with open(triton_root, 'r') as f:
        data = json.load(f)
        for item in data:
            if item['output'] == label:
                return item['file'], item['difficulty']
    return None, None

def process_code(code: str):
    if "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
    
    try:
        tree = ast.parse(code)
        imports = []
        function_definitions = []

        # Traverse the AST to find import statements and function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Collect the import statements
                imports.append(ast.unparse(node))  # Convert the AST node back to code
            elif isinstance(node, ast.FunctionDef):
                # Collect function definitions
                function_code = ast.unparse(node)  # Get the Python code for the function
                function_definitions.append(function_code)

        return "\n".join(imports) + "\n\n" + "\n".join(function_definitions)

    except:
        return code


def code_call_exec_success_allclose(code, fname, py_folder, temp_root="tmp2", atol=1e-3, rtol=1e-1, timeout=2*60, verbose=False):
    tmp_gen_folder = os.path.join(temp_root, "gen")
    os.makedirs(tmp_gen_folder, exist_ok=True)
    
    triton_root = py_folder
    triton_file = os.path.join(triton_root, fname)

    gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
    gen_file = os.path.join(tmp_gen_folder, gen_file)

    hash_line = "#"*146

    with open(triton_file, 'r') as f:
        lines = f.readlines()
        for iL, line in enumerate(lines):
            if line.strip() == hash_line:
                break
        test_code_lines = lines[iL+1:]
        test_code_lines_procs = test_code_lines

    # code = process_code(code)

    code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)

    with open(gen_file, 'w') as f:
        f.write(code)

    try:
        ## Just to a simple call to the generated code
        result_call = subprocess.run([f'python3 {gen_file}'], capture_output=True, text=True, timeout=timeout, shell=True)
        call_status = result_call.returncode == 0

        # Check for correctness
        result_corr = subprocess.run([f'python3 dataloaders/TB_eval/correctness.py --gen_file {gen_file} --ref_file {triton_file} --atol {atol} --rtol {rtol}'], capture_output=True, text=True, timeout=timeout, shell=True)
        stdout_corr = result_corr.stdout
        stderr_corr = result_corr.stderr

    except Exception as e:
        if verbose:
            print(f"File: {fname}, Execution error: {e}")
        return None, None, None, str(e), None, None
    
    # Clean up the temporary file
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"File: {fname} timed out!")
        return None, None, None, "Time out", None, None
    finally:
        pass

    with open(gen_file+".stdout", 'w') as f:
        f.write(stdout_corr)

    with open(gen_file+".stderr", 'w') as f:
        f.write(stderr_corr)

    # Check if the generated code executed successfully
    if result_corr.returncode != 0:
        if verbose:
            print(f"Error in generated code: {stderr_corr}")
        return call_status, None, result_call.stdout, result_call.stderr, stdout_corr, stderr_corr
    else:
        if verbose:
            print(f"Success in generated code: {stdout_corr}")
        _, exec_status, gen_stdout, gen_stderr = stdout_corr.split("*#*#")
        return call_status, exec_status, result_call.stdout, result_call.stderr, gen_stdout, gen_stderr

    

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def green_or_red(status):
    if status:
        return bcolors.OKGREEN
    else:
        return bcolors.FAIL

def color_end():
    return bcolors.ENDC

def bool_colorize(status):
    if status:
        return bcolors.OKGREEN + str(status) + bcolors.ENDC
    else:
        return bcolors.FAIL + str(status) + bcolors.ENDC