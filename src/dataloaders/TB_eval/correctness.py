import json
import argparse
import os
import sys
from random import randint
from glob import glob
import importlib.util
import random
import numpy as np
import torch
from collections import namedtuple

torch.set_printoptions(profile="full")

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def import_variable_from_file(file_path, variable_name):
    """
    Dynamically imports a variable from a Python file.

    Parameters:
    - file_path (str): The path to the Python file.
    - variable_name (str): The name of the variable to import.

    Returns:
    - The value of the specified variable, or None if not found.
    """
    set_seed()

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a module spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load specification for module {module_name}")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Could not create module {module_name} from spec")

    # Execute the module in its own namespace
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Could not execute module {module_name} due to {e}")

    # Retrieve the variable from the module
    return getattr(module, variable_name, None)

def _compare(tri, pyt, fname, atol=1e-3, rtol=1e-3, verbose=False):
    if type(pyt) == np.ndarray:
        if np.allclose(tri, pyt,  atol=atol, rtol=rtol):
            if verbose:
                print(f"PyTorch and Triton matched for file: {fname}")
            return True
        else:
            if verbose:
                diff = np.amax(np.abs(tri - pyt))
                print(f"Test failed for file: {fname} with abs max diff: {diff}")
            return False
    elif type(pyt) == torch.Tensor:
        if torch.allclose(tri, pyt,  atol=atol, rtol=rtol):
            if verbose:
                print(f"PyTorch and Triton matched for file: {fname}")
            return True
        else:
            if verbose:
                diff = (tri - pyt).abs().max()
                print(f"Test failed for file: {fname} with abs max diff: {diff}")
            return False
    elif type(pyt) == namedtuple:
        if tri._fields != pyt._fields:
            return False
        for field in tri._fields:
            if not torch.equal(getattr(tri, field), getattr(pyt, field)):
                return False
        return True
    else:
        return tri == pyt
    return None

def compare(ref, gen, fname, atol=1e-3, rtol=1e-3, verbose=False):
    ret_val = True
    # import pdb; pdb.set_trace()
    if (type(gen) == list) or (type(gen) == tuple):
        for tri, pyt in zip(ref, gen):
            ret_val &= compare(tri, pyt, fname)
    elif type(gen) == dict:
        return compare( list(ref.values()), list(gen.values()), fname, atol=atol, rtol=rtol, verbose=verbose)
    else:
        ret_val &= _compare(ref, gen, fname, atol=atol, rtol=rtol, verbose=verbose)
    return ret_val

def test_correctness(ref_file, gen_file, var_name, atol=1e-3, rtol=1e-3, verbose=False):
    fname = os.path.basename(gen_file)
    gen_call_acc, ref_call_acc = False, False
    gen_stderr, ref_stderr = None, None
    try:
        gen_result_golden = import_variable_from_file(gen_file, var_name)
        if verbose:
            with open(gen_file+".out", "w") as f:
                f.write(f"file: {fname}\n")
                json.dump(str(gen_result_golden), f)
                f.write("\n\n\n")
                f.write("#"*146)
        gen_call_acc = True
    except Exception as e:
        gen_stderr = e
        return gen_call_acc, None, None, gen_stderr
    try:
        ref_result_golden = import_variable_from_file(ref_file, var_name)
        if verbose:
            with open(gen_file+".out_ref", "w") as f:
                f.write(f"file: {fname}\n")
                json.dump(str(ref_result_golden), f)
                f.write("\n\n\n")
                f.write("#"*146)
        ref_call_acc = True
    except Exception as e:
        ref_stderr = e
        return gen_call_acc, None, None, ref_stderr
    # assert (ref_result_golden is not None), f'Reference output is None for file: {ref_file}'
    # assert (gen_result_golden is not None), f'Generated output is None for file: {gen_file}'
    # assert type(gen_result_golden) == type(ref_result_golden), f"Reference and Generated output results should be of the same type but generated is: {type(gen_result_golden)}, and reference is: {type(ref_result_golden)}"
    if gen_result_golden is None:
        return gen_call_acc, False, None, "Generated output is None"
    if ref_result_golden is None:
        return gen_call_acc, False, None, "Reference output is None"
    if type(gen_result_golden) != type(ref_result_golden):
        return gen_call_acc, False, None, f"Reference and Generated output results should be of the same type but generated is: {type(gen_result_golden)}, and reference is: {type(ref_result_golden)}"
    exec_acc = compare(ref_result_golden, gen_result_golden, fname, atol=atol, rtol=rtol, verbose=verbose)
    if not exec_acc:
        gen_stderr = f"Generated output does not match reference output for file: {fname}"
    return gen_call_acc, exec_acc, None, gen_stderr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", "-pf", type=str, required=True)
    parser.add_argument("--ref_file", "-tf", type=str, required=True)
    parser.add_argument("--var_name", type=str, default="result_gold")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    gen_call_acc, exec_acc, stdout, gen_stderr = test_correctness(args.ref_file, args.gen_file, args.var_name, atol=args.atol, rtol=args.rtol, verbose=args.verbose)
    print(f"{gen_call_acc}*#*#{exec_acc}*#*#{stdout}*#*#{gen_stderr}")
