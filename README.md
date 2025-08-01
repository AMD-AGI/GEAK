## Technical report [![arXiv](https://img.shields.io/badge/arXiv-2507.23194-b31b1b.svg)](https://arxiv.org/abs/2507.23194)

## Introduction

This is an LLM-based multi-agent framework, which can generate functional and efficient gpu kernels automatically.

The framework is extendable and flexible. You can easily make you own coding agent and test it on our GEAK-eval https://github.com/AMD-AIG-AIMA/GEAK-eval.
We provide a baseline agent to let you run directly.

## GEAK-agent
<img width="443" alt="image" src="https://github.com/user-attachments/assets/f5841a54-e3f1-4256-a380-0c75cff086e4" />

It contains a Generator, a Reflector, an Evaluator and an Optimizer. The actor generates codes according to the query and context information. The Reflector is responsible for reflecting on the generated code and the error trace if the code failed to run. The Evaluator has a cascade structure. It tests the generated code for the functionality first. If the generated code doesn't pass the functionality test, the error trace will be fedback to the Reflector. Otherwise, the Evaluator will evaluate the performance including latency and efficiency. The Optimizer gets the generated codes, which pass the evaluator's tests, and gives a strategy to optimize the code in terms of latency and efficiency.

### the Optimizer
We provide previous generated codes as reference codes with their corresponding performance to the Optimizer. The number of reference codes is controlled by the arg  `ancestor_num`. The reference codes are arranged in ascending order to help the Optimizer LLM find the optimization direction. We don't ask the LLM to generate new codes directly from the reference codes, instead we ask the Optimizer to analyze the reference codes first and to generate a promising strategy to optimize the code. Then we feed the generated optimization stratgey to the Generator to generate new codes.

### debugging trap
LLMs frequently get caught in debugging traps. When their generated code has bugs, we provide the error trace to the Reflector correction. However, we've observed that sometimes code can undergo several reflection cycles while still being plagued by the same bug. We refer to this as a debugging trap.

To prevent the LLM from getting stuck in a debugging trap, we limit debugging attempts per code snippet using `max_perf_debug_num`. If the code fails after this many fixes, the agent must abandon the current approach and generate a fresh strategy and code.

## Run the Agents
1. prepare the environment
   ```
   python3 -m pip install -r requirements.txt
   ```

2. go to the src/ folder
   ```
   cd src
   ```

3. edit config file. You need to give your API key and TritonBench data path in your config file.
   ```
   cp configs/tritonbench_optimagent_config.yaml configs/tritonbench_optimagent_config_new.yaml
   ```
   
4. put the config file in the main_optimagent.py and run the script
   ```
   python main_optimagent.py
   ```

### Resuming from Checkpoints
Result and memories will be stored in the `output_path` specified in the config file for each iteration. You can resume from any iter you want by specifying the `result_file`, `mem_file` and `start_iter` in the config file. For example:
```
result_path: "../outputs/optimagent_10.jsonl"
mem_file: "../outputs/optimagent_mem_10.json"
start_iter: 11
```

## How to use your own data
1. create a new file for your own dataloader in dataloaders
   ```
   touch dataloaders/OnlineData.py
   ```

2. In your own dataloader, define a new data class
   ```
   class OnlineData:
   ```

3. In the OnlineData, you need load `problem_states`, which is a list of ProblemState instances. To meet the minimum requirement, each ProblemState instance should include the following fields: `instruction`, `filename`, `test_code`.

4. In order to use OptimAgent, the OnlineData class must implement the following methods:
   ```
   __len__() -> int
   test_opt_correctness(code, filename, tmp_dir, exe_dir) -> pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr
   write_perf_file(input_folder_path, results_path, tmp_dir) -> None
   run_perf_scripts(gpu_id, script_dir, log_dir) -> None
   write_file(file_path) -> None

   ```

   ### Method Description
   - \_\_len\_\_()
     
     Returns the number of problem_states in the dataset.

   - test_opt_correctness(code, filename, tmp_dir, exe_dir)
   
     Tests whether the generated code is functionally correct.

     Parameters:
     
         code: The generated code to be tested.
     
         filename: Name of the test script file.
     
         tmp_dir: Directory to save the script (generated code + unit test).
     
         exe_dir: Directory to store scripts that pass execution tests.
   
     Returns:
     
         ass_call: True if the script runs without errors.
     
         pass_exe: True if the script produces the correct output.
     
         call_stdout, call_stderr: Stdout and stderr from the initial script execution.
     
         exe_stdout, exe_stderr: Stdout and stderr from the execution verification step.
   
     Note:
     
         You might need to provide a Python file containing both the golden code and unit test code, separated by a line of exactly 146 hash (#) symbols. The unit test will be appended to the           generated code for validation.
   
   - write_perf_file(input_folder_path, results_path, tmp_dir)
     
     Generates performance test scripts using the code in input_folder_path (which is typically the same as exe_dir from test_opt_correctness). The function will append unit tests for performance tests to the generated codes and store the scripts in tmp_dir. The performance test results will be saved in results_path.

   - run_perf_scripts(gpu_id, script_dir, log_dir)

     Executes the performance test scripts found in script_dir on the specified GPU (gpu_id) and stores the logs in log_dir.

   - write_file(file_path)
     
     Serializes and saves the current result to the specified file path.

#### Citation
If you find this work useful in your research or applications, please consider citing:

```bibtex
@misc{wang2025geakintroducingtritonkernel,
      title={Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks}, 
      author={Jianghui Wang and Vinay Joshi and Saptarshi Majumder and Xu Chao and Bin Ding and Ziqiong Liu and Pratik Prabhanjan Brahma and Dong Li and Zicheng Liu and Emad Barsoum},
      year={2025},
      eprint={2507.23194},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.23194}, 
}
```
