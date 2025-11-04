This is the Original README.md for GEAK triton, please checkout the README_HIP.md for GEAK Hip kernel optimization.

## Introduction

This is an LLM-based multi-agent framework, which can generate functional and efficient gpu kernels automatically.

The framework is extendable and flexible. You can easily make you own coding agent and test it on our update TritonBench (some bugs are fixed) https://github.com/AMD-AIG-AIMA/TB-eval

We also provide two predefined agents, Reflexion_oneshot and OptimAgent to let you run directly.

## OptimAgent
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
