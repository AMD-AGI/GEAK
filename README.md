## Introduction

This is an LLM-based multi-agent framework, which can generate functional and efficient gpu kernels automatically.

The framework is extendable and flexible. You can easily make you own coding agent and test it on our TritonBench-revised Benchmark and Cambricon MLU Benchmark.

We also provide a baseline agent, GEAK-Agent to let you run directly.

## Profiler and Analyzer Features

If you need hardware-level profiling and performance analysis capabilities, please visit the **Profiler Analyzer branch**:
- **Branch**: [`profiler-analyzer`](https://github.com/AMD-AGI/GEAK-agent/tree/profiling) 
- Includes Cambricon MLU profiling tools and analyzer for detailed hardware metrics
- Provides insights into memory bandwidth, compute unit occupancy, and kernel optimization strategies

## GEAK-OptimAgent V2
<img width="443" alt="image" src="img/geakOAv2.png" />

It contains a Generator, a Reflector, an Evaluator and an Optimizer. The actor generates codes according to the query and context information. The Reflector is responsible for reflecting on the generated code and the error trace if the code failed to run. The Evaluator has a cascade structure. It tests the generated code for the functionality first. If the generated code doesn't pass the functionality test, the error trace will be fedback to the Reflector. Otherwise, the Evaluator will evaluate the performance including latency and efficiency. The Optimizer gets the generated codes, which pass the evaluator's tests, and gives a strategy to optimize the code in terms of latency and efficiency.

## Quick Start: Run the Agent on TritonBench
1. prepare the Agent environment
   ```
   git clone https://github.com/AMD-AGI/GEAK-agent.git
   cd GEAK-agent
   python3 -m pip install -r requirements.txt
   ```
2. prepare the TritonBench environment
   ```
   cd ..
   git clone https://github.com/AMD-AGI/GEAK-eval.git
   cd GEAK-eval
   python3 -m pip install -r requirements.txt
   python3 -m pip install -e .
   ```

3. edit config file. You need to give your API key, dataloader path and agent parameters in your config file.
   ```
   cd ../GEAK-eval/src
   cp configs/tritonbench_gaagent_config.yaml configs/tritonbench_gaagent_config_new.yaml
   ```
   You can modify dataloader paths to the above downloaded TritonBench.
   
4. put the config file in the main_gaagent.py and run the script
   ```
   python main_gaagent.py
   ```

### Resuming from Checkpoints
Result and memories will be stored in the `output_path` specified in the config file for each iteration. You can resume from any iter you want by specifying the `result_file`, `mem_file` and `start_iter` in the config file. For example:
```
result_path: "../outputs/optimagent_10.jsonl"
mem_file: "../outputs/optimagent_mem_10.json"
start_iter: 11
```

## Guide: use your own data
1. create a new file for your own dataloader in dataloaders
   ```
   touch dataloaders/YourData.py
   ```

2. In your own dataloader, define a new data class
   ```
   class YourData:
   ```

3. In the YourData class, you need to load `problem_states`, which is a list of ProblemState instances. The agent will run loops over all `problem_states`. Your can define your own ProblemState class in `dataloaders/ProblemState.py`. To meet the minimum requirement, each ProblemState instance should include the fields of `instruction` and `filename`. Providing `label` field (golden code) may be helpful for the Agent.

4. In order to use our Agent, the YourData class must implement the following methods:
   ```
   __len__() -> int
   load_ps(path) -> problem_states
   test_opt_correctness(code, filename, tmp_dir, exe_dir) -> pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr

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
     
         pass_call: True if the script runs without errors.
     
         pass_exe: True if the script produces the correct output.

         speedup: float, defined as the latency of golden code compared to that of generated code.
     
         stdout, stderr: Stdout and stderr from the test script execution.



     For our other agents, please look at the corresponding branches in the same repo

[`GEAK-HIP`](https://github.com/AMD-AGI/GEAK-agent/tree/GEAK-HIP)

[`GEAK-OpenEvolve for Triton `](https://github.com/AMD-AGI/GEAK-agent/tree/geak-openevolve)

[`GEAK-V1 for Triton`](https://github.com/AMD-AGI/GEAK-agent/tree/GEAK-v1)


