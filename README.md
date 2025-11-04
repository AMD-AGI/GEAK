## Introduction

This README extends the original LLM-based multi-agent framework (check out the README_ori.md), focusing on generating optimized GPU kernels from existing HIP kernel/code implementations.

example_hip is provided for reference, including various repos, applications, and use cases. Please check it out here: 
   ```
   ./example_hip
   ```
By default, instruction is stored in example_hip/FromRe_instructions.json, and is loaded in 
   ```
   src/configs/hipbench_gaagent_config.yaml
   ```
Additionally, you can customize your own coding agent to test code optimization capabilities, simply following the example provided there e.g. silu.


## OptimAgent_v2_HIP
OptimAgent_HIP follows a pipeline similar to the original OptimAgent, comprising a Generator, Reflector & Evaluator, and Optimizer.

### Key Updates

- Takes original HIP code and instructions for generating optimized code. 
- Use previously generated codes as references, including their performance, for the Optimizer. 
- Controlled by the `ancestor_num` argument, set higher for HIP code optimization (recommended to match iteration count unless face issues). 
- Reference codes are sorted in ascending latency order to guide the Optimizer LLM toward optimal solutions. 

## Run the Agent, OptimAgent_v2_HIP
1. Prepare the environment and head to src/ folder
   ```
   python3 -m pip install -r requirements.txt
   cd src
   export MODEL_API_URL=<api.openai.com or other API URL>
   ```
   By default the model we use is GPT-5 from openai.

2. Edit config file. You need to put your API keys, instruction json path, and output_dir to the config file in src/ :
   ```
   configs/hipbench_gaagent_config.yaml
   ```
   in which descendant_num is 2 means 2 offsprings. The more offspring you use would slows down the iteration.
   For instruction json, check for examples in the example_hip, which contains the example usage as reported in our blog.
   
3. [Optional] Prepare dataset and setup the dataset path.
   By default, this repo will try to optimize the target file under example_hip folder.
   If you would like to try DIY on new examples, you might want to change the path in the following py file. 
   ```
   dataloaders/HIPBench.py
   ```
   
4. [Optional] To allow the Agent evaluate properly on your own use case, prepare the unittest file (More details regards to how the unittest file is written and formatted in example_hip). Please read and modify the utils file carefully under:
   ```
   dataloaders/HB_eval/utils.py
   ```
   In which there are two functions that needs some DIY: 
   extract_perf_validation: extract and get perf/validation result from the output of execution. 
   code_call_exec_success_allclose: compile and run the unittest of the generated code. 

   In the case of not generating perfs. Plz debug into the utils file for more info.

5. Run your target kernel and record the performance criterion into the utils.py. Check the base_perf variable in:
   ```
   dataloaders/HB_eval/utils.py
   ```
   function: extract_perf_validation.
   For example, if the kernel runtime printed in shell is 13ms, then the base_perf sould be set to 13.
   Without this step, the optimization performance might suffer. 
   
6. Run the agent via:
   ```
   python main_gaagent_hip.py
   ```
***Note: since the OptimalAgent requires proper latency measurement, it is strongly suggested that setting HIP_VISIBLE_DEVICES to allocate empty GPUs as an environment variable 

### Debugging trap & Resuming from Checkpoints
HIP OptimalAgent currently do not support resume from checkpoints, and debugging trap is currently not found with HIP code compile.


## Quick runing script & Benchmarking script
1. Quick runing:
   ```
   cd src
   bash run_gaagent_hip.sh
   ```

2. Benchmarking script:
   Change the output dir in ./src/run_gaagent_hip_benchmark.sh
   ```
   OUTPUT_BASE="/home/username/github_release"
   ```
   and run the script with:
   ```
   cd src
   bash run_gaagent_hip_benchmark.sh
   ```
