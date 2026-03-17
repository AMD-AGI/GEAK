# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from geak_agent.agents.reflexion_oneshot import Reflexion_Oneshot
from geak_agent.utils.utils import clear_code, extract_function_signatures, clear_json
from geak_agent.memories.Memory import MemoryClassMeta
from geak_agent.prompts import prompt_for_generation, prompt_for_reflection
from loguru import logger
from tenacity import RetryError
from geak_agent.dataloaders.ProblemState import tempCode
from typing import List, Optional
from geak_eval.evaluators.interface import get_evaluators

class GaAgent(Reflexion_Oneshot):
    def __init__(self, model, dataset, corpus_path, max_perf_debug_num=5, mem_file=None, descendant_num=1):
        super().__init__(model, dataset, corpus_path, mem_file, descendant_num)
        self.max_perf_debug_num = max_perf_debug_num

    def memory_init(self, mem_file=None, descendant_num=1):
        """
        Args:
            mem_file: previous stored memories, which can be loaded to continue run
        """
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "call_err_msg", 
                                                             "exe_err_msg",
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot", 
                                                             "perf_candidates",
                                                             "perf_strategy",
                                                             "raw_codes",
                                                             "call_candidate",
                                                             "exe_candidate",
                                                             "temp_strategy",
                                                             "perf_debug_num",
                                                             "pass_call", 
                                                             "pass_exe",
                                                             "pass_perf",
                                                             "history"]):
            pass
        
        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)
            assert len(input_mems) == len(self.dataset), f"expect {len(self.dataset)} samples, but got {len(input_mems)} instead"

        for ps in self.dataset.problem_states:

            if ps.label:
                fs_mem = extract_function_signatures(ps.label)
            else:
                fs_mem = None
            raw_codes =None
            if mem_file is None:
                os_mem = self.instruction_retriever.query(ps.instruction)[0]
                tmp_mem = Memory(ps=ps, 
                                call_err_msg=None,
                                exe_err_msg=None, 
                                reflection=None, 
                                function_signatures=fs_mem, 
                                oneshot=os_mem["code"], 
                                perf_candidates=[],
                                perf_strategy=None,
                                raw_codes=raw_codes,
                                call_candidate=None,
                                exe_candidate=None,
                                temp_strategy=None,
                                perf_debug_num=0,
                                pass_call=False,
                                pass_exe=False,
                                pass_perf=False,
                                history=[[] for _ in range(descendant_num)]
                                )
            else:
                input_mem = input_mems[ps.filename]
                tmp_mem = Memory(
                    ps=ps,
                    call_err_msg=input_mem["call_err_msg"],
                    exe_err_msg=input_mem["exe_err_msg"], 
                    reflection=input_mem["reflection"], 
                    function_signatures=fs_mem, 
                    oneshot=input_mem["oneshot"], 
                    perf_candidates=input_mem["perf_candidates"],
                    perf_strategy=input_mem["perf_strategy"],
                    raw_codes=raw_codes,
                    call_candidate=input_mem["call_candidate"],
                    exe_candidate=input_mem["exe_candidate"],
                    temp_strategy=input_mem["temp_strategy"],
                    perf_debug_num=input_mem["perf_debug_num"],
                    pass_call=input_mem["pass_call"],
                    pass_exe=input_mem["pass_exe"],
                    pass_perf=input_mem["pass_perf"],
                    history=[[] for _ in range(descendant_num)]
                )

            self.memories.append(tmp_mem)
    
    def write_memories(self, file_path):
        output_dict = {}
        with open(file_path, "w") as f:
            for mem in self.memories:
                output = {
                    "call_err_msg": str(mem.call_err_msg),
                    "exe_err_msg": str(mem.exe_err_msg),
                    "reflection": mem.reflection, 
                    "oneshot": mem.oneshot, 
                    "perf_candidates": [list(cand) for cand in mem.perf_candidates],
                    "perf_strategy": mem.perf_strategy,
                    "call_candidate": mem.call_candidate,
                    "exe_candidate": mem.exe_candidate,
                    "temp_strategy": mem.temp_strategy,
                    "perf_debug_num": mem.perf_debug_num,
                    "pass_call": mem.pass_call, 
                    "pass_exe": mem.pass_exe,
                    "pass_perf": mem.pass_perf
                }
                output_dict[mem.ps.filename] = output
            json.dump(output_dict, f)
    
    def run(self, output_path=None, multi_thread=True, datalen=None, iteration_num=0, temperature=0, ancestor_num=5, descendant_num=1, mutation=False, start_idx=0, gpu_id=0, start_iter=0, descendant_debug=1, target_mlu='MLU590', profiling=False):
        """
        Args:
            output_path: the folder to store the final result
            multi_thread: whether use multithreading for generating
            datalen: for debug, to specify how many data from the dataset you want to use
            iteration_num: how many iterations you want to run
            temperature: LLM temperature
            ancestor_num: how many samples you want to add in the prompt when optimize the code
            descendant_num: how many codes you want to generate in one try
            start_idx: start idx of the data rows
            gpu_id: which gpu you want to use when you test the scripts
            start_iter: which iteration you want to start with. useful when you load previous result and memory
        """
        assert ancestor_num >= 0, f"expect ancestor_num to be larger than 0, bug got {ancestor_num}"
        assert descendant_num >= 0, f"expect descendant_num to be larger than 0, bug got {descendant_num}"
        assert descendant_debug >= 0, f"expect descendant_debug to be larger than 0, bug got {descendant_debug}"
        data_len = datalen if datalen else len(self.dataset)
        evaluator = get_evaluators['tbg']()
        for iter in range(start_iter, iteration_num):
            logger.info(f"\n=== Iteration {iter} ===")
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{iter}{extension}"
                mem_output_path = f"{root}_mem_{iter}.json"

            if multi_thread:
                thread_num = 3
            # generate solution
            logger.info(f"\ngenerate solution")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_solution, mem, temperature, descendant_num, mutation): mem for mem in self.memories[start_idx:(start_idx + data_len)]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:data_len]:
                        self.generate_solution(mem, temperature=temperature, descendant_num=descendant_num, mutation=mutation)
                        pbar.update(1)
            
            # generate LLM evaluation
            logger.info(f"\ngenerate LLM evaluation")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_llm_evaluate, mem, temperature): mem for mem in self.memories[start_idx:(start_idx + data_len)]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:data_len]:
                        self.generate_llm_evaluate(mem, temperature=temperature)
                        pbar.update(1)
            
            # run scripts
            logger.info(f"\nrun call scripts on gpu")
            if output_path is None or (hasattr(self.dataset, 'MLU_tests') and self.dataset.MLU_tests):
                tmp_dir = "tmp"
                exe_dir = "pass_exe"
                perf_result_dir = "perf_results"
                perf_log_dir = "perf_logs"
            else:
                root, extension = os.path.splitext(output_path)
                tmp_dir = f"{root}_tmp"
                exe_dir = f"{root}_pass_exe"
                perf_result_dir = f"{root}_perf_results"
                perf_log_dir = f"{root}_perf_logs"
            for mem in tqdm(self.memories[start_idx:data_len]):
                if mem.raw_codes:
                    for i in range(len(mem.raw_codes)):
                        raw_code = mem.raw_codes[i]
                        if raw_code.pass_exe:
                            continue
                        try:
                            if raw_code.code :
                                opname = mem.ps.opname + f"_{i}" + ".py"
                                # ref="/home/chaox/kernel_agent/TB-eval/tb_eval/data/MLU/data/MLU_v1"
                                # tests_sep_line = '#'*146
                                # ref_file = os.path.join(ref,mem.ps.filename)
                                # with open(ref_file,'r') as f:
                                #     lines = f.readlines()
                                #     for iL, line in enumerate(lines):
                                #         if line.strip() == tests_sep_line:
                                #             break
                                #     print("get the ref code")
                                #     code_lines = lines[:iL]
                                # raw_code.code = ''.join(code_lines)
                                pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = self.dataset.test_opt_correctness(raw_code.code, mem.ps.filename, opname, tmp_dir, exe_dir=exe_dir, gpu_id=gpu_id)
                                # pass_call, pass_exe, speedup, call_stdout, call_stderr = evaluator(raw_code.code, tmp_dir, exe_dir, mem.ps.filename, atol=1e-2, rtol=1e-1, custom_tests_path=None, gpu_id=gpu_id)
                            else:
                                pass_call, pass_exe, speedup, call_stdout, call_stderr, exe_stdout, exe_stderr = False, False, 0.0, "", "Code is empty", "", "Code is empty"
                            # pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = self.dataset.test_opt_correctness(raw_code.code, filename, tmp_dir, exe_dir=exe_dir, gpu_id=gpu_id)
                        except Exception as e:
                            print(f"failed to test the code for {mem.ps.filename}")
                            raw_code.test_stdout = f"failed to test the code due to: {e}"
                            raw_code.test_stderr = f"failed to test the code due to: {e}"
                            continue
                        if not pass_call:
                            raw_code.test_stdout = call_stdout
                            raw_code.test_stderr = call_stderr
                            raw_code.profilig = None
                        elif pass_call and not pass_exe:
                            raw_code.pass_call = True
                            raw_code.test_stdout = exe_stdout
                            raw_code.test_stderr = exe_stderr
                            mem.call_candidate = raw_code.code
                            mem.temp_strategy = raw_code.strategy
                            mem.pass_call = True
                            raw_code.profilig= None
                        else:
                            raw_code.pass_call = True
                            raw_code.pass_exe = True
                            mem.pass_call = True
                            mem.exe_candidate = raw_code.code
                            mem.call_candidate = raw_code.code
                            mem.temp_strategy = raw_code.strategy
                            if profiling:
                                pass_prfiler, stdout_profile, stderr_profile, stdout_analyze = self.dataset.test_kernel_profiling(raw_code.code, mem.ps.filename, tmp_dir, exe_dir=exe_dir, target_mlu=target_mlu, timeout=30*60)
                                raw_code.profilig = stdout_analyze
                        mem.call_err_msg = raw_code.test_stdout
                        mem.exe_err_msg = raw_code.test_stderr
                    descendant_debug = min(descendant_debug,len(mem.raw_codes))
                    if sum(rc.pass_exe for rc in mem.raw_codes)>=descendant_debug:
                        mem.pass_exe = True
            
            
            # run perf scripts
            logger.info(f"\nrun call scripts on gpu")

            root, extension = os.path.splitext(output_path)
            exe_dir = os.path.join(root,exe_dir) if output_path else exe_dir
            perf_result_dir = os.path.join(root, perf_result_dir) if output_path else perf_result_dir
            if not os.listdir(exe_dir):
                pass
                logger.warning(f"No scripts passed correctness checks in iteration {iter}. Skipping performance evaluation.")
            else:
                perf_results_dict = {}

                if hasattr(self.dataset, 'MLU_tests') and self.dataset.MLU_tests:
                    perf_results_dict = self.dataset.run_perf_evaluation(
                        exec_folder=exe_dir, 
                        gen_perf_folder=perf_result_dir,
                        gpu_id=gpu_id
                    )
            
            for mem in tqdm(self.memories[start_idx:(start_idx + data_len)],desc="Performance Evaluation"):
                if mem.raw_codes:
                    for i in range(len(mem.raw_codes)):
                        raw_code = mem.raw_codes[i]
                        if not raw_code.pass_exe or raw_code.pass_perf:
                            continue
                        opname = mem.ps.opname + f"_{i}" + ".json"
                        speedup = None
                        eff=None
                        if opname in perf_results_dict.keys():
                            # import pdb;pdb.set_trace()
                            speedup = perf_results_dict[opname].get("ms")
                            eff = perf_results_dict[opname].get("efficiency")
                        if speedup is not None and eff is not None:
                            raw_code.pass_perf = True
                            mem.pass_perf = True
                            raw_code.latency = speedup
                            raw_code.eff = eff
                        else:
                            raw_code.pass_perf = False
                            mem.pass_perf = False
                            raw_code.latency = None
                            raw_code.eff = None


            # generate reflections
            logger.info(f"\ngenerate reflections")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_reflexion, mem, temperature): mem for mem in self.memories[start_idx:(start_idx + data_len)]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:data_len]:
                        self.generate_reflexion(mem, temperature=temperature)
                        pbar.update(1)

            # update perf_candidates

            for mem in self.memories[start_idx:data_len]:
                if mem.raw_codes:
                    for i in range(len(mem.raw_codes)):
                        raw_code = mem.raw_codes[i]
                        mem.history[i].append(raw_code)
                        codes_sorted = sorted(mem.history[i], key=lambda x: x.llm_metric, reverse=True)
                        mem.history[i] = codes_sorted[:5]
                        if raw_code.pass_perf and raw_code.strategy:
                            raw_code.strategy = None
                            self.update_perf_candidates(mem=mem, raw_code=raw_code, ancestor_num=ancestor_num)
                if len(mem.perf_candidates) > 0:
                    mem.ps.solution = mem.perf_candidates[0][0]
                    mem.ps.speedup = mem.perf_candidates[0][1]
                elif mem.exe_candidate:
                    mem.ps.solution = mem.exe_candidate
                elif mem.call_candidate:
                    mem.ps.solution = mem.call_candidate
                elif mem.raw_codes:
                    mem.ps.solution = mem.raw_codes[0].code

            if output_path is not None:
                self.dataset.write_file(iter_path)
                self.write_memories(mem_output_path)

            os.system(f'rm -rf {exe_dir}')
            os.system(f'rm -rf {perf_result_dir}')
            os.system(f'rm -rf {perf_log_dir}')
            os.system(f'rm -rf {tmp_dir}')
    
    def generate_solution(self, mem, temperature=0, descendant_num=1, mutation=False):

        tab = "\n"
        fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        text = prompt_for_generation.prompt_MLU.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text
        )
        
        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if (mem.perf_debug_num >= self.max_perf_debug_num) or mem.pass_exe:
            mem.perf_debug_num = 0
            mem.raw_codes =None
        if len(mem.perf_candidates) > 0 and not mem.raw_codes:
            text += """\nThere are some Optimized codes(NO.1, NO.2 and so on) to solve the Problem. The Optimized codes are arranged in ascending order based on their performance, where higher speedup and higher efficiencies indicate better performance. According to their performance(speedup is the latency compared with golden reference code and efficiency in TFLOPS or GB/s) and the corresponding analysis, you need to generate a new code with better performance. You should maintain code correctness during optimization."""
            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables."    
            for i, cand in enumerate(mem.perf_candidates):
                text += f"\n### Reference {i+1}"
                text += f"\nOptimized code: {cand[0]}"
                text += f"\nOptimized speedup: {cand[1]}"
                text += f"\nOriginal efficiency(TFLOPS, GB/s): {cand[2]}"
                if cand[3]:
                    text += f"\nStrategy: {cand[3]}"
                if cand[4]:
                    text += f"\nRocprof-compute profiling result:{cand[4]}"
            if mutation:
                text += "\nGenerate a better strategy completely different from Optimized Implementation. Based on the better strategy generate a better optimization code."
            else:    
                text += "\nAnalyze and compare all optimization strategies based on Optimized Implementation codes and give a better strategy motivated by them. Based on the better strategy generate a better optimization code to get a higher speedup."
        else:           
            if not mem.exe_candidate and not mem.call_candidate and not mem.raw_codes:
                text += f"\nHere is an example snippet of code: {mem.oneshot}"
            elif mem.raw_codes:
                one_shot = self.code_retriever.query(mem.raw_codes[0].code)[0]["code"]
                text += f"\nHere is an example snippet of code: {one_shot}"
        
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if not raw_code.pass_perf:
                    text_temp = text
                    history_text = self._build_history_prompt(mem.history[i])
                    text_temp += f"\nPrevious attempt implementations:{history_text}"
                    text_temp += prompt_for_generation.system_prompt
                    if raw_code.reflections:
                        raw_code.reflections = None
                    try:
                        raw_code.code, raw_code.strategy = self.call_llm_code(prompt=text_temp, temperature=temperature)
                    except:
                        logger.info(f"failed to call LLM for {mem.ps.filename}")
            mem.perf_debug_num +=1
            return
        
        gens_codes: List[tempCode] = []
        for i in range(descendant_num):
            gen_code = tempCode()
            try:
                text_temp = text
                text_temp += prompt_for_generation.system_prompt
                gen_code.code, gen_code.strategy = self.call_llm_code(prompt=text_temp, temperature=temperature)
            except:
                logger.info(f"failed to call LLM for {mem.ps.filename}")
            gens_codes.append(gen_code)
        mem.raw_codes = gens_codes
        mem.pass_exe = False
        mem.pass_call = False
        mem.pass_perf = False
        return
    
    
    def generate_reflexion(self, mem, temperature):
        
        tab = "\n"
        fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        m_info = """
- runnable test: test if the code can be successfully executed.
- correctness test: test if the output of the code is correct, i.e. if the code does implement the functionality required in the original problem.
- speedup: measures the total time from kernel launch to completion, reflecting the responsiveness and overhead of executing a single instance of the kernel on the GPU. And compare the time with golden reference code to get speedup.
"""
        
        if mem.raw_codes :
            # Extend history if needed (same check as in generate_solution)
            while len(mem.history) < len(mem.raw_codes):
                mem.history.append([])
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if  raw_code.reflections:
                    continue
                history_text = self._build_history_prompt(mem.history[i])
                if raw_code.pass_exe:
                    result_txt = f"""
- runnable test: Succeed
- correctness test: Succeed
- speedup: {raw_code.latency}
"""                 
                    reflect_txt = prompt_for_reflection.prompt_evolve_strategy_optimize_MLU.format(
                        instruction=mem.ps.instruction,
                        function_signatures=fss_text,
                        metrics_info=m_info,
                        evolution_history=history_text,
                        current_program=raw_code.code,
                        test_result=result_txt,
                        reflection=raw_code.reflections
                    )
                else:
                    if raw_code.pass_call:
                        result_txt = f"""
- runnable test: Succeed
- correctness test: Failed
Error Message: {raw_code.test_stderr}
"""
                    else:
                        result_txt = f"""
- runnable test: Failed
Error Message: {raw_code.test_stderr}
- correctness test: Failed
Error Message: {raw_code.test_stderr}
"""

                    reflect_txt = prompt_for_reflection.prompt_evolve_reflect_MLU.format(
                        instruction=mem.ps.instruction,
                        function_signatures=fss_text,
                        metrics_info=m_info,
                        evolution_history=history_text,
                        current_program=raw_code.code,
                        test_result=result_txt,
                        reflection=raw_code.reflections
                    )

                
                reflect_msg = [
                    {
                        "role": "user",
                        "content": reflect_txt
                    }
                ]
                raw_code.reflections = self.model.generate(reflect_msg, temperature=temperature)

    
    def call_llm_code(self, prompt, temperature):
        msg = [{"role": "user", "content": prompt}]
        try:
            # import pdb; pdb.set_trace()
            response = self.model.generate(msg, temperature=temperature, max_tokens=30000)
            opti = clear_json(response)
            if 'code' in opti.keys() and  'strategy' in opti.keys():
                code = clear_code(opti['code'])
                strategy = opti['strategy']
                return code, strategy 
        except:
            logger.info(f"failed to call LLM")
            raise ValueError("failed to call LLM")

    def call_llm_reflecion(self, prompt, temperature):
        msg = [{"role": "user", "content": prompt}]
        try:
            response = self.model.generate(msg, temperature=temperature, max_tokens=30000)
            opti = clear_json(response)
            if 'reflection' in opti.keys():
                reflection = opti['reflection']
                return reflection 
        except:
            logger.info(f"failed to call LLM")
            raise ValueError("failed to call LLM")

    def update_perf_candidates(self,mem, raw_code: tempCode, ancestor_num):
        if len(mem.perf_candidates) < ancestor_num:
            candidate = [raw_code.code, raw_code.latency, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates.append(tuple(candidate))
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)

        elif mem.perf_candidates[-1][1] <= raw_code.latency:
            candidate = [raw_code.code, raw_code.latency, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates[-1] = tuple(candidate)
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)
            
    def _build_history_prompt(self, history):
        text = ""
        history_template = """
### Attempt {attempt_number}
- Code: 
```python
{code}
```

- Test Results: 
{test_results}


- Analysis:
{reflection}
"""
        for i, raw_code in enumerate(history):
            if raw_code.pass_perf:
                test_txt = """
runnable test: Succeed
correctness test: Succeed
speedup: {latency}
"""
                test_txt = test_txt.format(
                    speedup=raw_code.latency
                )
            elif raw_code.pass_exe and not raw_code.pass_perf:
                test_txt = """
runnable test: Succeed
correctness test: Succeed
"""
            elif raw_code.pass_call and not raw_code.pass_exe:
                test_txt = """
runnable test: Succeed
correctness test: {err_msg}
"""
                test_txt = test_txt.format(
                    err_msg=raw_code.test_stderr
                )
            elif not raw_code.pass_call:
                test_txt = """
runnable test: {err_msg}
"""
                test_txt = test_txt.format(
                    err_msg=raw_code.test_stderr
                )
            text += history_template.format(
                attempt_number=i+1,
                code=raw_code.code,
                test_results=test_txt,
                reflection=raw_code.reflections
            )
        
        return text
    
    
    def generate_llm_evaluate(self, mem, temperature=1.0):
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if not raw_code.pass_perf:
                    text = ""
                    text += prompt_for_generation.llm_evaluate_prompt.format(current_program=raw_code.code)
                    msg = [{"role": "user", "content": text}]
                    try:
                        response = self.model.generate(msg, temperature=temperature, max_tokens=30000)
                        llm_eval = clear_json(response)
                        metric = 0.0
                        for k, v in llm_eval.items():
                            if k == "reasoning":
                                continue
                            if isinstance(v, float) or isinstance(v, int):
                                metric += float(v)
                        raw_code.llm_metric = metric
                        raw_code.llm_eval = llm_eval
                    except:
                        logger.info(f"failed to generate LLM evaluation")
                        raise ValueError("failed to generate LLM evaluation")
