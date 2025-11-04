from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.reflexion_oneshot import Reflexion_Oneshot
from utils.utils import clear_code, extract_function_signatures, clear_json
from memories.Memory import MemoryClassMeta
from prompts import prompt_for_generation, prompt_for_reflection
from loguru import logger
from tenacity import RetryError
from dataloaders.ProblemState import tempCode
from typing import List, Optional

class GaAgent(Reflexion_Oneshot):
    def __init__(self, model, dataset, corpus_path, max_perf_debug_num=5, mem_file=None):
        super().__init__(model, dataset, corpus_path, mem_file)
        self.max_perf_debug_num = max_perf_debug_num

    def memory_init(self, mem_file=None):
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
                                history=[]
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
                    history=input_mem["history"]
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
    
    def run(self, output_path=None, multi_thread=True, datalen=None, iteration_num=0, temperature=0, ancestor_num=5, descendant_num=1, mutation=False, start_idx=0, gpu_id=0, start_iter=0, descendant_debug=1, target_gpu='MI250'):
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
        assert descendant_debug > 0, f"expect descendant_debug to be larger than 0, bug got {descendant_debug}"
        data_len = datalen if datalen else len(self.dataset)
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
            
            # run scripts
            logger.info(f"\nrun scripts on gpu")
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                tmp_dir = f"{root}_tmp"
                exe_dir = f"{root}_pass_exe"
                perf_result_dir = f"{root}_perf_results"
                perf_log_dir = f"{root}_perf_logs"

            else:
                tmp_dir = "tmp"
                exe_dir = "pass_exe"
                perf_result_dir = "perf_results"
                perf_log_dir = "perf_logs"
            
            for mem in tqdm(self.memories[start_idx:data_len]):
                if mem.raw_codes:
                    for i in range(len(mem.raw_codes)):
                        raw_code = mem.raw_codes[i]
                        if raw_code.pass_perf:
                            continue
                        try:
                            filename = mem.ps.filename[:-3] + f"_{i}" + ".py"
                            pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = self.dataset.test_opt_correctness(raw_code.code, filename, tmp_dir, exe_dir=exe_dir, gpu_id=gpu_id)
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
                            
                            pass_prfiler, stdout_profile, stderr_profile, stdout_analyze = self.dataset.test_kernel_profiling(raw_code.code, mem.ps.filename, tmp_dir, exe_dir=exe_dir, target_gpu=target_gpu, timeout=20*60)
                            raw_code.profilig = stdout_analyze
                    descendant_debug = min(descendant_debug,len(mem.raw_codes))
                    if sum(rc.pass_exe for rc in mem.raw_codes)>=descendant_debug:
                        mem.pass_exe = True
            script_dir = os.path.join(tmp_dir, "perf_gen")
            self.dataset.write_perf_file(input_folder_path=exe_dir, results_path=perf_result_dir, tmp_dir=script_dir) 

            self.dataset.run_perf_scripts(gpu_id=gpu_id, script_dir=script_dir, log_dir=perf_log_dir)

            # get ms and efficiency
            print("===="*40)
            
            for mem in self.memories[start_idx:data_len]:

                for i in range(len(mem.raw_codes)):
                    raw_code = mem.raw_codes[i]
                    if raw_code.pass_perf:
                        continue
                    filename = mem.ps.filename[:-3] + f"_{i}"
                    path_gen = os.path.join(perf_result_dir, filename + ".json")
                    path_ref = None
                    if self.dataset.perf_ref_folder is not None:
                        path_ref = os.path.join(self.dataset.perf_ref_folder, mem.ps.filename[:-3] + ".json")
                    if not os.path.exists(path_gen):
                        continue
                    try:
                        _, efficiency, ms = self.dataset.calculate(path_gen, path_ref)

                        raw_code.pass_perf = True
                        mem.pass_perf = True
                        print(f"{mem.ps.filename}: {ms}")
                        print(f"{mem.ps.filename}: {efficiency}\n")
                        
                        raw_code.latency = ms
                        raw_code.eff = efficiency
                        
                    except Exception as e:
                        print(f"{mem.ps.filename} failed due to {e}")
                        raw_code.pass_perf = False

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
                mem.history.append(mem)
                if mem.raw_codes:
                    for raw_code in mem.raw_codes:
                        if raw_code.pass_perf and raw_code.strategy:
                            raw_code.strategy = None
                            self.update_perf_candidates(mem=mem, raw_code=raw_code, ancestor_num=ancestor_num)
                if len(mem.perf_candidates) > 0:
                    mem.ps.solution = mem.perf_candidates[-1][0]
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
        text = prompt_for_generation.prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text
        )
        
        history_text = self._build_history_prompt(mem)
        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if (mem.perf_debug_num >= self.max_perf_debug_num) or mem.pass_exe:
            mem.perf_debug_num = 0
            mem.raw_codes =None
        if len(mem.perf_candidates) > 0 and not mem.raw_codes:
            text += f"\nPrevious attempt implementation:{history_text}"
            text += """\nThere are some Optimized codes(NO.1, NO.2 and so on) to solve the Problem. The Optimized codes are arranged in ascending order based on their performance, where lower latencies and higher efficiencies indicate better performance. According to their performance(latency in ms and efficiency in TFLOPS or GB/s) and the corresponding analysis, you need to generate a new code with better performance. You should maintain code correctness during optimization."""
            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables."    
            for i, cand in enumerate(mem.perf_candidates):
                text += f"\n### Reference {i+1}"
                text += f"\nOptimized code: {cand[0]}"
                text += f"\nOptimized latency(ms): {cand[1]}"
                text += f"\nOriginal efficiency(TFLOPS, GB/s): {cand[2]}"
                text += f"\nStrategy: {cand[3]}"
                if cand[4]:
                    text += f"\nThere are some profiling info of this kernel generated by the tool of rocprof-compute on AMD MI250 gpu, you can reference these info to analyze and generate better kernel.Rocprof-compute profiling result:{cand[4]}"
            if mutation:
                text += "\nGenerate a better strategy completely different from Optimized Implementation. Based on the better strategy generate a better optimization code."
            else:    
                text += "\nAnalyze and compare all optimization strategies based on Optimized Implementation codes and give a better strategy motivated by them. Based on the better strategy generate a better optimization code to get a lower latency."
        else:           
            if not mem.exe_candidate and not mem.call_candidate and not mem.raw_codes:
                text += f"\nHere is an example snippet of code: {mem.oneshot}"
            elif mem.raw_codes:
                one_shot = self.code_retriever.query(mem.raw_codes[0].code)[0]["code"]
                text += f"\nHere is an example snippet of code: {one_shot}"
                text += f"\nPrevious attempt implementations:{history_text}"
        
        if mem.raw_codes :
            for raw_code in mem.raw_codes:
                if not raw_code.pass_perf:
                    text += prompt_for_generation.system_prompt
                    if raw_code.reflections:
                        raw_code.reflections = None
                    try:
                        raw_code.code, raw_code.strategy = self.call_llm_code(prompt=text, temperature=temperature)
                    except:
                        logger.info(f"failed to call LLM for {mem.ps.filename}")
            mem.perf_debug_num +=1
            return
        
        gens_codes: List[tempCode] = []
        for i in range(descendant_num):
            gen_code = tempCode()
            try:
                text += prompt_for_generation.system_prompt
                gen_code.code, gen_code.strategy = self.call_llm_code(prompt=text, temperature=temperature)
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
- latency: measures the total time from kernel launch to completion, reflecting the responsiveness and overhead of executing a single instance of the kernel on the GPU.
"""
        history_text = self._build_history_prompt(mem)
        if mem.raw_codes :
            for raw_code in mem.raw_codes:
                if  raw_code.reflections:
                    continue
                if raw_code.pass_exe:
                    result_txt = f"""
- runnable test: Succeed
- correctness test: Succeed
- latency: {raw_code.latency}
"""                 
                    reflect_txt = prompt_for_reflection.prompt_evolve_strategy_optimize.format(
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

                    reflect_txt = prompt_for_reflection.prompt_evolve_reflect.format(
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

        elif mem.perf_candidates[0][1] > raw_code.latency:
            candidate = [raw_code.code, raw_code.latency, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates[0] = tuple(candidate)
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)
            
    def _build_history_prompt(self, mem):
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
        for i, his in enumerate(mem.history):
            if his.raw_codes :
                for raw_code in mem.raw_codes:
                    if raw_code.pass_perf:
                        test_txt = """
runnable test: Succeed
correctness test: Succeed
latency: {latency}
"""
                        test_txt = test_txt.format(
                            latency=raw_code.latency
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