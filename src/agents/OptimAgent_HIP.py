from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.reflexion_oneshot_hip import Reflexion_Oneshot_HIP
from utils.utils import clear_code, extract_function_signatures, clear_json
from memories.Memory import MemoryClassMeta
from prompts import prompt_for_reflection_hip
from loguru import logger


class OptimAgent_HIP(Reflexion_Oneshot_HIP):
    def __init__(self, model, dataset, corpus_path, max_perf_debug_num=5):
        super().__init__(model, dataset, corpus_path)
        self.max_perf_debug_num = max_perf_debug_num

    def memory_init(self):
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "call_err_msg", 
                                                             "exe_err_msg",
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot", 
                                                             "perf_candidates",
                                                             "perf_strategy",
                                                             "raw_code",
                                                             "call_candidate",
                                                             "exe_candidate",
                                                             "perf_debug_num",
                                                             "pass_call", 
                                                             "pass_exe",
                                                             "pass_perf"]):
            pass
        
        for ps in self.dataset.problem_states:

            tmp_mem = Memory(ps=ps, 
                             call_err_msg=None,
                             exe_err_msg=None, 
                             reflection=None, 
                             function_signatures=None, 
                             oneshot=ps.test_code,  ##USE hip file
                             perf_candidates=[],
                             perf_strategy=None,
                             raw_code=None,
                             call_candidate=None,
                             exe_candidate=None,
                             perf_debug_num=0,
                             pass_call=False,
                             pass_exe=False,
                             pass_perf=False,
                             )
            self.memories.append(tmp_mem)

    
    def run(self, output_path=None, multi_thread=True, verbose=False, datalen=None, iteration_num=0, temperature=0, ancestor_num=8, start_idx=0, gpu_id=0):
        # import pdb; pdb.set_trace()

        data_len = datalen if datalen else len(self.dataset)
        for iter in range(iteration_num):
            logger.info(f"\n=== Iteration {iter} ===")
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{iter}{extension}"

            if multi_thread:
                thread_num = 3
            
            # generate solution
            logger.info(f"\ngenerate solution")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_solution, mem, temperature): mem for mem in self.memories[start_idx:(start_idx + data_len)]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:(start_idx + data_len)]:
                        self.generate_solution(mem, temperature=temperature)
                        pbar.update(1)
            
            # run scripts
            logger.info(f"\nrun scripts on gpu")
            tmp_dir = "tmp"
            exe_dir = "pass_exe"
            perf_result_dir = "perf_results"
            perf_log_dir = "perf_logs"
            
            for mem in tqdm(self.memories[start_idx:(start_idx + data_len)]):
                try:
                    pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr, perf, efficiency = self.dataset.test_opt_correctness(mem.raw_code[0], mem.ps.filename, tmp_dir, task_name=mem.ps.label, compile_path=mem.ps.build_dir, exe_dir=exe_dir)
                except Exception as e:
                    print(f"failed to test the code for {mem.ps.filename}")
                    mem.call_err_msg = f"failed to test the code due to: {e}"
                    mem.exe_err_msg = f"failed to test the code due to: {e}"
                    continue

                if not pass_call:
                    mem.call_err_msg = call_stdout+call_stderr
                    mem.exe_err_msg = exe_stdout+exe_stderr
                elif pass_call and not pass_exe:
                    mem.pass_call = True
                    mem.exe_err_msg = exe_stdout+exe_stderr
                    mem.call_candidate = mem.raw_code[0]
                    print()
                else:
                    mem.pass_call = True
                    mem.pass_exe = True
                    mem.exe_err_msg = exe_stdout+exe_stderr
                    mem.exe_candidate = mem.raw_code[0]

                if perf is not None and pass_exe:
                    _, efficiency, ms = None, efficiency, perf
                    mem.pass_perf = True
                    mem.raw_code.extend([ms, efficiency])
            
            print("===="*40)
            
            # generate reflections
            logger.info(f"\ngenerate reflections")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_reflexion, mem, temperature): mem for mem in self.memories[start_idx:(start_idx + data_len)]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:(start_idx + data_len)]:
                        self.generate_reflexion(mem, temperature=temperature)
                        pbar.update(1)

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                if not mem.pass_perf:
                    print("no passing_perf in this round")
                    continue

                ### Appending reflection and raw_code to candidates.
                mem.raw_code.append(mem.reflection)
                mem.perf_candidates.append(tuple(mem.raw_code))
                
                def get_sort_key(x):
                    efficiencies = x[2]
                    if isinstance(efficiencies, list):
                        total_speedup = sum(1.0 - e for e in efficiencies)
                    else:
                        total_speedup = 1.0 - efficiencies
                    return -total_speedup

                mem.perf_candidates = sorted(mem.perf_candidates, key=get_sort_key)
                
                mem.perf_candidates = mem.perf_candidates[:ancestor_num]
                
            for mem in self.memories[start_idx:(start_idx + data_len)]:
                if len(mem.perf_candidates) > 0:
                    mem.ps.solution = mem.perf_candidates[0][0]
                elif mem.pass_exe:
                    mem.ps.solution = mem.exe_candidate
                elif mem.pass_call:
                    mem.ps.solution = mem.call_candidate
                else:
                    mem.ps.solution = mem.raw_code[0]
            
            for i in range(len(mem.perf_candidates)):
                print("Candidate {} perf".format(i+1), mem.perf_candidates[i][1])
            if output_path is not None:
                self.dataset.write_file(iter_path)

            os.system(f'rm -rf {exe_dir}')
            os.system(f'rm -rf {perf_result_dir}')
            os.system(f'rm -rf {perf_log_dir}')
    
    def generate_solution(self, mem, temperature=0):

        text = mem.ps.instruction
        pytest = ''

        if "gsplat" in mem.ps.label:
            with open("/home/username/Dataset_hip/gsplat/test.py", "r") as f:
                pytest = f.read()

        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if len(mem.perf_candidates) > 0 and (mem.pass_exe or (not mem.pass_exe and mem.perf_debug_num >= self.max_perf_debug_num)):
            mem.perf_debug_num = 0

            tmp_perf = mem.perf_candidates[0][2]

            if "gemm_multiply_multiply_xdl_fp8_ab_scale" in mem.ps.label:
                text += f"\nHere is an example snippet of baseline code with heuristic: {mem.oneshot}"
                text += """\nThere are some reference codes(NO.1, NO.2 and so on). According to their performance(latency in ms) and the corresponding analysis, you need to optimize the heuristic with better performance. You should maintain code correctness during optimization."""
            elif "gsplat" in mem.ps.label:
                text += f"\nHere is an example snippet of baseline code for RasterizeToPixels3DGSBwd: {mem.oneshot}"
                
                text += "\nThe python unittest script used for validation and latency measurement (this script tests workflow of forward and backward, our goal is to focus only on optimize the backward. The latency reported is directly from the output print of this script): " + pytest
                text += "\nThe unittest script is not subject to change, you should focus only on generate/optimize the implementation of baseline code. "
                
                text += """\nThere are some reference codes(NO.1, NO.2 and so on). According to their performance(latency in s) and the corresponding analysis, you need to continue optimize the code with better performance. You should maintain code correctness during optimization."""
                text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy."

            else:
                text += f"\nHere is an example snippet of baseline code: {mem.oneshot}"
                text += """\nThere are some reference codes(NO.1, NO.2 and so on). According to their performance(latency in ms) and the corresponding analysis, you need to continue optimize the code with better performance. You should maintain code correctness during optimization."""
                text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy."

            for i, cand in enumerate(mem.perf_candidates):
                text += f"\nreference code No.{i}: {cand[0]}"
                if "gemm_multiply_multiply_xdl_fp8_ab_scale" in mem.ps.label:
                    text += f"\nreference code latency(ms) (corresponds to the MNK sizes (problem_shapes) in the code): {cand[1]}"
                elif type(cand[1]) is list:
                    text += f"\nreference code latency(ms) (multiple latency number corresponds to different input or forward/backward in order if implemented): {cand[1]}"
                else:
                    text += f"\nreference code latency(ms): {cand[1]}"
                
                text += f"\nreference code latency ratio to original baseline: {cand[2]}"
                text += f"\nreference code Analysis: {cand[3]}"
            
            #CK gemm
            if "gemm_multiply_multiply_xdl_fp8_ab_scale" in mem.ps.label:
                text += "\nAnalyze and compare all heuristic strategies based on these reference code/perf/analysis and give a better heuristic strategy motivated by them. Generate a better heuristic stategy with if/else based on the ref codes and their latency towards the given problem sizes."
            else:
                text += "\nAnalyze and compare all reference code, based on these reference code/perf/analysis and give a optimized version of code motivated by them."
            
            text += "\nThink before writing the optimization and no more explanation is required after the thinking."
            text += "\nYou should not suggest changes to the name of the function and parameter names, counts, or order."
        else:
            if not mem.raw_code or mem.raw_code[0] == "":
                if "gsplat" in mem.ps.label:
                    text += f"\nHere is an example snippet of baseline code for RasterizeToPixels3DGSBwd: {mem.oneshot}"
                    
                    text += "\nThe python unittest script used for validation and latency measurement (this script tests workflow of forward and backward, our goal is to focus only on optimize the backward. The latency reported is directly from the output print of this script): " + pytest
                    text += "\nThe unittest script is not subject to change, you should focus only on generate/optimize the implementation of baseline code. "
                else:   
                    text += f"\nHere is an example snippet of baseline code: {mem.oneshot}"
            else:
                
                if "gsplat" in mem.ps.label:
                
                    text += f"\nHere is an example snippet of baseline code for RasterizeToPixels3DGSBwd: {mem.oneshot}"
                    
                    text += "\nThe python unittest script used for validation and latency measurement (this script tests workflow of forward and backward, our goal is to focus only on optimize the backward. The latency reported is directly from the output print of this script): " + pytest
                    text += "\nThe unittest script is not subject to change, you should focus only on generate/optimize the implementation of baseline code. "
                else:
                    text += f"\nHere is an example snippet of baseline code: {mem.oneshot}"

                text += f"\nPrevious attempt implementation:{mem.raw_code[0]}"
                
                if not mem.pass_call:
                    text += f"\nTest messages for previous attempt:{mem.call_err_msg}"
                    text += f"\nTest messages for correctness check of previous attempt:{mem.exe_err_msg}"
                
                elif not mem.pass_exe:
                    text += "\nThe previous attempt implementation can be run successfully."
                    text += f"\nTest messages for correctness check of previous attempt:{mem.exe_err_msg}"
                
                if len(mem.perf_candidates) > 0:
                    mem.perf_debug_num += 1
            
            
            if mem.reflection:
                text += f"\nReflection on previous attempt:{mem.reflection}"

        text += "\nOutput your answer in json format, with the format as follows: {\"thought\": \"\", \"code\": \"\"}. Please strictly output in JSON format."
        
        if "gsplat" in mem.ps.label:
            text += "\nGenerate the correct and optimized RasterizeToPixels3DGSBwd code without explanation, which we can run directly in the \"code\" field. "
        else:
            text += "\nGenerate the correct and optimized code without explanation, which we can run directly in the \"code\" field."

        msg = [
            {"role": "user", "content": text},
        ]

        response = self.model.generate(msg, temperature=temperature, max_tokens=30000)

        try:
            mem.raw_code = [clear_json(response)["code"]]
        except:
            print(f"failed to extract code for {mem.ps.filename}")
            fail_dir = "failed_to_extract"
            fail_path = os.path.join(fail_dir, mem.ps.filename)
            os.makedirs(fail_dir, exist_ok=True)

            with open(fail_path, "w") as f:
                f.write(response)

            raw_code = response.split("\"code\":")[1]
            raw_code = raw_code.split("}")[0]
            mem.raw_code = [clear_code(raw_code)]
        
        if mem.raw_code is None:
            print(f"raw code for {mem.ps.filename} is None")
            mem.raw_code = [""]

        mem.pass_call = False
        mem.pass_exe = False
        mem.pass_perf = False

        return
    
    def generate_reflexion(self, mem, temperature):

        if mem.pass_perf:
            if "gemm_multiply_multiply_xdl_fp8_ab_scale" in mem.ps.label:
                reflect_txt = prompt_for_reflection_hip.prompt_ga_hip_gemm.format(
                    problem=mem.ps.instruction,
                    original_code=mem.oneshot,
                    code=mem.raw_code[0],
                    latency=mem.raw_code[1],
                    latency_ratio=mem.raw_code[2],
                    exe_test_result=mem.exe_err_msg
                )
            else:
                reflect_txt = prompt_for_reflection_hip.prompt_ga_hip.format(
                    problem=mem.ps.instruction,
                    original_code=mem.oneshot,
                    code=mem.raw_code[0],
                    latency=mem.raw_code[1],
                    latency_ratio=mem.raw_code[2],
                    exe_test_result=mem.exe_err_msg
                )

        elif mem.pass_call:
            reflect_txt = prompt_for_reflection_hip.prompt_exe_hip.format(
                problem=mem.ps.instruction,
                original_code=mem.oneshot,
                solution=mem.raw_code[0],
                call_test_result="succeed",
                exe_test_result=mem.exe_err_msg
            )
        else:
            reflect_txt = prompt_for_reflection_hip.prompt_hip.format(
                problem=mem.ps.instruction,
                original_code=mem.oneshot,
                solution=mem.raw_code[0],
                test_result=mem.call_err_msg
            )
        
        reflect_msg = [
            {
                "role": "user",
                "content": reflect_txt
            }
        ]
        mem.reflection = self.model.generate(reflect_msg, temperature=temperature)


