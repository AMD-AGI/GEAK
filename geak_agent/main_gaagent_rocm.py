# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from geak_agent.agents.GaAgent_mlu import GaAgent
from geak_agent.models.OpenAI import OpenAIModel
from geak_agent.models.Gemini import GeminiModel
from geak_agent.models.Claude import ClaudeModel
from geak_agent.dataloaders.TritonBench import TritonBench
from geak_agent.args_config import load_config
from geak_agent.dataloaders.mlu import MLU
import os

def main():
    args = load_config("configs/MLU_gaagent_config.yaml")
    args.log_root = os.path.abspath(args.output_path).replace(".jsonl", "")
    os.makedirs(args.log_root, exist_ok=True)
    print(args)

    # setup LLM model
    model = ClaudeModel(api_key=args.api_key, model_id=args.model_id)

    # setup dataset
    # dataset = TritonBench(statis_path=args.statis_path, 
    #                       py_folder=args.py_folder, 
    #                       instruction_path=args.instruction_path, 
    #                       py_interpreter=args.py_interpreter, 
    #                       golden_metrics=args.golden_metrics,
    #                       perf_ref_folder=args.perf_ref_folder,
    #                       perf_G_path=args.perf_G_path,
    #                       result_path=args.result_path)
    dataset = MLU(statis_path=args.statis_path, 
                        py_folder=args.py_folder, 
                        instruction_path=args.instruction_path, 
                        py_interpreter=args.py_interpreter,
                        log_root=args.log_root)
    # setup agent
    agent = GaAgent(model=model, dataset=dataset, corpus_path=args.corpus_path, mem_file=args.mem_file, descendant_num=args.descendant_num)

    # run the agent
    agent.run(output_path=args.output_path, 
              multi_thread=args.multi_thread, 
              iteration_num=args.max_iteration, 
              temperature=args.temperature, 
              datalen=args.datalen,
              gpu_id=args.gpu_id,
              start_iter=args.start_iter,
              ancestor_num=args.ancestor_num,
              descendant_num=args.descendant_num,
              descendant_debug=args.descendant_debug,
              target_mlu=args.target_mlu,
              profiling=args.profiling,
              start_idx=args.start_idx)


if __name__ == "__main__":
    main()