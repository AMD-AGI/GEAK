# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from agents.GaAgent_HIP import GaAgent_HIP
from models.OpenAI import OpenAIModel
from models.Claude import ClaudeModel
from models.Gemini import GeminiModel
from dataloaders.HIPBench import HIPBench 
from args_config import load_config


def main():
    args = load_config("configs/hipbench_gaagent_config.yaml")

    # setup LLM model
    model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)
    # model = ClaudeModel(api_key=args.api_key, model_id=args.model_id)
    # model = GeminiModel(api_key=args.api_key, model_id=args.model_id)

    dataset = HIPBench(instruction_path=args.instruction_path, 
                        result_path=args.result_path)

    # setup agent
    agent = GaAgent_HIP(model=model, dataset=dataset, corpus_path=args.corpus_path)

    # run the agent
    agent.run(output_path=args.output_path, 
                multi_thread=args.multi_thread, 
                iteration_num=args.max_iteration, 
                temperature=args.temperature, 
                datalen=None,
                ancestor_num=args.ancestor_num,
                descendant_num=args.descendant_num)


if __name__ == "__main__":
    main()