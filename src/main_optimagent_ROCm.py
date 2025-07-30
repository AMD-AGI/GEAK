from agents.OptimAgent import OptimAgent
from models.OpenAI import OpenAIModel
from dataloaders.ROCm import ROCm
from args_config import load_config
import os

def main():
    args = load_config("configs/rocm_optimagent_config.yaml")
    args.log_root = os.path.abspath(args.output_path).replace(".jsonl", "")
    os.makedirs(args.log_root, exist_ok=True)
    print(args)

    # setup LLM model
    model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)

    # setup dataset
    dataset = ROCm(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path, 
                          py_interpreter=args.py_interpreter,
                          log_root=args.log_root)

    # setup agent
    agent = OptimAgent(model=model, dataset=dataset, corpus_path=args.corpus_path, mem_file=args.mem_file)

    # run the agent
    agent.run(output_path=args.output_path, 
              multi_thread=args.multi_thread, 
              iteration_num=args.max_iteration, 
              temperature=args.temperature, 
              datalen=args.datalen,
              start_iter=args.start_iter,
              start_idx=args.start_idx,
              ancestor_num=args.ancestor_num)


if __name__ == "__main__":
    main()