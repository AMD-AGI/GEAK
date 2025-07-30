from agents.OptimAgent import OptimAgent
from models.OpenAI import OpenAIModel
from dataloaders.TritonBench import TritonBench
from args_config import load_config


def main():
    args = load_config("configs/tritonbench_optimagent_config.yaml")

    # setup LLM model
    model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)

    # setup dataset
    dataset = TritonBench(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path, 
                          py_interpreter=args.py_interpreter, 
                          golden_metrics=args.golden_metrics,
                          perf_ref_folder=args.perf_ref_folder,
                          perf_G_path=args.perf_G_path,
                          result_path=args.result_path,
                          target_kernels=args.target_kernels)

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