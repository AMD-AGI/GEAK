import json
from tqdm import tqdm
from args_config import load_config
from dataloaders.TritonBench import TritonBench


def main():
    args = load_config("configs/parallel_scaling_config.yaml")
    file_list = args.file_list
    output_file = args.output_file
    assert output_file.endswith(".jsonl"), f"expect output file to be a jsonl file, but got {file} instead"

    # setup dataset
    dataset = TritonBench(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path,
                          py_interpreter=args.py_interpreter, 
                          golden_metrics=args.golden_metrics,
                          perf_ref_folder=args.perf_ref_folder,
                          perf_G_path=args.perf_G_path)


    result_dict = {}
    call_num = [0 for _ in range(args.data_len)]
    exe_num = [0 for _ in range(args.data_len)]
    for file in file_list:
        assert file.endswith(".jsonl"), f"expect jsonl file, but got {file} instead"
        with open(file, "r") as f:
            lines = f.readlines()
            assert args.data_len == len(lines), f"expect {args.data_len} entries, but got {len(lines)} entries instead for file {file}"
            with tqdm(total=args.data_len) as pbar:
                for i, line in enumerate(lines):
                    content = json.loads(line)
                    filename = content["filename"]
                    # if filename in result_dict and result_dict[filename]["pass_exe"]:
                    #     pbar.update(1)
                    #     continue
                    
                    tmp_dir = "par_scaling_tmp"
                    exe_dir = "par_scaling_exe"
                    pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = dataset.test_opt_correctness(content["predict"], 
                                                                                                                            filename=filename, 
                                                                                                                            tmp_dir=tmp_dir, 
                                                                                                                            save_scripts=True, 
                                                                                                                            exe_dir=exe_dir)
                    ms = None
                    if pass_call:
                        call_num[i] = 1
                    if pass_exe:
                        exe_num[i] = 1
                        # import ipdb
                        # ipdb.set_trace()
                        result_dir = os.path.join(exe_dir, "result")
                        script_dir = os.path.join(exe_dir, "script")
                        log_dir = os.path.join(exe_dir, "log")
                        path_gen = os.path.join(result_dir, filename[:-3] + ".json")
                        path_ref = os.path.join(dataset.perf_ref_folder, filename[:-3] + ".json")

                        dataset.write_perf_file(input_folder_path=exe_dir, results_path=result_dir, tmp_dir=script_dir)
                        dataset.run_perf_scripts(script_dir=script_dir, log_dir=log_dir)
                        try:
                            _, _, ms = dataset.calculate(path_gen=path_gen, path_ref=path_ref)
                        except:
                            pass

                    if not filename in result_dict:
                        result_dict[filename] = {
                            "predict": content["predict"],
                            "pass_call": pass_call,
                            "pass_exe": pass_exe,
                            "instruction": content["instruction"],
                            "label": content["label"] if "label" in content else None,
                            "latency": ms
                        }
                    else:
                        if (not result_dict[filename]["pass_exe"]) and pass_exe:
                            result_dict[filename]["predict"] = content["predict"]
                            result_dict[filename]["pass_call"] = pass_call
                            result_dict[filename]["pass_exe"] = pass_exe
                            result_dict[filename]["latency"] = ms
                        elif (not result_dict[filename]["pass_call"]) and pass_call:
                            result_dict[filename]["predict"] = content["predict"]
                            result_dict[filename]["pass_call"] = pass_call
                            result_dict[filename]["pass_exe"] = pass_exe
                            result_dict[filename]["latency"] = ms
                        elif result_dict[filename]["pass_exe"] and ms:
                            if result_dict[filename]["latency"] is None or result_dict[filename]["latency"] > ms:
                                result_dict[filename]["predict"] = content["predict"]
                                result_dict[filename]["latency"] = ms
                    
                    os.system(f'rm -rf {tmp_dir}')
                    os.system(f'rm -rf {exe_dir}')
                    pbar.update(1)
    
    call_acc = sum(call_num) / 184.0
    exe_acc = sum(exe_num) / 184.0

    print(f"call acc: {call_acc}")
    print(f"exe acc: {exe_acc}")
    
    with open(output_file, "w") as f:
        for filename, result in result_dict.items():
            output = {
                "instruction": result["instruction"],
                "label": result["label"],
                "filename": filename,
                "predict": result["predict"]
            }
            f.write(json.dumps(output) + "\n")
    

if __name__ == "__main__":
    main()