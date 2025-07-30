import os
from utils import get_time, code_call_exec_success_allclose, extract_code_from_llm_output, get_fname_difficulty_from_label, passk
import json
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser(description="Check correctness of the code.")
parser.add_argument('--folder_or_file', '-f', type=str, required=True, help='Folder to check')
parser.add_argument('--outfile', '-o', type=str, required=True, help='Output file to save results')

parser.add_argument('--file_pat', '-p', type=str, default="*", help='Folder to check')
parser.add_argument('--k_vals', '-k', type=str, default="1,2,3,5,10,15", help='Folder to check')

parser.add_argument('--debug', '-d', type=int, default=0, help='Folder to check')
args = parser.parse_args()

## check if HIP_VISIBLE_DEVICES environment variable is set
assert 'HIP_VISIBLE_DEVICES' in os.environ, "HIP_VISIBLE_DEVICES environment variable is not set. Please set it to the GPU you want to use."

FAILED_FILES  = [] #["attention_fwd_triton1.py", "attention_llama.py", "block_sparse_attn.py", "chunk_bwd_dqkg.py", "context_attn_fwd.py", "chunk_gated_attention.py", "dequantize_matmul.py", "chunk_gla_simple.py", "decay_cumsum.py", "flash_decode2_llama.py", "index_select_bwd.py", "fused_layernorm_triton.py", "int4_matmul.py", "matmul_dequant_int4.py", "layer_norm_welfold.py", "matmul_dequantize_int4.py", "quantize_copy_kv.py", "fused_recurrent_delta.py", "parallel_attention.py", "softmax_flaggems.py", "mixed_sparse_attention.py", "quantize_kv_copy.py", "rotary_emb_nopad.py", "cross_entropy2.py", "max_reduction.py","chunk_delta_fwd.py","chunk_retention_ops.py","fp4_to_bf16_conversion.py","layer_norm_fwd.py","layer_norm_ops.py","int8_dequant_matmul.py","isfinite_kernel.py","logsumexp_fwd.py","llama_ff_triton.py","ksoftmax_triton.py","chunk_gla_fwd.py","pow_scalar_tensor.py","rbe_triton_transform.py","quantize_global.py","rms_matmul_rbe.py","quant_transpose_kernel.py","token_attn_mistral.py","triton_linear_activation.py","token_attn_llama2.py","masked_select.py","rowwise_quantization_triton.py","token_attn_reduceV.py","rmsnorm_triton.py","rms_rbe_matmul.py","softmax_reducev.py","spinning_lock_reduction.py"]

is_folder = os.path.isdir(args.folder_or_file)

if is_folder:
    files = glob(os.path.join(args.folder_or_file, f'{args.file_pat}.json'), recursive=True)
    assert len(files) > 0, f"No files found in {args.folder_or_file} with pattern {args.file_pat}.json"
else:
    files = [args.folder_or_file]

print(files)

data_across_passes = []
total_passes = len(files)
pass_num = -1
for file in tqdm(files, desc="Testing a folder", unit="file"):
    if args.debug > 0:
        if pass_num > 1:
            break
    if file.endswith(".jsonl"):
        tmp_folder = os.path.join(file.replace(".jsonl",""), 'tmp')
        out_file = os.path.join(file.replace(".jsonl",""), args.outfile)
    elif file.endswith(".json"):
        tmp_folder = os.path.join(file.replace(".json",""), 'tmp')
        out_file = os.path.join(file.replace(".json",""), args.outfile)
    os.makedirs(tmp_folder, exist_ok=True)
    logs = []
    call_acc, exec_acc = 0, 0
    eval_data_for_file = []
    pass_num += 1
    with open(file, 'r') as f:
        if file.endswith(".json"):
            data = json.load(f)
        elif file.endswith(".jsonl"):
            data = [json.loads(line) for line in f.readlines()]
        num_files = 0
        for item in tqdm(data, desc="Testing a file", unit="item"):
            if args.debug > 0:
                if num_files >2:
                    break
                num_files += 1
            response = item['predict']
            code = extract_code_from_llm_output(response)
            fname, difficulty = get_fname_difficulty_from_label(item['label'])
            if fname in FAILED_FILES:
                print(f"Skipping {fname} as it is known to fail.")
                continue
            assert fname is not None, f"File name is None for {item['label']}"
            assert difficulty is not None, f"Difficulty is None for {item['label']}"
            assert code is not None, f"Code is None for {item['label']}"
            call_status, exec_status, stdout, stderr = code_call_exec_success_allclose(code, fname, tmp_folder, atol=1e-2, rtol=1e-1)
            eval_data = {
                'pass_num': pass_num,
                'file_name': fname,
                'call_status': 1 if "True" in str(call_status) else 0,
                'exec_status': 1 if "True" in str(exec_status) else 0,
                'stdout': stdout,
                'stderr': stderr,
                'difficulty': int(difficulty)
            }
            eval_data_for_file.append(eval_data)
            call_acc += 1 if call_status else 0 
            exec_acc += 1 if exec_status else 0 
            log = f"{get_time()} => File: {fname}, Call Status: {call_status}, Exec Status: {exec_status}, difficulty: {difficulty}, stderr: {stderr}"
            logs.append(log)
            print(log.split("stderr")[0])
            with open(out_file, 'w') as out_f:
                for _log in logs:
                    out_f.write(_log + '\n')
        call_acc /= len(data)
        exec_acc /= len(data)
        with open(out_file, 'a') as out_f:
            _log = f"{get_time()} => File: {file}, Call Accuracy: {call_acc}, Exec Accuracy: {exec_acc}"
            out_f.write(_log + '\n')
    data_across_passes += eval_data_for_file

# Save the data across passes to a file
with open(args.outfile.replace(".json", "_all_passes.json"), 'w') as out_f:
    json.dump(data_across_passes, out_f, indent=4)
# Save the data across passes to a CSV file
df = pd.DataFrame(data_across_passes)

df.to_csv(args.outfile.replace(".json", "_all_passes.csv"), index=False)
# Save the data across passes to a pickle file
df.to_pickle(args.outfile.replace(".json", "_all_passes.pkl"))
## For each unique value in file_name column, calculate sum(call_status) and sum(exec_status) columns
# df = df.explode('file_name')
# df = df.explode('call_status')
# df = df.explode('exec_status')
df = df.groupby('file_name').agg({'call_status': 'sum', 'exec_status': 'sum', 'difficulty': 'first'}).reset_index()
df['call_status'] = df['call_status']
df['exec_status'] = df['exec_status']

## now return a dictionary with file_name as key and call_status and exec_status as values
df = df.set_index('file_name').T.to_dict()
df = {k: {'call_status': v['call_status'], 'exec_status': v['exec_status'], 'difficulty': v['difficulty']} for k, v in df.items()}

call_acc = 0
exec_acc = 0

for k_val in [int(_k) for _k in args.k_vals.split(",")]:
    for k, v in df.items():
        _call_pass = passk(total_passes, v['call_status'], k_val)
        _exec_pass = passk(total_passes, v['exec_status'], k_val)
        call_acc += _call_pass
        exec_acc += _exec_pass
    call_acc /= len(df)
    exec_acc /= len(df)
    print(f"Call Accuracy for pass@{k_val}: {100* call_acc}")
    print(f"Exec Accuracy for pass@{k_val}: {100* exec_acc}")
    with open(args.outfile, 'a') as out_f:
        out_f.write(f"Call Accuracy for k={k_val}: {100 * call_acc}\n")
        out_f.write(f"Exec Accuracy for k={k_val}: {100 * exec_acc}\n")
