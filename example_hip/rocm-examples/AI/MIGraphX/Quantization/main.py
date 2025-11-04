import os, re, time, json
from tqdm import tqdm
import openai
import subprocess  

#########################
# 1. 公共工具
#########################
SYSTEM_MSG = (
    "You are an expert HIP kernel engineer. "
    "Return ONLY compilable HIP/C++/HIP code. "
    "NO markdown fences, NO commentary."
)

def build_messages(user_inst: str):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_inst}
    ]

def clear_code(raw: str) -> str:
    """提取 ```code``` 中的第一段；若没有 fence，则原样返回"""
    m = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.S)
    return (m.group(1) if m else raw).strip()

#########################
# 2. 主流程
#########################
def main(set_name: str, suffix: str):
    
    url = "https://llm-api.amd.com"
    headers = {"REMOVED_SECRET}
    client = openai.AzureOpenAI(
        api_key="dummy",
        api_version="2024-06-01",
        azure_endpoint=url,
        default_headers=headers,
    )
    model_id = "GPT4o"

    # -------- 读指令集 --------
    inst_path = f"/home/fangyuan/codeagent/gpu-kernel-agent-main/{set_name}.json"
    instructions = json.load(open(inst_path, encoding="utf-8"))

    # -------- 读取已有输出 --------
    output_file = f"{set_name}_output_{suffix}.jsonl"
    outputs_tmp = {}
    if os.path.exists(output_file):
        for line in open(output_file, encoding="utf-8"):
            obj = json.loads(line)
            outputs_tmp[obj["instruction"]] = (obj["predict"], obj["label"])

    # -------- 循环生成 --------
    for item in tqdm(instructions):
        orig_inst = item["instruction"]       # 原始指令
        if orig_inst in outputs_tmp:          # 已生成过
            continue

        messages = build_messages(orig_inst)

        # --- 调 LLM，带指数退避 ---
        response_txt = None
        for retry in range(6):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0,
                    max_tokens=1200,
                )
                response_txt = resp.choices[0].message.content
                break
            except Exception as e:
                print(f"[retry {retry}] error: {e}")
                time.sleep(2 ** retry)
        if response_txt is None:
            print(f"✘ 生成失败：{orig_inst[:60]}...")
            continue

        outputs_tmp[orig_inst] = (
            clear_code(response_txt),
            item.get("output", ""),
        )

    # -------- 写结果 --------
    with open(output_file, "w", encoding="utf-8") as f:
        for inst, (pred, label) in outputs_tmp.items():
            f.write(json.dumps({"instruction": inst,
                                "predict": pred,
                                "label": label},ensure_ascii=False) + "\n")

    print(f"✓ 共写入 {len(outputs_tmp)} 条")
    return len(outputs_tmp)


def process_predict_code(predict_code: str) -> str:
    """处理predict部分的代码，去除多余的空行和缩进"""
    lines = predict_code.split('\n')
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:  # 跳过空行
            processed_lines.append(stripped)
    return '\n'.join(processed_lines)

def update_main_hip(jsonl_path: str, target_path: str) -> bool:
    """从JSONL文件中提取predict代码并更新main.hip文件"""
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                predict_code = data['predict']
                processed_code = process_predict_code(predict_code)
                
                # 写入目标文件
                with open(target_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(processed_code)
                print(f"✓ 已更新文件: {target_path}")
                return True  # 返回成功状态
    except Exception as e:
        print(f"更新文件失败: {e}")
        return False  # 返回失败状态
    return False

def execute_build_and_run(app_dir="bitonic_sort"):
    """执行构建和运行命令"""
    try:
        # 切换到目标目录
        os.chdir("/home/fangyuan/codeagent/rocm-examples/Applications")
        
        # 执行构建命令
        build_cmd = f"python3 build.py --app {app_dir}"
        print(f"执行命令: {build_cmd}")
        subprocess.run(build_cmd, shell=True, check=True)
        
        # 等待3秒确保构建完成
        time.sleep(3)
        
        # 执行运行命令
        run_cmd = f"python3 run.py --app {app_dir}"
        print(f"执行命令: {run_cmd}")
        subprocess.run(run_cmd, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 原来的主函数调用
    main(set_name="HipBench_alpac_v1", suffix="20250507")
    
    # 新增的代码替换功能
    #jsonl_file = "/home/fangyuan/codeagent/gpu-kernel-agent-main/HipBench_alpac_v1_output_20250427.jsonl"
    #target_file = "/home/fangyuan/codeagent/rocm-examples/Applications/bitonic_sort/main.hip"
    #update_success = update_main_hip(jsonl_file, target_file)
    
    # 只有在成功更新文件后才执行构建和运行
    #if update_success:
     #   execute_build_and_run()