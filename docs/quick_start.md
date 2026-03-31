# Quick start

Minimal steps to install GEAK and run the **`geak`** CLI against a kernel or repository.

## Prerequisites

- **Python** 3.10+
- **Git** (parallel runs use worktrees)
- **GPU** and the stack your kernels use — e.g. **Triton**, **PyTorch**, **CUDA**, or compiled **HIP**.
- **AMD Instinct / Radeon (ROCm):** install a normal **ROCm** user-space environment so tools like **`rocminfo`** / **`rocm-smi`** work when the agent inspects hardware. For **HIP C++** you also need **`hipcc`** (and friends). **`HIP_VISIBLE_DEVICES`** is often set by the scheduler or your shell when pinning a card.

## 1. Install

From the repository root:

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK

# Docker-based
AMD_LLM_API_KEY=<YOUR_KEY> bash scripts/run-docker.sh
# (or)
# Local
pip install -e .
```

## 2. Configure the model
In the case of docker-based setup, export the API key before running scripts/run-docker.sh.

`geak` resolves the model name in this order (first hit wins): CLI **`-m` / `--model`**, then YAML **`model.model_name`**, then env **`GEAK_MODEL`**, then **`MSWEA_MODEL_NAME`**.

YAML **`model.model_class`** selects the backend. If it is **missing or empty**, **`get_model_class`** in **`src/minisweagent/models/__init__.py`** still returns **`LitellmModel`**. You can also set it **explicitly** to **`litellm`** — that is the registered shortcut for the same class in **`_MODEL_CLASS_MAPPING`**.

| `model_class` (YAML) | Backend |
|----------------------|---------|
| **`litellm`**  | **`LitellmModel`** — any `provider/model` string supported by [LiteLLM](https://docs.litellm.ai/) |
| **`amd_llm`** | **`AmdLlmModel`** — AMD LLM gateway; **`model_name`** examples: `claude-opus-4.5`, `claude-sonnet-4.5`, `gpt-5`, `gpt-5-codex`, Gemini-style names with `gemini` |
| **`anthropic_model`** | Direct Anthropic SDK |

Optional global override: **`MSWEA_MODEL_API_KEY`** is copied into **`model_kwargs.api_key`** when set.

You can configure the model in two ways: **CLI** (flags and environment variables, applied after YAML is loaded) or **config** (a YAML file passed with **`--config`**, merged over the base strategy file). The next two subsections follow that split.

### CLI and environment variables


**CLI flags**

- **`-m` / `--model`** — forces **`model_name`** for this run. Default **`model_class`** is **`amd_llm`**.
- **`--model-class`** — forces **`model_class`** for this run (**`litellm`**, **`amd_llm`**, …).

**Example 1 — AMD LLM gateway**


```bash
export AMD_LLM_API_KEY="YOUR_KEY"
# or: export LLM_GATEWAY_KEY="YOUR_KEY"

geak --yolo --model claude-sonnet-4.5 -t "Your task here"
```

**Example 2 — LiteLLM + OpenAI**


```bash
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"
geak --model-class litellm --kernel-path /path/to/kernel/file --repo /path/to/kernel/repo
```

**Example 3 — LiteLLM + Anthropic**

```bash
export MSWEA_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="YOUR_KEY"
geak --model-class litellm --kernel-path /path/to/kernel/file --repo /path/to/kernel/repo
```

Other LiteLLM providers (**Azure**, **Vertex**, …): set the **`MSWEA_MODEL_NAME`** / **`GEAK_MODEL`** string and provider env vars per [LiteLLM](https://docs.litellm.ai/) and pass **`--model-class litellm`** when the merged YAML is not already LiteLLM.

### Config file (`--config`)


**AMD LLM gateway**

```yaml
model:
  model_class: amd_llm
  model_name: claude-opus-4.5
  api_key: ""
```

**LiteLLM**

```yaml
model:
  model_class: litellm
  model_name: openai/gpt-5
  api_key: ""
  # or set OPENAI_API_KEY / ANTHROPIC_API_KEY / … in the environment instead of api_key
```

**Mixing:** keep secrets in **`export …`** and YAML only for **`model_name`**, **`model_kwargs`**, **`agent:`**, etc.; still pass **`--config`** so the file merges without storing keys in git.


## 3. Run the agent

### Typical kernel optimization (natural language input)

```bash
geak -t "Optimize the kernel from /path/to/aiter, specifically aiter/ops/triton/topk.py. Use the harness at /path/to/test_topk_harness.py. Use four GPUs with IDs 0-3 simultaneously."
```

### Typical kernel optimization (single agent)

```bash
geak --kernel-path /path/to/kernel/file \
  --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel" \
```

### Parallel agents

Pass **`--gpu-ids`** as a comma-separated list of device indices (**`0,1,2,3`**). **Each parallel agent is bound to one GPU:** agent **`i`** uses **`gpu_ids[i]`** (0-based). For full isolation, set **`--num-parallel`** to the **same count** as the IDs you list; if you supply fewer IDs than agents, some runs share or fall back without per-agent GPU pinning (the CLI prints a warning).

```bash
geak --num-parallel 4 \
  --repo /path/to/kernel/repo \
  --kernel-path /path/to/kernel/file \
  --task "Optimize block_reduce. Metric: Extract Bandwidth in GB/s (higher is better)" \
  --gpu-ids 0,1,2,3 
```


### CLI reference

Options match the Typer **`Option`** definitions in **`main`** (same names in **`geak`** / **`mini`**).

| Option | Meaning |
|--------|---------|
| **`-m`**, **`--model`** | Model name. |
| **`--model-class`** | e.g. **`litellm`**, **`amd_llm`**. |
| **`-t`**, **`--task`** | Task string. If it equals an existing file path, **`geak`** reads that file as the task body. |
| **`-y`**, **`--yolo`** | Non-interactive / auto-confirm tool execution (sets **`agent.mode`** to **`yolo`**). **Parallel runs** already force **`yolo`** on each worker; this flag mainly affects single-agent **`geak`**. |
| **`-l`**, **`--cost-limit`** | Agent cost limit (use **`0`** to disable). |
| **`-c`**, **`--config`** | path to the config file. It **overrides** the default config file `geak.yaml`. |
| **`-o`**, **`--output`** | Trajectory **file** or output **directory**. Default is `./optimization_logs/kernel_name_timestamp` |
| **`--exit-immediately`** | Sets **`agent.confirm_exit`** to **`False`** in config. |
| **`--repo`** | Repository root for kernel. Even if the kernel code is in a single file, it needs to be put into a repository. |
| **`--kernel-url`**, **`--kernel-path`** | Kernel **source file** path or URL. **Required** unless **`kernel_target`** is supplied another way (e.g. parsed from **`--task "kernel url is xxx"`**). **URLs** are resolved by **`run/preprocess/resolve_kernel_url.py`** (clone/checkout under run output). |
| **`--num-parallel`** | Number of parallel agent runs. |
| **`--gpu-ids`** | Comma-separated GPU device indices. |
| **`--test-command`**, **`--test_command`** | test command used to test the correctness and performance of the kernel. |

## 4. Outputs

Default artifact root: **`optimization_logs/`**, with a per-run directory like **`optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`**.

Parallel layout example:

```text
optimization_logs/<kernel>_<timestamp>/
├── parallel_0/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   └── agent_0.log
├── parallel_1/
│   └── ...
├── best_results.json
└── select_agent.log
```