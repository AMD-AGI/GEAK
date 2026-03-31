# Configuration files

How GEAK loads **YAML** for **`geak`**, where builtin files live, and how **`--config`** resolves paths. For **model / CLI env** quick reference, use **[Quick start](quick_start.md)** §2.

## Main CLI

1. **Base** — **`src/minisweagent/config/geak.yaml`** is always loaded first.
2. **Override** — If you pass **`-c` / `--config`**, that file is **deep-merged** on top. Keys you set in the user file replace or merge into the result.


## What’s in the default config file (**`src/minisweagent/config/geak.yaml`**)


### **`model:`**

| Key | Purpose |
|-----|---------|
| **`model_class`** | Backend short name for **`get_model_class`** (here **`amd_llm`** — AMD LLM gateway). |
| **`model_name`** | Gateway model id (e.g. **`claude-opus-4.5`**, **`claude-sonnet-4.5`**, **`gpt-5`**, **`gpt-5.1`**, **`gpt-5-codex`**). Routed inside **`AmdLlmModel`** to Claude / OpenAI / Gemini clients by name pattern. |
| **`api_key`** | Empty string **`""`** means “read **`AMD_LLM_API_KEY`** or **`LLM_GATEWAY_KEY`** from the environment”; a non-empty value is sent to the gateway instead. |
| **`model_kwargs`** | Passed through to the vendor implementation: **`temperature`**, **`max_tokens`**, plus gateway-specific blocks. **`reasoning.effort`** and **`text.verbosity`** apply to **GPT**-style models on the gateway (see inline comments in the YAML). |

### **`agent:`**

| Key | Purpose |
|-----|---------|
| **`step_limit`** | Step cap for **`DefaultAgent`**. **`0`** means **disabled** (limits apply only when **`0 < step_limit`**). |
| **`cost_limit`** | Cost cap (same class). **`0`** means **disabled** (limits apply only when **`0 < cost_limit`**). |
| **`mode`** | **`confirm`** = interactive confirmation for tool actions; **`yolo`** = auto-run. Parallel workers force **`yolo`** regardless. |


### **`env:`**

| Key | Purpose |
|-----|---------|
| **`env`** | Nested map of **process environment** variables forwarded to the tool runtime / subprocesses (e.g. **`PAGER`**, **`MANPAGER`**, **`LESS`**, **`PIP_PROGRESS_BAR`**, **`TQDM_DISABLE`**) so logs stay non-interactive in automation. |
| **`timeout`** | Default **command timeout** in seconds (here **`3600`**) for environment executions where applicable. |

*(The large **`system_template`** / **`instance_template`** blocks live in **`mini_kernel_strategy_list.yaml`** unless you override them in another **`--config`** file.)*
