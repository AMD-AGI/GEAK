---
myst:
    html_meta:
        "description": "Install GEAK and run the geak CLI against a GPU kernel or repository. Covers Docker, local pip install, and model configuration for AMD LLM and LiteLLM backends."
        "keywords": "GEAK, install, ROCm, GPU kernel optimization, pip, Docker, LiteLLM, AMD LLM gateway"
---

# Install GEAK

Install GEAK and run the `geak` CLI against a GPU kernel or repository. This topic covers Docker and local pip installation, along with model configuration for AMD LLM and LiteLLM backends.

## Prerequisites

Before installing GEAK, ensure the following are installed on your system:

- [**ROCm** 7.2 or later](../compatibility.md#rocm-stack)
- **Python** 3.10 or later
- **Git** (parallel runs use worktrees)
- **An AMD GPU** and the stack your kernels use — for example, Triton, PyTorch, CUDA, or compiled HIP.
- **AMD Instinct or Radeon software stack (ROCm)**: Install a normal ROCm user-space environment so tools like `rocminfo` and `rocm-smi` work when the agent inspects hardware. For HIP C++, you also need `hipcc`. `HIP_VISIBLE_DEVICES` is often set by the scheduler or your shell when pinning a card. See [Compatibility Matrix](../compatibility.md) and [ROCm stack](../compatibility.md#rocm-stack) for more information.

## Install

Clone the repository and install using pip or Docker. Run these commands from the repository root:

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK

# Docker-based
AMD_LLM_API_KEY=<YOUR_KEY> bash scripts/run-docker.sh
# (or)
# Local
pip install -e .
# Or install everything at once:
pip install -e '.[full]'
```

```{note}
`fastmcp` and `mcp[cli]` are included in the core dependencies and installed automatically with `pip install -e .`. GEAK launches the shipped MCP servers from `mcp_tools/` directly from the repository, so there's no separate `.[mcp]` extra to install.
```

## Configure the model

For Docker-based setup, export the API key before running `scripts/run-docker.sh`.

`geak` resolves the model name in this order (first hit wins): 

1. CLI `-m` / `--model` 
2. YAML `model.model_name`
3. env `GEAK_MODEL` 
4. `MSWEA_MODEL_NAME`.

YAML `model.model_class` selects the backend. If it's missing or empty, `get_model_class` in `src/minisweagent/models/__init__.py` returns `LitellmModel`. You can also set it explicitly to `litellm`, which is the registered alias for the `LitellmModel` class in `_MODEL_CLASS_MAPPING`.

| `model_class` (YAML) | Backend |
|----------------------|---------|
| `litellm` | `LitellmModel`—any `provider/model` string supported by [LiteLLM](https://docs.litellm.ai/) |
| `amd_llm` | `AmdLlmModel`—AMD LLM gateway; `model_name` examples: `claude-opus-4.6`, `claude-sonnet-4.5`, `gpt-5`, `gpt-5-codex`, Gemini-style names starting with `gemini` |
| `anthropic_model` | Direct Anthropic SDK |

`MSWEA_MODEL_API_KEY` is an optional global override: when set, it overrides `model_kwargs.api_key`.

You can configure the model in two ways: via CLI flags and environment variables (applied after YAML is loaded), or via a YAML file passed with `--config` (merged over the base strategy file). The following two sections cover each approach.

### CLI and environment variables

Use CLI flags or environment variables to select the model for a single run, without modifying any config file.

CLI flags:

- `-m` / `--model` — forces `model_name` for this run. Default `model_class` is `amd_llm`.
- `--model-class` — forces `model_class` for this run (`litellm`, `amd_llm`, …).

**Example 1 — AMD LLM gateway:**

```bash
export AMD_LLM_API_KEY="YOUR_KEY"
# or: export LLM_GATEWAY_KEY="YOUR_KEY"

geak --yolo --model claude-sonnet-4.5 -t "Your task here"
```

**Example 2 — LiteLLM + OpenAI:**

```bash
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"
geak --model-class litellm --kernel-url /path/to/kernel/file --repo /path/to/kernel/repo
```

**Example 3 — LiteLLM + Anthropic:**

```bash
export MSWEA_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="YOUR_KEY"
geak --model-class litellm --kernel-url /path/to/kernel/file --repo /path/to/kernel/repo
```

For other LiteLLM providers (Azure, Vertex, and so on): Set the `MSWEA_MODEL_NAME` or `GEAK_MODEL` string and provider environment variables per [LiteLLM](https://docs.litellm.ai/) and pass `--model-class litellm` when the merged YAML doesn't already specify LiteLLM.

### Config file (`--config`)

Pass a YAML file with `--config` to set model options persistently across runs. The file is deep-merged over the base strategy file, so you only need to include keys you want to override.

AMD LLM gateway:

```yaml
model:
  model_class: amd_llm
  model_name: claude-opus-4.6
  api_key: ""
```

LiteLLM:

```yaml
model:
  model_class: litellm
  model_name: openai/gpt-5
  api_key: ""
  # or set OPENAI_API_KEY / ANTHROPIC_API_KEY / … in the environment instead of api_key
```

Keep secrets in exported variables and YAML only for `model_name`, `model_kwargs`, `agent`, and so on. Always pass `--config` so the file merges without storing keys in git.

## Related topics

- [GEAK compatibility matrix](../compatibility.md) — verify your GPU, ROCm version, and Python version before installing.
- [Run the agent](../how-to/run-agent.md) — invoke `geak` from the command line after installation.



