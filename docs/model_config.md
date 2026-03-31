# Model configuration

This document explains how GEAK selects and configures the LLM backend.

> If you only want to **run GEAK quickly**, see `docs/quick_start.md` and keep secrets (API keys) in environment variables.

## Model selection precedence

`geak` resolves the model name in this order (first hit wins):

1. CLI `-m` / `--model`
2. YAML `model.model_name`
3. Environment `GEAK_MODEL`
4. Environment `MSWEA_MODEL_NAME`

## Model backend (`model_class`)

YAML `model.model_class` selects the backend. If it is **missing or empty**, `get_model_class` in `src/minisweagent/models/__init__.py` still returns `LitellmModel`.

You can also set it explicitly to `litellm` — that is the registered shortcut for the same class in `_MODEL_CLASS_MAPPING`.

| `model_class` (YAML) | Backend |
|----------------------|---------|
| `litellm` | `LitellmModel` — any `provider/model` string supported by [LiteLLM](https://docs.litellm.ai/) |
| `amd_llm` | `AmdLlmModel` — AMD LLM gateway; `model_name` examples: `claude-opus-4.6`, `claude-sonnet-4.5`, `gpt-5`, `gpt-5-codex`, Gemini-style names with `gemini` |
| `anthropic_model` | Direct Anthropic SDK |

## API keys

Optional global override: `MSWEA_MODEL_API_KEY` is copied into `model_kwargs.api_key` when set.

In Docker-based setup, export the API key before running `scripts/run-docker.sh`.

## Configure via CLI and environment variables

### CLI flags

- `-m` / `--model`: forces `model_name` for this run. Default `model_class` is `amd_llm`.
- `--model-class`: forces `model_class` for this run (`litellm`, `amd_llm`, …).

### Example 1 — AMD LLM gateway

```bash
export AMD_LLM_API_KEY="YOUR_KEY"
# or: export LLM_GATEWAY_KEY="YOUR_KEY"

geak --yolo --model claude-sonnet-4.5 -t "Your task here"
```

### Example 2 — LiteLLM + OpenAI

```bash
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"
geak --model-class litellm --kernel-url /path/to/kernel/file --repo /path/to/kernel/repo
```

### Example 3 — LiteLLM + Anthropic

```bash
export MSWEA_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="YOUR_KEY"
geak --model-class litellm --kernel-url /path/to/kernel/file --repo /path/to/kernel/repo
```

Other LiteLLM providers (Azure, Vertex, …): set the `MSWEA_MODEL_NAME` / `GEAK_MODEL` string and provider env vars per the [LiteLLM docs](https://docs.litellm.ai/), and pass `--model-class litellm` when the merged YAML is not already LiteLLM.

## Configure via config file (`--config`)

You can configure the model in a YAML file and pass it with `--config` (merged over the base strategy file).

### AMD LLM gateway

```yaml
model:
  model_class: amd_llm
  model_name: claude-opus-4.6
  api_key: ""
```

### LiteLLM

```yaml
model:
  model_class: litellm
  model_name: openai/gpt-5
  api_key: ""
  # or set OPENAI_API_KEY / ANTHROPIC_API_KEY / … in the environment instead of api_key
```

### Recommendation: keep secrets out of YAML

Keep secrets in `export ...` and use YAML only for non-secret configuration like `model_name`, `model_kwargs`, and `agent`. Still pass `--config` so the file merges without storing keys in git.

