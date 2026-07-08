---
myst:
    html_meta:
        "description": "How GEAK loads and merges YAML configuration files. Explains the base geak.yaml, --config override, and the model, agent, and env configuration sections."
        "keywords": "GEAK, configuration, YAML, geak.yaml, --config, model class, agent mode, environment variables"
---

# Configuration files

GEAK uses a layered YAML configuration system. A built-in base file sets sensible defaults, and an optional `--config` file you supply is deep-merged on top. This page explains where the built-in files live, how the merge works, and what each configuration key does. For model and backend selection specifically, see [Model configuration](model-config.md).

## Main CLI

GEAK loads configuration in two steps:

1. Base—`src/minisweagent/config/geak.yaml` is always loaded first.
2. Override—If you pass `-c` / `--config`, that file is deep-merged on top. Keys you set in the user file replace or merge into the result.


## What’s in the default config file (`src/minisweagent/config/geak.yaml`)


### `model:`

| Key | Purpose |
|-----|---------|
| `model_class` | Backend short name for `get_model_class` (here `amd_llm` — AMD LLM gateway). |
| `model_name` | Gateway model id (for example, `claude-opus-4.6`, `claude-sonnet-4.5`, `gpt-5`, `gpt-5.1`, `gpt-5-codex`). Routed inside `AmdLlmModel` to Claude, OpenAI, or Gemini clients by name pattern. |
| `api_key` | Empty string `””` means “read `AMD_LLM_API_KEY` or `LLM_GATEWAY_KEY` from the environment”; a non-empty value is sent to the gateway instead. |
| `model_kwargs` | Passed through to the vendor implementation: `temperature`, `max_tokens`, plus gateway-specific blocks. `reasoning.effort` and `text.verbosity` apply to GPT-style models on the gateway (see inline comments in the YAML). |

### `agent:`

| Key | Purpose |
|-----|---------|
| `step_limit` | Step cap for `DefaultAgent`. `0` means disabled (limits apply only when `0 < step_limit`). |
| `cost_limit` | Cost cap (same class). `0` means disabled (limits apply only when `0 < cost_limit`). |
| `mode` | `confirm` = interactive confirmation for tool actions; `yolo` = auto-run. Parallel workers force `yolo` regardless. |


### `env:`

| Key | Purpose |
|-----|---------|
| `env` | Nested map of process environment variables forwarded to the tool runtime and subprocesses (for example, `PAGER`, `MANPAGER`, `LESS`, `PIP_PROGRESS_BAR`, `TQDM_DISABLE`) so logs stay non-interactive in automation. |
| `timeout` | Default command timeout in seconds (here `3600`) for environment executions where applicable. |

*(The large `system_template` / `instance_template` blocks live in `mini_kernel_strategy_list.yaml` unless you override them in another `--config` file.)*

## Related topics

- [Model configuration](model-config.md)—select a model backend and configure API keys.
- [API reference](../docs/reference/api-reference.md) — full CLI flag reference including `--config` and `--mode`.
- [Install GEAK](../docs/install/install.md) — install GEAK and run the initial configuration.
