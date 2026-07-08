# System & instance prompts

How GEAK assembles the prompts it sends to the model: where prompt files live, how YAML configs merge, and which Jinja variables are available.

## Where prompts live

Prompt templates are stored as YAML alongside the agent and pipeline-worker modules under `src/minisweagent/`. Each agent loads a base prompt set, which a run-specific `--config` can override.

## Merge order

Prompt and model configuration merge in the same precedence order as the rest of the GEAK config: built-in defaults are overlaid by the resolved `--config` YAML, then by any environment-driven overrides. See **[Configuration files](../configuration.md)** for the full resolution rules.

## Jinja variables

Prompt templates are rendered with Jinja. Template variables are supplied by the model's `get_template_vars()` (for example `n_model_calls` and `model_cost`) together with the per-task context the agent passes in.

## Related

- **[MCP and native tools](mcp-tools.md)** — how tools are exposed to the model.
- **[Configuration files](../configuration.md)** — YAML merge order and `--config` resolution.
