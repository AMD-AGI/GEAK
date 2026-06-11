# Subagents

This directory holds **subagent definitions** that GEAK discovers at runtime. Each subagent is a self-contained agent that the main agent (homoagent) can delegate specific tasks to.

## Where to put a subagent

1. Create a **new folder** under this directory:

   `GEAK/subagents/<your-subagent-folder>/`

2. Add a file named **`SUBAGENT.yaml`** inside that folder (exact name, case-sensitive).

GEAK scans **immediate subdirectories** of `subagents/` and only picks up folders that contain `SUBAGENT.yaml`. Files at the top level of `subagents/` (not inside a folder) are ignored for discovery.

## SUBAGENT.yaml format

A single `SUBAGENT.yaml` file contains **both** the subagent metadata and all agent/model/env configuration — no separate config file needed.

```yaml
# ── Metadata (required) ──────────────────────────────────────────────
name: my-subagent              # Unique identifier (lowercase, hyphens)
description: >-                # When to use this subagent (shown to the LLM)
  One line explaining the subagent's purpose.

execution_mode: inprocess      # "inprocess" (sync, shares model/env) or "subprocess" (async, own config)

# For subprocess mode: entry script
entry_script: scripts/my_script.sh   # Relative to GEAK root

# Parameters the subagent accepts (used to generate tool schema)
parameters:
  - name: task
    type: string
    description: "Task description for the subagent"
    required: true

# ── Agent configuration (embedded, same keys as mini_*.yaml) ─────────
agent:
  system_template: |
    You are a helpful assistant.
  instance_template: |
    Your task: {{task}}
  step_limit: 0    # 0 = use default
  cost_limit: 0.0

model:
  model_class: amd_llm
  model_name: claude-opus-4.5
  api_key: null
  model_kwargs:
    temperature: 0.0
    max_tokens: 16000

env:
  env:
    PAGER: cat
  timeout: 3600

tools:
  profiling: false
  rag: false
```

### Metadata fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Stable identifier, must be unique across all subagents |
| `description` | Yes | Written for the LLM -- explains *when* to use this subagent |
| `execution_mode` | Yes | `inprocess` (sync, shares parent model/env) or `subprocess` (async, independent process) |
| `entry_script` | No | Script to run in subprocess mode (relative to GEAK root). Required for `subprocess` mode. |
| `parameters` | No | List of parameters the subagent accepts. Each has `name`, `type`, `description`, `required`. |

### Embedded configuration sections

| Section | Description |
|---------|-------------|
| `agent` | Agent config: `system_template`, `instance_template`, `step_limit`, `cost_limit`, `mode`, etc. |
| `model` | Model config: `model_class`, `model_name`, `api_key`, `model_kwargs`. |
| `env` | Environment config: env vars, timeout, cwd. |
| `tools` | Tool toggles: `profiling`, `rag`, etc. |

## Execution modes

### `inprocess` (default)

The subagent runs inside the same process as the parent agent. It shares the parent's model and environment but gets its own step/cost budget and system prompt from the embedded `agent` section.

Best for: focused sub-tasks like algorithm rewrites, cross-file edits, code analysis.

### `subprocess`

The subagent runs as a separate process via `entry_script`. It has its own model configuration from the embedded `model` section. Communication is via stdout/stderr and exit code.

Best for: heavy, long-running tasks like repository analysis (reverse-knowledge), batch processing, or tasks that need a different model.

## How the agent uses subagents

1. **Discovery**: At startup, GEAK reads each `subagents/<folder>/SUBAGENT.yaml` and registers the subagent by `name`.
2. **Listing**: Subagent `name` and `description` are advertised in the system prompt inside `<available_subagents>`.
3. **Invocation**: The LLM calls the `sub_agent` tool with `agent_name` set to a registered subagent's name. The framework handles execution mode selection, parameter passing, and result collection.
4. **Ad-hoc**: The LLM can also call `sub_agent` without `agent_name` for one-off child agents with a free-form task description.

## Programmatic creation

The top agent can create custom subagents at runtime via `SubAgentRegistry`:

```python
registry.create_subagent(
    name="my-custom-agent",
    description="Analyze kernel performance",
    agent_config={"system_template": "You are a kernel analyst..."},
    model_config={"model_class": "amd_llm", "model_name": "claude-opus-4.5"},
    persist=True,  # writes subagents/my-custom-agent/SUBAGENT.yaml
)
```

Or register from a dictionary (in-memory only, no disk write):

```python
registry.register_from_dict({
    "name": "ephemeral-agent",
    "description": "One-off task agent",
    "execution_mode": "inprocess",
    "agent": {"system_template": "..."},
})
```

## Checklist for a new subagent

| Step | Action |
|------|--------|
| 1 | Create `GEAK/subagents/<folder>/` |
| 2 | Add `SUBAGENT.yaml` with metadata + embedded config |
| 3 | Ensure `name` is unique across all subagents |
| 4 | For subprocess mode: create and test the entry script |
