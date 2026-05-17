# Preprocess subagents (v3)

This directory holds the **v3 preprocess subagent definitions** consumed by
`minisweagent.run.preprocess_v3.registry.SubagentRegistry`.

> **Status (PR 1 — foundation only):** placeholder. No `SUBAGENT.yaml` files
> live here yet — they land in **PR 3**.

## Planned subagents

The preprocess v3 pipeline (see the locked design table in the PR 1 description)
delegates these steps to LLM subagents that will be defined here:

| Subagent          | Pipeline step                        | PR  |
|-------------------|--------------------------------------|-----|
| `pytorch-to-flydsl` | Step 2 (translation, conditional)  | 3   |
| `harness-generator` | Step 3a (replaces today's UTA)      | 3   |
| `harness-verifier`  | Step 3b (replaces today's ShapeFixer) | 3 |
| `speedup-verify`    | Step 5 (post-optimization gate)     | 3   |

## SUBAGENT.yaml shape (v3)

Each subagent lives under `subagents/preprocess/<name>/SUBAGENT.yaml`.
Required top-level keys (validated by the registry):

- `name` — unique identifier (lowercase, hyphens).
- `description` — one-line purpose, shown to the orchestrator LLM for routing.
- `system_prompt` — system prompt string (inline YAML block scalar).

Optional keys:

- `model` — model identifier override (e.g. `claude-opus-4.6`); defaults to
  the pipeline's configured model when absent.
- `tools` — list of tool names the subagent receives. Defaults to `[]`
  (orchestrator/runtime picks defaults).
- `max_steps` — step budget; defaults to `30`.

Any additional keys are preserved in `SubagentSpec.extras` so this directory
remains forward-compatible without rev-locking the registry schema.

## Relationship to the top-level `subagents/` directory

The repository already has `subagents/<name>/SUBAGENT.yaml` definitions
(`harness-generator`, `speedup-verify`, `pytorch-to-flydsl`, …) that follow
the **mini-swe-agent** YAML schema (`agent.system_template_file`, `env`,
`model`, …). Those entries remain the canonical definitions for the existing
flat homoagent dispatch path and are **not** modified by the v3 redesign.

`subagents/preprocess/` is a separate, v3-only namespace with a slimmer
schema (`system_prompt` as an inline string, no `agent`/`env` nesting). The
two registries point at different roots and never collide — the existing
top-level discovery rule of "scan immediate subdirectories that contain a
`SUBAGENT.yaml`" naturally skips this folder because it has no
`SUBAGENT.yaml` of its own.
