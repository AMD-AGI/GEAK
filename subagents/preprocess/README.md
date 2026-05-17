# Preprocess subagents (v3)

This directory holds the **v3 preprocess subagent definitions** consumed by
`minisweagent.run.preprocess_v3.registry.SubagentRegistry` and dispatched
by `minisweagent.run.preprocess_v3.tools.PreprocessSubagentDispatcher`.

## Locked design — 3 always-on subagents

The v3 orchestrator (`PreprocessOrchestratorAgent`) calls these three
subagents via `dispatch_subagent`; everything else in the 6-step flow is a
deterministic tool call.

> **Note on the Path-A short-circuit (commit set 7).** When the user's
> task prompt already carries explicit run instructions (e.g. `run via
> python my_kernel.py --benchmark`), the orchestrator's `Step 0` system
> prompt section instructs the LLM to call the
> `commandment_from_user_command` tool instead of dispatching
> `harness-generator`. That short-circuit skips all three of the
> subagents listed below. The subagents themselves are unchanged — Path
> A is purely an orchestrator-side bypass. See the orchestrator's `Step
> 0 — Path A vs Path B decision` section
> (`src/minisweagent/run/preprocess_v3/orchestrator.py`) for the
> selection logic.

| Subagent           | Pipeline step                      | `max_steps`  | `tools`                                  |
|--------------------|------------------------------------|--------------|------------------------------------------|
| `harness-generator`| Step 3a — build the test harness    | `-1` (unlimited) | `bash`, `str_replace_editor`, `save_and_test` |
| `harness-verifier` | Step 3b — verify the harness        | `30`         | `bash`, `str_replace_editor`             |
| `speedup-verify`   | Step 5 — write `compute_speedup.py` | `30`         | `bash`, `str_replace_editor`             |

`max_steps: -1` is the `UNLIMITED_MAX_STEPS` sentinel from
`preprocess_v3.registry`; only `harness-generator` uses it because legitimate
harness generation can take many tool-call rounds (read README, install
package, inspect tests, write harness, iterate against verifier feedback)
and a hard cap would pessimise slow-but-correct runs.

## Translation is a tool call, not a subagent

PyTorch -> FlyDSL translation (step 2 of the orchestrator) is handled by the
**deterministic tool** `preprocess_v3.translate.translate_to_flydsl`, which
wraps the legacy `run_translation` function. There is **no**
`pytorch-to-flydsl` subagent in this directory and `dispatch_subagent`'s
schema explicitly omits that name from its enum.

## SUBAGENT.yaml shape

Each subagent lives under `subagents/preprocess/<name>/SUBAGENT.yaml`.
Required top-level keys (validated by `SubagentRegistry`):

- `name` — unique identifier (lowercase, hyphens).
- `description` — one-line purpose, surfaced to the orchestrator LLM.
- `system_prompt` — system prompt string (inline YAML block scalar).

Required-in-practice (the dispatcher needs an explicit tool list):

- `tools` — list of tool names the subagent may call. Whitelisted against
  the v3 tool registry (`bash`, `str_replace_editor`, `save_and_test`).
  Unknown names raise `UnknownToolError` at child-agent construction time.

Required-where-applicable:

- `max_steps` — positive integer step cap, or `-1` for unlimited. Anything
  else (including `0`) is rejected at parse time.

Optional:

- `model` — model identifier override; defaults to the pipeline's
  configured model. Per commit-set decision 4, the AMD LLM router is
  globally wired, so this field is recorded for audit but the orchestrator
  passes its own model to every dispatch.

Any additional keys survive in `SubagentSpec.extras` so this directory
remains forward-compatible.

## Relationship to the top-level `subagents/` directory

The repository's existing top-level `subagents/<name>/SUBAGENT.yaml`
definitions (`harness-generator`, `speedup-verify`, `pytorch-to-flydsl`, …)
follow the mini-swe-agent schema (`agent.system_template_file`, `env`,
`model`, …). Those entries remain the canonical definitions for the legacy
flat heterogeneous-orchestrator dispatch path and are **not** modified by
v3. The two registries point at different roots and never collide — the
legacy registry's "scan immediate subdirectories that contain a
`SUBAGENT.yaml`" rule naturally skips this folder because it has no
`SUBAGENT.yaml` of its own.
