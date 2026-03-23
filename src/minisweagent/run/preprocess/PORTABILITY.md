# Preprocess Portability

`run/preprocess/` is the owned preprocessing stage.

Its goal is to keep kernel resolution, discovery interpretation, harness
generation/validation, shape fixing, testcase caching, profiling launch,
baseline creation, and COMMANDMENT generation inside one portable directory.

## What is owned here

- `preprocessor.py`
- `resolve_kernel_url.py`
- `codebase_context.py`
- `discovery_types.py`
- `run_harness.py`
- `harness_utils.py`
- `unit_test_agent.py`
- `shape_fixer_agent.py`
- `testcase_cache.py`
- `kernel_profile.py`
- `baseline.py`
- `benchmark_parsing.py`
- `commandment.py`
- `validate_commandment.py`
- `config/`

## Expected shared dependencies

These are intentionally kept shared and should already exist on the target repo:

- `minisweagent.config`
- `minisweagent.agents.default`
- `minisweagent.environments.local`
- `minisweagent.models`
- `minisweagent.debug_runtime`
- `minisweagent.run.utils.git_safe_env`

## External adapters

Preprocess still expects the following adapter packages or source trees to be
available:

- `automated_test_discovery`
- `profiler_mcp`
- `metrix_mcp`

The helper in `repo_paths.py` is the single preprocess-owned place that
resolves the repository root and adds `mcp_tools/...` entries to `sys.path`.

## Main branch copy set

If `GEAK-main/GEAK` already has the shared dependencies above, preprocessing
should land by copying:

- `src/minisweagent/run/preprocess/`

If main does not already contain a shared dependency, copy only the missing
support file(s), not unrelated stage directories.
