"""Preprocess Orchestrator: LLM-driven preprocessing pipeline.

Instead of the hardcoded sequential pipeline in ``preprocessor.py``,
this orchestrator uses an LLM agent that dispatches to registered subagents
(codebase-explore, harness-generator, speedup-verify) via the ``sub_agent``
tool.  The orchestrator decides the flow; no steps are hardcoded here.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── System prompt for the preprocess orchestrator ─────────────────────

PREPROCESS_SYSTEM_PROMPT = """\
You are the **GEAK Preprocess Orchestrator**. Your job is to prepare a GPU
kernel repository for optimization.

You have access to these tools:
- **bash**: Execute shell commands
- **sub_agent**: Delegate tasks to specialized subagents

{subagent_catalog}

## Pipeline

Achieve these goals in order. Use `sub_agent` to delegate each step to the
appropriate subagent. Use `bash` when you need to run simple commands
(e.g. validate harness, collect benchmark output, install deps).

1. **Explore codebase & resolve kernel** — use `codebase-explore` to find the kernel file, analyze dependencies, and generate CODEBASE_CONTEXT.md.
2. **Generate test harness** — produce a harness with `--correctness`, `--benchmark`, `--profile`, `--full-benchmark` modes. Validate it by running correctness + benchmark.
3. **Collect baseline** — run the harness to collect benchmark output.
4. **Generate speedup script** — produce a `compute_speedup.py` from the benchmark output format.
5. **Generate COMMANDMENT.md** — write the evaluation contract for the optimizer.

Pass relevant context (kernel path, repo root, output dir, benchmark output,
harness path, etc.) to each subagent via the `task` parameter.

## Output

Output directory: `{output_dir}`

When all steps are done, output:
```
PREPROCESS_COMPLETE: {{"kernel_path": "...", "repo_root": "...", "harness_path": "...", "test_command": "...", "speedup_script_path": "...", "commandment_path": "...", "codebase_context_path": "..."}}
```

## Rules
- Use absolute paths
- Do NOT modify existing kernel source files
- If a step fails, diagnose and retry before skipping
"""


# ── Orchestrator runner ───────────────────────────────────────────────


def _build_system_prompt(output_dir: Path) -> str:
    """Build the system prompt with the subagent catalog."""
    from minisweagent.subagents.subagent_registry import SubAgentRegistry

    registry = SubAgentRegistry()
    subagent_catalog = registry.build_system_prompt_section()

    return PREPROCESS_SYSTEM_PROMPT.format(
        subagent_catalog=subagent_catalog,
        output_dir=output_dir,
    )


def _extract_preprocess_result(text: str) -> dict[str, Any] | None:
    """Extract the PREPROCESS_COMPLETE JSON from orchestrator output."""
    match = re.search(r"PREPROCESS_COMPLETE:\s*(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        logger.warning("Failed to parse PREPROCESS_COMPLETE JSON")
        return None


def run_preprocess_orchestrator(
    task: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    model=None,
    model_factory=None,
) -> dict[str, Any]:
    """Run the agentic preprocessing pipeline.

    The orchestrator LLM decides the flow: it dispatches to registered
    subagents (codebase-explore, harness-generator, speedup-verify) and
    uses bash for simple commands.  All artefacts (harness, baseline,
    profile, commandment, speedup script) are written to *output_dir*.

    Parameters
    ----------
    task:
        User prompt describing the kernel / repo to preprocess.  The
        orchestrator passes this to ``codebase-explore`` which discovers
        the kernel file, repo root, etc.
    output_dir:
        Directory to write all artefacts.
    gpu_id:
        GPU device to use for profiling.
    model:
        LLM model instance.
    model_factory:
        Callable returning a new model instance.

    Returns
    -------
    dict with preprocessing context (same keys as run_preprocessor).
    """
    _t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _model = model or (model_factory() if model_factory else None)
    if _model is None:
        raise RuntimeError("Preprocess orchestrator requires a model (model or model_factory)")

    # Build system prompt
    system_prompt = _build_system_prompt(output_dir)

    # Build instance message from the user's task
    instance_msg = (
        f"{task}\n\n"
        f"Output directory: {output_dir}\n"
        f"GPU device: {gpu_id}"
    )

    # Set up model with tool schemas
    from minisweagent.subagents.subagent_registry import SubAgentRegistry

    registry = SubAgentRegistry()

    tools_schema = _build_orchestrator_tools(registry)
    model_impl = getattr(_model, "_impl", _model)
    _orig_tools = getattr(model_impl, "tools", None)
    model_impl.tools = tools_schema

    # Build conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_msg},
    ]

    # Run LLM step loop
    logger.info(
        "\n%s\n  Preprocess Orchestrator (agentic mode)\n%s",
        "=" * 60,
        "=" * 60,
    )

    ctx: dict[str, Any] = {
        "output_dir": str(output_dir),
        "gpu_id": gpu_id,
        "registry": registry,
        "model": _model,
        "model_factory": model_factory,
    }

    result = _run_orchestrator_steps(_model, messages, ctx)

    # Restore original tools
    if _orig_tools is not None:
        model_impl.tools = _orig_tools

    _elapsed = time.monotonic() - _t0
    logger.info("Preprocess orchestrator completed in %.0fs", _elapsed)

    # Parse result into preprocessor-compatible context
    if result:
        return _build_preprocess_context(result, output_dir)

    # Fallback: build context from output_dir artifacts
    return _build_context_from_artifacts(output_dir)


def _build_orchestrator_tools(registry) -> list[dict]:
    """Build tool schemas for the preprocess orchestrator.

    Includes bash and sub_agent tools.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command and return stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": registry.build_tool_schema(),
        },
    ]
    return tools


def _dispatch_tool(ctx: dict[str, Any], tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call from the orchestrator."""
    import subprocess

    if tool_name == "bash":
        command = tool_args.get("command", "")
        logger.info("  [bash] %s", command[:200])
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=600,
                cwd=ctx.get("output_dir"),
                env={
                    **__import__("os").environ,
                    "HIP_VISIBLE_DEVICES": str(ctx.get("gpu_id", 0)),
                },
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output[:10000]
        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 600s"
        except Exception as exc:
            return f"[ERROR] {exc}"

    if tool_name == "sub_agent":
        return _dispatch_subagent(ctx, tool_args)

    return f"[ERROR] Unknown tool: {tool_name}"


def _dispatch_subagent(ctx: dict[str, Any], tool_args: dict) -> str:
    """Dispatch a sub_agent tool call."""
    from minisweagent.tools.sub_agent_tool import SubAgentTool

    agent_name = tool_args.get("agent_name")
    task = tool_args.get("task", "")
    step_limit = tool_args.get("step_limit", 150)
    system_prompt = tool_args.get("system_prompt")

    registry = ctx.get("registry")
    model = ctx.get("model")

    if agent_name and registry:
        desc = registry.get(agent_name)
        if desc:
            loaded_prompt = registry.load_system_prompt(desc)
            if loaded_prompt:
                system_prompt = loaded_prompt
            if desc.step_limit > 0:
                step_limit = desc.step_limit

    logger.info("  [sub_agent] %s (steps=%d)", agent_name or "ad-hoc", step_limit)

    from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

    env_config = LocalEnvironmentConfig(cwd=ctx.get("output_dir") or ".")
    env = LocalEnvironment(**env_config.__dict__)

    sub_agent_tool = SubAgentTool(model=model, env=env)
    result = sub_agent_tool(
        task=task,
        step_limit=step_limit,
        system_prompt=system_prompt,
    )
    return result.get("output", str(result))


def _run_orchestrator_steps(
    model,
    messages: list[dict],
    ctx: dict[str, Any],
) -> dict[str, Any] | None:
    """Run the orchestrator LLM step loop.

    Returns the parsed PREPROCESS_COMPLETE result, or None.
    """
    import os

    max_steps = int(os.getenv("GEAK_PREPROCESS_STEP_LIMIT", "100"))
    step = 0

    while step < max_steps:
        step += 1
        logger.debug("Preprocess orchestrator step %d", step)

        _t0 = time.monotonic()
        response = model.query(messages)
        _elapsed = time.monotonic() - _t0
        logger.debug("Step %d: model.query returned in %.1fs", step, _elapsed)

        content_text = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        # Check for completion
        if "PREPROCESS_COMPLETE:" in content_text:
            result = _extract_preprocess_result(content_text)
            if result:
                logger.info("Preprocess orchestrator completed at step %d", step)
                return result

        if not tool_call:
            if content_text:
                _first_line = content_text.strip().split("\n", 1)[0][:200]
                logger.info("  Orchestrator: %s", _first_line)
            messages.append({"role": "assistant", "content": content_text})

            # If no tool call and no completion marker, prompt to continue
            messages.append({
                "role": "user",
                "content": "Continue with the next preprocessing step. "
                "When all steps are done, output PREPROCESS_COMPLETE with the results JSON.",
            })
            continue

        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("function", {}).get("arguments", {})
        tool_id = tool_call.get("id", f"call_preprocess_{step}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

        messages.append({
            "role": "assistant",
            "content": content_text,
            "tool_calls": tool_call,
        })

        result_str = _dispatch_tool(ctx, tool_name, tool_args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result_str,
        })

        # Check if tool result contains completion marker
        if "PREPROCESS_COMPLETE:" in result_str:
            result = _extract_preprocess_result(result_str)
            if result:
                return result

    logger.warning("Preprocess orchestrator hit step limit (%d)", max_steps)
    return None


def _build_preprocess_context(
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Convert orchestrator result to preprocessor-compatible context dict."""
    ctx: dict[str, Any] = {}

    ctx["kernel_path"] = result.get("kernel_path", "")
    ctx["repo_root"] = result.get("repo_root", "")
    ctx["harness_path"] = result.get("harness_path", "")
    ctx["test_command"] = result.get("test_command", "")
    ctx["speedup_script_path"] = result.get("speedup_script_path")

    # Load artifacts from output_dir
    for name, key in [
        ("resolved.json", "resolved"),
        ("discovery.json", "discovery"),
        ("profile.json", "profiling"),
        ("baseline_metrics.json", "baseline_metrics"),
        ("harness_results.json", "harness_results"),
    ]:
        path = output_dir / name
        if path.exists():
            try:
                ctx[key] = json.loads(path.read_text())
            except json.JSONDecodeError:
                pass

    # Load text artifacts
    for name, key in [
        ("benchmark_baseline.txt", "benchmark_baseline"),
        ("full_benchmark_baseline.txt", "full_benchmark_baseline"),
        ("COMMANDMENT.md", "commandment"),
    ]:
        path = output_dir / name
        if path.exists():
            ctx[key] = path.read_text()

    cbc_path = output_dir / "CODEBASE_CONTEXT.md"
    if cbc_path.exists():
        ctx["codebase_context_path"] = str(cbc_path)

    return ctx


def _build_context_from_artifacts(output_dir: Path) -> dict[str, Any]:
    """Fallback: build context dict from whatever artifacts exist on disk."""
    ctx = _build_preprocess_context({}, output_dir)
    logger.warning(
        "Preprocess orchestrator did not return structured result; "
        "built context from artifacts in %s",
        output_dir,
    )
    return ctx
