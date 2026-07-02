"""Claude Agent SDK backend for GEAK agents.

This module replaces GEAK's hand-rolled ``model.query() -> parse_action() ->
tool dispatch`` loop with the `Claude Agent SDK <https://code.claude.com/docs/en/agent-sdk/overview>`_.

Instead of GEAK owning the agent loop (querying the model, parsing bash/tool
actions and executing them), the SDK owns the loop: it talks to Claude, decides
which tool to call and drives the turn-by-turn conversation. GEAK's existing
domain tools (``bash``, ``str_replace_editor``, ``save_and_test``, ``submit``,
``strategy_manager`` and any MCP-bridged tools) are exposed to the SDK as an
**in-process MCP server**, so all tool execution still flows through GEAK's
:class:`~minisweagent.tools.tools_runtime.ToolRuntime` (preserving protected
files, per-worktree ``cwd``, GPU env isolation and the save/benchmark loop).

Authentication reuses whatever the ambient environment provides. On AMD hosts
that means ``ANTHROPIC_BASE_URL=https://llm-api.amd.com/Anthropic`` together with
``ANTHROPIC_CUSTOM_HEADERS`` / ``ANTHROPIC_AUTH_TOKEN`` — the same gateway the
legacy ``amd_claude`` path used. No credentials are handled here; the bundled
Claude Code CLI picks them up from the environment.

The public entry point is :func:`run_sdk_agent`, a drop-in replacement for an
agent's ``run(task) -> (exit_status, message)`` method. It preserves GEAK's
message history, trajectory logging and ``Submitted`` / ``LimitsExceeded``
exit-status contract so the rest of the pipeline (dispatch, postprocess, patch
selection) is unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: Config value (``agent.backend``) that activates this backend.
SDK_BACKEND = "claude_sdk"

#: Sentinels that historically signalled task completion when emitted as the
#: first line of a bash observation (see ``OptimizationAgent.has_finished``).
_SUBMIT_SENTINELS = ("MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


def backend_enabled(config: Any) -> bool:
    """Return True when the agent config opts into the Claude Agent SDK backend.

    The switch honours ``agent.backend`` from config and a ``GEAK_AGENT_BACKEND``
    environment override (handy for A/B testing without editing YAML).
    """
    env_override = os.getenv("GEAK_AGENT_BACKEND")
    if env_override:
        return env_override.strip().lower() == SDK_BACKEND
    return str(getattr(config, "backend", "") or "").strip().lower() == SDK_BACKEND


def _map_model_name(raw_name: str | None) -> str | None:
    """Map a GEAK/litellm model name onto the name the SDK/gateway expects.

    GEAK configs use provider-qualified or bare names such as
    ``openai/claude-opus-4.8`` or ``claude-opus-4.8``. The AMD gateway exposes
    Anthropic models under names like ``Claude-Opus-4.6`` which the environment
    already advertises via ``ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL``.
    We map by family and fall back to the ambient ``ANTHROPIC_MODEL`` default
    (returning ``None`` lets the CLI use that default).
    """
    if os.getenv("GEAK_SDK_MODEL"):
        return os.getenv("GEAK_SDK_MODEL")
    name = (raw_name or "").lower()
    if "opus" in name:
        return os.getenv("ANTHROPIC_DEFAULT_OPUS_MODEL") or None
    if "sonnet" in name:
        return os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL") or None
    if "haiku" in name:
        return os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL") or None
    # Unknown family: let the CLI use its default ($ANTHROPIC_MODEL).
    return None


def _result_to_text(result: Any) -> tuple[str, bool]:
    """Normalise a GEAK tool result into ``(text, is_error)``.

    GEAK tools return dicts shaped like ``{"output": str, "returncode": int}``
    but some MCP-bridged tools return other shapes; handle both.
    """
    if isinstance(result, dict):
        rc = result.get("returncode", 0)
        if "output" in result:
            text = result.get("output", "")
        else:
            text = json.dumps(result, ensure_ascii=False, default=str)
        return (str(text), bool(rc))
    return (str(result), False)


def build_geak_mcp_server(agent: Any):
    """Wrap the agent's :class:`ToolRuntime` tools as an in-process SDK MCP server.

    Returns ``(server_config, submit_state, tool_names)`` where ``submit_state``
    is a mutable dict the ``submit`` tool (and bash completion sentinel) flips to
    signal task completion to the driver loop.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    toolruntime = agent.toolruntime
    table = dict(toolruntime._tool_table)
    schemas = {t["name"]: t for t in toolruntime.get_tools_list()}

    submit_state: dict[str, Any] = {"submitted": False, "summary": ""}
    sdk_tools = []

    def _make_handler(tool_name: str, tool_fn: Any):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            # The submit tool historically raises ``Submitted`` to unwind the
            # loop. Under the SDK we instead record completion and return a
            # terminal message; the driver stops the client afterwards.
            if tool_name == "submit":
                submit_state["submitted"] = True
                submit_state["summary"] = args.get("summary", "") or ""
                return {
                    "content": [
                        {"type": "text", "text": "Final result submitted. The task is complete."}
                    ]
                }
            try:
                # GEAK tools are synchronous and some (bash, save_and_test) block
                # on long benchmarks; run them off the event loop.
                result = await asyncio.to_thread(lambda: tool_fn(**args))
            except Exception as exc:  # keep the agent loop alive (see SDK docs)
                from minisweagent.tools.submit import Submitted as ToolSubmitted

                if isinstance(exc, ToolSubmitted):
                    submit_state["submitted"] = True
                    submit_state["summary"] = str(exc)
                    return {"content": [{"type": "text", "text": "Final result submitted."}]}
                logger.warning("SDK tool %r raised: %s", tool_name, exc)
                return {
                    "content": [{"type": "text", "text": f"Tool {tool_name} failed: {exc}"}],
                    "is_error": True,
                }

            text, is_error = _result_to_text(result)

            # Preserve the legacy bash completion sentinel: if a command's output
            # begins with the final-output marker, treat it as a submit.
            if tool_name == "bash":
                stripped = text.lstrip().splitlines()
                if stripped and stripped[0].strip() in _SUBMIT_SENTINELS:
                    submit_state["submitted"] = True
                    submit_state["summary"] = "\n".join(stripped[1:])

            from minisweagent.agents.optimization_agent import truncate_observation

            return {
                "content": [{"type": "text", "text": truncate_observation(text)}],
                "is_error": is_error,
            }

        return handler

    for name, fn in table.items():
        schema = schemas.get(name)
        if schema is None:
            continue
        input_schema = schema.get("parameters") or {"type": "object", "properties": {}}
        description = schema.get("description", name)
        sdk_tools.append(tool(name, description, input_schema)(_make_handler(name, fn)))

    server = create_sdk_mcp_server(name="geak", version="1.0.0", tools=sdk_tools)
    return server, submit_state, list(table.keys())


def build_options(
    *,
    system_prompt: str,
    server: Any,
    server_name: str,
    tool_names: list[str],
    model_name: str | None,
    cwd: str | None,
    step_limit: int = 0,
    cost_limit: float = 0.0,
):
    """Assemble :class:`ClaudeAgentOptions` shared by all GEAK SDK loops."""
    from claude_agent_sdk import ClaudeAgentOptions

    allowed = [f"mcp__{server_name}__{n}" for n in tool_names]
    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={server_name: server},
        allowed_tools=allowed,
        # Remove all built-in tools: GEAK's own tools are the sanctioned surface
        # (they enforce protected files, per-worktree cwd and GPU env isolation).
        tools=[],
        permission_mode="bypassPermissions",
        cwd=str(cwd or os.getcwd()),
        model=_map_model_name(model_name),
        max_turns=step_limit if step_limit > 0 else None,
        max_budget_usd=cost_limit if cost_limit > 0 else None,
        # NOTE: we intentionally do NOT set ``setting_sources=[]``. On the AMD
        # gateway the working credentials/base-url live in the Claude settings
        # that the CLI loads by default; stripping them breaks auth. The run is
        # still scoped by the explicit ``system_prompt`` (which replaces the
        # default) plus ``tools=[]`` + ``allowed_tools`` restricting the surface.
    )


async def _stream_client(options: Any, prompt: str, on_event, stop_when=None) -> None:
    """Drive a :class:`ClaudeSDKClient` and forward parsed events to ``on_event``.

    ``on_event(kind, payload)`` is called with:
      * ``("assistant", {"text": str, "tool_calls": list})``
      * ``("tool_result", {"text": str, "tool_use_id": str})``
      * ``("result", {"text": str, "subtype": str, "cost": float | None})``

    ``stop_when()`` (optional) is polled after each event; when it returns True
    the client is interrupted (used to stop promptly after submit/finish).
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(
                            {"id": block.id, "function": {"name": block.name, "arguments": block.input}}
                        )
                on_event("assistant", {"text": "".join(text_parts), "tool_calls": tool_calls})
            elif isinstance(msg, UserMessage):
                for block in getattr(msg, "content", []) or []:
                    if isinstance(block, ToolResultBlock):
                        raw = block.content
                        if isinstance(raw, list):
                            text = "".join(
                                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                                for b in raw
                            )
                        else:
                            text = str(raw)
                        on_event(
                            "tool_result",
                            {"text": text, "tool_use_id": getattr(block, "tool_use_id", "") or ""},
                        )
            elif isinstance(msg, ResultMessage):
                on_event(
                    "result",
                    {
                        "text": getattr(msg, "result", "") or "",
                        "subtype": getattr(msg, "subtype", "") or "",
                        "cost": getattr(msg, "total_cost_usd", None),
                    },
                )

            if stop_when is not None and stop_when():
                try:
                    await client.interrupt()
                except Exception:
                    pass
                break


async def _drive(agent: Any, prompt: str, options: Any, submit_state: dict[str, Any]) -> tuple[str, str]:
    """Run the SDK loop, mirroring messages into GEAK's history + trajectory."""
    from minisweagent.agents.optimization_agent import truncate_observation

    state = {"status": "Submitted", "final": ""}

    def on_event(kind: str, payload: dict) -> None:
        if kind == "assistant":
            try:
                agent.model.n_calls += 1
            except Exception:
                pass
            msg_kwargs: dict[str, Any] = {}
            tcs = payload["tool_calls"]
            if tcs:
                msg_kwargs["tool_calls"] = tcs[0] if len(tcs) == 1 else tcs
            agent.add_message("assistant", payload["text"], **msg_kwargs)
            agent._save_traj()
        elif kind == "tool_result":
            agent.add_message(
                "tool",
                truncate_observation(payload["text"]),
                tool_call_id=payload["tool_use_id"],
                name="tool",
            )
            agent._save_traj()
        elif kind == "result":
            if payload["cost"] is not None:
                try:
                    agent.model.cost = float(payload["cost"])
                except Exception:
                    pass
            state["final"] = payload["text"]
            subtype = payload["subtype"]
            if submit_state.get("submitted"):
                state["status"] = "Submitted"
                state["final"] = submit_state.get("summary") or payload["text"]
            elif "max_turns" in subtype or "budget" in subtype or "limit" in subtype:
                state["status"] = "LimitsExceeded"
            else:
                # Natural stop without explicit submit == completed run; GEAK's
                # downstream reads produced patches from disk.
                state["status"] = "Submitted"

    await _stream_client(options, prompt, on_event, stop_when=lambda: submit_state.get("submitted"))
    return state["status"], state["final"]


def run_sdk_agent(agent: Any, task: str, **kwargs) -> tuple[str, str]:
    """Drop-in replacement for ``agent.run`` that uses the Claude Agent SDK.

    Preserves the ``(exit_status, message)`` contract and GEAK's message /
    trajectory bookkeeping, then runs the same post-termination patch selection.
    """
    agent.extra_template_vars |= {"task": task, **kwargs}
    agent.extra_template_vars["tool_names"] = set(agent.toolruntime._tool_table.keys())
    agent.messages = []
    agent._traj_last_saved_idx = -1
    if getattr(agent.config, "use_skills", False):
        agent.config.system_template += agent.skillruntime.build_system_prompt()

    system_prompt = agent.render_template(agent.config.system_template)
    instance_prompt = agent.render_template(agent.config.instance_template)

    # Mirror the classic loop's opening messages for trajectory parity.
    agent.add_message("system", system_prompt)
    agent.add_message("user", instance_prompt)

    server, submit_state, tool_names = build_geak_mcp_server(agent)
    options = build_options(
        system_prompt=system_prompt,
        server=server,
        server_name="geak",
        tool_names=tool_names,
        model_name=getattr(agent.model.config, "model_name", None),
        cwd=getattr(agent.env.config, "cwd", None),
        step_limit=int(getattr(agent.config, "step_limit", 0) or 0),
        cost_limit=float(getattr(agent.config, "cost_limit", 0) or 0),
    )

    try:
        status, message = asyncio.run(_drive(agent, instance_prompt, options, submit_state))
    except Exception as exc:
        logger.exception("Claude Agent SDK run failed: %s", exc)
        status, message = "SDKError", str(exc)
    finally:
        agent._save_traj()

    # Preserve GEAK's post-run patch selection when available.
    select = getattr(agent, "_run_select_patch_agent", None)
    if callable(select):
        try:
            select()
        except Exception as exc:
            logger.debug("run_sdk_agent: select-patch step failed: %s", exc)

    return status, message


# ---------------------------------------------------------------------------
# Preprocess orchestrator support
#
# The v3 PreprocessOrchestratorAgent is a standalone class with its own tool
# surface (``self._tools`` of ``ToolEntry``) and a ``finish_preprocess`` tool
# that raises ``FinishedSuccessfully`` to terminate with a structured payload.
# We drive its tool loop with the SDK while reusing its own ``_dispatch_tool``
# (side effects populate ``_collected`` / ``_tool_calls``) and its
# ``_build_result`` finalisation.
# ---------------------------------------------------------------------------


def run_preprocess_via_sdk(agent: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Drive :class:`PreprocessOrchestratorAgent`'s tool loop via the SDK.

    Returns ``(finish_payload, errors)`` mirroring the classic loop's outcome so
    the caller can hand them straight to ``_build_result``. Tool side effects
    (``agent._collected`` etc.) accrue through ``agent._dispatch_tool``.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    from minisweagent.run.preprocess_v3.orchestrator import FinishedSuccessfully

    finish_state: dict[str, Any] = {"finished": False, "payload": None}
    errors: list[str] = []

    sdk_tools = []
    for name, entry in agent._tools.items():
        schema = entry.schema or {}
        input_schema = schema.get("parameters") or {"type": "object", "properties": {}}
        description = schema.get("description", name)

        def _make(tool_name: str):
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    result = await asyncio.to_thread(lambda: agent._dispatch_tool(tool_name, args))
                except FinishedSuccessfully as fin:
                    finish_state["finished"] = True
                    finish_state["payload"] = fin.payload
                    return {"content": [{"type": "text", "text": "Preprocess finished."}]}
                except Exception as exc:  # keep the loop alive; surface as data
                    logger.warning("preprocess tool %r raised: %s", tool_name, exc)
                    return {
                        "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                        "is_error": True,
                    }
                text = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
                is_error = isinstance(result, dict) and bool(result.get("error"))
                return {"content": [{"type": "text", "text": text}], "is_error": is_error}

            return handler

        sdk_tools.append(tool(name, description, input_schema)(_make(name)))

    server = create_sdk_mcp_server(name="geak_preprocess", version="1.0.0", tools=sdk_tools)
    system_prompt = agent.render_template(agent.config.system_template)
    instance_prompt = agent.render_template(agent.config.instance_template)

    cwd = agent._extra_template_vars.get("repo_root") or agent._extra_template_vars.get("output_dir")
    options = build_options(
        system_prompt=system_prompt,
        server=server,
        server_name="geak_preprocess",
        tool_names=list(agent._tools.keys()),
        model_name=getattr(getattr(agent.model, "config", None), "model_name", None),
        cwd=cwd,
        step_limit=int(getattr(agent.config, "step_limit", 0) or 0),
        cost_limit=float(getattr(agent.config, "cost_limit", 0) or 0),
    )

    def on_event(kind: str, payload: dict) -> None:
        if kind == "assistant":
            try:
                agent.model.n_calls += 1
            except Exception:
                pass
            msg_kwargs: dict[str, Any] = {}
            tcs = payload["tool_calls"]
            if tcs:
                msg_kwargs["tool_calls"] = tcs[0] if len(tcs) == 1 else tcs
            agent.add_message("assistant", payload["text"], **msg_kwargs)
        elif kind == "tool_result":
            agent.add_message("tool", payload["text"], tool_call_id=payload["tool_use_id"], name="tool")
        elif kind == "result":
            if payload["cost"] is not None:
                try:
                    agent.model.cost = float(payload["cost"])
                except Exception:
                    pass
            subtype = payload["subtype"]
            if not finish_state["finished"] and (
                "max_turns" in subtype or "budget" in subtype or "limit" in subtype
            ):
                errors.append(f"Limits exceeded (step_limit/cost_limit): {subtype}")

    try:
        asyncio.run(
            _stream_client(options, instance_prompt, on_event, stop_when=lambda: finish_state["finished"])
        )
    except Exception as exc:
        logger.exception("Claude Agent SDK preprocess run failed: %s", exc)
        errors.append(f"{type(exc).__name__}: {exc}")

    return finish_state["payload"], errors
