"""Tool implementations for the v3 preprocess orchestrator.

Each tool here is the LLM-callable bridge between an
``OpenAI/LiteLLM``-style tool call and one of the v3 preprocess modules.
Tools come in two flavours:

* **Deterministic tools** (``codebase_explore``, ``translate_to_flydsl``,
  ``collect_baseline``, ``collect_profile``, ``render_commandment``) —
  call directly into the v3 module with no LLM step of their own.

* **LLM-dispatch tool** (``dispatch_subagent``) — looks up the named
  subagent in :class:`SubagentRegistry`, builds an
  :class:`PreprocessSubagentDispatcher`, runs it, and returns a
  structured summary.

* **Completion sentinel** (``finish_preprocess``) — populates the
  orchestrator's pending :class:`PreprocessResult` payload and raises
  :class:`FinishedSuccessfully` to terminate the loop.

Why a v3-native dispatcher + a v3-native subagent?
--------------------------------------------------

The legacy :class:`minisweagent.tools.sub_agent_tool.SubAgentTool` plus
its :class:`minisweagent.agents.default.DefaultAgent` child:

1. Hard-codes the rich legacy descriptor model
   (:class:`minisweagent.subagents.SubAgentRegistry`) as its routing
   source.
2. Has no concept of the v3 ``max_steps == -1`` sentinel — it normalises
   step limits via ``MIN_CHILD_STEP_LIMIT = 150``.
3. Has no concept of a per-subagent restricted tool set — child agents
   inherit the parent's tool surface (~30 tools).
4. Drags in strategy manager, save/test, working memory, RAG
   postprocessor, MCP bridges, and a select-patch agent — none of which
   is appropriate for a short-lived preprocess subagent that produces a
   single deliverable.

The v3 contract needs all four points to be different.
:class:`PreprocessSubagentDispatcher` (this module) plus
:class:`minisweagent.run.preprocess_v3.subagent.PreprocessSubagent`
together provide the v3-native path. The legacy ``SubAgentTool`` +
``DefaultAgent`` stay untouched — important because the legacy
heterogeneous orchestrator still uses them.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.baseline import (
    BaselineMetrics,
    ProfileResult,
    collect_baseline_metrics,
    collect_profile,
)
from minisweagent.run.preprocess_v3.commandment import (
    CommandmentContext,
    render_commandment,
    render_commandment_from_sections,
)
from minisweagent.run.preprocess_v3.explore import CodebaseContext, explore_codebase
from minisweagent.run.preprocess_v3.harness_kb import load_harness_kb
from minisweagent.run.preprocess_v3.orchestrator import (
    FinishedSuccessfully,
    PreprocessOrchestratorAgent,
)
from minisweagent.run.preprocess_v3.registry import SubagentRegistry, SubagentSpec
from minisweagent.run.preprocess_v3.translate import TranslationResult, translate_to_flydsl

logger = logging.getLogger(__name__)


#: Subagent names the orchestrator may dispatch. Anything outside this
#: set is rejected at the schema layer (enum) AND at the dispatcher (an
#: extra defensive check, in case the LLM bypasses the schema).
ALLOWED_SUBAGENT_NAMES: tuple[str, ...] = (
    "harness-generator",
    "harness-verifier",
)

#: Mode names the Path-A short-circuit understands. Maps 1:1 to the four
#: harness CLI flags (``--correctness``, ``--profile``, ``--benchmark``,
#: ``--full-benchmark``). Used to validate ``modes_covered`` /
#: ``inferred_modes`` arguments on the
#: ``commandment_from_user_command`` tool and to drive the per-section
#: body assembly when projecting the user's command into a
#: ``COMMANDMENT.md``.
PATH_A_MODES: tuple[str, ...] = ("correctness", "profile", "benchmark", "full_benchmark")


# ---------------------------------------------------------------------------
# Path-A dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunInstructions:
    """Parsed user-provided run instructions for the Path-A short-circuit.

    When the orchestrator detects (via LLM judgment, see ``Step 0`` in the
    system prompt) that the user's task prompt already contains explicit
    run instructions, it bypasses the harness-generator / harness-verifier
    subagent dispatches and emits a ``COMMANDMENT.md`` directly from those
    instructions. This dataclass captures the parsed
    shape of that decision so :class:`PreprocessResult` consumers have an
    audit trail of which command was used and which modes were inferred.

    Attributes:
        raw_command:
            The user-provided shell command verbatim (whitespace
            trimmed). Required, non-empty.
        modes_covered:
            Subset of :data:`PATH_A_MODES` the LLM concluded the
            command directly covers (e.g. ``("benchmark",)`` for a
            command that ends in ``--benchmark``).
        inferred_modes:
            Subset of :data:`PATH_A_MODES` the LLM asked the
            orchestrator to fill in by inference from the covered
            modes (e.g. ``("correctness",)`` inferred from a
            ``--benchmark`` command by swapping the flag).
        notes:
            Free-form LLM justification, recorded for audit. Empty
            string is the no-notes default.
    """

    raw_command: str
    modes_covered: tuple[str, ...]
    inferred_modes: tuple[str, ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# v3-native subagent dispatcher
#
# Replaces the legacy SubAgentTool for the v3 preprocess flow.
# ---------------------------------------------------------------------------


class PreprocessSubagentDispatcher:
    """Run a v3 preprocess subagent and project the outcome to a dict.

    Mirrors the legacy ``SubAgentTool.__call__`` interface (takes a
    ``task`` string + budget hints + an optional system prompt override,
    returns ``{output, returncode}``) but reads its routing exclusively
    from :class:`SubagentRegistry` and honors v3 sentinels:

    * ``spec.max_steps == -1`` (UNLIMITED) — pass ``step_limit=0`` to the
      child (matches ``DefaultAgent.AgentConfig`` convention: 0 = "no
      step cap").
    * ``spec.tools`` — restrict the child agent's tool surface via
      ``ToolRuntime.disable_tools`` for tools NOT in the spec (white-list
      semantics).
    * ``spec.system_prompt`` — inject as the child's ``system_template``.

    The class is intentionally constructed lazily inside
    :func:`build_dispatch_subagent_tool` so unit tests can swap in a
    mock factory without spinning up the full ``ToolRuntime``.
    """

    def __init__(
        self,
        registry: SubagentRegistry,
        *,
        agent_factory: Callable[..., Any] | None = None,
        env_factory: Callable[[], Any] | None = None,
        kernel_language: KernelLanguage | None = None,
    ) -> None:
        """Build the dispatcher.

        Args:
            registry:
                v3 :class:`SubagentRegistry` instance. The dispatcher
                resolves subagent names through ``registry.get(name)``.
            agent_factory:
                Optional callable returning a fresh agent for the
                subagent run. Defaults to building a
                :class:`minisweagent.run.preprocess_v3.subagent.PreprocessSubagent`
                (the v3-native class — imported lazily inside
                ``_run_child`` so the import graph stays minimal).
                Tests inject a stub here.
            env_factory:
                Optional callable. Reserved for future use; the v3
                subagent does not need a separate environment object
                (it routes commands through its own tool whitelist).
                Kept on the constructor for backward compatibility
                with the previous signature.
            kernel_language:
                Optional :class:`KernelLanguage` instance. When the
                dispatched subagent's spec carries
                ``knowledge_base_template == "from_kernel_language"``
                and ``kernel_language`` is set, the dispatcher calls
                :func:`load_harness_kb` and injects the result as
                ``{"knowledge_base": <string>}`` into the child's
                ``extra_template_vars`` (which the child's
                ``render_template`` consumes when rendering the
                system prompt). Defaults to ``None`` — graceful
                degradation: the child's ``{{knowledge_base}}``
                placeholder, if present, renders as the empty string.
        """
        self._registry = registry
        self._agent_factory = agent_factory
        self._env_factory = env_factory
        self.kernel_language = kernel_language

    def __call__(
        self,
        *,
        name: str,
        task: str,
        model: Any,
        cwd: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a single subagent run.

        Args:
            name:
                One of the names declared in
                :data:`ALLOWED_SUBAGENT_NAMES`. Anything else returns a
                structured error rather than raising — the orchestrator
                surfaces the error to the LLM.
            task:
                Free-form task string handed to the child agent's
                ``run`` method.
            model:
                Model instance the child uses. Per commit-set decision
                4, the orchestrator passes its own ``self.model`` here
                (global AMD-router routing).
            cwd:
                Working directory for the child. Defaults to the
                process CWD.
            context:
                Optional extra dict merged into the child's task as a
                preamble. Useful for passing the codebase context text
                or harness path without inflating the task string.

        Returns:
            ``{name, success, output, returncode, system_prompt_used,
              max_steps, elapsed_s}``.
        """
        if name not in ALLOWED_SUBAGENT_NAMES:
            return {
                "name": name,
                "success": False,
                "error": f"Subagent {name!r} is not in the v3 allow-list {list(ALLOWED_SUBAGENT_NAMES)}",
                "elapsed_s": 0.0,
            }

        try:
            spec = self._registry.get(name)
        except KeyError as exc:
            return {
                "name": name,
                "success": False,
                "error": f"Subagent {name!r} not found in registry: {exc}",
                "elapsed_s": 0.0,
            }

        full_task = task if not context else self._format_context_preamble(context) + "\n\n" + task

        extra_template_vars = self._resolve_extra_template_vars(spec)

        t0 = time.monotonic()
        try:
            exit_status, message = self._run_child(
                spec=spec,
                task=full_task,
                model=model,
                cwd=cwd,
                extra_template_vars=extra_template_vars,
            )
        except Exception as exc:
            logger.exception("Subagent %r dispatch failed", name)
            return {
                "name": name,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - t0, 3),
                "max_steps": spec.max_steps,
            }
        elapsed_s = round(time.monotonic() - t0, 3)

        success = exit_status in {"Submitted", "FinishedSuccessfully"} or "VERIFIED=true" in message
        return {
            "name": name,
            "success": success,
            "exit_status": exit_status,
            "output": message,
            "elapsed_s": elapsed_s,
            "max_steps": spec.max_steps,
            "is_unlimited_steps": spec.is_unlimited_steps,
        }

    @staticmethod
    def _format_context_preamble(context: Any) -> str:
        """Render an optional context value as a Markdown preamble.

        Defensive: the orchestrator LLM has been observed to pass a bare
        string (or, more rarely, ``None`` / another scalar) for the
        ``context`` argument of ``dispatch_subagent`` despite the schema
        declaring it as an object. Coerce before the ``.items()`` call so
        the preamble assembly never raises ``AttributeError``.
        """
        if context is None:
            return ""
        if isinstance(context, str):
            context = {"context": context}
        elif not isinstance(context, dict):
            logger.warning(
                "dispatch_subagent context expected dict, got %s; coercing",
                type(context).__name__,
            )
            context = {"context": str(context)}
        if not context:
            return ""
        lines = ["## Context"]
        for key, value in context.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)

    def _resolve_extra_template_vars(self, spec: SubagentSpec) -> dict[str, Any]:
        """Compute the per-dispatch ``extra_template_vars`` for the child.

        Recognises the ``knowledge_base_template == "from_kernel_language"``
        routing tag on :class:`SubagentSpec`. When the tag is present AND
        the dispatcher was constructed with a ``kernel_language``,
        ``{"knowledge_base": load_harness_kb(self.kernel_language)}`` is
        added. In every other case (tag missing, tag unknown, language
        unset, language has no on-disk KB) the placeholder is filled
        with the empty string so the child's Jinja
        ``StrictUndefined`` renderer does not crash on the unresolved
        ``{{knowledge_base}}`` variable.
        """
        vars_out: dict[str, Any] = {}
        tag = spec.knowledge_base_template
        if tag == "from_kernel_language":
            if self.kernel_language is not None:
                vars_out["knowledge_base"] = load_harness_kb(self.kernel_language)
            else:
                logger.debug(
                    "knowledge_base_template=from_kernel_language on subagent %r but "
                    "dispatcher has no kernel_language; injecting empty KB",
                    spec.name,
                )
                vars_out["knowledge_base"] = ""
        elif tag is not None:
            logger.warning(
                "Unknown knowledge_base_template tag %r on subagent %r; no KB injected",
                tag,
                spec.name,
            )
        return vars_out

    def _run_child(
        self,
        *,
        spec: SubagentSpec,
        task: str,
        model: Any,
        cwd: str | None,
        extra_template_vars: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Build + run the child agent. Returns ``(exit_status, message)``."""
        if self._agent_factory is not None:
            kwargs: dict[str, Any] = {"spec": spec, "model": model, "cwd": cwd}
            # Only forward ``extra_template_vars`` when the factory
            # accepts it (either as a named parameter or via ``**kwargs``).
            # This keeps older test fakes that pin a strict kwarg list
            # working unchanged.
            sig = inspect.signature(self._agent_factory)
            has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "extra_template_vars" in sig.parameters or has_kwargs:
                kwargs["extra_template_vars"] = extra_template_vars or {}
            agent = self._agent_factory(**kwargs)
            return agent.run(task)

        # v3-native default: build a PreprocessSubagent. The class
        # natively honours ``spec.tools`` (whitelist), ``spec.max_steps``
        # (with the ``UNLIMITED_MAX_STEPS == -1`` -> ``step_limit=0``
        # projection), and ``spec.system_prompt``. The legacy
        # ``DefaultAgent`` is no longer imported from this module.
        from minisweagent.run.preprocess_v3.subagent import PreprocessSubagent

        step_limit = 0 if spec.is_unlimited_steps else int(spec.max_steps)
        agent = PreprocessSubagent(
            model=model,
            system_prompt=spec.system_prompt,
            tools=list(spec.tools) if spec.tools else [],
            step_limit=step_limit,
            cost_limit=0.0,
            cwd=cwd,
            extra_template_vars=extra_template_vars or {},
        )
        return agent.run(task)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


def _schema_codebase_explore() -> dict[str, Any]:
    return {
        "name": "codebase_explore",
        "type": "function",
        "description": (
            "Step 1 — deterministic. Walk the repo from the kernel and produce "
            "CODEBASE_CONTEXT.md plus a list of in-repo dependencies. Always "
            "call this first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_root": {"type": "string", "description": "Absolute path to the repository root."},
                "kernel_path": {"type": "string", "description": "Absolute path to the target kernel."},
                "out_path": {
                    "type": "string",
                    "description": "Where to write CODEBASE_CONTEXT.md.",
                },
            },
            "required": ["repo_root", "kernel_path", "out_path"],
        },
    }


def _schema_translate_to_flydsl() -> dict[str, Any]:
    return {
        "name": "translate_to_flydsl",
        "type": "function",
        "description": (
            "Step 2 — translate a PyTorch kernel to FlyDSL. Call ONLY when "
            "source_language != target_language and target_language == 'flydsl'. "
            "Wraps run_translation; not idempotent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "description": "Source kernel (e.g. PyTorch nn.Module file)."},
                "output_dir": {"type": "string", "description": "Where the candidate FlyDSL kernel is written."},
            },
            "required": ["source_path", "output_dir"],
        },
    }


def _schema_dispatch_subagent() -> dict[str, Any]:
    return {
        "name": "dispatch_subagent",
        "type": "function",
        "description": (
            "Step 3a / 3b — dispatch one of the two v3 preprocess subagents "
            "(harness-generator, harness-verifier) with a focused task string. "
            "Use them in alternation during step 3 (max 3 generator attempts). "
            "On step 3b (verifier), pass the four COMMANDMENT_<MODE> commands "
            "from harness-generator's output as ``commandment_commands`` in the "
            "context so the verifier runs the declared commands verbatim."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": list(ALLOWED_SUBAGENT_NAMES),
                    "description": "Subagent name. Must be one of the three v3 preprocess subagents.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description forwarded to the child agent.",
                },
                "context": {
                    "type": "object",
                    "description": "Optional context dict (rendered as a Markdown preamble before the task).",
                },
            },
            "required": ["name", "task"],
        },
    }


def _schema_collect_baseline() -> dict[str, Any]:
    return {
        "name": "collect_baseline",
        "type": "function",
        "description": (
            "Step 4 part 1 — deterministic. Run the harness in --benchmark mode "
            "``repeats`` times, parse the latency markers, return the median + "
            "samples + raw outputs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "harness_path": {"type": "string", "description": "Absolute path to the verified harness."},
                "repeats": {"type": "integer", "description": "Number of benchmark invocations.", "default": 5},
                "work_dir": {"type": "string", "description": "Working directory + PYTHONPATH prefix."},
                "gpu_id": {"type": "integer", "description": "HIP_VISIBLE_DEVICES value.", "default": 0},
            },
            "required": ["harness_path"],
        },
    }


def _schema_collect_profile() -> dict[str, Any]:
    return {
        "name": "collect_profile",
        "type": "function",
        "description": (
            "Step 4 part 2 — deterministic. Profile the harness in --profile mode "
            "via profiler-mcp. Returns a structured profile dict."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "harness_path": {"type": "string", "description": "Absolute path to the verified harness."},
                "work_dir": {"type": "string", "description": "Working directory."},
                "gpu_id": {"type": "integer", "description": "HIP_VISIBLE_DEVICES value.", "default": 0},
                "out_path": {"type": "string", "description": "Optional path to write the profile JSON."},
            },
            "required": ["harness_path"],
        },
    }


def _schema_render_commandment() -> dict[str, Any]:
    return {
        "name": "render_commandment",
        "type": "function",
        "description": (
            "Step 6 — deterministic. Render COMMANDMENT.md via the per-language "
            "Jinja template (or the legacy generator on fallback). Call last, "
            "after all other artifacts exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kernel_path": {"type": "string", "description": "Absolute path to the kernel."},
                "harness_path": {"type": "string", "description": "Absolute path to the verified harness."},
                "repo_root": {"type": "string", "description": "Repository root."},
                "out_path": {
                    "type": "string",
                    "description": (
                        "Optional. Where to write COMMANDMENT.md. When omitted, the tool "
                        "defaults to ``<output_dir>/COMMANDMENT.md`` using the orchestrator's "
                        "run-time ``output_dir``."
                    ),
                },
                "baseline_metrics": {
                    "type": "object",
                    "description": "Optional baseline metrics dict (median_ms, samples_ms, etc.).",
                },
            },
            "required": ["kernel_path", "harness_path", "repo_root"],
        },
    }


def _schema_finish_preprocess() -> dict[str, Any]:
    return {
        "name": "finish_preprocess",
        "type": "function",
        "description": (
            "Completion sentinel — call when all 6 steps succeeded (or you have "
            "exhausted retries and want to surface a partial result). The "
            "orchestrator terminates the LLM loop after this returns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "harness_path": {"type": "string", "description": "Absolute path to the verified harness."},
                "commandment_path": {"type": "string", "description": "Absolute path to COMMANDMENT.md."},
                "errors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Collected non-fatal errors.",
                },
                "summary": {"type": "string", "description": "One-paragraph summary of the run."},
            },
            "required": [],
        },
    }


def _schema_commandment_from_user_command() -> dict[str, Any]:
    return {
        "name": "commandment_from_user_command",
        "type": "function",
        "description": (
            "Path-A short-circuit (Step 0 alternative to the 6-step flow). Call "
            "ONLY when the user's task prompt already contains explicit run "
            "instructions (a literal command-line invocation, a reference to an "
            "existing harness file, or a make-target). Renders a "
            "COMMANDMENT.md directly from the user's command — skipping "
            "harness-generator and harness-verifier subagent dispatches "
            "entirely. Mutually exclusive with calling "
            "dispatch_subagent('harness-generator', ...) — the orchestrator's "
            "path is determined by which of these two tools is called first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "run_command": {
                    "type": "string",
                    "description": (
                        "The user-provided shell command, verbatim. Must be non-empty. "
                        "Example: 'cd /repo && python my_kernel.py --benchmark --shape 4096'."
                    ),
                },
                "out_path": {
                    "type": "string",
                    "description": "Where to write COMMANDMENT.md.",
                },
                "modes_covered": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(PATH_A_MODES),
                    },
                    "description": (
                        "Subset of ('correctness', 'profile', 'benchmark', 'full_benchmark') "
                        "the run_command directly covers."
                    ),
                },
                "inferred_modes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(PATH_A_MODES),
                    },
                    "description": (
                        "Modes the LLM is asking the tool to fill in from the covered modes "
                        "by inference (e.g. swap --benchmark -> --correctness). Each entry "
                        "in this list produces a PATH_A_PARTIAL_COVERAGE warning marker in "
                        "the rendered COMMANDMENT."
                    ),
                },
                "notes": {
                    "type": "string",
                    "description": (
                        "Short audit note explaining the Path-A choice (e.g. 'benchmark-only "
                        "command; inferring the other modes by flag substitution')."
                    ),
                },
            },
            "required": ["run_command", "out_path"],
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _make_tool_codebase_explore(
    agent: PreprocessOrchestratorAgent,
    kernel_language: KernelLanguage,
) -> Callable[..., dict[str, Any]]:
    """Bind ``codebase_explore`` to the agent's collected-state mutator.

    The tool stores the produced :class:`CodebaseContext` on the agent's
    private ``_collected`` dict so ``finish_preprocess`` can copy it into
    the final result without the LLM having to re-pass it through the
    final tool call.
    """

    def _impl(
        repo_root: str,
        kernel_path: str,
        out_path: str,
    ) -> dict[str, Any]:
        ctx: CodebaseContext = explore_codebase(
            Path(repo_root),
            Path(kernel_path),
            kernel_language,
            out_path=Path(out_path),
        )
        agent._collected["codebase_context"] = ctx
        return {
            "ok": True,
            "out_path": str(ctx.out_path) if ctx.out_path else None,
            "n_files": len(ctx.files),
            "files_preview": ctx.files[:10],
            "text_length": len(ctx.text),
        }

    return _impl


def _make_tool_translate_to_flydsl(
    agent: PreprocessOrchestratorAgent,
) -> Callable[..., dict[str, Any]]:
    def _impl(source_path: str, output_dir: str) -> dict[str, Any]:
        result: TranslationResult = translate_to_flydsl(
            source_path=Path(source_path),
            output_dir=Path(output_dir),
            gpu_id=agent.config.gpu_id,
            model=agent.model,
            repo=agent.config.repo,
            flydsl_repo=agent.config.flydsl_repo,
        )
        agent._collected["translation"] = result
        return {
            "ok": result.success,
            "translated_kernel_path": str(result.translated_kernel_path) if result.translated_kernel_path else None,
            "speedup": result.speedup,
            "self_review": result.self_review,
            "errors": result.errors,
            "elapsed_s": result.elapsed_s,
        }

    return _impl


def _make_tool_dispatch_subagent(
    agent: PreprocessOrchestratorAgent,
    dispatcher: PreprocessSubagentDispatcher,
) -> Callable[..., dict[str, Any]]:
    def _impl(name: str, task: str, context: dict | None = None) -> dict[str, Any]:
        result = dispatcher(name=name, task=task, model=agent.model, context=context)
        agent._subagent_runs.append(result)
        # Populate orchestrator state from the subagent output where
        # possible — the LLM still needs to call later tools, but we
        # surface the parsed harness path / verifier verdict so it
        # doesn't have to re-derive them.
        output = result.get("output", "") or ""
        if name == "harness-generator":
            for line in output.splitlines():
                if line.startswith("TEST_COMMAND:"):
                    agent._collected.setdefault("test_command", line.split(":", 1)[1].strip())
        if name == "harness-verifier":
            for line in output.splitlines():
                if line.startswith("HARNESS_PATH="):
                    agent._collected["harness_path"] = line.split("=", 1)[1].strip()
        return result

    return _impl


def _make_tool_collect_baseline(
    agent: PreprocessOrchestratorAgent,
) -> Callable[..., dict[str, Any]]:
    def _impl(
        harness_path: str,
        repeats: int = 5,
        work_dir: str | None = None,
        gpu_id: int | None = None,
        **_extra_ignored: Any,
    ) -> dict[str, Any]:
        # Defensive: the orchestrator LLM occasionally invents kwargs the
        # schema does not declare (observed in the field: ``out_path``,
        # ``repo_root``, ``output_dir``). Drop them with a debug-log
        # breadcrumb instead of raising ``TypeError``, which would
        # otherwise abort the whole baseline step.
        if _extra_ignored:
            logger.debug("collect_baseline ignored extra kwargs: %s", list(_extra_ignored))
        baseline: BaselineMetrics = collect_baseline_metrics(
            Path(harness_path),
            repeats=repeats,
            work_dir=Path(work_dir) if work_dir else None,
            gpu_id=gpu_id if gpu_id is not None else agent.config.gpu_id,
        )
        agent._collected["baseline"] = baseline
        return {
            "ok": baseline.success,
            "median_ms": baseline.median_ms,
            "samples_ms": baseline.samples_ms,
            "stdev_ms": baseline.stdev_ms,
            "repeats": baseline.repeats,
            "command": baseline.command,
        }

    return _impl


def _make_tool_collect_profile(
    agent: PreprocessOrchestratorAgent,
) -> Callable[..., dict[str, Any]]:
    def _impl(
        harness_path: str,
        work_dir: str | None = None,
        gpu_id: int | None = None,
        out_path: str | None = None,
        **_extra_ignored: Any,
    ) -> dict[str, Any]:
        # See ``_make_tool_collect_baseline._impl`` — same defensive
        # accept-and-log policy for LLM-invented kwargs.
        if _extra_ignored:
            logger.debug("collect_profile ignored extra kwargs: %s", list(_extra_ignored))
        profile: ProfileResult = collect_profile(
            Path(harness_path),
            work_dir=Path(work_dir) if work_dir else None,
            gpu_id=gpu_id if gpu_id is not None else agent.config.gpu_id,
            out_path=Path(out_path) if out_path else None,
        )
        agent._collected["profile"] = profile
        return {
            "ok": profile.success,
            "command": profile.command,
            "backend": profile.backend,
            "profile_path": str(profile.profile_path) if profile.profile_path else None,
        }

    return _impl


def _make_tool_render_commandment(
    agent: PreprocessOrchestratorAgent,
    kernel_language: KernelLanguage,
) -> Callable[..., dict[str, Any]]:
    def _impl(
        kernel_path: str,
        harness_path: str,
        repo_root: str,
        out_path: str | None = None,
        baseline_metrics: dict | None = None,
        **_extra_ignored: Any,
    ) -> dict[str, Any]:
        if _extra_ignored:
            logger.debug("render_commandment ignored extra kwargs: %s", list(_extra_ignored))

        # ``out_path`` used to be required positionally. When the LLM
        # forgets it (observed in the field), fall back to
        # ``<output_dir>/COMMANDMENT.md`` if the orchestrator's run
        # context carries an ``output_dir``. This is a graceful default,
        # not silent guessing: every Path-B orchestrator run sets
        # ``output_dir`` explicitly in :meth:`PreprocessOrchestratorAgent.run`.
        if not out_path:
            output_dir = (
                agent._extra_template_vars.get("output_dir") if hasattr(agent, "_extra_template_vars") else None
            )
            if not output_dir:
                raise ValueError(
                    "render_commandment: out_path was not provided and no output_dir "
                    "is available on the orchestrator. Pass out_path explicitly."
                )
            out_path = str(Path(output_dir) / "COMMANDMENT.md")

        ctx = CommandmentContext(
            kernel_path=Path(kernel_path),
            harness_path=Path(harness_path),
            repo_root=Path(repo_root),
            baseline_metrics=baseline_metrics,
        )
        text = render_commandment(kernel_language, ctx, out_path=Path(out_path))
        agent._collected["commandment_path"] = out_path
        return {
            "ok": True,
            "out_path": out_path,
            "text_length": len(text),
        }

    return _impl


def _make_tool_commandment_from_user_command(
    agent: PreprocessOrchestratorAgent,
) -> Callable[..., dict[str, Any]]:
    """Bind ``commandment_from_user_command`` — the Path-A short-circuit tool.

    Behaviour:

    1. Validate that ``run_command`` is a non-empty shell command.
       (Empty / whitespace-only commands raise ``ValueError`` so the
       orchestrator's dispatch loop turns the failure into a clear
       ``{"error": ...}`` observation for the LLM to react to.)
    2. Project the single ``run_command`` into the 5-section
       COMMANDMENT structure (``Setup``, ``Correctness``, ``Benchmark``,
       ``Full Benchmark``, ``Profile``) by treating each entry in
       ``modes_covered`` as a direct copy of the command into that
       section's body, each entry in ``inferred_modes`` (and not in
       ``modes_covered``) as the command with a
       ``PATH_A_PARTIAL_COVERAGE`` warning marker prepended, and modes
       in neither list as a bare warning marker (no command).
    3. Write the rendered ``COMMANDMENT.md`` to ``out_path`` via
       :func:`render_commandment_from_sections`.
    4. Record the parsed :class:`RunInstructions` on
       ``agent._collected["run_instructions"]`` for audit + downstream
       consumers, and the commandment path on
       ``agent._collected["commandment_path"]`` so
       ``finish_preprocess`` doesn't have to repeat it.

    Returns:
        ``{ok, commandment_path, modes_emitted, warnings, text_length}``.
    """

    def _impl(
        run_command: str,
        out_path: str,
        modes_covered: list[str] | None = None,
        inferred_modes: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        if not isinstance(run_command, str) or not run_command.strip():
            raise ValueError(
                "commandment_from_user_command: run_command must be a non-empty "
                "shell command. Path A is meaningless without an explicit "
                "user-provided run instruction — if you intended Path B, call "
                "dispatch_subagent('harness-generator', ...) instead."
            )
        if not isinstance(out_path, str) or not out_path.strip():
            raise ValueError(
                "commandment_from_user_command: out_path must be a non-empty "
                "filesystem path for the rendered COMMANDMENT.md."
            )

        cmd = run_command.strip()
        modes_covered_tup = _normalise_modes(modes_covered)
        inferred_modes_tup = _normalise_modes(inferred_modes)
        source_mode = modes_covered_tup[0] if modes_covered_tup else None

        sections: dict[str, str] = {
            "setup": ("# Path-A: user-provided run command assumes the environment is ready.\ntrue"),
        }
        warnings: list[str] = []
        modes_emitted: list[str] = []

        for mode in PATH_A_MODES:
            if mode in modes_covered_tup:
                sections[mode] = cmd
                modes_emitted.append(mode)
            elif mode in inferred_modes_tup:
                src = source_mode or "<unspecified>"
                marker_line = f"# PATH_A_PARTIAL_COVERAGE: {mode} inferred from {src}"
                sections[mode] = f"{marker_line}\n{cmd}"
                warnings.append(f"PATH_A_PARTIAL_COVERAGE: {mode} inferred from {src}")
                modes_emitted.append(mode)
            else:
                sections[mode] = f"# PATH_A_PARTIAL_COVERAGE: {mode} not covered"
                warnings.append(f"PATH_A_PARTIAL_COVERAGE: {mode} not covered")

        preamble_lines = ["<!-- Path-A short-circuit: rendered from user-provided run command -->"]
        preamble_lines.append(f"<!-- raw_command: {cmd} -->")
        if notes:
            preamble_lines.append(f"<!-- notes: {notes} -->")
        preamble = "\n".join(preamble_lines)

        out_path_obj = Path(out_path)
        text = render_commandment_from_sections(
            sections,
            out_path=out_path_obj,
            preamble=preamble,
        )

        instructions = RunInstructions(
            raw_command=cmd,
            modes_covered=modes_covered_tup,
            inferred_modes=inferred_modes_tup,
            notes=notes or "",
        )
        agent._collected["commandment_path"] = str(out_path_obj)
        agent._collected["run_instructions"] = instructions

        return {
            "ok": True,
            "commandment_path": str(out_path_obj),
            "modes_emitted": modes_emitted,
            "warnings": warnings,
            "text_length": len(text),
        }

    return _impl


def _normalise_modes(value: Any) -> tuple[str, ...]:
    """Coerce a possibly-``None`` list of mode strings to a stripped tuple.

    Unknown values (modes not in :data:`PATH_A_MODES`) are dropped with a
    debug log so a noisy LLM doesn't crash the tool; the contract is
    enforced by the schema's enum, this is just defence-in-depth.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        if stripped not in PATH_A_MODES:
            logger.debug("commandment_from_user_command: dropping unknown mode %r", stripped)
            continue
        if stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _make_tool_finish_preprocess(
    agent: PreprocessOrchestratorAgent,
) -> Callable[..., dict[str, Any]]:
    def _impl(
        harness_path: str | None = None,
        commandment_path: str | None = None,
        errors: list | None = None,
        summary: str = "",
    ) -> dict[str, Any]:
        if harness_path:
            agent._collected["harness_path"] = harness_path
        if commandment_path:
            agent._collected["commandment_path"] = commandment_path
        payload = {
            "harness_path": harness_path,
            "commandment_path": commandment_path,
            "errors": list(errors or []),
            "summary": summary,
        }
        agent._finish_payload = payload
        raise FinishedSuccessfully(payload)

    return _impl


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_default_tools(
    agent: PreprocessOrchestratorAgent,
    *,
    kernel_language: KernelLanguage,
    registry: SubagentRegistry | None = None,
    dispatcher: PreprocessSubagentDispatcher | None = None,
) -> None:
    """Register all 7 v3 preprocess tools on ``agent``.

    Args:
        agent:
            Target agent. Tools are bound to its ``_collected`` /
            ``_subagent_runs`` mutable state so the eventual
            :class:`PreprocessResult` carries every artifact.
        kernel_language:
            Language resolved by step 0b. Threaded into
            ``codebase_explore`` and ``render_commandment``.
        registry:
            v3 :class:`SubagentRegistry`. Defaults to a fresh
            registry that points at the in-repo
            ``subagents/preprocess/`` tree.
        dispatcher:
            Pre-built :class:`PreprocessSubagentDispatcher`. When
            ``None``, one is constructed from ``registry``. Tests
            inject a stub dispatcher here to keep
            ``dispatch_subagent`` deterministic.
    """
    registry = registry or SubagentRegistry()
    dispatcher = dispatcher or PreprocessSubagentDispatcher(registry, kernel_language=kernel_language)

    agent.register_tool(
        "codebase_explore",
        _schema_codebase_explore(),
        _make_tool_codebase_explore(agent, kernel_language),
    )
    agent.register_tool(
        "translate_to_flydsl",
        _schema_translate_to_flydsl(),
        _make_tool_translate_to_flydsl(agent),
    )
    agent.register_tool(
        "dispatch_subagent",
        _schema_dispatch_subagent(),
        _make_tool_dispatch_subagent(agent, dispatcher),
    )
    agent.register_tool(
        "collect_baseline",
        _schema_collect_baseline(),
        _make_tool_collect_baseline(agent),
    )
    agent.register_tool(
        "collect_profile",
        _schema_collect_profile(),
        _make_tool_collect_profile(agent),
    )
    agent.register_tool(
        "render_commandment",
        _schema_render_commandment(),
        _make_tool_render_commandment(agent, kernel_language),
    )
    agent.register_tool(
        "commandment_from_user_command",
        _schema_commandment_from_user_command(),
        _make_tool_commandment_from_user_command(agent),
    )
    agent.register_tool(
        "finish_preprocess",
        _schema_finish_preprocess(),
        _make_tool_finish_preprocess(agent),
    )


# ---------------------------------------------------------------------------
# Schema validation helpers (used by tests; small + dependency-free)
# ---------------------------------------------------------------------------


def validate_call_against_schema(schema: dict[str, Any], args: dict[str, Any]) -> tuple[bool, str]:
    """Lightweight argument validation against an OpenAI tool schema.

    Implements just enough of JSON Schema to catch:

    * missing required fields,
    * top-level type mismatch (object vs other),
    * ``enum`` violations on string properties.

    Returns ``(ok, message)``. Used by the orchestrator tests to ensure
    each tool's schema is well-formed without dragging in a full JSON
    Schema implementation.
    """
    parameters = schema.get("parameters", {}) or {}
    if parameters.get("type") != "object":
        return False, f"schema {schema.get('name')!r}: parameters.type must be 'object'"
    if not isinstance(args, dict):
        return False, f"args must be a dict (got {type(args).__name__})"

    required = parameters.get("required", []) or []
    for key in required:
        if key not in args:
            return False, f"missing required field {key!r}"

    properties = parameters.get("properties", {}) or {}
    for key, value in args.items():
        prop = properties.get(key)
        if not prop:
            continue
        enum = prop.get("enum")
        if enum is not None and value not in enum:
            return False, f"field {key!r} value {value!r} is not in enum {enum}"
    return True, "ok"


__all__ = [
    "ALLOWED_SUBAGENT_NAMES",
    "PATH_A_MODES",
    "PreprocessSubagentDispatcher",
    "RunInstructions",
    "register_default_tools",
    "validate_call_against_schema",
]


def _serialize_json(payload: Any) -> str:
    """Module-level JSON helper kept for symmetry with future tool result handlers."""
    return json.dumps(payload, default=str)
