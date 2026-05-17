"""LLM-driven orchestrator for the v3 preprocess pipeline.

This module ships the agent class + system prompt + result dataclass for
the v3 preprocessing flow. The actual tool registration (the bridges from
LLM tool calls to the deterministic v3 modules and to subagent dispatch)
lands in :mod:`minisweagent.run.preprocess_v3.tools` (next commit) and is
attached here via :meth:`PreprocessOrchestratorAgent.register_tools`.

Design notes
------------

* **Standalone class, no inheritance.** Mirrors
  :class:`minisweagent.agents.optimization_agent.OptimizationAgent`, which
  is itself standalone (the legacy 4-layer chain was collapsed already).
  Inheriting from ``OptimizationAgent`` would drag in the
  ``ToolRuntime`` + bash/save_and_test/submit tool surface, which is
  wrong here — the v3 orchestrator has its own purpose-built tool set
  (codebase_explore, translate_to_flydsl, dispatch_subagent,
  collect_baseline, collect_profile, render_commandment,
  finish_preprocess).

* **Loop shape mirrors ``OptimizationAgent.run()``** — message history,
  ``model.query`` per step, ``parse_action`` -> tool dispatch -> append
  observation -> repeat until termination. Termination is the
  ``finish_preprocess`` tool call (the LLM signals it's done) or
  exceeding ``step_limit`` / ``cost_limit``.

* **Model routing = global default.** Per commit-set decision 4: the AMD
  LLM router is wired globally for all preprocess subagent dispatches.
  v3 YAMLs only specify ``model: <name>``; the orchestrator does not
  override per-subagent at the model layer (the registry's ``model``
  string is recorded for audit only — the actual model the subagent
  executes against is the orchestrator's ``self.model``).

* **No tool registration in this commit.** The agent's ``tools`` list is
  empty until commit 5 wires the 7-tool surface. Constructing the agent
  with no tools is still useful for unit tests that exercise the config
  shape and the system-prompt contract.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from jinja2 import StrictUndefined, Template

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.baseline import BaselineMetrics, ProfileResult
from minisweagent.run.preprocess_v3.explore import CodebaseContext
from minisweagent.run.preprocess_v3.translate import TranslationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (mirroring the OptimizationAgent's primitives)
# ---------------------------------------------------------------------------


class OrchestratorError(Exception):
    """Base error for v3 orchestrator-internal exceptions."""


class FormatError(OrchestratorError):
    """The model produced a response we could not parse."""


class TerminatingException(OrchestratorError):
    """A condition that ends the orchestrator loop cleanly."""


class FinishedSuccessfully(TerminatingException):
    """``finish_preprocess`` was called — the LLM declared completion.

    Carries the parsed result payload as :attr:`payload`.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("finish_preprocess called")
        self.payload = payload


class LimitsExceeded(TerminatingException):
    """The orchestrator hit its step or cost cap."""


# ---------------------------------------------------------------------------
# System prompt (80-180 lines)
# ---------------------------------------------------------------------------

#: System prompt for ``PreprocessOrchestratorAgent``.
#:
#: Encodes the full 6-step preprocess flow plus the routing rules between
#: deterministic tool calls and LLM-driven subagent dispatch. The prompt is
#: a Jinja template; ``render_template`` substitutes the run-time variables
#: (``kernel_path``, ``repo_root``, etc.) so the LLM sees a fully-resolved
#: prompt rather than ``{{...}}`` placeholders.
_SYSTEM_PROMPT_TEMPLATE: str = """\
You are the v3 GEAK Preprocess Orchestrator.

# Your Mission

Drive a kernel-optimization repository through the 6-step v3 preprocess
flow and emit a final ``PreprocessResult`` carrying every artifact path
the downstream optimization loop needs (codebase context, translated
kernel if applicable, harness, baseline metrics, profile, COMMANDMENT.md).

You operate by calling tools. Each tool returns a structured JSON
observation that you read, reason about, and use to decide your next
call. The 7 tools available to you are described below. When you are
done, you call ``finish_preprocess`` to produce the final result.

# Inputs you start with

* ``kernel_path``       — absolute path to the target kernel.
* ``repo_root``         — absolute path to the cloned repository
                          (``baseline/`` tree).
* ``kernel_language``   — pre-detected by step 0b. One of: ``triton``,
                          ``hip``, ``flydsl``, or ``unknown``.
* ``source_language``   — same as ``kernel_language`` initially.
* ``target_language``   — what the optimization loop will work on. For
                          PyTorch -> FlyDSL workflows, this is
                          ``flydsl`` while ``source_language`` is
                          ``triton``/``pytorch``. Otherwise equal to
                          ``source_language`` and step 2 is skipped.
* ``output_dir``        — where artifacts (CODEBASE_CONTEXT.md, harness,
                          baseline_metrics.json, profile.json, and
                          COMMANDMENT.md) must be written.
* ``gpu_id``            — GPU index for harness invocations.

# The 6 steps (must be executed in this order)

## Step 1 — codebase-explore (deterministic)

Call ``codebase_explore`` with ``repo_root`` and ``kernel_path``. The
tool runs a deterministic dependency BFS and writes
``CODEBASE_CONTEXT.md`` to ``output_dir``. Returns the rendered text +
the discovered file list. **Do NOT dispatch a subagent for this** — the
walk is purely AST-based and language-agnostic.

## Step 2 — translate (conditional)

If ``source_language != target_language`` and ``target_language ==
"flydsl"``, call ``translate_to_flydsl`` with the source kernel path
and an output sub-directory. The tool wraps ``run_translation`` (which
itself drives a translation agent under the hood) and returns a typed
projection. On success, the **translated** kernel path becomes the new
``kernel_path`` for downstream steps.

If the languages already match, skip Step 2 entirely. Do not call
``translate_to_flydsl`` defensively — it is not idempotent (each call
spins up the translation agent).

## Step 3 — harness generation + verification (LLM subagents)

Step 3a: Dispatch ``harness-generator`` via ``dispatch_subagent`` with
the kernel path + repo_root + the codebase context's text. The
subagent produces a ``TEST_COMMAND`` and writes a harness file.

Step 3b: Verify by dispatching ``harness-verifier`` via
``dispatch_subagent`` with the harness path + kernel_path + repo_root.
The verifier returns either ``HARNESS_VERIFIED=true`` or a structured
correction directive.

If the verifier rejects, you MAY retry by re-dispatching
``harness-generator`` with the verifier's correction directive as the
new task. **Maximum 3 generator attempts total** — if attempt 3 fails,
record the failure in the final result's ``errors`` list and proceed
with whatever harness was produced (do not deadlock the pipeline).

## Step 4 — baseline + profile (deterministic)

Once the harness is verified:

1. Call ``collect_baseline`` with the harness path. Returns
   ``BaselineMetrics`` (median latency, samples, etc.).
2. Call ``collect_profile`` with the harness path. Returns
   ``ProfileResult`` (kernel-level profiler-mcp output).

Both are deterministic subprocess calls; do not dispatch a subagent.

## Step 5 — speedup-verify (LLM subagent)

Dispatch ``speedup-verify`` via ``dispatch_subagent`` with the baseline
output path + harness path + output_dir. The subagent writes
``compute_speedup.py`` and verifies it parses the baseline cleanly. The
script is consumed by the optimization loop later, not by you.

## Step 6 — render COMMANDMENT.md (deterministic)

Call ``render_commandment`` with ``kernel_language``, kernel_path,
harness_path, repo_root, and the baseline metrics dict. The tool
renders ``COMMANDMENT.md`` to ``output_dir`` via Jinja (or a language-
specific Python generator fallback).

## Final — finish_preprocess

Once all 6 steps succeeded, call ``finish_preprocess`` with the full
artifact bundle. Provide:

* every artifact path you produced,
* the ``BaselineMetrics`` and ``ProfileResult`` payloads,
* the ``CodebaseContext`` text,
* the ``TranslationResult`` (or ``null`` if step 2 was skipped),
* a list of subagent run summaries (you can copy these from each
  ``dispatch_subagent`` return),
* the cumulative ``errors`` list (empty on a clean run).

After ``finish_preprocess`` returns, the orchestrator loop terminates.
There is no further turn.

# Routing rules (decision tree)

1. **Codebase-explore is deterministic.** Call ``codebase_explore``
   directly. Never dispatch a subagent for it.
2. **Translation is a tool call**, not a subagent dispatch. The legacy
   ``run_translation`` function drives its own LLM agent — your job is
   to call ``translate_to_flydsl`` once and read the result.
3. **Harness generation and verification are subagents.** Use
   ``dispatch_subagent`` with subagent name == ``harness-generator``
   or ``harness-verifier``. Allowed retries: up to 3 generator attempts
   before recording the failure and proceeding.
4. **Baseline + profile are deterministic.** Two separate tool calls;
   never dispatch.
5. **Speedup-verify is a subagent.** Dispatch only after baseline
   metrics exist (the subagent reads ``baseline_metrics.json`` and the
   harness file).
6. **COMMANDMENT renders last, deterministically.** Pass every artifact
   path you collected so the template has full context.

# Output discipline

* Every tool you call returns a JSON observation. Read it before the
  next call.
* If a tool fails (returns an error), **do not silently retry** — read
  the error message, decide whether the failure is recoverable
  (e.g. transient subprocess flake) or terminal (e.g. missing
  ``repo_root``), and only retry recoverable failures.
* If you exhaust the harness-generator retry budget, record the
  failure in your scratch reasoning and proceed; the final
  ``finish_preprocess`` carries the partial results.
* Do **not** invent tool calls outside the 7 listed. Do **not** call
  shell or write files directly — every artifact has a deterministic
  tool that owns its writing.

# Available tools (full list)

1. ``codebase_explore`` — deterministic; step 1.
2. ``translate_to_flydsl`` — deterministic-from-orchestrator; step 2 (conditional).
3. ``dispatch_subagent`` — LLM dispatch; steps 3a, 3b, 5. The ``name``
   argument must be one of ``harness-generator``, ``harness-verifier``,
   ``speedup-verify``.
4. ``collect_baseline`` — deterministic; step 4 part 1.
5. ``collect_profile`` — deterministic; step 4 part 2.
6. ``render_commandment`` — deterministic; step 6.
7. ``finish_preprocess`` — completion sentinel; terminates the loop.

Begin with step 1 (``codebase_explore``).
"""


# ---------------------------------------------------------------------------
# Config + Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessOrchestratorConfig:
    """Configuration for :class:`PreprocessOrchestratorAgent`.

    Mirrors :class:`minisweagent.agents.optimization_agent.AgentConfig`'s
    shape where the fields apply, with v3-only additions for the
    deterministic-tool surface (``gpu_id``, ``repo``, ``flydsl_repo``).

    Attributes:
        model:
            Model name (string identifier the AMD router resolves to a
            concrete model). The actual model instance is supplied at
            agent construction time; this field exists for audit /
            template rendering, not for routing.
        model_class:
            Model-class identifier. Defaults to ``"amd_llm"`` per
            commit-set decision 4 (global AMD-router routing for all
            preprocess subagents).
        step_limit:
            Maximum number of LLM turns the orchestrator will run. ``0``
            disables the limit (matches the convention from
            ``OptimizationAgent.AgentConfig``). Defaults to 200 — enough
            for the 6-step flow plus retries, well below any per-run
            cost ceiling.
        cost_limit:
            Maximum cumulative cost in USD. ``0.0`` disables the limit.
            Defaults to ``0.0`` because the orchestrator-side cost is
            small (it's the subagents that spend tokens) and the global
            budget is owned by the surrounding pipeline.
        gpu_id:
            ``HIP_VISIBLE_DEVICES`` value forwarded to baseline /
            profile / translate calls.
        repo:
            Path to a parent repo, threaded through to
            ``translate_to_flydsl``.
        flydsl_repo:
            Optional path to a local FlyDSL clone for
            ``translate_to_flydsl``'s KB loader.
        system_template:
            Jinja template for the system prompt. Defaults to the
            module-level :data:`_SYSTEM_PROMPT_TEMPLATE`.
        instance_template:
            Jinja template for the initial user message. Receives the
            ``task`` and the orchestration context vars.
    """

    model: str = "amd-llm-router"
    model_class: str = "amd_llm"
    step_limit: int = 200
    cost_limit: float = 0.0
    gpu_id: int = 0
    repo: Path | None = None
    flydsl_repo: Path | None = None
    system_template: str = _SYSTEM_PROMPT_TEMPLATE
    instance_template: str = (
        "Begin the v3 preprocess flow.\n\n"
        "## Inputs\n"
        "- kernel_path: {{kernel_path}}\n"
        "- repo_root: {{repo_root}}\n"
        "- kernel_language: {{kernel_language}}\n"
        "- source_language: {{source_language}}\n"
        "- target_language: {{target_language}}\n"
        "- output_dir: {{output_dir}}\n"
        "- gpu_id: {{gpu_id}}\n\n"
        "## Task\n"
        "{{task}}\n\n"
        "Execute the 6 steps in order. Begin with codebase_explore."
    )


@dataclass(frozen=True)
class PreprocessResult:
    """Final result emitted by ``finish_preprocess``.

    Mirrors the artifact bundle the v3 orchestrator hands off to the
    optimization loop. Fields are typed instead of free-form so the
    downstream consumer does not have to dict-walk.

    Attributes:
        success:
            ``True`` when every required step produced an artifact and
            no fatal error was recorded.
        kernel_language:
            The :class:`KernelLanguage` resolved by step 0b. Carried
            forward so downstream callers don't need to re-detect.
        kernel_path:
            Path to the kernel after step 2 (the translated FlyDSL
            file when translation ran, otherwise the original).
        harness_path:
            Path to the verified harness from step 3, or ``None`` when
            harness generation failed and ``success`` is ``False``.
        baseline:
            ``BaselineMetrics`` from step 4 part 1, or ``None`` when
            collection failed.
        profile:
            ``ProfileResult`` from step 4 part 2, or ``None`` when the
            profiler was unavailable.
        codebase_context:
            ``CodebaseContext`` from step 1.
        commandment_path:
            Path to ``COMMANDMENT.md`` from step 6, or ``None`` on a
            partial run.
        translation:
            ``TranslationResult`` from step 2, or ``None`` when step 2
            was skipped (source language already == target language).
        subagent_runs:
            One dict per ``dispatch_subagent`` call summarising name,
            success, and short output. Order matches dispatch order.
        tool_calls:
            One dict per dispatched tool call (every entry in the LLM
            loop's tool history, not just subagent dispatches). Each
            entry carries ``{name, args}``. This is the canonical
            audit trail for which tools the LLM actually invoked —
            ``path_taken`` is derived from it.
        path_taken:
            Which structural path the orchestrator followed.

            * ``"A"`` — Path-A short-circuit: the LLM called
              ``commandment_from_user_command`` (the user's task
              prompt carried explicit run instructions). The
              harness-generator / harness-verifier / speedup-verify
              subagents are skipped on this path.
            * ``"B"`` — Path-B / standard 6-step flow (the default
              for legacy / descriptive task prompts).

            Computed from ``tool_calls`` in :meth:`_build_result`;
            never set directly by callers.
        elapsed_s:
            Wall-clock seconds spent inside ``run()``.
        errors:
            Cumulative error strings collected across all steps. Empty
            on a clean success.
    """

    success: bool
    kernel_language: KernelLanguage
    kernel_path: Path
    harness_path: Path | None = None
    baseline: BaselineMetrics | None = None
    profile: ProfileResult | None = None
    codebase_context: CodebaseContext | None = None
    commandment_path: Path | None = None
    translation: TranslationResult | None = None
    subagent_runs: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    path_taken: Literal["A", "B"] = "B"
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool registration shim
#
# The actual tool implementations live in
# :mod:`minisweagent.run.preprocess_v3.tools` (commit 5). This skeleton
# exposes the registration hook so unit tests can construct the agent
# with an empty (or mocked) tool table.
# ---------------------------------------------------------------------------


@dataclass
class ToolEntry:
    """One LLM-callable tool: its OpenAI-style schema + a Python callable.

    The orchestrator's dispatch loop:

    1. Reads ``model.query(messages)`` and looks for a tool call.
    2. Dispatches by ``name`` into ``self._tools[name]``.
    3. Appends the result to message history as a ``role="tool"`` message.

    Schemas conform to the OpenAI function-tool format (a ``parameters``
    JSON Schema object) so ``model.set_tools(...)`` can pass them
    through to LiteLLM unchanged.
    """

    schema: dict[str, Any]
    callable: Any  # Callable[[**kwargs], dict[str, Any]] but typing this is awkward.


# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------


class PreprocessOrchestratorAgent:
    """Standalone LLM-driven orchestrator for the v3 preprocess flow.

    Construction:

        agent = PreprocessOrchestratorAgent(
            model=...,
            config=PreprocessOrchestratorConfig(...),
        )

    The agent's tool table is empty by default; call
    :meth:`register_tool` to wire individual tools, or
    :func:`register_default_tools` (commit 5) to wire the full 7-tool
    surface against an existing v3 module set.

    Public API:

    * :meth:`run(task: str, **context) -> PreprocessResult` — entry
      point. Drives the LLM loop until ``finish_preprocess`` is called
      (or limits are exceeded), then returns the parsed result.
    * :meth:`register_tool(name, schema, callable_)` — register one
      tool. The schema must declare ``name`` matching the registry key.
    * :attr:`messages` — message history for tests / debugging.
    """

    def __init__(
        self,
        model: Any,
        *,
        config: PreprocessOrchestratorConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PreprocessOrchestratorConfig()
        self.messages: list[dict[str, Any]] = []
        self._tools: dict[str, ToolEntry] = {}
        self._extra_template_vars: dict[str, Any] = {}
        self._subagent_runs: list[dict[str, Any]] = []
        # Generic tool-call audit log — populated by ``_dispatch_tool`` for
        # every LLM-invoked tool. Distinct from ``_subagent_runs`` (which
        # only captures ``dispatch_subagent`` outputs). Used to compute
        # ``PreprocessResult.path_taken`` (Path A iff the LLM called
        # ``commandment_from_user_command`` at least once).
        self._tool_calls: list[dict[str, Any]] = []
        self._collected: dict[str, Any] = {}
        self._finish_payload: dict[str, Any] | None = None

    # -----------------------------------------------------------------
    # Tool registration
    # -----------------------------------------------------------------

    def register_tool(self, name: str, schema: dict[str, Any], callable_: Any) -> None:
        """Register one tool with an OpenAI-style schema and a Python callable.

        The callable must accept the schema's parameters as keyword
        arguments and return a dict (the orchestrator JSON-serialises
        the return value into the ``role="tool"`` message). Raising is
        allowed — exceptions become ``{"error": str(exc)}`` observations
        for the LLM to recover from.
        """
        if not isinstance(schema, dict) or schema.get("name") != name:
            raise ValueError(f"register_tool: schema must be a dict with name=={name!r} (got {schema!r})")
        self._tools[name] = ToolEntry(schema=schema, callable=callable_)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tool schemas (for ``model.set_tools``)."""
        return [t.schema for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of registered tool names. Stable for tests."""
        return sorted(self._tools)

    # -----------------------------------------------------------------
    # Template rendering + message history
    # -----------------------------------------------------------------

    def render_template(self, template: str, **kwargs: Any) -> str:
        """Render a Jinja template with config + context vars merged in."""
        config_vars: dict[str, Any] = asdict(self.config)
        all_vars = {**config_vars, **self._extra_template_vars, **kwargs}
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        """Append a message to the history. ``extra`` survives onto the dict."""
        self.messages.append({"role": role, "content": content, **extra})

    # -----------------------------------------------------------------
    # Step loop
    # -----------------------------------------------------------------

    def step(self) -> dict[str, Any]:
        """Run one LLM turn: query model, parse + dispatch, return observation."""
        n_calls = int(getattr(self.model, "n_calls", 0) or 0)
        cost = float(getattr(self.model, "cost", 0.0) or 0.0)
        if 0 < self.config.step_limit <= n_calls:
            raise LimitsExceeded(f"step_limit reached: {n_calls} >= {self.config.step_limit}")
        if 0 < self.config.cost_limit <= cost:
            raise LimitsExceeded(f"cost_limit reached: ${cost:.2f} >= ${self.config.cost_limit:.2f}")

        response = self.model.query(self.messages)
        return self._handle_response(response)

    def _handle_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Append the assistant turn, dispatch any tool call, return observation."""
        content = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        msg_kwargs: dict[str, Any] = {}
        if tool_call:
            msg_kwargs["tool_calls"] = tool_call
        self.add_message("assistant", content, **msg_kwargs)

        if not tool_call:
            raise FormatError(
                "Orchestrator response had no tool call. The orchestrator must "
                "drive every step via tool calls — text-only responses are not "
                "actionable. Please call one of the registered tools."
            )

        function = tool_call.get("function") or {}
        name = function.get("name", "")
        raw_args = function.get("arguments", {})
        args = self._parse_args(raw_args)

        result = self._dispatch_tool(name, args)
        self.add_message(
            "tool",
            json.dumps(result, default=str),
            tool_call_id=tool_call.get("id", ""),
            name=name,
        )
        return result

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        """Coerce tool-call arguments to a dict (LiteLLM may pass them as a JSON string)."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Tool args were not valid JSON; using empty dict (raw=%r)", raw[:200])
                return {}
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Tool args parsed to non-dict (%s); using empty dict", type(parsed).__name__)
            return {}
        logger.warning("Tool args were neither dict nor str (%s); using empty dict", type(raw).__name__)
        return {}

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Look up the tool by name and invoke it.

        Every dispatch is appended to ``self._tool_calls`` as a
        ``{name, args}`` entry — including dispatches to unknown tools
        and dispatches that raise — so the eventual
        :class:`PreprocessResult.tool_calls` audit trail reflects what
        the LLM actually tried, not what worked. The
        :class:`PreprocessResult.path_taken` flag reads this list to
        decide A vs B.
        """
        if not name:
            return {"error": "Tool call missing 'name' field"}
        self._tool_calls.append({"name": name, "args": dict(args)})
        if name not in self._tools:
            return {
                "error": (
                    f"Unknown tool {name!r}. Available tools: {', '.join(sorted(self._tools)) or '<none registered>'}"
                ),
            }
        try:
            result = self._tools[name].callable(**args)
        except FinishedSuccessfully:
            raise
        except Exception as exc:
            logger.exception("Tool %r raised", name)
            return {"error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(result, dict):
            return {"output": str(result)}
        return result

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def run(self, task: str, **context: Any) -> PreprocessResult:
        """Drive the LLM loop until ``finish_preprocess`` (or limits) terminate.

        Args:
            task:
                Free-form task description rendered into the instance
                template's ``{{task}}`` slot.
            **context:
                Run-time variables for the template (``kernel_path``,
                ``repo_root``, ``kernel_language``, ``source_language``,
                ``target_language``, ``output_dir``).

        Returns:
            A :class:`PreprocessResult` projecting the artifact bundle
            ``finish_preprocess`` produced. On limit-exceeded or fatal
            error, returns a partial result with ``success=False`` and
            populated ``errors``.
        """
        self.messages = []
        self._extra_template_vars = {"task": task, **_stringify_paths(context)}
        self._subagent_runs = []
        self._tool_calls = []
        self._collected = {}
        self._finish_payload = None

        self._inject_tools_into_model()

        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        t0 = time.monotonic()
        finish_payload: dict[str, Any] | None = None
        errors: list[str] = []

        try:
            while True:
                try:
                    self.step()
                except FinishedSuccessfully as fin:
                    finish_payload = fin.payload
                    break
                except FormatError as fmt:
                    self.add_message("user", str(fmt))
                except LimitsExceeded as lim:
                    errors.append(str(lim))
                    break
                except TerminatingException as term:
                    errors.append(str(term))
                    break
        except Exception as exc:
            logger.exception("Orchestrator run() crashed")
            errors.append(f"{type(exc).__name__}: {exc}")

        elapsed_s = round(time.monotonic() - t0, 3)
        return self._build_result(
            context=context,
            finish_payload=finish_payload,
            errors=errors,
            elapsed_s=elapsed_s,
        )

    # -----------------------------------------------------------------
    # Result assembly
    # -----------------------------------------------------------------

    def _build_result(
        self,
        *,
        context: dict[str, Any],
        finish_payload: dict[str, Any] | None,
        errors: list[str],
        elapsed_s: float,
    ) -> PreprocessResult:
        """Assemble the :class:`PreprocessResult` from finish payload + context.

        Success semantics (commit set 5a — fix for open question #7):

        * Loop-level ``errors`` (e.g. ``LimitsExceeded``) always invalidate
          ``success``.
        * ``finish_preprocess(errors=[...])`` — i.e. the LLM gracefully
          gave up after exhausting the harness-verifier retry budget — is
          treated as failure: the per-step errors are folded into
          ``result.errors`` and ``success`` is ``False``.
        * Missing required artifacts (``harness_path`` or ``baseline``)
          downgrade a nominally clean run to ``success=False`` too — a
          finish payload with neither is meaningless to the downstream
          pipeline.

        Path-A vs Path-B (commit set 7):

        * ``path_taken`` is computed from ``self._tool_calls``: Path A
          iff the LLM called ``commandment_from_user_command`` at least
          once during the run. Otherwise Path B (the default 6-step
          flow).
        * Success criteria branch on ``path_taken``: Path A requires a
          ``commandment_path`` artifact but does NOT require a
          ``harness_path`` (the harness is the user's command, not a
          file we own). Path B keeps the existing strict criteria.
        """
        kernel_language = context.get("kernel_language")
        kernel_path = _coerce_path(context.get("kernel_path")) or Path()

        collected = self._collected
        merged_errors = list(errors)
        if finish_payload is not None:
            for err in finish_payload.get("errors") or []:
                if err and err not in merged_errors:
                    merged_errors.append(str(err))

        translated_path = None
        translation: TranslationResult | None = collected.get("translation")
        if translation is not None and translation.translated_kernel_path is not None:
            translated_path = translation.translated_kernel_path

        harness_path = _coerce_path(collected.get("harness_path"))
        baseline = collected.get("baseline")
        commandment_path = _coerce_path(collected.get("commandment_path"))
        path_taken = _derive_path_taken(self._tool_calls)
        success = self._finalize_success(
            finish_payload=finish_payload,
            errors=merged_errors,
            harness_path=harness_path,
            baseline=baseline,
            path_taken=path_taken,
            commandment_path=commandment_path,
        )

        return PreprocessResult(
            success=success,
            kernel_language=kernel_language,  # type: ignore[arg-type]
            kernel_path=translated_path or kernel_path,
            harness_path=harness_path,
            baseline=baseline,
            profile=collected.get("profile"),
            codebase_context=collected.get("codebase_context"),
            commandment_path=commandment_path,
            translation=translation,
            subagent_runs=list(self._subagent_runs),
            tool_calls=list(self._tool_calls),
            path_taken=path_taken,
            elapsed_s=elapsed_s,
            errors=merged_errors,
        )

    @staticmethod
    def _finalize_success(
        *,
        finish_payload: dict[str, Any] | None,
        errors: list[str],
        harness_path: Path | None,
        baseline: Any,
        path_taken: Literal["A", "B"] = "B",
        commandment_path: Path | None = None,
    ) -> bool:
        """Compute the final ``success`` flag for the :class:`PreprocessResult`.

        Universal preconditions (both paths):

        * ``finish_preprocess`` was actually called (so we have a payload).
        * No accumulated ``errors`` (loop-level or finish-payload-level).

        Path-specific criteria:

        * **Path A** — the harness IS the user's run command, not a file
          artifact we own; ``harness_path`` may be ``None``. Required
          artifact is ``commandment_path`` (the rendered COMMANDMENT.md
          that captures the user's command). ``baseline`` is helpful but
          not required (the LLM may opt to skip it on a Path-A run if
          the user only asked for a one-shot command).
        * **Path B** — existing strict criteria preserved from commit
          set 5a: ``harness_path`` AND ``baseline`` must both be
          populated. Missing either downgrades the run to
          ``success=False`` because the downstream optimisation loop
          ``KeyError``s on the missing artefact.

        Any other combination is a partial / failed run — the rest of
        the pipeline will refuse to proceed without these artefacts,
        so surface the failure early instead of letting downstream
        crash.
        """
        if finish_payload is None:
            return False
        if errors:
            return False
        if path_taken == "A":
            return commandment_path is not None
        # Path B (default).
        if harness_path is None:
            return False
        if baseline is None:
            return False
        return True

    # -----------------------------------------------------------------
    # Model wiring
    # -----------------------------------------------------------------

    def _inject_tools_into_model(self) -> None:
        """Mirror ``OptimizationAgent``: tell the model which tools it can call."""
        schemas = self.get_tool_schemas()
        if hasattr(self.model, "set_tools"):
            try:
                self.model.set_tools(schemas)
                return
            except Exception as exc:
                logger.debug("model.set_tools failed: %s; falling back to attribute set", exc)
        impl = getattr(self.model, "_impl", self.model)
        try:
            impl.tools = schemas
        except Exception as exc:
            logger.debug("model.tools attribute set failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Name of the tool whose presence in the LLM's tool-call history toggles
#: the orchestrator from Path B (6-step flow) to Path A (short-circuit).
#: Defined here (not in ``tools.py``) so the orchestrator can derive
#: ``path_taken`` without importing ``tools.py`` (which would create a
#: circular dependency).
_PATH_A_TRIGGER_TOOL_NAME: str = "commandment_from_user_command"


def _derive_path_taken(tool_calls: list[dict[str, Any]]) -> Literal["A", "B"]:
    """Return ``"A"`` iff the LLM called the Path-A short-circuit tool.

    The decision is purely structural: if
    ``commandment_from_user_command`` appears anywhere in the recorded
    tool-call audit log, the run is Path A; otherwise Path B. This is
    intentionally LLM-driven (the LLM picks the tool based on Step 0 of
    the system prompt), not regex-based on the task prompt.
    """
    for call in tool_calls:
        if call.get("name") == _PATH_A_TRIGGER_TOOL_NAME:
            return "A"
    return "B"


_PATH_KEY_RE = re.compile(r"(?:^|_)(path|dir|root|repo)$")


def _stringify_paths(context: dict[str, Any]) -> dict[str, Any]:
    """Coerce Path values in context to strings for Jinja rendering.

    StrictUndefined doesn't like ``Path.parent`` etc. inside templates,
    and we want path keys to render as plain strings consistently.
    """
    out: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif _PATH_KEY_RE.search(key) and value is not None:
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _coerce_path(value: Any) -> Path | None:
    """Best-effort string -> Path coercion preserving ``None``."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


__all__ = [
    "FinishedSuccessfully",
    "FormatError",
    "LimitsExceeded",
    "OrchestratorError",
    "PreprocessOrchestratorAgent",
    "PreprocessOrchestratorConfig",
    "PreprocessResult",
    "TerminatingException",
    "ToolEntry",
]
