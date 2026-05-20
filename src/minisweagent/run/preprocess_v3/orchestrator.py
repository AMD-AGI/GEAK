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
  (run_discovery, codebase_explore, translate_to_flydsl, dispatch_subagent,
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

Drive a kernel-optimization repository through the v3 preprocess flow and emit a final ``PreprocessResult`` carrying every artifact path the downstream optimization loop needs (codebase context, translated kernel if applicable, harness, baseline metrics, profile, COMMANDMENT.md).

You operate by calling tools. Each tool returns a structured JSON observation that you read, reason about, and use to decide your next call. When you are done, call ``finish_preprocess`` to produce the final result.

## Step 0 — classify the user's task

Before any other action, read the task prompt and classify it into exactly ONE of these cases. The classification is based only on the user's task text, not on YAML test-plan metadata, discovery output, or file names you happen to see later.

**Case A — user provided explicit run instructions / commands.**
Indicators: a literal command-line invocation (``python <script>``, ``pytest ... -k ...``, ``make ...``, shell script, existing custom harness command). The command is opaque: it may NOT support GEAK's four harness flags.

Action: run ``run_discovery`` because it is the standard cheap deterministic front step, then call ``commandment_from_user_command`` with the extracted user command. Discovery/ATD is IRRELEVANT for this case: do not inspect it to alter the user command, and do not generate a harness.

**Important exception**: if the "Hints from the call site" section says the harness is **pre-validated** and supports the four standard modes (``--correctness``, ``--benchmark``, ``--full-benchmark``, ``--profile``), you MUST list all four modes in ``modes_covered`` when calling ``commandment_from_user_command``. The tool will substitute the correct flag for each COMMANDMENT section automatically. Do NOT put all modes in ``inferred_modes`` — use ``modes_covered``.

**After ``commandment_from_user_command`` succeeds**: if the return value includes a ``harness_path`` (i.e. the command references a standard harness file), call ``collect_baseline(harness_path=<path>)`` and ``collect_profile(harness_path=<path>)`` before calling ``finish_preprocess``. These are fast deterministic subprocess calls (~30-60s total) and their output is required for downstream verified-speedup evaluation. If either call fails, proceed anyway — record the failure and call ``finish_preprocess``.

**Case B — user provided explicit shapes/configs but no runnable command.**
Indicators: the task names exact shapes, dims, dtype/config tuples, model-production configs, or says "use only this shape/config". The user's shapes/configs are authoritative.

Action: run ``run_discovery`` because it is the standard cheap deterministic front step, but discovery/ATD is IRRELEVANT for this case. Dispatch ``harness-generator`` with the user-provided shapes/configs. Do not pass, inspect, or rely on ``DISCOVERY_CONTEXT.md``. The generated harness must use ONLY the user-provided shapes/configs while still satisfying the universal four-mode harness contract.

**Case C — normal descriptive task, no command and no explicit shapes/configs.**
This is the default Path B. The user is asking to optimize a kernel but did not supply run commands or shape/config overrides.

Action: run ``run_discovery`` and use its legacy ATD output as authoritative. The top relevant benchmark/test from ``discovery.json`` / ``DISCOVERY_CONTEXT.md`` determines the source file, exact shapes/configs, ordering, reference logic, tolerances, dtype/device/layout, and helper semantics. The generator must not invent representative shapes while discovery found a relevant source.

# Inputs you start with

* ``kernel_path``, ``repo_root``, ``kernel_language``, ``source_language``, ``target_language``, ``output_dir``, ``gpu_id``.

# Common deterministic step

## Step 1 — legacy discovery front half (deterministic, all cases)

Call ``run_discovery`` with ``repo_root``, ``kernel_path``, and ``output_dir``. This reuses the legacy deterministic discovery pipeline: resolve kernel/repo, write ``CODEBASE_CONTEXT.md``, run automated-test-discovery, write ``discovery.json``, and write ``DISCOVERY_CONTEXT.md`` with the legacy UTA ``FILES YOU MUST READ`` block.

Important: discovery/ATD has only two states: AUTHORITATIVE or IRRELEVANT. It is AUTHORITATIVE only in Case C. It is IRRELEVANT in Case A and Case B. Never treat discovery as an auxiliary hint in Case A or B.

## Step 2 — translate (conditional)

If ``source_language != target_language`` and ``target_language == "flydsl"``, call ``translate_to_flydsl`` once. The translated kernel path becomes the new ``kernel_path`` for downstream steps. Otherwise skip.

## Step 3 — harness generation + verification (LLM subagents)

**Step 3a.** Dispatch ``harness-generator`` via ``dispatch_subagent`` only for Case B or Case C.

For Case B (user shape/config override): pass ``kernel_path``, ``repo_root``, and the exact user-provided shapes/configs. Do NOT pass ``discovery_context_path``.

For Case C (normal Path B): pass ``kernel_path``, ``repo_root``, ``codebase_context_path``, and ``discovery_context_path`` from Step 1. The dispatcher inlines those files for the subagent, and the discovery context is authoritative.

The subagent's final message contains exactly one structured line:

    HARNESS_PATH: <absolute path to the harness file>

That is the ENTIRE handoff. The harness owns the four CLI modes (``--correctness``, ``--profile``, ``--benchmark``, ``--full-benchmark``) and any build step it needs (e.g. a HIP harness invokes ``make`` internally). Do NOT look for or parse ``COMMANDMENT_<MODE>:`` lines — they no longer exist in the contract; deterministic Python renders the COMMANDMENT in Step 5 from ``harness_path`` alone.

The dispatcher auto-populates ``harness_path`` for you from the subagent's output, so you do not need to extract it manually.

**Step 3b.** Verify by dispatching ``harness-verifier`` via ``dispatch_subagent``. Pass only ``harness_path`` in the context. The verifier runs the four CLI modes against the harness directly and returns either ``HARNESS_VERIFIED=true`` (possibly after applying its own in-place mechanical fixes) or a structured correction directive.

If the verifier rejects after its own repair loop, re-dispatch ``harness-generator`` with the existing ``harness_path`` and the verifier's exact evidence/correction directive. This is a repair pass over the current harness, not a from-scratch rewrite: the generator wrote the code and has the most context, so it owns the first broad fix. After the generator edits or replaces the harness, dispatch ``harness-verifier`` again. **Maximum 3 generator attempts total** — if attempt 3 fails, record the failure in the final result's ``errors`` list and proceed.

## Step 4 — baseline + profile (deterministic)

Once the harness is verified:

1. Call ``collect_baseline`` with ``harness_path``. Returns ``BaselineMetrics``.
2. Call ``collect_profile`` with ``harness_path``. Returns ``ProfileResult``.

Both are deterministic subprocess calls; do not dispatch a subagent.

## Step 5 — render COMMANDMENT.md (deterministic)

Call ``render_commandment`` with these arguments ONLY:

* ``kernel_path``
* ``harness_path``
* ``repo_root``
* ``baseline_metrics`` (optional — the dict returned by ``collect_baseline``)

The tool renders the canonical 5-section COMMANDMENT.md (Setup / Correctness / Benchmark / Full Benchmark / Profile) by template substitution from those inputs alone. Do NOT pass any ``correctness_command``, ``benchmark_command``, ``full_benchmark_command``, ``profile_command``, ``compile_command``, ``out_path``, or ``output_dir`` arguments — the four mode commands are derived deterministically from ``harness_path`` by the renderer.

## Final — finish_preprocess

Call ``finish_preprocess`` with no arguments (or at most an optional one-paragraph ``summary``). All artifacts are already recorded on the orchestrator from the deterministic tool calls above; the finish call is purely a completion sentinel. Do NOT pass ``harness_path``, ``commandment_path``, ``output_dir``, ``kernel_path``, ``artifacts``, ``codebase_context_path``, or ``path_taken`` — those fields are derived from the orchestrator state.

After ``finish_preprocess`` returns, the orchestrator loop terminates.

# Output discipline

* Every tool call returns a JSON observation. Read it before the next call.
* If a tool fails, decide whether the failure is recoverable; don't silently retry on terminal failures.
* If you exhaust the harness-generator retry budget, record the failure and proceed; ``finish_preprocess`` carries partial results.
* Do not invent tool calls outside the tools listed below.

# Available tools

1. ``run_discovery`` — deterministic legacy discovery front half; always runs first, but its output is authoritative only in Case C and irrelevant in Cases A/B.
2. ``codebase_explore`` — deterministic legacy codebase context only; compatibility fallback if ``run_discovery`` fails before writing context.
3. ``translate_to_flydsl`` — deterministic; step 2 (conditional, Path B only).
4. ``dispatch_subagent`` — LLM dispatch. ``name`` argument must be ``harness-generator`` or ``harness-verifier``. Path B steps 3a and 3b only.
5. ``collect_baseline`` — deterministic; step 4 (Path A when harness is available, and Path B).
6. ``collect_profile`` — deterministic; step 4 (Path A when harness is available, and Path B).
7. ``render_commandment`` — deterministic; Path B step 5.
8. ``commandment_from_user_command`` — Path A short-circuit. Mutually exclusive with ``dispatch_subagent("harness-generator", ...)``.
9. ``finish_preprocess`` — completion sentinel; terminates the loop only when final invariants pass.

Begin by classifying the task into Case A, B, or C, then call ``run_discovery``. After discovery, follow the case-specific action above.
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
        "Classify the task into Case A, B, or C, then begin with run_discovery."
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
              harness-generator / harness-verifier subagents are
              skipped on this path.
            * ``"B"`` — Path-B / standard 5-step flow (the default
              for descriptive task prompts).

            Computed from ``tool_calls`` in :meth:`_build_result`;
            never set directly by callers.
        elapsed_s:
            Wall-clock seconds spent inside ``run()``.
        errors:
            Cumulative error strings collected across all steps. Empty
            on a clean success. On Path A, ``collect_baseline`` /
            ``collect_profile`` failures are NOT included here (they
            land on :attr:`warnings` instead — see ``_finalize_success``
            for the rationale).
        warnings:
            Non-fatal issues that should not invalidate ``success`` but
            are worth surfacing to the operator. Populated on Path A
            when the orchestrator LLM reported a ``collect_baseline`` /
            ``collect_profile`` failure: the harness on Path A is the
            user's own command, so baseline / profile failures are
            informational rather than disqualifying. Always empty on
            Path B (Path B keeps the strict legacy criteria — any error
            invalidates ``success``).
    """

    success: bool
    kernel_language: KernelLanguage
    kernel_path: Path
    harness_path: Path | None = None
    baseline: BaselineMetrics | None = None
    full_benchmark_stdout: str | None = None
    profile: ProfileResult | None = None
    codebase_context: CodebaseContext | None = None
    commandment_path: Path | None = None
    translation: TranslationResult | None = None
    subagent_runs: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    path_taken: Literal["A", "B"] = "B"
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
        real_errors, warnings = self._partition_errors_by_path(merged_errors, path_taken)
        success = self._finalize_success(
            finish_payload=finish_payload,
            errors=real_errors,
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
            full_benchmark_stdout=collected.get("full_benchmark_stdout"),
            profile=collected.get("profile"),
            codebase_context=collected.get("codebase_context"),
            commandment_path=commandment_path,
            translation=translation,
            subagent_runs=list(self._subagent_runs),
            tool_calls=list(self._tool_calls),
            path_taken=path_taken,
            elapsed_s=elapsed_s,
            errors=real_errors,
            warnings=warnings,
        )

    @staticmethod
    def _partition_errors_by_path(
        errors: list[str],
        path_taken: Literal["A", "B"],
    ) -> tuple[list[str], list[str]]:
        """Split error strings into ``(errors, warnings)`` for the given path.

        On **Path A**, baseline / profile failures are inevitable: the
        harness IS the user's run command, not a file the orchestrator
        owns, so ``collect_baseline`` / ``collect_profile`` will fail in
        the standard "no harness file present" way and that does NOT
        mean preprocessing failed. Any error string starting with
        ``"collect_baseline"`` or ``"collect_profile"`` is downgraded to
        a warning; everything else stays in the errors list and
        invalidates ``success`` as usual.

        On **Path B**, nothing is downgraded — the legacy strict
        criteria are preserved (any error invalidates ``success``). The
        returned ``warnings`` list is always empty on Path B.

        Args:
            errors: Cumulative error strings (loop + finish-payload).
            path_taken: ``"A"`` or ``"B"``.

        Returns:
            ``(real_errors, warnings)`` — the partition. The two lists
            together contain exactly the same strings as ``errors``,
            preserving order within each bucket.
        """
        if path_taken != "A":
            return list(errors), []
        real_errors: list[str] = []
        warnings: list[str] = []
        for err in errors:
            text = err if isinstance(err, str) else str(err)
            stripped = text.lstrip()
            if stripped.startswith("collect_baseline") or stripped.startswith("collect_profile"):
                warnings.append(text)
            else:
                real_errors.append(text)
        return real_errors, warnings

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
        * No accumulated ``errors`` (loop-level or finish-payload-level)
          AFTER :meth:`_partition_errors_by_path` has stripped out
          Path-A's baseline / profile warnings.

        Path-specific criteria:

        * **Path A** — the harness IS the user's run command, not a file
          artifact we own; ``harness_path`` may be ``None``. Required
          artifact is ``commandment_path`` (the rendered COMMANDMENT.md
          that captures the user's command). ``baseline`` is helpful but
          not required (the LLM may opt to skip it on a Path-A run if
          the user only asked for a one-shot command). Baseline /
          profile failures are downgraded to warnings by
          :meth:`_partition_errors_by_path` before they reach this
          function.
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
