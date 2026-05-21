"""Tool implementations for the v3 preprocess orchestrator.

Each tool here is the LLM-callable bridge between an
``OpenAI/LiteLLM``-style tool call and one of the v3 preprocess modules.
Tools come in two flavours:

* **Deterministic tools** (``run_discovery``, ``codebase_explore``,
  ``translate_to_flydsl``, ``collect_baseline``, ``collect_profile``,
  ``render_commandment``) —
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
import os
import shlex
import shutil
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
from minisweagent.run.preprocess_v3.discovery import DiscoveryContext, run_legacy_discovery
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

_MODE_TO_FLAG: dict[str, str] = {
    "correctness": "--correctness",
    "profile": "--profile",
    "benchmark": "--benchmark",
    "full_benchmark": "--full-benchmark",
}


def _substitute_mode_flag(cmd: str, target_mode: str) -> str:
    """Ensure *cmd* contains *target_mode*'s harness flag.

    Three cases:
    1. *cmd* already has the right flag → return unchanged.
    2. *cmd* has a different known flag → replace it.
    3. *cmd* has no known flag at all → return unchanged (safe fallback).
    """
    dst_flag = _MODE_TO_FLAG.get(target_mode)
    if not dst_flag:
        return cmd
    if dst_flag in cmd:
        return cmd
    for other_mode, other_flag in _MODE_TO_FLAG.items():
        if other_mode != target_mode and other_flag in cmd:
            return cmd.replace(other_flag, dst_flag)
    return cmd


_DEFAULT_PROFILE_REPLAYS = 5


def _build_profile_section(profile_cmd: str) -> str:
    """Wrap a ``--profile`` harness invocation with warmup + ``kernel-profile``.

    Matches the PROFILE section that Path B's ``render_commandment`` produces:
    one warmup pass (suppressed stdout), then ``kernel-profile`` wrapping the
    same command to capture hardware counters and write ``profile.json``.
    """
    warmup = f"{profile_cmd} > /dev/null 2>&1 || true"
    kernel_profile = (
        f'kernel-profile "{profile_cmd}"'
        f" --gpu-devices ${{GEAK_GPU_DEVICE}}"
        f" --replays {_DEFAULT_PROFILE_REPLAYS}"
        f" --json -o ${{GEAK_WORK_DIR}}/profile.json"
    )
    return f"{warmup}\n{kernel_profile}"


def _extract_harness_from_command(cmd: str) -> str | None:
    """Extract a harness file path from a shell command if it looks like a standard harness.

    Scans for ``python[3] <path>`` and returns the path when the command
    also contains a known harness flag (``--correctness``, ``--benchmark``,
    etc.), indicating the file supports the standard four-mode CLI contract.
    Returns ``None`` when the command is opaque (no flag or no python invocation).
    """
    has_harness_flag = any(flag in cmd for flag in _MODE_TO_FLAG.values())
    if not has_harness_flag:
        return None
    import shlex

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    for i, tok in enumerate(tokens):
        if tok in ("python", "python3") and i + 1 < len(tokens):
            candidate = tokens[i + 1]
            if not candidate.startswith("-"):
                return candidate
    return None


def _try_synthesize_shell_contract_harness(
    cmd: str,
    *,
    out_path: str,
    repo_root_str: str,
) -> str | None:
    """Wrap a compound shell ``eval_command`` into the 4-mode harness contract.

    When the user's task description carries a runner-style compound command
    (canonical AKA pattern: ``python3 scripts/task_runner.py compile && correctness
    && performance``) without any GEAK harness flag, the legacy preprocess used
    :func:`eval_contract_adapter.materialize_shell_contract_harness` to write a
    Python wrapper exposing ``--correctness``/``--profile``/``--benchmark``/
    ``--full-benchmark`` so v3's ``collect_baseline`` / ``collect_profile`` can
    treat it as a normal harness.

    This helper:

    1. ``rsplit(\"&&\", 1)`` mirrors the legacy ``resolve_shell_eval_commands``
       fallback — left half becomes the correctness/setup body, right half
       becomes the performance body.
    2. ``infer_compile_command_from_eval`` (also legacy) extracts the leading
       compile/build prefix from the full command and re-prepends it to the
       performance body so a standalone ``--benchmark`` invocation rebuilds
       the binary if needed.
    3. ``materialize_shell_contract_harness`` writes the wrapper into the
       COMMANDMENT.md output directory and returns its path.

    Returns ``None`` when no confident split exists, when the legacy module is
    unavailable, or when the materialize call raises.
    """
    if "&&" not in cmd:
        return None
    if not repo_root_str:
        return None
    left, right = cmd.rsplit("&&", 1)
    correctness_shell = left.strip()
    performance_shell = right.strip()
    if not correctness_shell or not performance_shell:
        return None
    try:
        from minisweagent.run.preprocess.contract_normalize import (
            infer_compile_command_from_eval,
        )

        compile_prefix = infer_compile_command_from_eval(cmd)
        if compile_prefix and compile_prefix not in performance_shell:
            performance_shell = f"{compile_prefix} && {performance_shell}"
    except ImportError:
        pass
    try:
        from minisweagent.run.preprocess.eval_contract_adapter import (
            materialize_shell_contract_harness,
        )
    except ImportError:
        return None
    try:
        output_dir = Path(out_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        synthesized = materialize_shell_contract_harness(
            output_dir=output_dir,
            repo_root=repo_root_str,
            correctness_shell=correctness_shell,
            performance_shell=performance_shell,
        )
        logger.info(
            "commandment_from_user_command: synthesized shell-contract harness at %s (correctness=%r, performance=%r)",
            synthesized,
            correctness_shell,
            performance_shell,
        )
        return str(synthesized)
    except Exception as exc:
        logger.warning(
            "commandment_from_user_command: shell-contract harness synthesis failed: %s",
            exc,
        )
        return None


def _validate_harness_or_warn(harness_path: str) -> bool:
    """Run legacy ``validate_harness`` on the candidate; log + return validity.

    Static-analyses the harness for the four required GEAK CLI flags
    (``--correctness``, ``--profile``, ``--benchmark``, ``--full-benchmark``)
    and a recognised arg parser (argparse / click / typer). Returns ``True``
    on validation success (with optional warnings logged) so callers can gate
    accepting the path. Returns ``True`` (treated as valid) when the legacy
    module is unavailable, mirroring the previous "no validation" behaviour.
    """
    try:
        from minisweagent.run.preprocess.harness_utils import validate_harness
    except ImportError:
        return True
    try:
        valid, messages = validate_harness(harness_path)
    except Exception as exc:
        logger.debug("validate_harness raised on %s: %s", harness_path, exc)
        return True
    if not valid:
        logger.warning(
            "commandment_from_user_command: harness validation FAILED for %s: %s",
            harness_path,
            messages,
        )
    elif messages:
        logger.info(
            "commandment_from_user_command: harness validation passed with warnings for %s: %s",
            harness_path,
            messages,
        )
    return valid


def _copy_repo_sandbox(repo_root: Path, sandbox_root: Path, output_dir: Path) -> None:
    """Copy a non-git repo into a preprocess subagent sandbox."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()

    def _ignore(dir_path: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", ".ruff_cache"}
        current = Path(dir_path).resolve()
        for name in names:
            child = current / name
            try:
                child_resolved = child.resolve()
            except OSError:
                continue
            # Avoid recursively copying the active GEAK output directory
            # when users place outputs under the target repo.
            if child_resolved == output_dir or output_dir in child_resolved.parents:
                ignored.add(name)
        return ignored

    shutil.copytree(repo_root, sandbox_root, symlinks=True, ignore=_ignore)


def _ensure_preprocess_subagent_sandbox(agent: PreprocessOrchestratorAgent) -> tuple[Path | None, dict[str, str]]:
    """Create a repo sandbox for preprocess subagents and return tool env."""

    repo_raw = agent._extra_template_vars.get("repo_root") if hasattr(agent, "_extra_template_vars") else None
    output_raw = agent._extra_template_vars.get("output_dir") if hasattr(agent, "_extra_template_vars") else None
    gpu_raw = agent._extra_template_vars.get("gpu_id") if hasattr(agent, "_extra_template_vars") else None
    if not repo_raw or not output_raw:
        return None, {}

    repo_root = Path(str(repo_raw)).expanduser().resolve()
    output_dir = Path(str(output_raw)).expanduser().resolve()
    if not repo_root.is_dir():
        return None, {}

    sandbox_root = output_dir / "_preprocess_subagent_worktree"
    if not sandbox_root.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        if (repo_root / ".git").exists():
            from minisweagent.run.task_file import create_worktree

            create_worktree(repo_root, sandbox_root)
        else:
            _copy_repo_sandbox(repo_root, sandbox_root, output_dir)

    gpu_id = str(gpu_raw if gpu_raw is not None else 0)
    env = {
        "GEAK_REPO_ROOT": str(repo_root),
        "GEAK_WORK_DIR": str(sandbox_root.resolve()),
        "GEAK_GPU_DEVICE": gpu_id,
        "HIP_VISIBLE_DEVICES": gpu_id,
    }
    return sandbox_root.resolve(), env


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

        tool_env = None
        if isinstance(context, dict):
            tool_env = context.pop("_tool_env", None)

        full_task = task if not context else self._format_context_preamble(context) + "\n\n" + task

        extra_template_vars = self._resolve_extra_template_vars(spec)
        if isinstance(tool_env, dict):
            extra_template_vars["_tool_env"] = tool_env

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

        if name == "harness-verifier":
            success = "HARNESS_VERIFIED=true" in message
        elif name == "harness-generator":
            success = exit_status in {"Submitted", "FinishedSuccessfully"} and "HARNESS_PATH:" in message
        else:
            success = exit_status in {"Submitted", "FinishedSuccessfully"}
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
            if key in {"codebase_context_path", "discovery_context_path"} and value:
                path = Path(str(value))
                if path.is_file():
                    try:
                        content = path.read_text(encoding="utf-8")
                    except Exception as exc:  # noqa: BLE001
                        lines.append(f"  (failed to read {path}: {exc})")
                    else:
                        lines.append("")
                        lines.append(f"### Contents of `{path}`")
                        lines.append("")
                        lines.append(content)
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


def _schema_run_discovery() -> dict[str, Any]:
    return {
        "name": "run_discovery",
        "type": "function",
        "description": (
            "Step 1 — deterministic legacy discovery front half. Reuses legacy "
            "DiscoveryPhase semantics: resolve kernel/repo, write CODEBASE_CONTEXT.md, "
            "run automated-test-discovery, write discovery.json and DISCOVERY_CONTEXT.md. "
            "Call this before dispatching harness-generator on Path B."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_root": {"type": "string", "description": "Absolute path to the repository root."},
                "kernel_path": {"type": "string", "description": "Absolute path to the target kernel."},
                "output_dir": {
                    "type": "string",
                    "description": "Directory where CODEBASE_CONTEXT.md, discovery.json, and DISCOVERY_CONTEXT.md are written.",
                },
            },
            "required": ["repo_root", "kernel_path", "output_dir"],
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
            "The harness-generator emits a single ``HARNESS_PATH:`` line — the "
            "dispatcher auto-populates ``harness_path`` on the orchestrator from "
            "it. On step 3b (verifier), the only context the verifier needs is "
            "``harness_path``."
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
            "path is determined by which of these two tools is called first.\n\n"
            "STRICT ARGUMENT NAMING: this tool accepts EXACTLY these keyword "
            "arguments — `run_command`, `out_path`, `modes_covered`, "
            "`inferred_modes`, `notes`. Do NOT invent synonyms. In particular, "
            "the user's shell invocation MUST be passed as `run_command` — "
            "NOT as `command`, `cmd`, `user_command`, `raw_command`, "
            "`harness_command`, or `kernel_path`. The COMMANDMENT.md output "
            "path MUST be `out_path` — NOT `output`, `output_path`, `path`, "
            "or `commandment_path`. Calls with any other argument names will "
            "raise TypeError; the dispatcher will surface the expected schema "
            "in its error reply so you can self-correct on the next turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "run_command": {
                    "type": "string",
                    "description": (
                        "The user-provided shell command, verbatim. Must be non-empty. "
                        "The argument name MUST be exactly `run_command` (not `command`, "
                        "`cmd`, `user_command`, `raw_command`, or `harness_command`). "
                        "Example: 'cd /repo && python my_kernel.py --benchmark --shape 4096'."
                    ),
                },
                "out_path": {
                    "type": "string",
                    "description": (
                        "Where to write COMMANDMENT.md. The argument name MUST be exactly "
                        "`out_path` (not `output`, `output_path`, `path`, `commandment_path`, "
                        "or `kernel_path`)."
                    ),
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


def _make_tool_run_discovery(
    agent: PreprocessOrchestratorAgent,
    kernel_language: KernelLanguage,
) -> Callable[..., dict[str, Any]]:
    """Bind ``run_discovery`` to the legacy deterministic discovery front half."""

    def _impl(
        repo_root: str,
        kernel_path: str,
        output_dir: str,
    ) -> dict[str, Any]:
        ctx: DiscoveryContext = run_legacy_discovery(
            kernel_path=Path(kernel_path),
            repo_root=Path(repo_root),
            output_dir=Path(output_dir),
            kernel_language=kernel_language,
        )
        # Preserve the PreprocessResult surface: codebase_context remains
        # the codebase briefing artifact, while discovery artifacts are
        # kept separately for subagent context / audit.
        codebase_ctx = CodebaseContext(
            text=ctx.codebase_context_text,
            files=[],
            out_path=ctx.codebase_context_path,
            kernel_language=kernel_language,
        )
        agent._collected["codebase_context"] = codebase_ctx
        agent._collected["discovery"] = ctx.discovery
        agent._collected["discovery_path"] = str(ctx.discovery_path) if ctx.discovery_path else None
        agent._collected["discovery_context_text"] = ctx.discovery_context_text
        discovery_context_path = Path(output_dir) / "DISCOVERY_CONTEXT.md"
        agent._collected["discovery_context_path"] = (
            str(discovery_context_path) if discovery_context_path.is_file() else None
        )
        return {
            "ok": True,
            "kernel_path": str(ctx.kernel_path),
            "repo_root": str(ctx.repo_root),
            "codebase_context_path": str(ctx.codebase_context_path) if ctx.codebase_context_path else None,
            "discovery_path": str(ctx.discovery_path) if ctx.discovery_path else None,
            "discovery_context_path": agent._collected.get("discovery_context_path"),
            "tests_found": len((ctx.discovery or {}).get("tests") or []),
            "benchmarks_found": len((ctx.discovery or {}).get("benchmarks") or []),
            "focused_test": bool((ctx.discovery or {}).get("focused_test")),
        }

    return _impl


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
        out_path: str | None = None,
        output_dir: str | None = None,
        **_extra_ignored: Any,
    ) -> dict[str, Any]:
        if _extra_ignored:
            logger.debug("codebase_explore ignored extra kwargs: %s", list(_extra_ignored))
        if not out_path:
            if not output_dir:
                raise ValueError("codebase_explore requires out_path or output_dir")
            out_path = str(Path(output_dir) / "CODEBASE_CONTEXT.md")
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
    def _impl(name: str, task: str, context: Any = None) -> dict[str, Any]:
        if context is None:
            context = {}
        elif isinstance(context, str):
            # The orchestrator LLM sometimes passes a JSON-ish or free-form
            # string despite the schema declaring an object. Preserve it as
            # text rather than raising and spinning in the tool loop.
            try:
                parsed = json.loads(context)
            except json.JSONDecodeError:
                parsed = {"context": context}
            context = parsed if isinstance(parsed, dict) else {"context": context}
        elif not isinstance(context, dict):
            context = {"context": str(context)}
        else:
            context = dict(context)
        generator_attempts = int(agent._collected.get("_harness_generator_attempts", 0) or 0)
        if name == "harness-generator":
            if generator_attempts >= 3:
                return {
                    "name": name,
                    "success": False,
                    "error": "harness-generator retry budget exhausted after 3 attempts",
                    "output": (
                        "HARNESS_VERIFIED=false\n"
                        "ESCALATE=true\n"
                        "PHASE=runtime\n"
                        "FAILED_RULE=generator-retry-budget-exhausted\n"
                        "FAILED_MODE=n/a\n"
                        "EVIDENCE=harness-generator was requested after 3 failed attempts\n"
                    ),
                    "elapsed_s": 0.0,
                    "max_steps": None,
                }
            generator_attempts += 1
            agent._collected["_harness_generator_attempts"] = generator_attempts
            context.setdefault("attempt", generator_attempts)
        elif name == "harness-verifier":
            context.setdefault("attempt", max(generator_attempts, 1))
        codebase_ctx = agent._collected.get("codebase_context")
        if (
            codebase_ctx is not None
            and getattr(codebase_ctx, "out_path", None)
            and "codebase_context_path" not in context
        ):
            context["codebase_context_path"] = str(codebase_ctx.out_path)
        if agent._collected.get("discovery_path") and "discovery_path" not in context:
            context["discovery_path"] = agent._collected["discovery_path"]
        if agent._collected.get("discovery_context_path") and "discovery_context_path" not in context:
            context["discovery_context_path"] = agent._collected["discovery_context_path"]
        if agent._collected.get("discovery") and "discovery" not in context:
            discovery = agent._collected["discovery"]
            context["discovery_summary"] = {
                "tests_found": len((discovery or {}).get("tests") or []),
                "benchmarks_found": len((discovery or {}).get("benchmarks") or []),
                "focused_test": bool((discovery or {}).get("focused_test")),
            }
        _scoring = agent._extra_template_vars.get("scoring_target")
        if _scoring and "scoring_target" not in context:
            context["scoring_target"] = _scoring
        sandbox_cwd, tool_env = _ensure_preprocess_subagent_sandbox(agent)
        if sandbox_cwd is not None:
            context.setdefault("sandbox_repo_root", str(sandbox_cwd))
            context["_tool_env"] = tool_env
        result = dispatcher(
            name=name, task=task, model=agent.model, cwd=str(sandbox_cwd) if sandbox_cwd else None, context=context
        )
        agent._subagent_runs.append(result)
        # Auto-populate orchestrator state from the subagent's output so
        # the LLM doesn't have to extract structured fields by regex.
        # Legacy parity: the harness-generator emits exactly one
        # structured line (``HARNESS_PATH: <path>``); the harness-verifier
        # echoes ``HARNESS_PATH=<path>`` on its success block.
        output = result.get("output", "") or ""
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("HARNESS_PATH:"):
                agent._collected["harness_path"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("HARNESS_PATH="):
                agent._collected["harness_path"] = stripped.split("=", 1)[1].strip()
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
        resolved_gpu = gpu_id if gpu_id is not None else agent.config.gpu_id
        resolved_work_dir = Path(work_dir) if work_dir else None
        baseline: BaselineMetrics = collect_baseline_metrics(
            Path(harness_path),
            repeats=repeats,
            work_dir=resolved_work_dir,
            gpu_id=resolved_gpu,
        )
        agent._collected["baseline"] = baseline

        from minisweagent.run.preprocess_v3.baseline import capture_full_benchmark_stdout

        fb_stdout = capture_full_benchmark_stdout(
            Path(harness_path),
            work_dir=resolved_work_dir,
            gpu_id=resolved_gpu,
        )
        if fb_stdout:
            agent._collected["full_benchmark_stdout"] = fb_stdout

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
        # Extract the harness path from the original command (before
        # ${GEAK_WORK_DIR} substitution) so collect_baseline/collect_profile
        # can use the real filesystem path.
        original_harness_path = _extract_harness_from_command(cmd)
        repo_root = str(agent.config.repo) if agent.config.repo else os.environ.get("GEAK_REPO_ROOT", "")
        # Fallback: if the user's command is a compound shell pipeline (e.g.
        # ``task_runner.py compile && correctness && performance``) without any
        # standard GEAK harness flag, synthesize a 4-mode wrapper harness using
        # the legacy eval_contract_adapter so collect_baseline / collect_profile
        # have a real harness_path to point at.
        if not original_harness_path:
            synthesized = _try_synthesize_shell_contract_harness(
                cmd,
                out_path=out_path,
                repo_root_str=repo_root,
            )
            if synthesized:
                original_harness_path = synthesized
        # Static-validate the harness (whether user-supplied or synthesized) so
        # a malformed file doesn't cause silent baseline/profile failures later.
        # Failed validation clears the harness_path; the COMMANDMENT.md is still
        # rendered from the user's command and finish_preprocess can fall back
        # to extracting a path from the command body.
        if original_harness_path and not _validate_harness_or_warn(original_harness_path):
            original_harness_path = None
        if original_harness_path:
            agent._collected["harness_path"] = original_harness_path
        # Rewrite hardcoded repo-root paths to ${GEAK_WORK_DIR} so the
        # COMMANDMENT references the agent's worktree at runtime.
        # Use the orchestrator's config.repo (available at preprocess time)
        # rather than GEAK_REPO_ROOT (only set later for agent subprocesses).
        if repo_root and repo_root in cmd:
            cmd = cmd.replace(repo_root, "${GEAK_WORK_DIR}")
        modes_covered_tup = _normalise_modes(modes_covered)
        inferred_modes_tup = _normalise_modes(inferred_modes)
        source_mode = modes_covered_tup[0] if modes_covered_tup else None

        setup_body = (
            "printf '#!/bin/bash\\nexport PYTHONPATH=%s:%s:${PYTHONPATH}\\n"
            "export HIP_VISIBLE_DEVICES=%s\\n"
            'cd "%s" && exec bash -lc "$*"\\n\' '
            '"${GEAK_WORK_DIR}" "${GEAK_REPO_ROOT}" "${GEAK_GPU_DEVICE}" '
            '"${GEAK_WORK_DIR}" '
            "> ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh"
        )
        sections: dict[str, str] = {
            "setup": setup_body,
        }
        warnings: list[str] = []
        modes_emitted: list[str] = []

        # Invoke through run.sh so every Path-A section uses the same
        # PYTHONPATH/HIP_VISIBLE_DEVICES/worktree contract as legacy
        # COMMANDMENT generation while still supporting compound shell
        # commands such as "compile && correctness && performance".
        wrapped_cmd = f"${{GEAK_WORK_DIR}}/run.sh {shlex.quote(cmd)}"
        for mode in PATH_A_MODES:
            if mode in modes_covered_tup:
                mode_cmd = _substitute_mode_flag(wrapped_cmd, mode)
                if mode == "profile":
                    mode_cmd = _build_profile_section(mode_cmd)
                sections[mode] = mode_cmd
                modes_emitted.append(mode)
            elif mode in inferred_modes_tup:
                src = source_mode or "<unspecified>"
                inferred_cmd = _substitute_mode_flag(wrapped_cmd, mode)
                if mode == "profile":
                    inferred_cmd = _build_profile_section(inferred_cmd)
                marker_line = f"# PATH_A_PARTIAL_COVERAGE: {mode} inferred from {src}"
                sections[mode] = f"{marker_line}\n{inferred_cmd}"
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
            "harness_path": original_harness_path,
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
        **_extra_ignored: Any,
    ) -> dict[str, Any]:
        # Defensive: the orchestrator LLM has been observed to invent
        # kwargs the schema does not declare (``artifacts``,
        # ``codebase_context_path``, ``path_taken``, ``kernel_path``,
        # ``output_dir``). Drop them with a debug breadcrumb so the
        # completion sentinel never fails on TypeError. The new prompt
        # contract instructs the LLM to call ``finish_preprocess()`` with
        # no arguments — the kwargs above are accepted for backwards
        # compatibility but their values are ignored (all artifacts are
        # already on ``agent._collected`` from prior tool calls).
        if _extra_ignored:
            logger.debug("finish_preprocess ignored extra kwargs: %s", list(_extra_ignored))

        # Path-A: the orchestrator skipped the harness-generator subagent.
        # If the user's command references a standard harness file (has a
        # known flag like --correctness), extract and preserve the path so
        # downstream evaluation can run Metrix profiling. Otherwise clear
        # harness_path — the command is opaque and the evaluation reads it
        # from COMMANDMENT.md via commandment_path.
        on_path_a = agent._collected.get("run_instructions") is not None
        if on_path_a:
            instructions = agent._collected["run_instructions"]
            harness_path = _extract_harness_from_command(instructions.raw_command)

        if harness_path:
            agent._collected["harness_path"] = harness_path
        if commandment_path:
            agent._collected["commandment_path"] = commandment_path

        # Harness-path derivation fallback (Path B only): when the LLM
        # forgot to thread ``harness_path`` AND the dispatcher didn't
        # already populate it (e.g. the subagent omitted its
        # ``HARNESS_PATH:`` line), scan ``output_dir`` for a likely
        # harness file before downgrading the run to ``success=False``.
        # Mirrors legacy ``extract_harness_path`` + the discovery fallback
        # Layer 7. Skipped on Path A so we don't accidentally pick up a
        # stale file from a prior run.
        if not on_path_a and not agent._collected.get("harness_path"):
            output_dir = (
                agent._extra_template_vars.get("output_dir") if hasattr(agent, "_extra_template_vars") else None
            )
            if output_dir:
                for candidate_name in ("test_harness.py", "harness.py", "_geak_auto_harness.py", "harness.hip"):
                    candidate = Path(output_dir) / candidate_name
                    if candidate.is_file():
                        agent._collected["harness_path"] = str(candidate)
                        logger.info("finish_preprocess derived harness_path from output_dir: %s", candidate)
                        break

        payload = {
            "harness_path": agent._collected.get("harness_path"),
            "commandment_path": agent._collected.get("commandment_path"),
            "errors": list(errors or []),
            "summary": summary,
        }
        blockers = _finish_blockers(agent=agent, on_path_a=on_path_a, payload=payload)
        if blockers:
            return {
                "ok": False,
                "error": "finish_preprocess blocked: unresolved preprocess invariants",
                "blockers": blockers,
                "next_action": (
                    "Do not call finish_preprocess again yet. Fix the listed blockers by "
                    "rerunning the failed deterministic tool or redispatching the relevant "
                    "subagent, then call finish_preprocess only after all blockers are gone."
                ),
            }
        agent._finish_payload = payload
        raise FinishedSuccessfully(payload)

    return _impl


def _path_a_deterministic_blockers(
    agent: PreprocessOrchestratorAgent,
    harness_path: str,
) -> list[str]:
    """Check whether Path A still needs to call collect_baseline / collect_profile."""
    blockers: list[str] = []
    for tool_name in ("collect_baseline", "collect_profile"):
        attempted = any(tc.get("name") == tool_name for tc in agent._tool_calls)
        if agent._collected.get(tool_name.split("_", 1)[1]) is None and not attempted:
            blockers.append(
                f"Path A has harness_path but {tool_name} was not called. "
                f"Call {tool_name}(harness_path='{harness_path}') before finishing."
            )
    return blockers


def _finish_blockers(
    *,
    agent: PreprocessOrchestratorAgent,
    on_path_a: bool,
    payload: dict[str, Any],
) -> list[str]:
    """Return final-state blockers that should keep the LLM iterating.

    ``finish_preprocess`` is a completion sentinel, not a validation bypass.
    Path A is allowed to finish once the user-command COMMANDMENT exists
    and baseline/profile have been attempted (when a harness is available).
    Path B must have the generated harness verified and all deterministic
    artifacts collected successfully before the sentinel can terminate the
    orchestrator loop.
    """
    blockers: list[str] = []

    if on_path_a:
        if not payload.get("commandment_path"):
            blockers.append("Path A missing commandment_path")
        harness_path = agent._collected.get("harness_path")
        if harness_path:
            blockers.extend(_path_a_deterministic_blockers(agent, str(harness_path)))
        return blockers

    if payload.get("errors"):
        # A failed preprocess must be allowed to terminate. The final
        # PreprocessResult will carry success=False; blocking here causes the
        # orchestrator to spin until its global step limit.
        return blockers

    if not agent._collected.get("harness_path"):
        blockers.append("Path B missing harness_path")

    verifier_runs = [r for r in agent._subagent_runs if r.get("name") == "harness-verifier"]
    if not verifier_runs:
        blockers.append("Path B never dispatched harness-verifier")
    elif not any(r.get("success") is True for r in verifier_runs):
        blockers.append("Path B has no successful harness-verifier run")

    baseline = agent._collected.get("baseline")
    if baseline is None:
        blockers.append("Path B missing baseline metrics")
    elif getattr(baseline, "success", True) is False:
        blockers.append("Path B baseline collection failed")

    profile = agent._collected.get("profile")
    if profile is None:
        blockers.append("Path B missing profile result")
    elif getattr(profile, "success", True) is False:
        blockers.append("Path B profile collection failed")

    if not agent._collected.get("commandment_path"):
        blockers.append("Path B missing COMMANDMENT.md")

    return blockers


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
    """Register all v3 preprocess tools on ``agent``.

    Args:
        agent:
            Target agent. Tools are bound to its ``_collected`` /
            ``_subagent_runs`` mutable state so the eventual
            :class:`PreprocessResult` carries every artifact.
        kernel_language:
            Language resolved by step 0b. Threaded into
            discovery/context and ``render_commandment``.
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
        "run_discovery",
        _schema_run_discovery(),
        _make_tool_run_discovery(agent, kernel_language),
    )
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
