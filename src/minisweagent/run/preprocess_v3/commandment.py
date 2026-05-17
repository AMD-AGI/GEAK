"""Step 6 — render ``COMMANDMENT.md`` for the v3 preprocess pipeline.

Two-track rendering, matching the legacy ``ExplorePhase`` semantics:

1. **Jinja-first** — when ``KernelLanguage.commandment_template_path``
   resolves to an existing file, render via Jinja with the same
   variable set the legacy ``ExplorePhase._try_jinja_render`` used.
   This is the preferred path for Triton + HIP today and the only
   path that scales to future languages.

2. **Legacy fallback** — when no Jinja template is available (or
   rendering raises), fall through to
   :func:`minisweagent.run.preprocess.commandment.generate_commandment`
   (harness-path flow) or
   :func:`minisweagent.run.preprocess.commandment.generate_commandment_from_commands`
   (eval-command flow). The legacy module's auto-fix retry loop runs
   under the hood, so callers get the same auto-correction guarantees
   the legacy phase had.

Both paths produce a string. Output writing is the caller's choice
(via ``out_path``); we deliberately do NOT validate the rendered
output here — the dedicated contract validators in
:mod:`minisweagent.kernel_languages.contract` and
:mod:`minisweagent.run.preprocess_v3.contracts` (lands later in this
commit set) own that concern.

Strict: no LLM calls, no network access. Pure templating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.kernel_languages.base import KernelLanguage

# TODO(commit-set-5): inline; old preprocess/ goes away
from minisweagent.run.preprocess.commandment import (
    generate_commandment,
    generate_commandment_from_commands,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandmentContext:
    """Inputs needed to render a ``COMMANDMENT.md``.

    The field set is the union of what the Jinja templates and the
    legacy Python generators consume — kept tight so callers don't
    have to thread unused parameters. See the per-language Jinja
    templates under ``kernel_languages/<lang>/commandment.j2`` for
    the authoritative variable list.

    Attributes:
        kernel_path:
            Absolute path to the target kernel file. Required by both
            Jinja templates and the legacy generators (the latter use
            it to derive a default ``repo_root``).
        harness_path:
            Absolute path to the validated harness script. Required
            by the harness-path flow (Triton today). Optional in the
            eval-command flow (HIP) when ``performance_command``
            already encodes how the kernel is exercised.
        repo_root:
            Repository root for ``PYTHONPATH`` plumbing. When ``None``,
            the legacy generator falls back to ``kernel_path.parent``
            (matching its long-standing default).
        baseline_metrics:
            Optional ``baseline_metrics.json`` payload from
            :func:`collect_baseline_metrics`. Carried for templates
            that may want to embed baseline duration / bottleneck
            into the rendered commandment, and surfaced on
            :class:`CommandmentContext` so :class:`Phase` callers
            don't have to re-thread it as a separate kwarg.
        codebase_context_path:
            Optional path to the ``CODEBASE_CONTEXT.md`` produced by
            :func:`explore_codebase`. Templates that link back to the
            codebase briefing reference this path; passing ``None``
            simply means the link is omitted.
        compile_command:
            HIP-style compile step (string or shell snippet) for
            language flows whose ``SETUP`` section needs a build.
            ``None`` for harness-path flows (Triton).
        correctness_command:
            Override for the ``CORRECTNESS`` section. Templates that
            default to ``python3 {harness} --correctness`` look here
            first.
        performance_command:
            Override for ``BENCHMARK`` / ``FULL_BENCHMARK`` sections.
            HIP's eval-command flow leans on this; harness flows
            usually leave it ``None`` and let the template default
            kick in.
        inner_kernel:
            ``True`` when the kernel is an inner file imported by a
            wrapper. The legacy generator emits a different SETUP
            block in this case; carried through so v3 keeps parity.
        inner_kernel_relpath:
            Required when ``inner_kernel`` is ``True``: relative path
            from ``repo_root`` to the inner kernel file (e.g.
            ``aiter/ops/triton/_triton_kernels/rope/rope.py``).
        warmup_runs:
            Profiler warm-up invocations. Pinned to 2 by default to
            match the legacy generator and keep the agent-side and
            preprocessor-side profilers in sync.
        profile_replays:
            ``kernel-profile`` replay count. Default 3 mirrors the
            legacy ExplorePhase.
        kernel_language_kind:
            Coarse language family for the legacy generator's
            ``kernel_language`` parameter (``"python"``, ``"cpp"``,
            or ``"asm"``). This is *not* the v3
            :class:`KernelLanguage` (those map to the Jinja path) —
            it's the legacy enum that controls C++ build steps in
            ``SETUP``. Consumed only on the legacy fallback path.
        extras:
            Free-form extra variables propagated into the Jinja
            render context. Useful for templates that want optional
            knobs without bloating this dataclass.
    """

    kernel_path: Path
    harness_path: Path | None = None
    repo_root: Path | None = None
    baseline_metrics: dict[str, Any] | None = None
    codebase_context_path: Path | None = None
    compile_command: str | list[str] | None = None
    correctness_command: str | list[str] | None = None
    performance_command: str | list[str] | None = None
    inner_kernel: bool = False
    inner_kernel_relpath: str | None = None
    warmup_runs: int = 2
    profile_replays: int = 3
    kernel_language_kind: str = "python"
    extras: dict[str, Any] = field(default_factory=dict)


def _join_cmd(cmd: str | list[str] | None) -> str | None:
    """Normalize a string-or-list command into a single shell snippet."""
    if cmd is None:
        return None
    if isinstance(cmd, list):
        joined = " && ".join(c.strip() for c in cmd if c.strip())
        return joined or None
    stripped = cmd.strip()
    return stripped or None


def _try_jinja_render(
    kernel_language: KernelLanguage,
    context: CommandmentContext,
) -> str | None:
    """Render the per-language Jinja commandment template if available.

    Mirrors the idiom from
    :func:`minisweagent.run.preprocess.phases.explore._try_jinja_render`
    so behaviour is identical: same template variable names, same
    fall-through-to-legacy behaviour on import / missing-file /
    rendering errors.

    Returns:
        Rendered markdown string on success; ``None`` when no
        template is configured or rendering fails (the caller then
        falls back to the legacy Python generators).
    """
    template_path = kernel_language.commandment_template_path
    if template_path is None:
        return None

    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        logger.debug("jinja2 not installed; falling back to legacy commandment.py")
        return None

    template_path = Path(template_path)
    if not template_path.exists():
        logger.debug(
            "commandment template missing for language=%s at %s; falling back to legacy commandment.py",
            kernel_language.name,
            template_path,
        )
        return None

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    try:
        tmpl = env.get_template(template_path.name)
        # Variable set matches legacy ExplorePhase._try_jinja_render so
        # existing template files (Triton, HIP) render unchanged.
        return tmpl.render(
            kernel_path=str(context.kernel_path),
            harness_path=str(context.harness_path) if context.harness_path else "",
            repo_root=str(context.repo_root) if context.repo_root else "",
            inner_kernel=context.inner_kernel,
            profile_replays=context.profile_replays,
            warmup_runs=context.warmup_runs,
            correctness_command=_join_cmd(context.correctness_command),
            performance_command=_join_cmd(context.performance_command),
            compile_command=_join_cmd(context.compile_command),
            baseline_metrics=context.baseline_metrics,
            codebase_context_path=(str(context.codebase_context_path) if context.codebase_context_path else ""),
            **context.extras,
        )
    except Exception as exc:
        logger.warning(
            "Jinja commandment render failed for language=%s: %s; falling back to legacy commandment.py",
            kernel_language.name,
            exc,
        )
        return None


def _legacy_render(
    kernel_language: KernelLanguage,
    context: CommandmentContext,
) -> str:
    """Render via the legacy Python generators when no Jinja template is available.

    Picks between the harness-path generator and the eval-command
    generator the same way the legacy ``ExplorePhase`` did:

    * If a ``performance_command`` (or ``correctness_command``) is
      set, treat as the eval-command flow and call
      :func:`generate_commandment_from_commands`.
    * Otherwise treat as the harness-path flow and call
      :func:`generate_commandment`.

    The legacy auto-fix retry loop runs inside both functions, so
    output is already validated and corrected as it would have been
    in the legacy ``_validate_and_fix`` pipeline.
    """
    correctness_cmd = _join_cmd(context.correctness_command)
    perf_cmd = _join_cmd(context.performance_command)
    compile_cmd = _join_cmd(context.compile_command)

    if perf_cmd or correctness_cmd or compile_cmd:
        return generate_commandment_from_commands(
            kernel_path=context.kernel_path,
            compile_command=context.compile_command,
            correctness_command=context.correctness_command,
            performance_command=context.performance_command,
            repo_root=context.repo_root,
            warmup_runs=context.warmup_runs,
            profile_replays=context.profile_replays,
        )

    if context.harness_path is None:
        raise ValueError(
            "render_commandment: legacy fallback needs either a harness_path "
            "or one of correctness_command / performance_command / compile_command "
            f"(language={kernel_language.name!r})"
        )

    return generate_commandment(
        kernel_path=context.kernel_path,
        harness_path=context.harness_path,
        repo_root=context.repo_root,
        inner_kernel=context.inner_kernel,
        inner_kernel_relpath=context.inner_kernel_relpath,
        warmup_runs=context.warmup_runs,
        profile_replays=context.profile_replays,
        kernel_language=context.kernel_language_kind,
    )


def render_commandment(
    kernel_language: KernelLanguage,
    context: CommandmentContext,
    *,
    out_path: Path | None = None,
) -> str:
    """Render a ``COMMANDMENT.md`` for the given language and context.

    Resolution order:

    1. **Jinja** — render
       ``kernel_language.commandment_template_path`` via Jinja with
       the variable set used by the legacy ``ExplorePhase``.
    2. **Legacy Python** — call into
       :func:`minisweagent.run.preprocess.commandment.generate_commandment`
       (harness path) or
       :func:`generate_commandment_from_commands` (eval-command path),
       depending on whether the context carries a harness or a set
       of explicit commands.

    Args:
        kernel_language:
            :class:`KernelLanguage` produced by pre-step-0b. Drives
            template selection and (on the legacy path) language-
            specific build steps.
        context:
            Input bundle (kernel/harness paths, optional commands,
            warm-up + replay knobs). See :class:`CommandmentContext`.
        out_path:
            When supplied, the rendered text is also written here.
            Re-running with the same ``out_path`` is idempotent — the
            file is overwritten with identical content if inputs are
            unchanged.

    Returns:
        The rendered ``COMMANDMENT.md`` text.

    Raises:
        ValueError: When the legacy fallback can't proceed because
            neither a ``harness_path`` nor any command override is
            supplied (no way to construct a meaningful commandment).
    """
    text = _try_jinja_render(kernel_language, context)
    if text is None:
        text = _legacy_render(kernel_language, context)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    return text


__all__ = ["CommandmentContext", "render_commandment"]
