"""``TranslationAgent`` — standalone verify-retry subagent for kernel language porting.

Architectural contract (per execution plan §0.5(b) and user direction):

  - Translation is a **preprocess phase**, not a ``run_pipeline`` mode.
    It runs only when the user asks for a target language different
    from the source (``target_language``).  After the phase completes,
    ``ctx.kernel_path`` and ``ctx.language`` are swapped to the
    translated kernel and the normal fixed / planned / auto pipeline
    continues (or the pipeline exits, if ``translate_only`` was set).

  - ``TranslationAgent`` is a ``SubagentBase`` subclass overriding
    ``loop()`` (the multi-round verify-retry entry point).  It is
    **deliberately not derived from ``OptimizationAgent``** and does
    **not** compose ``OptimizationAgent`` via
    ``_make_optimization_agent``.  Translation is a narrow,
    verifier-gated task: one prompt → one candidate → verify.  No
    tool runtime, no strategy manager, no RAG wrapping, no
    patch-apply machinery.  Direct ``model.query`` is sufficient.

  - Success criterion: ``verify_fn(candidate)`` returns True, where
    ``verify_fn`` is normally a tensor ``allclose(src_out, tgt_out,
    atol=1e-5)`` against golden tensors captured by running the source
    harness once at the start of the phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from minisweagent.pipeline_workers.base import SubagentBase

logger = logging.getLogger(__name__)


class TranslationFailed(Exception):
    """Raised when all ``max_attempts`` translation attempts fail verification."""


@dataclass
class TranslationResult:
    """Structured return value of ``TranslationAgent.loop``."""

    ok: bool
    candidate_code: str = ""
    attempts_used: int = 0
    feedback_history: list[str] = field(default_factory=list)
    reason: str = ""


def _default_summarize_failure(attempt: int, error: str) -> str:
    """Default feedback summariser: wraps the verifier's error message.

    The verifier (``verify_fn``) is passed the candidate code and is
    expected to return True on success.  When it returns False we need
    a string describing *why* to feed back into the next attempt's
    prompt.  The default just reports attempt index + the raw error.
    Callers can override by passing a richer ``summarize_fn`` that
    inspects tensor diffs, shape mismatches, etc.
    """
    return f"Attempt {attempt}: {error or 'verification failed (no details supplied)'}"


class TranslationAgent(SubagentBase):
    """Verify-retry subagent that rewrites a kernel in a different language.

    Usage:

        agent = TranslationAgent(language=target_kernel_language, config=...)
        result = agent.loop(
            max_attempts=3,
            verify_fn=lambda candidate: tensor_allclose(golden, run(candidate)),
            source_code=...,
            source_language=...,
        )
        if result.ok:
            translated_code = result.candidate_code

    ``self.language`` holds the TARGET ``KernelLanguage`` (that is the
    language the agent is writing code in).  The SOURCE language is
    passed via ``**inputs`` because the subagent's "home" language is
    the one whose prompts / templates it uses.
    """

    # System prompt for the target-language translator.  Falls back to
    # a generic prompt when ``self.language.system_prompt_path`` is
    # None (PR-2 builds out the language-specific prompts).
    _DEFAULT_SYSTEM_PROMPT = (
        "You are a GPU kernel translator.  Given source code in one kernel "
        "language, produce a semantically equivalent kernel in the target "
        "language.  Preserve the input/output contract exactly (same tensor "
        "shapes, same dtypes, same function signature, same entry-point name).  "
        "Do NOT introduce any new dependencies beyond the target language's "
        "standard idioms.  Return ONLY the translated source code — no prose, "
        "no markdown fences, no explanation.  If a previous attempt failed "
        "verification, use the provided feedback to fix the failure."
    )

    def loop(
        self,
        *,
        max_attempts: int,
        verify_fn: Callable[[str], bool | tuple[bool, str]],
        **inputs: Any,
    ) -> TranslationResult:
        """Retry-until-verified translation loop.

        ``verify_fn(candidate)`` returns either a plain bool or a
        ``(bool, str)`` tuple where the string is an error explanation
        to feed back into the next attempt's prompt.

        Required inputs:
          - ``source_code``     (str): the source kernel to translate
          - ``source_language`` (str or KernelLanguage): canonical name
                                of the source language

        Optional inputs:
          - ``hints`` (str): extra translation guidance (e.g. contents
                              of ``<src>_to_<tgt>.md``)
          - ``summarize_fn`` (Callable[[int, str], str]): overrides the
                              default feedback summariser
        """
        source_code = inputs.get("source_code")
        source_language = inputs.get("source_language")
        if not source_code or not source_language:
            raise ValueError(
                "TranslationAgent.loop requires ``source_code`` and ``source_language`` "
                "in inputs."
            )
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        summarize_fn: Callable[[int, str], str] = inputs.get(
            "summarize_fn", _default_summarize_failure
        )
        hints = inputs.get("hints", "")

        src_lang_name = getattr(source_language, "name", str(source_language))
        tgt_lang_name = self.language.name

        feedback_history: list[str] = []
        last_feedback: str | None = None
        last_candidate: str = ""

        for attempt in range(1, max_attempts + 1):
            sys_p, inst_p = self._compose_translation_prompt(
                source_language=src_lang_name,
                target_language=tgt_lang_name,
                source_code=source_code,
                hints=hints,
                last_feedback=last_feedback,
            )
            logger.info(
                "TranslationAgent attempt %d/%d (src=%s, tgt=%s)",
                attempt,
                max_attempts,
                src_lang_name,
                tgt_lang_name,
            )
            candidate = self._query_model(sys_p, inst_p)
            last_candidate = candidate

            # verify_fn may return a plain bool or (bool, explanation)
            verdict = verify_fn(candidate)
            if isinstance(verdict, tuple):
                ok, failure_reason = verdict
            else:
                ok, failure_reason = bool(verdict), ""

            if ok:
                logger.info("TranslationAgent verified on attempt %d", attempt)
                return TranslationResult(
                    ok=True,
                    candidate_code=candidate,
                    attempts_used=attempt,
                    feedback_history=feedback_history,
                )

            last_feedback = summarize_fn(attempt, failure_reason)
            feedback_history.append(last_feedback)
            logger.info("  Attempt %d failed verification: %s", attempt, last_feedback[:200])

        return TranslationResult(
            ok=False,
            candidate_code=last_candidate,
            attempts_used=max_attempts,
            feedback_history=feedback_history,
            reason=f"exhausted {max_attempts} attempts without passing verify_fn",
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _compose_translation_prompt(
        self,
        *,
        source_language: str,
        target_language: str,
        source_code: str,
        hints: str = "",
        last_feedback: str | None = None,
    ) -> tuple[str, str]:
        """Render the (system, instance) prompt pair for one attempt.

        The instance prompt concatenates:
          - a language-pair header
          - the source code
          - optional pair-specific hints (``hints`` kwarg)
          - optional last-attempt feedback

        This intentionally sidesteps ``self._compose_prompt`` from
        ``SubagentBase`` because translation's prompt shape is
        pair-specific (source + target) rather than the single-
        language ``{language_name}`` substitution the base helper
        expects.  Once PR-2's Jinja prompt templates land, this can
        move into the base helper.
        """
        system = self._DEFAULT_SYSTEM_PROMPT
        if self.language.system_prompt_path is not None:
            try:
                system = self.language.system_prompt_path.read_text(encoding="utf-8")
            except Exception:
                pass  # fall back to the default

        parts = [
            f"Translate the following {source_language} kernel into {target_language}.",
            "",
            f"SOURCE ({source_language}):",
            "```",
            source_code.rstrip(),
            "```",
        ]
        if hints:
            parts += [
                "",
                f"TRANSLATION HINTS ({source_language} -> {target_language}):",
                hints.strip(),
            ]
        if last_feedback:
            parts += [
                "",
                "PREVIOUS ATTEMPT FAILED:",
                last_feedback,
                "",
                "Please fix the failure in your next attempt.  Return ONLY the "
                f"translated {target_language} source code.",
            ]
        else:
            parts += [
                "",
                f"Return ONLY the translated {target_language} source code.",
            ]
        return system, "\n".join(parts)

    def _query_model(self, sys_prompt: str, inst_prompt: str) -> str:
        """Direct LLM query — no tool loop, no OptimizationAgent.

        Attempts to use the model instance attached to the subagent
        via ``self.model`` (resolved from config).  If ``self.model``
        is absent (test contexts), falls back to constructing a new
        model via ``get_model`` from the config name.  The string
        result is returned verbatim; callers (``verify_fn`` /
        ``TranslationPhase``) handle any post-processing.
        """
        model = getattr(self, "model", None)
        if model is None:
            from minisweagent.models import get_model

            model = get_model(self.config.model_name, {})

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": inst_prompt},
        ]
        response = model.query(messages)
        if isinstance(response, dict):
            return str(response.get("content", ""))
        return str(response)


__all__ = ["TranslationAgent", "TranslationFailed", "TranslationResult"]
