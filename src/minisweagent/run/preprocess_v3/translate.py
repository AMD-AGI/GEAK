"""Step 2 — PyTorch -> FlyDSL translation wrapper for the v3 preprocess pipeline.

This module is a **thin typed projection** over the legacy
:func:`minisweagent.run.preprocess.translate.run_translation`. It exists so
the v3 orchestrator can call a single deterministic Python function whose
return shape is a frozen dataclass (rather than a free-form ``dict``) and
whose contract is explicit about FlyDSL being the only target language v3
currently supports.

The legacy ``run_translation`` is still authoritative — it owns the
multi-round translation agent, the harness construction, the LLM-driven
self-review, and the performance-regression gate. This wrapper does **not**:

* drive any LLM loop of its own,
* re-implement any part of the translation policy,
* mutate the legacy module.

It does:

1. Forward the v3-shaped arguments to ``run_translation``,
2. Pin ``target_language="flydsl"`` so v3 callers can never accidentally
   trigger a different translation pair,
3. Project the legacy ``dict`` into a typed :class:`TranslationResult` with
   stable field names (no ``translation_*`` prefixes),
4. Surface ``elapsed_s`` derived from a wall-clock timer around the legacy
   call so callers don't have to reach into ``raw["translation_elapsed_s"]``
   (which the legacy code only writes when at least one round runs).

The legacy import boundary is tagged with a ``TODO(commit-set-5)`` marker so
the cleanup pass that retires ``run/preprocess/`` knows to inline this
function's body once the legacy module goes away.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# TODO(commit-set-5): inline; old preprocess/ goes away
from minisweagent.run.preprocess.translate import run_translation

logger = logging.getLogger(__name__)


#: The only target language v3 currently supports. Pinned so misconfiguration
#: in the orchestrator surface (passing a different string upstream) can't
#: silently route into a different translation pair than the rest of the
#: pipeline expects.
TARGET_LANGUAGE: str = "flydsl"


@dataclass(frozen=True)
class TranslationResult:
    """Typed projection of the legacy ``run_translation`` dict.

    Attributes:
        success:
            ``True`` when the translation agent produced a candidate that
            passed both correctness and the legacy performance-regression
            gate. Mirrors ``raw["translation_success"]`` exactly.
        target_language:
            Resolved target language as reported by the legacy code.
            Always ``"flydsl"`` in v3 (we pass it explicitly), but echoed
            back in case the legacy registry ever rewrites it (e.g. an
            alias).
        translated_kernel_path:
            Filesystem path to the candidate kernel produced by the
            translation agent, or ``None`` when ``success`` is ``False``.
            Mirrors ``raw["translation_kernel_path"]``.
        speedup:
            FlyDSL/PyTorch speedup parsed from harness output, or ``None``
            when timing wasn't measured (e.g. the run failed before the
            performance check). Mirrors ``raw["translation_speedup"]``.
        self_review:
            Self-review verdict string (``"passed"``, ``"skipped"``,
            ``"passed_with_issues"``, ``"accepted_with_issues"``,
            ``"review_error"``, or ``""`` when the legacy code didn't
            populate it). Mirrors ``raw["translation_self_review"]``.
        errors:
            Per-attempt error strings collected by the legacy translation
            loop. Empty list on a clean success.
        elapsed_s:
            Wall-clock seconds spent inside :func:`translate_to_flydsl`.
            Computed in the wrapper rather than read from the legacy dict
            because the legacy ``translation_elapsed_s`` key is only set
            when at least one round runs (a fast no-op on "no translation
            pair found" returns without it).
        raw:
            The original ``dict`` returned by ``run_translation``, carried
            verbatim so callers that need a key the wrapper doesn't yet
            project (or want to write the entire dict to disk for audit)
            can still get it without re-running translation.
    """

    success: bool
    target_language: str
    translated_kernel_path: Path | None
    speedup: float | None
    self_review: str
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def _coerce_path(value: Any) -> Path | None:
    """Best-effort string -> Path coercion preserving ``None``."""
    if value is None:
        return None
    return Path(str(value))


def _project(raw: dict[str, Any], elapsed_s: float) -> TranslationResult:
    """Project a ``run_translation`` dict into a :class:`TranslationResult`.

    Carries every recognised field over with type normalisation:

    * ``translation_kernel_path`` (``str | None``) -> ``Path | None``,
    * ``translation_speedup`` (``float | None``) -> ``float | None``,
    * ``translation_self_review`` (optional string, may be missing) ->
      ``str`` (empty when absent — callers shouldn't have to worry about
      missing keys),
    * ``translation_errors`` (``list[str]``) -> ``list[str]`` (defaults to
      ``[]``).

    The full ``raw`` dict is preserved on the result so callers can fall
    through to legacy keys we haven't promoted (e.g.
    ``translation_pytorch_latency_ms``, ``translation_review_findings``).
    """
    return TranslationResult(
        success=bool(raw.get("translation_success", False)),
        target_language=str(raw.get("translation_target_language") or TARGET_LANGUAGE),
        translated_kernel_path=_coerce_path(raw.get("translation_kernel_path")),
        speedup=raw.get("translation_speedup"),
        self_review=str(raw.get("translation_self_review") or ""),
        errors=list(raw.get("translation_errors") or []),
        elapsed_s=round(elapsed_s, 3),
        raw=dict(raw),
    )


def translate_to_flydsl(
    *,
    source_path: Path,
    output_dir: Path,
    gpu_id: int = 0,
    model: Any = None,
    model_factory: Any = None,
    repo: Path | None = None,
    flydsl_repo: Path | None = None,
    console: Any = None,
) -> TranslationResult:
    """Translate a PyTorch kernel to FlyDSL via the legacy translation agent.

    This wrapper is **strictly deterministic from the v3 orchestrator's
    point of view** — it does not own an LLM loop, but the legacy
    ``run_translation`` does drive its own translation agent inside. The
    wrapper's job is to:

    1. Forward the v3 surface to ``run_translation``,
    2. Pin ``target_language="flydsl"``,
    3. Time the call,
    4. Project the dict into :class:`TranslationResult`.

    Args:
        source_path:
            Path to the PyTorch kernel to translate. Forwarded as
            ``run_translation``'s ``kernel_path`` parameter.
        output_dir:
            Directory where the translation agent should write candidate
            kernels and per-round logs. The legacy code creates it on
            demand.
        gpu_id:
            ``HIP_VISIBLE_DEVICES`` value used when running the
            translation harness. Defaults to GPU 0.
        model:
            Pre-built LLM model instance. When ``None`` and
            ``model_factory`` is also ``None``, the legacy code falls back
            to the agent-config model. The orchestrator typically passes
            the global AMD-router model directly.
        model_factory:
            Callable returning a fresh model instance. Forwarded as-is to
            ``run_translation`` for callers that want per-round model
            isolation.
        repo:
            Repository root for ``PYTHONPATH`` plumbing inside the
            translation harness. Defaults to ``source_path.parent`` in the
            legacy code.
        flydsl_repo:
            Optional path to a local FlyDSL clone. When set, the legacy KB
            loader pulls reference docs from the repo instead of the
            authored KB files. Forwarded unchanged.
        console:
            Optional Rich console for progress output. Forwarded
            unchanged.

    Returns:
        A :class:`TranslationResult` projecting every documented field of
        the legacy ``dict``. The full legacy dict is also available via
        :attr:`TranslationResult.raw` for callers that need keys this
        wrapper doesn't yet promote.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)

    t0 = time.monotonic()
    raw = run_translation(
        kernel_path=source_path,
        output_dir=output_dir,
        gpu_id=gpu_id,
        target_language=TARGET_LANGUAGE,
        model=model,
        model_factory=model_factory,
        repo=repo,
        flydsl_repo=flydsl_repo,
        console=console,
    )
    elapsed_s = time.monotonic() - t0

    if not isinstance(raw, dict):
        raise TypeError(
            f"translate_to_flydsl: run_translation returned {type(raw).__name__}, "
            f"expected dict (legacy contract violated)"
        )

    return _project(raw, elapsed_s)


__all__ = [
    "TARGET_LANGUAGE",
    "TranslationResult",
    "translate_to_flydsl",
]
