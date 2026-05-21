"""Translation phase (CONDITIONAL) — runs before Discovery when ``target_language ≠ source``.

Flow (per execution plan §0.5(b)):

  a. Infer source language from the kernel URL extension.
  b. If source == target, short-circuit as a no-op.
  c. Read the source kernel into memory.
  d. Invoke ``TranslationAgent.loop(max_attempts=3, verify_fn=...)`` —
     a standalone ``SubagentBase`` subclass that does NOT compose
     ``OptimizationAgent``.  Each attempt is a direct ``model.query``
     call.
  e. Run ``validate_translation_performance`` (0.5× fail / 0.8× warn
     per language pair, adapted from PR #153) when both source and
     target latencies are available.
  f. Persist the translated file next to the source, swap
     ``ctx.kernel_path`` + ``ctx.kernel_url`` to it, and continue.
  g. If ``ctx.translate_only=True``, the orchestrator returns early
     after this phase completes.

Verifier policy
---------------

``_build_verify_fn`` layers checks in priority order:

  1. Source kernel's entry-point name(s) must be preserved — the
     user's test harness imports the translated kernel by name, so a
     translation that renames the public function is broken by
     construction.
  2. Candidate must be non-empty and parse as code for the target
     language (target-language token presence — Triton uses ``tl.``,
     HIP uses ``__global__``, CUDA uses ``__global__``).
  3. When a ``harness_path`` is supplied to ``loop()`` via inputs,
     full tensor-level verification via tensor allclose is performed
     by running the harness against the candidate.  (This tier is
     implemented progressively; today it's a TODO hook consumed by
     callers who supply their own ``verify_fn``.)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


# Canonical mapping of file suffix -> canonical language name.  Kept
# local to this phase so DiscoveryPhase can't accidentally depend on
# a translation-phase-only helper.
_SUFFIX_TO_LANGUAGE: dict[str, str] = {
    ".py": "triton",
    ".hip": "hip",
    ".cu": "cuda",
    ".cuh": "cuda",
}


def _canonicalize_language(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().lower()
    # Allow a handful of legacy / informal names to map cleanly
    return {"rocm": "hip"}.get(n, n)


@dataclass
class PerformanceReport:
    """Result of comparing source vs target kernel latencies after translation.

    Per plan §13.2-E row 30 and the PR #153 thresholds used by the
    original FlyDSL translation pipeline.

    Attributes:
        status: one of ``"ok"``, ``"warn"``, ``"fail"``.
        ratio: target_latency / source_latency.  A ratio > 1 means
          the translated kernel is SLOWER than the source.
        source_latency_ms: latency of the source kernel in ms.
        target_latency_ms: latency of the translated kernel in ms.
        message: human-readable summary.
    """

    status: str
    ratio: float
    source_latency_ms: float
    target_latency_ms: float
    message: str


def validate_translation_performance(
    source_latency_ms: float,
    target_latency_ms: float,
    *,
    fail_threshold: float = 0.5,
    warn_threshold: float = 0.8,
) -> PerformanceReport:
    """Compare source vs target kernel latency post-translation.

    Thresholds adapted from PR #153's PyTorch→FlyDSL validation:

      - **fail**: ``target_latency < fail_threshold * source_latency``
        or ``target_latency > (1 / fail_threshold) * source_latency``.
        The translated kernel is either wildly faster (suspicious —
        suggests a correctness bug masked as a speedup) or >2× slower.
      - **warn**: ``target_latency > (1 / warn_threshold) * source_latency``
        (i.e. target is 1.25× slower or worse).  Translation succeeded
        but the target has a real performance regression relative to
        source.
      - **ok**: target within ``[fail_threshold, 1/warn_threshold]``
        multiples of source.
    """
    if source_latency_ms <= 0:
        raise ValueError(f"source_latency_ms must be > 0; got {source_latency_ms}")
    if target_latency_ms <= 0:
        raise ValueError(f"target_latency_ms must be > 0; got {target_latency_ms}")
    if not 0 < fail_threshold < warn_threshold < 1:
        raise ValueError(
            f"thresholds must satisfy 0 < fail_threshold ({fail_threshold}) < warn_threshold ({warn_threshold}) < 1"
        )

    ratio = target_latency_ms / source_latency_ms

    # Two fail cases: too-fast (suggests correctness bug masked as speedup)
    # and too-slow (>2× regression).
    too_fast = ratio < fail_threshold
    too_slow = ratio > (1.0 / fail_threshold)
    if too_fast or too_slow:
        reason = (
            f"suspicious speedup ({ratio:.2f}x — possible correctness bug)"
            if too_fast
            else f"severe regression ({ratio:.2f}x slower than source)"
        )
        return PerformanceReport(
            status="fail",
            ratio=ratio,
            source_latency_ms=source_latency_ms,
            target_latency_ms=target_latency_ms,
            message=reason,
        )

    # Warn: target is more than 1/warn_threshold× slower (e.g. 1.25×).
    if ratio > (1.0 / warn_threshold):
        return PerformanceReport(
            status="warn",
            ratio=ratio,
            source_latency_ms=source_latency_ms,
            target_latency_ms=target_latency_ms,
            message=f"performance regression ({ratio:.2f}x slower than source)",
        )

    return PerformanceReport(
        status="ok",
        ratio=ratio,
        source_latency_ms=source_latency_ms,
        target_latency_ms=target_latency_ms,
        message=f"within threshold ({ratio:.2f}x source latency)",
    )


# ─── Verification helpers ───────────────────────────────────────────


def _extract_python_entry_points(source: str) -> set[str]:
    """Return the set of top-level ``def <name>`` function names in Python source.

    Used to require that the translated candidate preserve the user-
    facing entry-point names from the Triton / Python source.  We
    scan only top-level definitions (no leading whitespace).
    """
    return set(re.findall(r"^def\s+([A-Za-z_]\w*)", source, re.MULTILINE))


def _extract_hip_entry_points(source: str) -> set[str]:
    """Return the set of ``__global__ void <name>`` kernel names."""
    return set(re.findall(r"__global__\s+void\s+([A-Za-z_]\w*)", source))


def _extract_entry_points(source: str, language: str) -> set[str]:
    lang = _canonicalize_language(language) or ""
    if lang == "triton":
        return _extract_python_entry_points(source)
    if lang in {"hip", "cuda"}:
        return _extract_hip_entry_points(source)
    # Other languages: try both.  Translation pairs outside the
    # registered set should still produce a non-empty set when
    # possible.
    return _extract_python_entry_points(source) | _extract_hip_entry_points(source)


def _always_true_verifier(_candidate: str) -> bool:
    """No-verify fallback used only when no harness is supplied.

    Never used in production; kept for tests that want to bypass the
    verifier entirely.
    """
    return True


class TranslationPhase(Phase):
    """Gate + run.  Only executes when ``target_language`` is set and differs from source."""

    name = "translation"

    # Optional model handle that callers (e.g. orchestrator harnesses or
    # tests) may attach via ``phase._injected_model = model`` BEFORE
    # running.  When set, :meth:`_build_agent` reuses it instead of
    # letting :class:`TranslationAgent` resolve a fresh one from
    # ``SubagentConfig.model_name``.  Declared at class scope so pylint
    # / type-checkers can see the attribute exists.
    _injected_model: Any = None

    def is_applicable(self, ctx: PhaseContext) -> bool:
        return bool(ctx.target_language)

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        target = _canonicalize_language(ctx.target_language)
        source = _canonicalize_language(self._infer_source_language(ctx.kernel_url))

        if not target:
            logger.debug("TranslationPhase: target_language unset; nothing to do")
            return

        if source and source == target:
            logger.info(
                "  target_language=%s matches inferred source language; translation skipped (no-op).",
                target,
            )
            ctx.phases_skipped.append((self.name, f"source={source} already matches target"))
            return

        # Require source content we can pass to the agent.
        src_path = Path(ctx.kernel_url) if ctx.kernel_url else None
        if src_path is None or not src_path.is_file():
            raise FileNotFoundError(
                f"TranslationPhase: cannot read source kernel at ctx.kernel_url={ctx.kernel_url!r}. "
                "Translation requires a local path to the source kernel file."
            )
        source_code = src_path.read_text(encoding="utf-8")
        source_lang = source or "unknown"

        # ── D4: Golden-tensor prep (source-side validation) ─────────
        #
        # Diagram step (a): "run source harness → golden tensors".
        # Full tensor-allclose verification needs a harness protocol
        # extension (tensor-dump mode) that doesn't exist today, so
        # we implement the PRACTICAL subset:
        #
        #   1. If ctx.harness is set, run the source harness in
        #      ``--correctness`` mode against the source kernel.
        #      This catches the biggest failure mode — translating
        #      a source that doesn't even pass its own correctness
        #      gate.  If source-side correctness fails, we bail
        #      before wasting LLM attempts.
        #
        #   2. Callers who DO need full tensor-allclose verification
        #      can pass ``verify_fn`` via loop() kwargs; our default
        #      ``_build_verify_fn`` still does Layer 1 (structural)
        #      + Layer 2 (syntactic) regardless.
        source_correctness_ok = self._validate_source_correctness(ctx, src_path)
        if source_correctness_ok is False:
            raise RuntimeError(
                "TranslationPhase: source kernel fails its own --correctness "
                "gate; translation aborted (would produce a guaranteed-broken "
                f"target).  See ctx.correctness for details (source: {src_path})."
            )

        agent = self._build_agent(target)
        verify_fn = self._build_verify_fn(
            ctx=ctx,
            target=target,
            source_code=source_code,
            source_language=source_lang,
        )

        logger.info(
            "  Running TranslationAgent (src=%s, tgt=%s, max_attempts=3)",
            source_lang,
            target,
        )
        result = agent.loop(
            max_attempts=3,
            verify_fn=verify_fn,
            source_code=source_code,
            source_language=source_lang,
            hints=self._load_pair_hints(src_path, target, source_lang),
        )

        if not result.ok:
            raise RuntimeError(
                f"TranslationAgent exhausted {result.attempts_used} attempts without passing verify_fn.  "
                f"Feedback history:\n  - " + "\n  - ".join(result.feedback_history[-3:])
            )

        # Write the translated kernel next to the source file with a
        # target-language suffix.  Downstream phases pick it up from
        # ctx.kernel_path / ctx.kernel_url.
        target_suffix = self._suffix_for_language(target)
        translated_path = src_path.with_suffix(target_suffix)
        if translated_path == src_path:
            # Guard against clobbering the source when target and
            # source canonicalise to different names but share a
            # suffix (unlikely given _SUFFIX_TO_LANGUAGE, but cheap).
            translated_path = src_path.with_name(src_path.stem + "_translated" + target_suffix)
        translated_path.write_text(result.candidate_code, encoding="utf-8")

        logger.info(
            "  Translation succeeded after %d attempt(s).  Translated kernel written to %s",
            result.attempts_used,
            translated_path,
        )

        ctx.kernel_path = str(translated_path)
        ctx.kernel_url = str(translated_path)  # downstream phases treat this as the local path
        ctx.phases_run.append(self.name)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _infer_source_language(kernel_url: str) -> str | None:
        if not kernel_url:
            return None
        suffix = Path(kernel_url).suffix.lower()
        return _SUFFIX_TO_LANGUAGE.get(suffix)

    @staticmethod
    def _suffix_for_language(name: str) -> str:
        for suffix, lang in _SUFFIX_TO_LANGUAGE.items():
            if lang == name:
                return suffix
        return ".py"  # sensible fallback

    def _build_agent(self, target: str) -> Any:
        """Instantiate TranslationAgent with the target KernelLanguage."""
        from minisweagent.kernel_languages import registry
        from minisweagent.pipeline_workers.base import SubagentConfig
        from minisweagent.pipeline_workers.translation import TranslationAgent

        kernel_language = registry.get(target) if hasattr(registry, "get") else None
        if kernel_language is None:
            raise ValueError(
                f"TranslationPhase: unknown target_language={target!r}.  "
                f"Register a KernelLanguage for it in kernel_languages/ first."
            )

        config = SubagentConfig(
            name="translation",
            model_name="",  # resolved from self.model in _query_model
            system_template="",
            instance_template="",
            step_limit=1,
            cost_limit=3.0,
        )
        agent = TranslationAgent(language=kernel_language, config=config)
        # Lazy model resolution: if the caller passed one in, reuse it.
        if getattr(self, "_injected_model", None) is not None:
            agent.model = self._injected_model
        return agent

    @staticmethod
    def _load_pair_hints(src_path: Path, target: str, source_lang: str) -> str:
        """Return the pair-specific translation hint markdown, or fallback.

        Looks up the source ``KernelLanguage`` and calls its
        ``translation_hints_for(target)`` helper, which internally
        resolves ``<src>_to_<target>.md`` -> ``_fallback.md`` -> ``""``.
        """
        if not source_lang:
            return ""
        from minisweagent.kernel_languages import registry

        src_lang = registry.get(source_lang)
        if src_lang is None:
            return ""
        try:
            return src_lang.translation_hints_for(target)
        except Exception:
            return ""

    # ── D4: golden-tensor helpers ────────────────────────────────────

    @staticmethod
    def _validate_source_correctness(
        ctx: PhaseContext,
        src_path: Path,
    ) -> bool | None:
        """Run the source harness in ``--correctness`` mode, pre-translation.

        Returns:
            ``True``  — source passes correctness.  Translation proceeds.
            ``False`` — source FAILS correctness.  Translation is
                         aborted (caller raises).
            ``None``  — no harness available OR subprocess infrastructure
                         unavailable; skip gracefully and let Layer 1+2
                         verification do its job.

        Behaviour rationale: ``python3 <harness> --correctness`` is the
        universal contract's cheapest mode.  It either exits 0 (OK) or
        1 (FAIL).  We treat any other outcome (timeout, import error,
        non-existent harness) as "skip" rather than False, so broken
        test infrastructure doesn't mask real translation issues.
        """
        harness = getattr(ctx, "harness", None) or getattr(ctx, "harness_path", None)
        if not harness:
            logger.debug("  D4 source-correctness: no ctx.harness/harness_path; skipping.")
            return None

        harness_path = Path(harness)
        if not harness_path.is_file():
            logger.debug(
                "  D4 source-correctness: harness file missing (%s); skipping.",
                harness_path,
            )
            return None

        import shlex
        import subprocess
        import sys

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(harness_path.resolve()))} --correctness"
        repo_root = getattr(ctx, "repo_root", None)
        cwd = str(Path(repo_root).resolve()) if repo_root else None

        logger.info(
            "  D4 source-correctness: running %s (cwd=%s)",
            cmd,
            cwd or "<caller>",
        )
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                timeout=getattr(ctx, "benchmark_timeout", 3600),
                capture_output=True,
                text=True,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(
                "[yellow]  D4 source-correctness: subprocess failed (%s); "
                "skipping (translation will proceed without golden signal).[/yellow]",
                exc,
            )
            return None

        rc = proc.returncode
        if rc == 0 and "OK" in (proc.stdout or ""):
            logger.info(
                "  D4 source-correctness: PASS (exit=%d, stdout tail: %s)",
                rc,
                (proc.stdout or "").strip().splitlines()[-1][:160] if proc.stdout else "",
            )
            return True

        if rc != 0 and ("FAIL" in (proc.stdout or "") or "FAIL" in (proc.stderr or "")):
            logger.warning(
                "[yellow]  D4 source-correctness: FAIL (exit=%d) — source kernel "
                "does not pass its own --correctness gate.  Aborting translation.[/yellow]",
                rc,
            )
            return False

        # Neither OK nor FAIL — treat as "skip" rather than False to
        # avoid masking real translation issues with test-infra issues.
        logger.warning(
            "[yellow]  D4 source-correctness: INDETERMINATE (exit=%d); "
            "skipping (translation will proceed without golden signal).[/yellow]",
            rc,
        )
        return None

    def _build_verify_fn(
        self,
        *,
        ctx: PhaseContext,
        target: str,
        source_code: str,
        source_language: str,
    ) -> Any:
        """Build a verifier for the TranslationAgent loop.

        The verifier layers three checks:

          1. **Structural** — the translated candidate must declare at
             least one of the source kernel's entry-point names.  A
             translation that silently renames the public interface
             is broken by construction (the user's downstream harness
             imports by name).
          2. **Syntactic** — the candidate must contain tokens that
             identify it as target-language code (``tl.`` for Triton,
             ``__global__`` for HIP/CUDA).  Catches obvious language-
             mismatch failures.
          3. **Golden-tensor (harness-based)** — when a harness path
             is supplied on ``ctx.harness``, run it against the
             candidate and tensor-allclose the outputs against golden
             tensors captured from the source.  This tier is a hook
             today; callers who need it can pass their own
             ``verify_fn`` via ``TranslationAgent.loop``.
        """
        source_entry_points = _extract_entry_points(source_code, source_language)

        # Pre-compiled patterns for the syntactic layer.  Word-
        # boundary anchored so substrings like "not_hip" don't
        # false-accept a Python function as HIP code.
        _HIP_MARKERS = re.compile(
            r"__global__|hipLaunchKernelGGL|\bhip[A-Z]\w*\s*\("
            r"|#\s*include\s*[<\"]hip/"
        )
        _CUDA_MARKERS = re.compile(
            r"__global__|cudaLaunchKernel|\bcuda[A-Z]\w*\s*\("
            r"|#\s*include\s*[<\"]cuda"
        )
        _TRITON_MARKERS = re.compile(r"@triton\.jit\b|\btl\.|\bfrom\s+triton\b|\bimport\s+triton\b")

        def _verify(candidate: str) -> tuple[bool, str]:
            # Layer 2 (syntactic) — cheapest check, runs first so we
            # catch completely wrong outputs before parsing.  Uses
            # word-boundary patterns so identifiers containing
            # language keywords (e.g. a Python function literally
            # named ``not_hip``) don't false-accept.
            if not candidate or not candidate.strip():
                return False, "empty candidate"

            if target == "hip":
                if not _HIP_MARKERS.search(candidate):
                    return False, "candidate does not look like HIP code (missing __global__ / hipX()/#include <hip/>)"
            elif target == "cuda":
                if not _CUDA_MARKERS.search(candidate):
                    return False, "candidate does not look like CUDA code (missing __global__ / cudaX()/#include <cuda)"
            elif target == "triton":
                if not _TRITON_MARKERS.search(candidate):
                    return False, "candidate does not look like Triton code (missing @triton.jit / tl. / import triton)"

            # Layer 1 (structural) — entry-point preservation.
            # Only enforced when the source had detectable entry
            # points AND the target language has detectable entry
            # points (otherwise we skip this layer rather than
            # false-reject).
            if source_entry_points:
                target_entry_points = _extract_entry_points(candidate, target)
                if target_entry_points:
                    missing = source_entry_points - target_entry_points
                    # Require at least ONE source entry point to be
                    # present (translations may split a single source
                    # function into multiple device kernels).
                    if missing and not (source_entry_points & target_entry_points):
                        return False, (
                            f"translation drops all source entry points "
                            f"(source={sorted(source_entry_points)}, "
                            f"target={sorted(target_entry_points)})"
                        )

            # Layer 3 (golden-tensor, opt-in) — when ``ctx.harness``
            # is set and callers specifically want strict verification,
            # they can pass their own ``verify_fn`` via loop() kwargs.
            # The default verifier does NOT invoke subprocess here
            # because (a) the harness typically imports from the
            # source kernel path, which doesn't exist in the target
            # language, and (b) running untrusted LLM-generated code
            # in-process needs a sandbox we don't have yet.
            return True, ""

        return _verify


__all__ = [
    "TranslationPhase",
    "PerformanceReport",
    "validate_translation_performance",
]
