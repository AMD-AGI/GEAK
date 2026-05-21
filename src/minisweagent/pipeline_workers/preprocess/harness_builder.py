"""``HarnessBuilder`` — one-shot LLM subagent that produces a universal-contract harness.

Architectural contract (per execution plan §0.5(b) Harness phase):

  - **Narrow task**: take a kernel + optional user tests + the target
    ``KernelLanguage``'s Jinja harness template + builder hints, emit a
    single ``harness.py`` file.
  - **Universal contract enforced by construction**: the produced
    harness MUST expose
    ``--correctness / --benchmark / --full-benchmark / --profile``
    CLI flags and MUST emit ``GEAK_RESULT_LATENCY_MS`` /
    ``GEAK_RESULT_SPEEDUP`` markers.  After the LLM returns, we run
    ``kernel_languages.contract.validate_harness`` on the output; on
    ``ContractViolation`` we retry ONCE with the validator's error
    message appended to the instance prompt.
  - **Peer of OptimizationAgent, not subclass**.  HarnessBuilder is
    a ``SubagentBase`` subclass overriding ``run()`` (one-shot).  It
    does NOT inherit from ``OptimizationAgent`` and does NOT compose
    one via ``_make_optimization_agent`` — the builder is a single
    LLM call + contract validation, no tool runtime / strategy
    manager / patch-apply loop needed.

The implementation follows the same direct-model-query pattern as
``pipeline_workers/translation/translator.py`` (one prompt, one candidate,
validate + retry on structural failure).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from minisweagent.pipeline_workers.base import SubagentBase

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


# Default wallclock budget for HarnessBuilder's validate-retry loop.
# Architectural intent (execution plan §0.5(b) Harness phase): keep
# trying — with validator feedback fed back into the next prompt —
# until the universal contract is satisfied OR the budget is exhausted.
# 30 min is generous enough that a sluggish backend (e.g. token-rate-
# limited LLM) can still converge, but bounded enough that a stuck
# pipeline reports failure instead of hanging indefinitely.
_DEFAULT_WALLCLOCK_BUDGET_S = 30 * 60


# Heuristic regexes used by ``_strip_code_fences`` to extract the actual
# Python from LLM responses that may include prose preamble, prose
# epilogue, and/or a single code-fenced block.
_FENCED_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+\-.]*)\s*\n(.*?)```",
    re.DOTALL,
)
_PYTHON_LINE_START_RE = re.compile(
    r"^(?:#!|#\s|import |from |def |class |@\w|if __name__|\"\"\"|''')",
    re.MULTILINE,
)


# ──────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────


class HarnessBuildFailed(RuntimeError):
    """Raised when all retries fail the universal harness contract."""


@dataclass
class HarnessBuildResult:
    """Structured return value of ``HarnessBuilder.run``.

    Attributes:
        ok: True when the produced harness passes ``validate_harness``.
        harness_path: absolute str path to the written harness.  When
            ``ok`` is False, the last candidate is still written (so
            callers can inspect it) but the path is marked with a
            ``.rejected`` suffix.
        attempts_used: how many LLM round-trips happened (1 on success,
            up to ``max_retries + 1`` on failure).
        contract_errors: list of the validator's error strings from the
            last attempt.  Empty when ``ok`` is True.
        candidate_code: the final harness source (written to disk too).
    """

    ok: bool
    harness_path: str = ""
    attempts_used: int = 0
    contract_errors: list[str] = field(default_factory=list)
    candidate_code: str = ""


# ──────────────────────────────────────────────────────────────────────
# HarnessBuilder
# ──────────────────────────────────────────────────────────────────────


class HarnessBuilder(SubagentBase):
    """One-shot LLM subagent that produces a universal-contract harness.

    Usage
    -----

        builder = HarnessBuilder(language=triton_language, config=config)
        builder.model = my_model  # optional; falls back to config.model_name
        result = builder.run(
            kernel_path=Path("/path/to/kernel.py"),
            repo_root=Path("/path/to/repo"),
            out_path=Path("/path/to/output/harness.py"),
            user_test_files=[Path("/path/to/test.py")],  # optional
            discovery_context="...",                       # optional
            max_retries=1,
        )

        if result.ok:
            # result.harness_path has a contract-satisfying harness.py
            ...

    Why ``run`` and not ``loop``?
    -----------------------------
    HarnessBuilder is a **one-shot** task — one LLM query, one candidate,
    validate-and-done.  The retry-on-contract-failure is internal to
    ``run`` (capped at ``max_retries`` round-trips) and does NOT use the
    general ``SubagentBase.loop`` verify-retry machinery, because:
      1. ``loop`` takes a caller-supplied ``verify_fn``, but here the
         verifier is fixed (``validate_harness`` from the universal
         contract module) and not a policy decision of the caller.
      2. The retry count is small (default 1) and deterministic, not a
         meaningful "budget" the caller tunes.
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are a GPU kernel test-harness builder.  Your job is to produce "
        "a single Python harness.py that evaluates a kernel against a "
        "UNIVERSAL CONTRACT.  Return ONLY the harness source code — no "
        "prose, no markdown fences, no triple-backticks, no explanation "
        "before or after the code."
    )

    # The absolute minimum the generated harness must satisfy; repeated
    # verbatim in every instance prompt so the LLM cannot forget it.
    _UNIVERSAL_CONTRACT = (
        "UNIVERSAL HARNESS CONTRACT (non-negotiable):\n"
        "  1. Use argparse with a mutually-exclusive group of four flags:\n"
        "       --correctness        run the kernel and verify against a reference;\n"
        "                             print 'OK' if allclose, 'FAIL' otherwise;\n"
        "                             exit(0) on OK, exit(1) on FAIL.\n"
        "       --benchmark          time the kernel over multiple iterations;\n"
        "                             print 'GEAK_RESULT_LATENCY_MS=<float>' on stdout.\n"
        "       --full-benchmark     time + verify; print both\n"
        "                             'GEAK_RESULT_LATENCY_MS=<float>' and\n"
        "                             'GEAK_RESULT_SPEEDUP=<float>' on stdout,\n"
        "                             plus 'OK'/'FAIL'.\n"
        "       --profile            run the kernel with a short warm-up loop\n"
        "                             (profiler-friendly; no timing output required).\n"
        "  2. Use 'torch.cuda.synchronize()' around every timing section.\n"
        "  3. Warmup 5 iterations before each timed measurement; report median\n"
        "     over at least 100 measured iterations (not mean — medians are\n"
        "     robust to scheduler noise).\n"
        "  4. Use 'torch.allclose(candidate, reference, atol=1e-4, rtol=1e-4)'\n"
        "     for correctness unless the user test specifies tighter tolerances.\n"
        "  5. Print the GEAK_RESULT_* markers as standalone lines so downstream\n"
        "     regex parsers (run/postprocess/benchmark_parsing.py) can find them.\n"
    )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, **inputs: Any) -> str | dict:  # type: ignore[override]
        """Build a validated harness from a kernel + (optional) user tests.

        Required inputs:
          - ``kernel_path``  (Path): the kernel being wrapped
          - ``out_path``     (Path): where the harness is written

        Optional inputs:
          - ``repo_root``                (Path): informational; embedded in prompt
          - ``user_test_files``          (list[Path]): user-supplied test files
          - ``discovery_context``        (str): codebase context snippet
          - ``max_wallclock_seconds``    (float, default 1800.0 = 30 min):
                                          total wallclock budget for the
                                          validate-retry loop.  When the
                                          budget is exhausted, the latest
                                          candidate is written to
                                          ``harness.py.rejected`` and the
                                          call returns ok=False (or raises
                                          ``HarnessBuildFailed`` from the
                                          public ``run`` wrapper).
          - ``max_retries``              (int | None, default None):
                                          OPTIONAL safety cap on attempt
                                          count.  ``None`` means unbounded
                                          (only the wallclock terminates).
                                          Tests pin small values for
                                          deterministic termination.
          - ``seed_harness_source``      (str | None): existing harness
                                          source to hand the LLM as a
                                          starting point ("fix the contract
                                          violations in this" rather than
                                          "build from scratch").  When the
                                          user supplies a harness that's
                                          close-but-not-quite compliant,
                                          seeding preserves their domain
                                          knowledge (shape tables,
                                          reference impls, tolerances)
                                          and converges faster than a
                                          cold start.

        Returns:
          A dict ``{"harness_path": str, "attempts_used": int, "ok": bool}``
          on success, with the written file satisfying the universal
          harness contract.

        Raises:
          HarnessBuildFailed if the loop exits without a valid harness
          (either wallclock budget exhausted OR ``max_retries`` cap hit).
        """
        kernel_path = inputs.get("kernel_path")
        out_path = inputs.get("out_path")
        if kernel_path is None or out_path is None:
            raise ValueError("HarnessBuilder.run requires 'kernel_path' and 'out_path' inputs.")
        kernel_path = Path(kernel_path)
        out_path = Path(out_path)
        if not kernel_path.is_file():
            raise FileNotFoundError(f"kernel_path does not exist: {kernel_path}")

        repo_root = inputs.get("repo_root")
        repo_root = Path(repo_root) if repo_root is not None else None
        user_test_files_raw = inputs.get("user_test_files") or []
        user_test_files = [Path(p) for p in user_test_files_raw]
        discovery_context = str(inputs.get("discovery_context") or "")
        # Seed can come via raw source string (``seed_harness_source``)
        # or a path to read from (``seed_harness_path``).  The path
        # variant is what HarnessPhase uses to forward a user's
        # non-compliant harness; the source variant is for tests and
        # direct programmatic calls.
        seed_harness_source_raw = inputs.get("seed_harness_source")
        seed_harness_source = str(seed_harness_source_raw) if seed_harness_source_raw else ""
        seed_harness_path_raw = inputs.get("seed_harness_path")
        if seed_harness_path_raw and not seed_harness_source:
            _sp = Path(seed_harness_path_raw)
            if _sp.is_file():
                try:
                    seed_harness_source = _sp.read_text(encoding="utf-8")
                    logger.info(
                        "HarnessBuilder: seeded with %s (%d bytes)",
                        _sp,
                        len(seed_harness_source),
                    )
                except OSError as exc:
                    logger.warning(
                        "HarnessBuilder: could not read seed path %s: %s",
                        _sp,
                        exc,
                    )

        # Resolve wallclock + retry caps.  Precedence (highest first):
        #   1. ``run`` kwargs
        #   2. ``config.extra`` (lets the YAML-loaded SubagentConfig set them)
        #   3. defaults: 30 min wallclock, no retry cap
        # Both can be set simultaneously; the loop stops on whichever
        # fires first.  ``max_retries`` exists primarily so tests can
        # pin a deterministic attempt count without relying on timing.
        cfg_extra = (getattr(self, "config", None) and self.config.extra) or {}

        wallclock_raw = inputs.get("max_wallclock_seconds")
        if wallclock_raw is None:
            wallclock_raw = cfg_extra.get("max_wallclock_seconds")
        retries_raw = inputs.get("max_retries")
        if retries_raw is None:
            retries_raw = cfg_extra.get("max_retries")

        max_wallclock_seconds = (
            float(wallclock_raw) if wallclock_raw is not None else float(_DEFAULT_WALLCLOCK_BUDGET_S)
        )
        max_retries: int | None = max(0, int(retries_raw)) if retries_raw is not None else None

        kernel_source = self._safe_read(kernel_path)
        user_tests_blob = self._read_user_tests(user_test_files)

        result = self._build_with_retry(
            kernel_path=kernel_path,
            kernel_source=kernel_source,
            user_tests_blob=user_tests_blob,
            discovery_context=discovery_context,
            out_path=out_path,
            repo_root=repo_root,
            max_wallclock_seconds=max_wallclock_seconds,
            max_retries=max_retries,
            seed_harness_source=seed_harness_source,
        )

        if not result.ok:
            raise HarnessBuildFailed(
                f"HarnessBuilder failed after {result.attempts_used} attempt(s); "
                f"contract errors: {result.contract_errors}"
            )

        return {
            "harness_path": result.harness_path,
            "attempts_used": result.attempts_used,
            "ok": True,
        }

    # ------------------------------------------------------------------
    # Internals — broken out for testability
    # ------------------------------------------------------------------

    def _build_with_retry(
        self,
        *,
        kernel_path: Path,
        kernel_source: str,
        user_tests_blob: str,
        discovery_context: str,
        out_path: Path,
        repo_root: Path | None,
        max_wallclock_seconds: float,
        max_retries: int | None,
        seed_harness_source: str = "",
    ) -> HarnessBuildResult:
        """Validate-retry loop bounded by wallclock + optional attempt cap.

        Runs until any of:
          - the candidate passes ``_validate`` (returns ok=True), OR
          - ``max_wallclock_seconds`` elapses since loop entry, OR
          - ``max_retries`` is set and ``max_retries + 1`` attempts have
            completed (deterministic cap for tests).

        Validator errors from each failed attempt are fed back into the
        next prompt so the LLM can fix them.  This is the "loop until
        the universal contract is satisfied" behaviour from execution
        plan §0.5(b) Harness phase.

        When ``seed_harness_source`` is non-empty it's threaded into
        every prompt as the "starting point" — the LLM is asked to
        FIX the seed to satisfy the contract rather than generate a
        harness from scratch.  This preserves the user's domain
        knowledge (shape tables, reference impls, tolerances) and
        converges faster than a cold start.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_errors: list[str] = []
        last_candidate: str = ""
        attempt = 0
        deadline = time.monotonic() + max(0.0, float(max_wallclock_seconds))
        max_attempts_cap = (max_retries + 1) if max_retries is not None else None

        logger.info(
            "HarnessBuilder loop start: budget=%.0fs, max_attempts=%s, language=%s, kernel=%s",
            max_wallclock_seconds,
            max_attempts_cap if max_attempts_cap is not None else "unbounded",
            self.language.name,
            kernel_path.name,
        )

        while True:
            remaining = deadline - time.monotonic()
            if attempt > 0 and remaining <= 0.0:
                logger.warning(
                    "HarnessBuilder wallclock budget exhausted after attempt %d "
                    "(budget=%.0fs); writing rejected candidate.",
                    attempt,
                    max_wallclock_seconds,
                )
                break
            if max_attempts_cap is not None and attempt >= max_attempts_cap:
                logger.warning(
                    "HarnessBuilder hit max_attempts cap (%d); writing rejected candidate.",
                    max_attempts_cap,
                )
                break

            attempt += 1
            sys_p, inst_p = self._compose_harness_prompt(
                kernel_path=kernel_path,
                kernel_source=kernel_source,
                user_tests_blob=user_tests_blob,
                discovery_context=discovery_context,
                repo_root=repo_root,
                last_errors=last_errors,
                attempt=attempt,
                seed_harness_source=seed_harness_source,
            )

            cap_str = f"/{max_attempts_cap}" if max_attempts_cap is not None else ""
            logger.info(
                "HarnessBuilder attempt %d%s (%.0fs of %.0fs budget remaining)",
                attempt,
                cap_str,
                max(0.0, remaining if attempt > 1 else max_wallclock_seconds),
                max_wallclock_seconds,
            )
            raw = self._query_model(sys_p, inst_p)
            candidate = self._strip_code_fences(raw)
            last_candidate = candidate

            # Write to disk so validate_harness (which reads from a
            # Path) can inspect it, and so callers can see the
            # artifact even when it fails.
            out_path.write_text(candidate, encoding="utf-8")

            errors = self._validate(out_path)
            if not errors:
                logger.info(
                    "HarnessBuilder succeeded on attempt %d (-> %s)",
                    attempt,
                    out_path,
                )
                return HarnessBuildResult(
                    ok=True,
                    harness_path=str(out_path),
                    attempts_used=attempt,
                    contract_errors=[],
                    candidate_code=candidate,
                )

            last_errors = errors
            logger.info(
                "  HarnessBuilder attempt %d failed validation (%d error(s)): %s",
                attempt,
                len(errors),
                "; ".join(errors)[:400],
            )

        # Loop exited without a valid candidate.  Persist the final
        # attempt next to ``out_path`` with a ``.rejected`` suffix so
        # callers can diff + debug.
        rejected_path = out_path.with_suffix(out_path.suffix + ".rejected")
        rejected_path.write_text(last_candidate, encoding="utf-8")
        return HarnessBuildResult(
            ok=False,
            harness_path=str(rejected_path),
            attempts_used=attempt,
            contract_errors=last_errors,
            candidate_code=last_candidate,
        )

    def _compose_harness_prompt(
        self,
        *,
        kernel_path: Path,
        kernel_source: str,
        user_tests_blob: str,
        discovery_context: str,
        repo_root: Path | None,
        last_errors: list[str],
        attempt: int,
        seed_harness_source: str = "",
    ) -> tuple[str, str]:
        """Render (system, instance) prompts for one attempt.

        The system prompt is a short role description (target-language
        system prompt if the KernelLanguage supplies one, else a generic
        fallback).  The instance prompt carries:

          - the universal contract (verbatim, non-negotiable)
          - the Jinja harness skeleton (if the language supplies one)
          - language-specific builder hints (if supplied)
          - the kernel source being wrapped
          - any user test files (informational — shape / reference hints)
          - discovery context (optional codebase hint)
          - when ``seed_harness_source`` is non-empty: the user's
            existing harness as a starting template with an explicit
            "fix the contract violations in this" directive.  This
            converges dramatically faster than generating from scratch
            when the user's harness is close-to-compliant.
          - on retry: the contract-validation errors from the previous
            attempt as explicit "fix these" directives.
        """
        system = self._DEFAULT_SYSTEM_PROMPT
        try:
            lang_system = self.language.system_prompt  # lazy-load from language
        except Exception:
            lang_system = ""
        if lang_system.strip():
            system = lang_system

        parts: list[str] = [
            f"TARGET LANGUAGE: {self.language.name}",
            "",
            self._UNIVERSAL_CONTRACT,
            "",
        ]

        try:
            template_blob = self.language.harness_template
        except Exception:
            template_blob = ""
        if template_blob.strip():
            parts += [
                "JINJA HARNESS SKELETON (fill in the placeholders — emit plain Python, NOT Jinja):",
                "```",
                template_blob.rstrip(),
                "```",
                "",
            ]

        try:
            builder_hints = self.language.builder_hints
        except Exception:
            builder_hints = ""
        if builder_hints.strip():
            parts += [
                f"{self.language.name.upper()} BUILDER HINTS:",
                builder_hints.strip(),
                "",
            ]

        parts += [
            f"KERNEL SOURCE ({kernel_path.name}):",
            "```",
            kernel_source.rstrip(),
            "```",
            "",
        ]

        if user_tests_blob.strip():
            parts += [
                "USER TEST FILE(S) (for reference — informs input shapes / reference impl):",
                user_tests_blob.rstrip(),
                "",
            ]

        if discovery_context.strip():
            parts += [
                "CODEBASE CONTEXT:",
                discovery_context.strip(),
                "",
            ]

        if repo_root is not None:
            parts += [
                f"REPO ROOT: {repo_root}",
                "",
            ]

        if seed_harness_source.strip():
            parts += [
                "STARTING HARNESS (user-provided — fix contract violations in this file, DO NOT rewrite from scratch):",
                "```",
                seed_harness_source.rstrip(),
                "```",
                "",
                "Preserve the user's domain knowledge (shape tables, "
                "reference implementations, tolerances) from the starting "
                "harness.  Only change what's necessary to satisfy the "
                "universal contract (all four argparse flags + both "
                "GEAK_RESULT markers + runtime pass on every mode).",
                "",
            ]

        if last_errors:
            parts += [
                f"PREVIOUS ATTEMPT (#{attempt - 1}) FAILED THE UNIVERSAL CONTRACT:",
                "",
                *(f"  - {err}" for err in last_errors),
                "",
                "Fix EVERY listed issue in your next attempt.  Return ONLY the full "
                "harness.py source code (no markdown fences, no prose).",
            ]
        else:
            parts += [
                "Return the full harness.py source code.  Plain Python only — NO "
                "markdown fences, NO triple-backticks, NO prose before or after.",
            ]

        return system, "\n".join(parts)

    def _query_model(self, sys_prompt: str, inst_prompt: str) -> str:
        """Direct model query — no OptimizationAgent, no tool runtime.

        Mirrors ``TranslationAgent._query_model`` so both narrow subagents
        share the same "one prompt, one response" shape.
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

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_user_tests(paths: list[Path]) -> str:
        """Concatenate user test files into a single blob with headers.

        Each file is prefixed with ``--- <filename> ---`` so the LLM
        can distinguish them.  Unreadable files are skipped with a
        logged warning rather than aborting the build.
        """
        chunks: list[str] = []
        for p in paths:
            try:
                body = p.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("HarnessBuilder: skipping unreadable test file %s: %s", p, exc)
                continue
            chunks.append(f"--- {p.name} ---\n{body.rstrip()}")
        return "\n\n".join(chunks)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Extract the harness Python from an LLM response.

        The system prompt forbids markdown fences and prose.  Reality
        is messier — we observe at least three response shapes:

          1. **Pure code** (no fences, starts with Python-looking
             syntax).  Returned as-is.
          2. **Code wrapped in a single ``` ...```` fence**, optionally
             with a language hint (``` ```python```).  The fence is
             stripped.
          3. **Prose preamble + one or more fenced blocks** (LLMs love
             to "explain" before emitting code).  We pick the LARGEST
             fenced block — the harness is by far the longest piece of
             code in the response, so size is a robust selector.

        If we still cannot find code (no fences, prose-first), we look
        for the first Python-looking line and discard everything before
        it.  When that also fails we return the raw text and let the
        contract validator surface the error.
        """
        if not text:
            return ""

        # Case 3: response contains explicit code fences anywhere in the
        # body.  Pick the largest fenced block — the harness is the
        # longest code in the response by construction.
        fenced = _FENCED_BLOCK_RE.findall(text)
        if fenced:
            largest = max(fenced, key=len).rstrip()
            return largest + "\n"

        # Case 1: response starts with code.  Return as-is.
        stripped = text.lstrip()
        if not stripped:
            return ""
        first_line = stripped.split("\n", 1)[0].strip()
        if (
            first_line.startswith(("#!", "#", '"""', "'''", "import ", "from ", "def ", "class ", "@"))
            or first_line == "if __name__"
        ):
            return text

        # Case 2 (no closing fence): response begins with ``` but the
        # closing fence is missing — strip the leading line.
        if stripped.startswith("```"):
            lines = stripped.splitlines()[1:]
            while lines and lines[-1].strip() in ("", "```"):
                lines.pop()
            return "\n".join(lines) + "\n"

        # Prose-first with NO fence: try to locate the first Python-
        # looking line and discard preamble.
        m = _PYTHON_LINE_START_RE.search(text)
        if m:
            return text[m.start() :].rstrip() + "\n"

        # Pure prose — give up; let the validator complain about the
        # missing flags/markers (which is the right error message
        # anyway, since the LLM clearly didn't produce a harness).
        return text

    @staticmethod
    def _validate(path: Path) -> list[str]:
        """Run the universal harness contract validator, returning error list.

        ``kernel_languages.contract.validate_harness`` raises
        ``ContractViolation`` on failure; we translate that into a
        string list the retry loop can feed back to the LLM.  An empty
        list means the harness passed.
        """
        from minisweagent.kernel_languages.contract import (
            REQUIRED_HARNESS_FLAGS,
            REQUIRED_HARNESS_MARKERS,
            ContractViolation,
            validate_harness,
        )

        try:
            validate_harness(path)
        except ContractViolation as exc:
            return [str(exc)]
        except Exception as exc:  # noqa: BLE001 — validator surprises
            return [f"unexpected validator error: {exc}"]

        # The shipped validate_harness is permissive (passes when EITHER
        # flags OR markers are present).  For HarnessBuilder we want
        # STRICT compliance so retries trigger on partial failures.
        text = path.read_text(encoding="utf-8", errors="ignore")
        missing_flags = [f for f in REQUIRED_HARNESS_FLAGS if f not in text]
        missing_markers = [m for m in REQUIRED_HARNESS_MARKERS if m not in text]
        errors: list[str] = []
        if missing_flags:
            errors.append(f"missing required argparse flags: {missing_flags}")
        if missing_markers:
            errors.append(f"missing required stdout markers: {missing_markers}")
        return errors


__all__ = [
    "HarnessBuilder",
    "HarnessBuildFailed",
    "HarnessBuildResult",
]
