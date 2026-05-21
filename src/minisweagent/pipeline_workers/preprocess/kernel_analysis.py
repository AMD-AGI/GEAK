"""``KernelAnalysisAgent`` — one-shot subagent producing the [A]-[D] analysis rubric.

Per execution plan §0.5(b) Explore phase, the rubric markdown summarises:

    [A] Primitives        — GEMM / reduce / elementwise / scatter / etc.
    [B] Shape Regimes     — tiles, head counts, sequence length bands
    [C] Profile Hotspots  — top-N kernels + roofline position
    [D] Attack Surfaces   — the ordered list of optimisations the
                             optimizer should try, derived from [A]+[B]+[C]

The output is consumed by ``compose_task_body`` (prepended to the task
body as structured context for both fixed and planned modes) and by
the ``CrossSessionMemoryAnalysisAgent`` when deciding whether KB
entries are transferable.

Like HarnessBuilder, this is a ``SubagentBase`` subclass that
overrides ``run()`` (one-shot) and does NOT inherit from or compose
``OptimizationAgent``.  It performs a direct ``model.query`` with a
single prompt, then lightly validates that the four rubric headers
appear in the output — on failure we retry once, on second failure we
return the best-effort markdown as-is (the analysis is advisory, not
a hard contract, so we do not raise).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from minisweagent.pipeline_workers.base import SubagentBase

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────


REQUIRED_RUBRIC_HEADERS = (
    "## [A] Primitives",
    "## [B] Shape Regimes",
    "## [C] Profile Hotspots",
    "## [D] Attack Surfaces",
)


@dataclass
class KernelAnalysisResult:
    """Structured return value of ``KernelAnalysisAgent.run``.

    Always returns successfully (analysis is advisory).  ``ok`` is
    True when all four rubric headers were present in the LLM output,
    False when we fell through to best-effort on retry exhaustion.
    """

    ok: bool
    analysis_path: str = ""
    attempts_used: int = 0
    missing_headers: list[str] = None  # type: ignore[assignment]
    markdown: str = ""

    def __post_init__(self) -> None:
        if self.missing_headers is None:
            self.missing_headers = []


# ──────────────────────────────────────────────────────────────────────
# KernelAnalysisAgent
# ──────────────────────────────────────────────────────────────────────


class KernelAnalysisAgent(SubagentBase):
    """One-shot producer of the [A]-[D] kernel analysis markdown.

    Usage
    -----

        agent = KernelAnalysisAgent(language=triton_language, config=config)
        agent.model = my_model  # optional; falls back to config.model_name
        result = agent.run(
            kernel_path=Path("/path/to/kernel.py"),
            out_path=Path("/path/to/output/kernel_analysis.md"),
            profile=profile_dict,              # optional
            baseline_metrics=metrics_dict,     # optional
            codebase_context_path=Path(...),   # optional
            max_retries=1,
        )

        if result.ok:
            # result.analysis_path has the [A]-[D] rubric markdown
            ...

    Why best-effort instead of strict?
    ----------------------------------
    Unlike HarnessBuilder (which must produce a contract-compliant
    executable harness), the analysis rubric is ADVISORY context fed
    to the downstream optimizer.  A partially-structured analysis is
    still useful; silently discarding it because of a missing header
    would regress prompt context.  We retry once for a clean output,
    then fall through to best-effort — and the ExplorePhase wrapper
    is responsible for dropping it entirely if the LLM returns garbage.
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are an expert GPU kernel analyst.  Given the full source of a "
        "GPU kernel, its baseline profile data, and any codebase context "
        "available, produce a structured [A]-[D] analysis rubric in "
        "markdown.  Return ONLY the four required sections — no prose "
        "before or after, no markdown fences, no additional headers."
    )

    _RUBRIC_TEMPLATE = (
        "Produce the following FOUR markdown sections VERBATIM, with these "
        "exact level-2 headers and in this exact order.  Content inside each "
        "section is yours to write (informed by the inputs below), but the "
        "headers themselves are non-negotiable.\n\n"
        "## [A] Primitives\n"
        "Identify the fundamental GPU operations this kernel performs.  "
        "Name them concretely (e.g. 'reduction over axis=1 producing "
        "softmax logits', 'two GEMMs back-to-back: [M,K] @ [K,N] -> [M,N] "
        "then [M,N] @ [N,D] -> [M,D]', 'scatter-add on int32 indices').  "
        "Classify as memory-bound / compute-bound / latency-bound / LDS-bound "
        "if the profile supports it.\n\n"
        "## [B] Shape Regimes\n"
        "Identify the shape regimes this kernel serves.  For matmul-like: "
        "tile dimensions (BLOCK_M, BLOCK_N, BLOCK_K).  For elementwise: "
        "BLOCK_SIZE.  For attention: head count, sequence length, head dim.  "
        "If the test harness exercises specific shapes, list them.\n\n"
        "## [C] Profile Hotspots\n"
        "From the profile data, identify:\n"
        "- top-N hottest sub-kernels by duration (with percentages if "
        "available)\n"
        "- roofline position (memory-bound / compute-bound / "
        "latency-bound / LDS-bound)\n"
        "- HBM utilization %, L2 hit rate %, any obvious stalls or "
        "bottlenecks\n"
        "If no profile data is supplied, say so explicitly — do not invent "
        "numbers.\n\n"
        "## [D] Attack Surfaces\n"
        "Ordered list (most promising first) of optimisations the optimizer "
        "should try, derived from [A] + [B] + [C].  Each entry: one-line "
        "hypothesis + expected impact magnitude (high / medium / low).  "
        "Prefer kernel-body rewrites over autotune sweeps."
    )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, **inputs: Any) -> str | dict:  # type: ignore[override]
        """Produce the [A]-[D] analysis rubric.

        Required inputs:
          - ``kernel_path`` (Path): the kernel source to analyse
          - ``out_path``    (Path): where the markdown is written

        Optional inputs:
          - ``profile``                  (dict): profile.json payload
          - ``baseline_metrics``         (dict): baseline_metrics.json payload
          - ``codebase_context_path``    (Path): CODEBASE_CONTEXT.md path
          - ``max_retries``              (int, default 1): extra attempts

        Returns:
          A dict ``{"analysis_path": str, "attempts_used": int,
                    "ok": bool, "missing_headers": list[str]}``.

        Never raises.  On total failure the markdown is still written
        (best-effort) and ``ok`` is False.
        """
        kernel_path = inputs.get("kernel_path")
        out_path = inputs.get("out_path")
        if kernel_path is None or out_path is None:
            raise ValueError("KernelAnalysisAgent.run requires 'kernel_path' and 'out_path' inputs.")
        kernel_path = Path(kernel_path)
        out_path = Path(out_path)
        if not kernel_path.is_file():
            raise FileNotFoundError(f"kernel_path does not exist: {kernel_path}")

        profile = inputs.get("profile") or {}
        baseline_metrics = inputs.get("baseline_metrics") or {}
        codebase_context_path = inputs.get("codebase_context_path")
        max_retries = max(0, int(inputs.get("max_retries", 1)))

        kernel_source = self._safe_read(kernel_path)
        codebase_context = (
            self._safe_read(Path(codebase_context_path))
            if codebase_context_path and Path(codebase_context_path).is_file()
            else ""
        )
        profile_blob = self._format_profile_blob(profile, baseline_metrics)

        result = self._build_with_retry(
            kernel_path=kernel_path,
            kernel_source=kernel_source,
            profile_blob=profile_blob,
            codebase_context=codebase_context,
            out_path=out_path,
            max_retries=max_retries,
        )

        return {
            "analysis_path": result.analysis_path,
            "attempts_used": result.attempts_used,
            "ok": result.ok,
            "missing_headers": result.missing_headers,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_with_retry(
        self,
        *,
        kernel_path: Path,
        kernel_source: str,
        profile_blob: str,
        codebase_context: str,
        out_path: Path,
        max_retries: int,
    ) -> KernelAnalysisResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_markdown = ""
        last_missing: list[str] = []
        total_attempts = max_retries + 1

        for attempt in range(1, total_attempts + 1):
            sys_p, inst_p = self._compose_analysis_prompt(
                kernel_path=kernel_path,
                kernel_source=kernel_source,
                profile_blob=profile_blob,
                codebase_context=codebase_context,
                last_missing=last_missing,
                attempt=attempt,
            )

            logger.info(
                "KernelAnalysisAgent attempt %d/%d (language=%s, kernel=%s)",
                attempt,
                total_attempts,
                self.language.name,
                kernel_path.name,
            )
            raw = self._query_model(sys_p, inst_p)
            markdown = self._strip_code_fences(raw)
            last_markdown = markdown

            missing = [h for h in REQUIRED_RUBRIC_HEADERS if h not in markdown]
            if not missing:
                out_path.write_text(markdown, encoding="utf-8")
                logger.info(
                    "KernelAnalysisAgent succeeded on attempt %d (-> %s)",
                    attempt,
                    out_path,
                )
                return KernelAnalysisResult(
                    ok=True,
                    analysis_path=str(out_path),
                    attempts_used=attempt,
                    missing_headers=[],
                    markdown=markdown,
                )

            last_missing = missing
            logger.info(
                "  KernelAnalysisAgent attempt %d missing headers: %s",
                attempt,
                ", ".join(missing),
            )

        # Best-effort fallback: write whatever we got and mark ok=False
        # so ExplorePhase can decide whether to feed it to the task body.
        out_path.write_text(last_markdown, encoding="utf-8")
        logger.warning(
            "KernelAnalysisAgent exhausted %d attempt(s); best-effort markdown written to %s (missing headers: %s)",
            total_attempts,
            out_path,
            last_missing,
        )
        return KernelAnalysisResult(
            ok=False,
            analysis_path=str(out_path),
            attempts_used=total_attempts,
            missing_headers=last_missing,
            markdown=last_markdown,
        )

    def _compose_analysis_prompt(
        self,
        *,
        kernel_path: Path,
        kernel_source: str,
        profile_blob: str,
        codebase_context: str,
        last_missing: list[str],
        attempt: int,
    ) -> tuple[str, str]:
        """Render (system, instance) prompts for one attempt."""
        system = self._DEFAULT_SYSTEM_PROMPT
        try:
            lang_system = self.language.system_prompt
        except Exception:
            lang_system = ""
        if lang_system.strip():
            system = lang_system

        parts: list[str] = [
            f"TARGET LANGUAGE: {self.language.name}",
            "",
            f"KERNEL SOURCE ({kernel_path.name}):",
            "```",
            kernel_source.rstrip(),
            "```",
            "",
        ]

        if profile_blob.strip():
            parts += [
                "PROFILE DATA:",
                profile_blob.rstrip(),
                "",
            ]
        else:
            parts += [
                "PROFILE DATA: (none supplied — skip the measured-numbers "
                "parts of [C] and say so explicitly rather than inventing "
                "figures)",
                "",
            ]

        if codebase_context.strip():
            parts += [
                "CODEBASE CONTEXT:",
                codebase_context.rstrip()[:8000],  # cap context size
                "",
            ]

        parts += ["", self._RUBRIC_TEMPLATE, ""]

        if last_missing:
            parts += [
                f"PREVIOUS ATTEMPT (#{attempt - 1}) MISSED THESE REQUIRED HEADERS:",
                *(f"  - {h}" for h in last_missing),
                "",
                "Include ALL four section headers with the EXACT text shown "
                "in the template.  Return ONLY the markdown.",
            ]
        else:
            parts += [
                "Return ONLY the markdown for the four sections.  No fences, no prose before or after.",
            ]

        return system, "\n".join(parts)

    def _query_model(self, sys_prompt: str, inst_prompt: str) -> str:
        """Direct model query — mirrors HarnessBuilder / TranslationAgent pattern."""
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
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _format_profile_blob(profile: Any, baseline_metrics: Any) -> str:
        """Render profile + baseline_metrics into a compact text blob.

        We pick the high-signal fields by convention rather than
        dumping the full JSON — the full dump is long and dilutes the
        LLM's attention.  Missing fields are silently omitted.
        """
        bm = baseline_metrics if isinstance(baseline_metrics, dict) else {}
        prof = profile if isinstance(profile, dict) else {}

        lines: list[str] = []

        # Baseline metrics summary
        if bm:
            dur = bm.get("duration_us")
            if dur is not None:
                lines.append(f"- Baseline duration: {dur} us")
            bottleneck = bm.get("bottleneck")
            if bottleneck:
                lines.append(f"- Bottleneck classification: {bottleneck}")
            metrics = bm.get("metrics") or {}
            if isinstance(metrics, dict):
                hbm = metrics.get("memory.hbm_bandwidth_utilization")
                if hbm is not None:
                    lines.append(f"- HBM utilization: {hbm}%")
                l2 = metrics.get("memory.l2_hit_rate")
                if l2 is not None:
                    lines.append(f"- L2 hit rate: {l2}%")
            top = bm.get("top_kernels") or []
            if isinstance(top, list) and top:
                lines.append("- Top sub-kernels:")
                for entry in top[:5]:
                    if isinstance(entry, dict):
                        name = entry.get("name", "?")
                        share = entry.get("duration_share") or entry.get("share")
                        if share is not None:
                            lines.append(f"    * {name} ({share})")
                        else:
                            lines.append(f"    * {name}")

        # Profile summary (only surface top-level facts so the LLM can
        # reason without drowning in raw rocprof records)
        if prof:
            success = prof.get("success")
            if success is False:
                lines.append("- NOTE: profiler run did not complete successfully")
            duration = prof.get("duration_us") or prof.get("duration_ms")
            if duration is not None and not lines:
                lines.append(f"- Profile duration: {duration}")
            replays = prof.get("replays") or prof.get("num_replays")
            if replays is not None:
                lines.append(f"- Replays: {replays}")

        if not lines:
            return ""

        return "\n".join(lines)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip outermost ```...``` if the model wrapped output despite
        instructions.  Mirrors HarnessBuilder._strip_code_fences."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        lines = stripped.splitlines()
        lines = lines[1:]  # drop opening fence line
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines and lines[-1].strip() == "```":
            lines.pop()
        return "\n".join(lines) + "\n"


__all__ = [
    "KernelAnalysisAgent",
    "KernelAnalysisResult",
    "REQUIRED_RUBRIC_HEADERS",
]
