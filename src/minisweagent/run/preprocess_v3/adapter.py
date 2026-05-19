"""Call-site adapter: v3 orchestrator dressed in the legacy preprocess shape.

Provides :func:`run_preprocess_v3`, a drop-in replacement for
:func:`minisweagent.run.preprocess.orchestrator.run_preprocessor_via_orchestrator`.
Same keyword arguments, same return shape (``dict[str, Any]`` with the
keys downstream consumers — ``run/mini.py``, ``run/unified.py``,
``run/orchestrator.py``, ``run/compose.py``, ``agents/heterogeneous/*`` —
already read).

The function:

1. Resolves the repo + kernel paths (local-only fast path; lazy-imports
   the legacy URL resolver only when ``kernel_url`` is non-local).
2. Detects the kernel language via :mod:`preprocess_v3.lang`.
3. Builds + runs :class:`PreprocessOrchestratorAgent`.
4. Raises :class:`RuntimeError` (the legacy preprocess failure type) when
   ``result.success is False`` so the surrounding pipeline's error
   handling continues to apply.
5. Otherwise projects :class:`PreprocessResult` plus the run-context
   inputs into the legacy ``preprocess_ctx`` dict shape.

Boundary notes
--------------

* The lazy legacy import (``resolve_kernel_url``) is tagged
  ``TODO(commit-set-5)`` because it crosses the v3 boundary. The other
  legacy imports in ``preprocess_v3/`` (``baseline``, ``commandment``,
  ``explore``, ``translate``) keep their existing markers — those are
  for a future commit set and are not touched here.
* The legacy ``run_preprocessor_via_orchestrator`` and the rest of
  ``run/preprocess/`` remain on disk and importable; we just stopped
  calling them from the CLI flow. That's per the locked decision: "Wire
  the routing to the new version but do not delete the old stuff right
  now. Test the preprocessing pipeline first."

Translation
-----------

The v3 orchestrator handles translation as a tool call (step 2) when
``target_language`` differs from the detected source. The ``translate_only``
flag from the legacy signature is currently unsupported by this adapter
because no production call site passes it — the standalone ``geak
translate`` CLI doesn't reach this adapter. Wire-up for that path lands
in a follow-up commit set when standalone translate gets re-routed too.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.lang import detect_language, detect_language_for_repo
from minisweagent.run.preprocess_v3.orchestrator import (
    PreprocessOrchestratorAgent,
    PreprocessOrchestratorConfig,
    PreprocessResult,
)
from minisweagent.run.preprocess_v3.tools import register_default_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_preprocess_v3(
    kernel_url: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    model: Any = None,
    model_factory: Any = None,
    console: Any = None,
    harness: str | None = None,
    repo: str | Path | None = None,
    eval_command: str | None = None,
    correctness_command: str | list[str] | None = None,
    performance_command: str | list[str] | None = None,
    benchmark_timeout: int = 3600,
    target_language: str | None = None,
    translate_only: bool = False,
    budget: Any = None,
    state: Any = None,
    user_task: str | None = None,
    scoring_target: str = "wall",
) -> dict[str, Any]:
    """Drop-in shim for ``run_preprocessor_via_orchestrator`` using v3.

    See module docstring for the contract. ``correctness_command`` /
    ``performance_command`` are accepted for signature compatibility and
    folded into the orchestrator's initial task body; the v3 orchestrator
    drives the harness generation itself, so those become *hints* rather
    than authoritative commands.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_path, repo_root = _resolve_kernel_and_repo(kernel_url, repo, console)

    if model is None and model_factory is not None:
        model = model_factory()
    if model is None:
        raise RuntimeError(
            "run_preprocess_v3: neither ``model`` nor ``model_factory`` was supplied; "
            "the v3 orchestrator cannot drive the LLM loop without a model instance."
        )

    detected_language = _resolve_kernel_language(kernel_path, repo_root)
    source_language = detected_language.name
    target_lang_name = (target_language or detected_language.name).lower()

    config = PreprocessOrchestratorConfig(
        gpu_id=gpu_id,
        repo=Path(repo_root) if repo_root else None,
    )
    agent = PreprocessOrchestratorAgent(model=model, config=config)
    register_default_tools(agent, kernel_language=detected_language)

    task = _build_orchestrator_task(
        user_task=user_task,
        harness=harness,
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
        benchmark_timeout=benchmark_timeout,
        translate_only=translate_only,
    )

    t0 = time.monotonic()
    result: PreprocessResult = agent.run(
        task=task,
        kernel_path=kernel_path,
        repo_root=repo_root,
        kernel_language=detected_language,
        source_language=source_language,
        target_language=target_lang_name,
        output_dir=output_dir,
        gpu_id=gpu_id,
        scoring_target=scoring_target,
    )
    elapsed = time.monotonic() - t0
    logger.info(
        "v3 preprocess completed in %.1fs (success=%s, errors=%d)",
        elapsed,
        result.success,
        len(result.errors),
    )

    if not result.success:
        raise RuntimeError(
            "v3 preprocess failed: " + ("; ".join(result.errors) if result.errors else "no artefacts produced")
        )

    return _preprocess_result_to_legacy_context(
        result=result,
        repo_root=repo_root,
        output_dir=output_dir,
        kernel_path_input=kernel_path,
    )


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _resolve_kernel_and_repo(
    kernel_url: str,
    repo: str | Path | None,
    console: Any,
) -> tuple[Path, str]:
    """Resolve ``kernel_url`` + ``repo`` into ``(kernel_path, repo_root_str)``.

    Local-path fast path: if ``kernel_url`` is an existing file on disk we
    skip URL resolution entirely. Otherwise fall back to the legacy
    ``resolve_kernel_url`` (which clones if necessary).
    """
    kernel_path_obj = Path(kernel_url).expanduser()
    # Resolve repo-relative kernel paths before falling back to the URL resolver.
    if not kernel_path_obj.is_absolute() and repo is not None:
        candidate = Path(repo).expanduser().resolve() / kernel_path_obj
        if candidate.is_file():
            kernel_path_obj = candidate
    if kernel_path_obj.is_file():
        repo_root = str(Path(repo).expanduser().resolve()) if repo is not None else _infer_repo_root(kernel_path_obj)
        return kernel_path_obj.resolve(), repo_root

    # TODO(commit-set-5): the legacy URL resolver still lives under
    # run/preprocess/; inline once that package is dismantled.
    from minisweagent.run.preprocess.resolve_kernel_url import resolve_kernel_url as _legacy_resolve

    resolved = _legacy_resolve(kernel_url, repo=str(repo) if repo is not None else None)
    if resolved.get("error"):
        raise RuntimeError(f"v3 preprocess: resolve-kernel-url failed: {resolved['error']}")
    kp = Path(str(resolved["kernel_path"])).resolve()
    rr = str(Path(resolved.get("repo_root") or _infer_repo_root(kp)).resolve())
    return kp, rr


def _infer_repo_root(kernel_path: Path) -> str:
    """Walk up from ``kernel_path`` looking for a ``.git`` dir; fall back to parent.

    Matches the legacy ``DiscoveryPhase`` rule: if a ``.git`` directory is
    found while walking up the tree, that's the repo root. Otherwise the
    kernel's parent directory is treated as the repo root (single-file
    repos / loose-script flows).
    """
    for candidate in (kernel_path, *kernel_path.parents):
        if (candidate / ".git").is_dir():
            return str(candidate.resolve())
    return str(kernel_path.parent.resolve())


def _resolve_kernel_language(kernel_path: Path, repo_root: str) -> KernelLanguage:
    """Detect the kernel language for the v3 orchestrator inputs.

    Tries the single-file detector first; falls back to a repo-wide
    majority vote when the file alone is ambiguous (the
    :data:`UNKNOWN` sentinel).
    """
    detected = detect_language(kernel_path)
    if detected.name != "unknown":
        return detected
    return detect_language_for_repo(Path(repo_root))


# ---------------------------------------------------------------------------
# Task body construction
# ---------------------------------------------------------------------------


def _build_orchestrator_task(
    *,
    user_task: str | None,
    harness: str | None,
    eval_command: str | None,
    correctness_command: str | list[str] | None,
    performance_command: str | list[str] | None,
    benchmark_timeout: int,
    translate_only: bool,
) -> str:
    """Assemble the orchestrator's free-form task body from legacy kwargs.

    Each non-default kwarg becomes a bullet in the task body so the
    LLM has the information it would have lost otherwise (the
    instance template only auto-renders ``kernel_path`` / ``repo_root``
    / language / output_dir / gpu_id).
    """
    lines: list[str] = ["Run the v3 preprocess pipeline end-to-end."]
    if user_task:
        lines.append("")
        lines.append("## User task (highest priority context)")
        lines.append(user_task.strip())
    hints: list[str] = []
    if harness:
        hints.append(f"- A user-supplied harness candidate is at: {harness}")
    if eval_command:
        hints.append(f"- Legacy eval_command (use only as a fallback hint): {eval_command}")
    if correctness_command:
        hints.append(f"- Suggested correctness command: {correctness_command}")
    if performance_command:
        hints.append(f"- Suggested performance command: {performance_command}")
    if benchmark_timeout and benchmark_timeout != 3600:
        hints.append(f"- benchmark_timeout (subprocess seconds): {benchmark_timeout}")
    if translate_only:
        hints.append(
            "- translate_only=True (the standalone `geak translate` flow); the v3 "
            "orchestrator does not yet support short-circuiting after translation. "
            "Run the full flow and surface translation artifacts in the result."
        )
    if hints:
        lines.append("")
        lines.append("## Hints from the call site")
        lines.extend(hints)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result projection
# ---------------------------------------------------------------------------


def _preprocess_result_to_legacy_context(
    *,
    result: PreprocessResult,
    repo_root: str,
    output_dir: Path,
    kernel_path_input: Path,
) -> dict[str, Any]:
    """Project a :class:`PreprocessResult` into the legacy ``preprocess_ctx`` dict.

    Downstream consumers we have to satisfy (audited from
    ``run/mini.py``, ``run/unified.py``, ``run/orchestrator.py``,
    ``run/compose.py``, ``agents/heterogeneous/orchestrator.py``):

    * ``kernel_path``, ``repo_root``, ``output_dir``
    * ``kernel_type`` (used by ``_normalize_kernel_type``) and
      ``discovery.kernel.type`` (used as a secondary signal)
    * ``test_command``, ``harness_path``
    * ``commandment`` (text), ``commandment_path``
    * ``baseline_metrics`` dict and ``baseline_metrics_path``
    * ``profiling`` (the profile JSON payload)
    * ``codebase_context_path``
    """
    kernel_path = result.kernel_path or kernel_path_input
    kernel_language_name = result.kernel_language.name if result.kernel_language is not None else "unknown"

    baseline_metrics = _project_baseline(result)
    baseline_metrics_path: str | None = None
    if baseline_metrics:
        target = output_dir / "baseline_metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(baseline_metrics, indent=2, default=str), encoding="utf-8")
        baseline_metrics_path = str(target)

    commandment_text: str | None = None
    commandment_path_str: str | None = None
    if result.commandment_path is not None:
        commandment_path_str = str(result.commandment_path)
        try:
            commandment_text = Path(commandment_path_str).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("v3 adapter: could not read commandment at %s: %s", commandment_path_str, exc)

    codebase_context_path: str | None = None
    if result.codebase_context is not None and result.codebase_context.out_path is not None:
        codebase_context_path = str(result.codebase_context.out_path)

    test_command = _extract_test_command(result)

    profiling = None
    if result.profile is not None:
        profiling = dict(result.profile.profile or {})
        if result.profile.backend and "backend" not in profiling:
            profiling["backend"] = result.profile.backend
        if result.profile.profile_path is not None and "profile_path" not in profiling:
            profiling["profile_path"] = str(result.profile.profile_path)

    discovery = {
        "kernel": {
            "type": kernel_language_name,
            "path": str(kernel_path),
        },
        "repo_root": str(repo_root),
    }
    if result.codebase_context is not None:
        discovery["codebase_files"] = list(result.codebase_context.files)

    legacy_ctx: dict[str, Any] = {
        "kernel_path": str(kernel_path),
        "kernel_type": kernel_language_name,
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "resolved": {
            "kernel_path": str(kernel_path),
            "repo_root": str(repo_root),
            "kernel_language": kernel_language_name,
        },
        "codebase_context_path": codebase_context_path,
        "discovery": discovery,
        "harness_path": str(result.harness_path) if result.harness_path else "",
        "test_command": test_command,
        "harness_results": None,
        "testcase_selection": None,
        "profiling": profiling,
        "baseline_metrics": baseline_metrics or None,
        "baseline_metrics_path": baseline_metrics_path,
        "benchmark_baseline": None,
        "full_benchmark_baseline": None,
        "correctness": None,
        "commandment": commandment_text,
        "commandment_path": commandment_path_str,
        "kernel_analysis_md": None,
        "evaluation_contract": None,
        "v3_subagent_runs": list(result.subagent_runs),
        "v3_elapsed_s": result.elapsed_s,
    }
    if result.translation is not None:
        legacy_ctx["v3_translation"] = asdict(result.translation)

    return legacy_ctx


def _project_baseline(result: PreprocessResult) -> dict[str, Any]:
    """Project :class:`BaselineMetrics` into the legacy ``baseline_metrics`` dict shape.

    The legacy preprocess wrote this dict to ``baseline_metrics.json`` and
    consumers (compose, planned-mode orchestrator, mini's heterogeneous
    routing) read fields like ``duration_us`` and ``median_ms``. We map
    what v3 produces; missing legacy fields stay absent rather than
    fabricated.
    """
    if result.baseline is None:
        return {}
    baseline = result.baseline
    out: dict[str, Any] = {
        "median_ms": baseline.median_ms,
        "samples_ms": list(baseline.samples_ms),
        "stdev_ms": baseline.stdev_ms,
        "repeats": baseline.repeats,
        "command": baseline.command,
    }
    if baseline.median_ms is not None:
        out["duration_us"] = baseline.median_ms * 1000.0
    return out


def _extract_test_command(result: PreprocessResult) -> str | None:
    """Recover ``TEST_COMMAND`` from the harness-generator's subagent output.

    The v3 orchestrator stashes this in its private ``_collected`` dict
    but doesn't surface it on :class:`PreprocessResult`. Re-extract from
    the recorded ``subagent_runs`` payload so the downstream pipeline
    still gets it.
    """
    for run in result.subagent_runs:
        if run.get("name") != "harness-generator":
            continue
        output = run.get("output") or ""
        for line in str(output).splitlines():
            if line.startswith("TEST_COMMAND:"):
                return line.split(":", 1)[1].strip()
    return None


__all__ = [
    "run_preprocess_v3",
]
