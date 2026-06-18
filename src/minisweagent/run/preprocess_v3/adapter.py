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
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minisweagent import get_data_dir
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
    kernel_url: str | None = None,
    output_dir: Path = Path("."),
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

    When ``kernel_url`` is ``None`` and ``repo`` is provided, the
    codebase-explore subagent is launched to auto-discover the kernel.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model BEFORE kernel resolution (codebase-explore needs it).
    if model is None and model_factory is not None:
        model = model_factory()

    kernel_path, repo_root = _resolve_kernel_and_repo(
        kernel_url,
        repo,
        console,
        user_task=user_task,
        output_dir=output_dir,
    )

    # Make the canonical SOURCE repo root visible to the worktree-bypass
    # detector (``kernel_languages.contract.find_source_repo_path_leaks`` ->
    # ``_resolve_repo_roots`` reads ``GEAK_REPO_ROOT``). Without this, harness
    # validation during preprocess only sees ``GEAK_WORK_DIR`` (the *subagent*
    # worktree), so a harness that hardcodes the original source tree (e.g.
    # ``"/repo/python"`` in a sys.path candidate list) is NOT flagged and the
    # optimizer silently evaluates the unpatched baseline (~1.00x). Force-set:
    # GEAK_REPO_ROOT is by definition the original repo root, never a worktree.
    if repo_root:
        os.environ["GEAK_REPO_ROOT"] = str(repo_root)

    if model is None:
        raise RuntimeError(
            "run_preprocess_v3: neither ``model`` nor ``model_factory`` was supplied; "
            "the v3 orchestrator cannot drive the LLM loop without a model instance."
        )

    detected_language = _resolve_kernel_language(kernel_path, repo_root)
    source_language = detected_language.name
    target_lang_name = (target_language or detected_language.name).lower()

    # Deterministic Path-A bypass for a PRE-VALIDATED harness.
    #
    # When the caller hands us a harness it already validated end-to-end, the
    # entire A1 preprocess (render COMMANDMENT -> collect_baseline ->
    # collect_profile) is deterministic — there is nothing for the LLM
    # orchestrator to decide. Driving it through the LLM loop anyway is not just
    # wasteful: the classifier can misroute (e.g. divert a shape-bearing task to
    # the harness-GENERATOR) or simply fail to converge, burning the whole
    # preprocess budget without ever producing a baseline. Run the deterministic
    # sequence directly and skip the LLM entirely. Opt-out: GEAK_NO_PREVALIDATED_BYPASS=1.
    #
    # IMPORTANT: a backend-REWRITE request (target_language differs from the detected
    # source) is NOT a no-op — the TranslationPhase must run to produce the rewritten
    # kernel. The pre-validated harness only attests the SOURCE kernel, so the bypass
    # must be skipped whenever a rewrite is requested.
    _is_rewrite = bool(target_language) and target_lang_name != source_language.lower()
    _bypass_disabled = os.environ.get("GEAK_NO_PREVALIDATED_BYPASS", "").strip().lower() in ("1", "true", "yes", "on")
    if harness and not translate_only and not _bypass_disabled and not _is_rewrite and Path(harness).is_file():
        t0 = time.monotonic()
        result = _run_prevalidated_path_a(
            harness=Path(harness),
            kernel_path=kernel_path,
            repo_root=repo_root,
            kernel_language=detected_language,
            output_dir=output_dir,
            gpu_id=gpu_id,
            correctness_command=correctness_command,
            performance_command=performance_command,
        )
        # PreprocessResult is a frozen dataclass; stamp elapsed via replace().
        from dataclasses import replace as _dc_replace

        result = _dc_replace(result, elapsed_s=time.monotonic() - t0)
        logger.info(
            "v3 preprocess (pre-validated Path-A bypass) completed in %.1fs (success=%s, errors=%d)",
            result.elapsed_s,
            result.success,
            len(result.errors),
        )
        if not result.success and not _can_proceed_despite_failure(result):
            raise RuntimeError(
                "v3 preprocess (pre-validated bypass) failed: "
                + ("; ".join(result.errors) if result.errors else "no artefacts produced")
            )
        return _preprocess_result_to_legacy_context(
            result=result,
            repo_root=repo_root,
            output_dir=output_dir,
            kernel_path_input=kernel_path,
            harness=harness,
            eval_command=eval_command,
            correctness_command=correctness_command,
            performance_command=performance_command,
        )

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
        # Backstop salvage: a pure step/cost-limit abort must not discard a run
        # whose essential artifacts (COMMANDMENT.md, and a harness on Path B)
        # were already produced. The orchestrator normally salvages this case
        # itself; this guards the residual path where success stayed False but
        # the artifacts are on disk, so a step_limit overshoot can't tear down
        # the whole GEAK pipeline. Genuine failures (no artifacts, or non-limit
        # errors) still raise.
        if _can_proceed_despite_failure(result):
            logger.warning(
                "v3 preprocess returned success=False (errors=%s) but required artifacts are "
                "present (harness=%s, commandment=%s); proceeding into optimization with the "
                "salvaged context instead of aborting.",
                result.errors,
                result.harness_path,
                result.commandment_path,
            )
        else:
            raise RuntimeError(
                "v3 preprocess failed: " + ("; ".join(result.errors) if result.errors else "no artefacts produced")
            )

    return _preprocess_result_to_legacy_context(
        result=result,
        repo_root=repo_root,
        output_dir=output_dir,
        kernel_path_input=kernel_path,
        harness=harness,
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
    )


def _run_prevalidated_path_a(
    *,
    harness: Path,
    kernel_path: Path,
    repo_root: str | None,
    kernel_language: KernelLanguage,
    output_dir: Path,
    gpu_id: int,
    correctness_command: str | list[str] | None,
    performance_command: str | list[str] | None,
) -> PreprocessResult:
    """Run the deterministic A1 preprocess for a pre-validated harness — no LLM.

    Mirrors exactly what the orchestrator's deterministic tools do on Path A
    (``collect_baseline`` -> ``collect_profile`` -> ``render_commandment``),
    but called directly so a pre-validated harness never depends on the LLM
    classifier converging. The same worktree-bypass gate the
    ``render_commandment`` tool enforces is applied here, so a harness that
    hardcodes the source-repo path is still rejected (it would otherwise
    measure the unpatched baseline at ~1.00x).
    """
    from minisweagent.run.preprocess_v3.baseline import (
        BaselineMetrics,
        ProfileResult,
        capture_full_benchmark_stdout,
        collect_baseline_metrics,
        collect_profile,
    )
    from minisweagent.run.preprocess_v3.commandment import (
        CommandmentContext,
        render_commandment,
    )

    work_dir = Path(repo_root) if repo_root else None
    errors: list[str] = []
    warnings: list[str] = []

    # Worktree-bypass gate (deterministic, final) — identical contract to the
    # render_commandment tool. A harness that imports the source repo directly
    # silently evaluates the UNPATCHED baseline, so refuse it up front.
    if not os.environ.get("GEAK_ALLOW_HARDCODED_PATHS") and repo_root:
        try:
            from minisweagent.kernel_languages.contract import (
                ContractViolation,
                validate_harness,
            )

            validate_harness(harness, repo_root=repo_root)
        except ContractViolation as exc:
            logger.error("pre-validated bypass REJECTED harness (worktree bypass): %s", exc)
            return PreprocessResult(
                success=False,
                kernel_language=kernel_language,
                kernel_path=kernel_path,
                harness_path=harness,
                path_taken="A",
                errors=[f"worktree_bypass: {exc}"],
            )
        except Exception as exc:  # noqa: BLE001 — never let the gate crash the bypass
            logger.debug("pre-validated bypass: worktree gate skipped (validator error): %s", exc)

    baseline: BaselineMetrics | None = None
    full_benchmark_stdout: str | None = None
    try:
        baseline = collect_baseline_metrics(
            harness,
            work_dir=work_dir,
            gpu_id=gpu_id,
        )
        full_benchmark_stdout = capture_full_benchmark_stdout(
            harness,
            work_dir=work_dir,
            gpu_id=gpu_id,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"collect_baseline failed: {exc}")
        logger.error("pre-validated bypass: collect_baseline failed: %s", exc)

    # Profiling is advisory (matches the orchestrator escape-hatch contract: a
    # run with a verified harness + baseline is salvageable even if profile fails).
    profile: ProfileResult | None = None
    try:
        profile = collect_profile(harness, work_dir=work_dir, gpu_id=gpu_id)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"collect_profile failed (non-fatal): {exc}")
        logger.warning("pre-validated bypass: collect_profile failed (non-fatal): %s", exc)

    commandment_path: Path | None = None
    try:
        ctx = CommandmentContext(
            kernel_path=kernel_path,
            harness_path=harness,
            repo_root=Path(repo_root) if repo_root else None,
            correctness_command=correctness_command,
            performance_command=performance_command,
        )
        out_path = output_dir / "COMMANDMENT.md"
        render_commandment(kernel_language, ctx, out_path=out_path)
        commandment_path = out_path
    except Exception as exc:  # noqa: BLE001
        errors.append(f"render_commandment failed: {exc}")
        logger.error("pre-validated bypass: render_commandment failed: %s", exc)

    success = baseline is not None and baseline.success and commandment_path is not None
    return PreprocessResult(
        success=success,
        kernel_language=kernel_language,
        kernel_path=kernel_path,
        harness_path=harness,
        baseline=baseline,
        full_benchmark_stdout=full_benchmark_stdout,
        profile=profile,
        commandment_path=commandment_path,
        path_taken="A",
        tool_calls=[
            {"name": "collect_baseline", "args": {"harness_path": str(harness)}},
            {"name": "collect_profile", "args": {"harness_path": str(harness)}},
            {"name": "render_commandment", "args": {"harness_path": str(harness)}},
        ],
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Codebase-explore kernel discovery
# ---------------------------------------------------------------------------


def _find_codebase_explore_prompt() -> Path:
    """Locate ``subagents/codebase-explore/SYSTEM_PROMPT.md`` via the shared resolver."""
    p = get_data_dir("subagents") / "codebase-explore" / "SYSTEM_PROMPT.md"
    if p.is_file():
        return p
    raise FileNotFoundError("Could not find subagents/codebase-explore/SYSTEM_PROMPT.md")


def _parse_explore_result(message: str) -> dict[str, Any] | None:
    """Parse ``CODEBASE_EXPLORE_RESULT: {...}`` JSON from subagent output."""
    for line in message.splitlines():
        if "CODEBASE_EXPLORE_RESULT:" in line:
            json_str = line.split("CODEBASE_EXPLORE_RESULT:", 1)[1].strip()
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse CODEBASE_EXPLORE_RESULT JSON: %s", json_str[:200])
                return None
            kp = result.get("kernel_path")
            if kp and Path(kp).is_file():
                return result
            logger.warning("Explored kernel_path %r does not exist on disk", kp)
            return None
    return None


def _load_codebase_explore_model() -> Any:
    """Create a model instance for the codebase-explore subagent.

    Reads model name + kwargs from ``subagents/codebase-explore/SUBAGENT.yaml``
    (falling back to geak.yaml defaults), matching the pattern used by
    :class:`SubagentRegistry` / :class:`PreprocessSubagentDispatcher`.
    """
    import yaml as _yaml

    from minisweagent.config import load_config
    from minisweagent.models import get_model

    # Load geak.yaml defaults
    try:
        geak_cfg = load_config("geak")
    except FileNotFoundError:
        geak_cfg = {}
    model_sec = geak_cfg.get("model", {})
    default_model = model_sec.get("model_name")
    default_model_class = model_sec.get("model_class", "")
    default_model_kwargs = dict(model_sec.get("model_kwargs", {}))

    # Load per-subagent overrides from SUBAGENT.yaml
    prompt_dir = _find_codebase_explore_prompt().parent
    subagent_yaml = prompt_dir / "SUBAGENT.yaml"
    spec_model: str | None = None
    spec_model_kwargs: dict[str, Any] = {}
    if subagent_yaml.is_file():
        data = _yaml.safe_load(subagent_yaml.read_text(encoding="utf-8")) or {}
        spec_model = data.get("model") or None
        spec_model_kwargs = data.get("model_kwargs") or {}

    resolved_name = spec_model or default_model
    resolved_kwargs = {**default_model_kwargs, **spec_model_kwargs}

    return get_model(resolved_name, {"model_class": default_model_class, "model_kwargs": resolved_kwargs})


def _run_codebase_explore(
    repo: Path,
    user_task: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Run the codebase-explore subagent to discover kernel files.

    Returns the parsed ``CODEBASE_EXPLORE_RESULT`` dict or ``None`` on failure.
    The model is loaded from geak.yaml / SUBAGENT.yaml automatically.
    """
    from minisweagent.run.preprocess_v3.subagent import PreprocessSubagent

    model = _load_codebase_explore_model()
    prompt_path = _find_codebase_explore_prompt()
    system_prompt = prompt_path.read_text(encoding="utf-8")

    _explore_timeout = int(os.environ.get("GEAK_EXPLORE_TIMEOUT", "600"))
    agent = PreprocessSubagent(
        model=model,
        system_prompt=system_prompt,
        tools=["bash"],
        step_limit=50,
        wall_timeout=_explore_timeout,
        cwd=str(repo),
    )

    task = f"Explore the repository at {repo}"
    if output_dir:
        task += f"\nWrite CODEBASE_CONTEXT.md to {output_dir}"
    if user_task:
        task += f"\nUser's optimization task: {user_task}"

    logger.info("Starting codebase-explore subagent (step_limit=50, cwd=%s)", repo)
    exit_status, message = agent.run(task)
    logger.info("Codebase-explore finished with status: %s", exit_status)

    return _parse_explore_result(message)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _resolve_kernel_and_repo(
    kernel_url: str | None,
    repo: str | Path | None,
    console: Any,
    *,
    user_task: str | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, str]:
    """Resolve ``kernel_url`` + ``repo`` into ``(kernel_path, repo_root_str)``.

    When ``kernel_url`` is ``None`` and ``repo`` is provided, runs the
    codebase-explore subagent to auto-discover the kernel file.

    Local-path fast path: if ``kernel_url`` is an existing file on disk we
    skip URL resolution entirely. Otherwise fall back to the legacy
    ``resolve_kernel_url`` (which clones if necessary).
    """
    if not kernel_url:
        if repo is None:
            raise RuntimeError("v3 preprocess: kernel_url not provided and no --repo for auto-discovery")
        logger.info("No kernel_url provided; running codebase-explore on %s", repo)
        result = _run_codebase_explore(
            Path(repo).resolve(),
            user_task=user_task,
            output_dir=output_dir,
        )
        if result is None or not result.get("kernel_path"):
            raise RuntimeError("v3 preprocess: codebase-explore failed to discover a kernel in " + str(repo))
        kernel_url = result["kernel_path"]
        logger.info("Codebase-explore discovered kernel: %s", kernel_url)

    kernel_path_obj = Path(kernel_url).expanduser()
    # Resolve repo-relative kernel paths before falling back to the URL resolver.
    if not kernel_path_obj.is_absolute() and repo is not None:
        candidate = Path(repo).expanduser().resolve() / kernel_path_obj
        if candidate.is_file():
            kernel_path_obj = candidate
        elif candidate.suffix == "":
            # LLM extractors (parse_task_info) sometimes drop the extension
            # ("silu" instead of "silu.hip"). When the bare-name candidate
            # doesn't resolve, probe each known kernel-type extension under
            # the repo and promote if exactly one matches.
            from minisweagent.run.utils.task_parser import _KERNEL_TYPE_TO_EXT

            ext_candidates: set[str] = {ext for exts in _KERNEL_TYPE_TO_EXT.values() for ext in exts}
            matches = [candidate.with_suffix(ext) for ext in ext_candidates if candidate.with_suffix(ext).is_file()]
            if len(matches) == 1:
                kernel_path_obj = matches[0]
                logger.info(
                    "_resolve_kernel_and_repo: bare-name kernel_url %r promoted to %s by extension probe",
                    kernel_url,
                    kernel_path_obj,
                )
            elif len(matches) > 1:
                logger.warning(
                    "_resolve_kernel_and_repo: bare-name kernel_url %r matched multiple files (%s); refusing to guess",
                    kernel_url,
                    [str(m.name) for m in matches],
                )
    if kernel_path_obj.is_file():
        repo_root = str(Path(repo).expanduser().resolve()) if repo is not None else _infer_repo_root(kernel_path_obj)
        return kernel_path_obj.resolve(), repo_root

    # TODO(commit-set-5): the legacy URL resolver still lives under
    # run/preprocess/; inline once that package is dismantled.
    from minisweagent.run.preprocess.resolve_kernel_url import resolve_kernel_url as _legacy_resolve

    resolved = _legacy_resolve(kernel_url, repo=str(repo) if repo is not None else None)
    if resolved.get("error"):
        raise RuntimeError(f"v3 preprocess: resolve-kernel-url failed: {resolved['error']}")
    kp = Path(str(resolved["local_file_path"])).resolve()
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
        hints.append(
            f"- A user-supplied harness is at: `{harness}`.\n"
            "  It passes the four-mode CLI contract (`--correctness`, `--benchmark`,\n"
            "  `--full-benchmark`, `--profile`). This is NOT a guarantee that its shapes are the\n"
            "  ones the task wants, nor that it is worktree-clean — it does NOT exempt the harness\n"
            "  from the shapes pre-check.\n"
            "  Route on whether the task prompt carries shapes the harness does not already pin:\n"
            "\n"
            "  **(A1) The task prompt names NO shapes/dims/dtype-tuple the harness lacks** — take\n"
            "  the fast path: **skip `run_discovery`**, then call `commandment_from_user_command`\n"
            f"  with the harness invocation (e.g. `python {harness} --correctness`) as\n"
            "  `run_command` and list ALL four modes in `modes_covered`:\n"
            "  `['correctness', 'profile', 'benchmark', 'full_benchmark']`. The tool substitutes\n"
            "  the correct flag for each COMMANDMENT section automatically.\n"
            "\n"
            "  **(A2-with-shapes) The task prompt carries shapes** (a `Shapes:` line, explicit\n"
            "  dims, or a dtype/quant tuple) — the shapes pre-check wins: do **NOT** call\n"
            "  `commandment_from_user_command`. Dispatch `harness-generator` (then\n"
            "  `harness-verifier`, then `render_commandment`) and pass the harness path in the\n"
            f"  `dispatch_subagent` context as `template_harness_path: {harness}` so the generator\n"
            "  uses it as a structural template. The generator reads the harness and decides\n"
            "  whether to reproduce it (shapes already match) or regenerate with the prompt shapes\n"
            "  (shapes differ). Do NOT instruct yourself to read the harness or compare shapes —\n"
            "  that comparison is the generator's job."
        )
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


def _can_proceed_despite_failure(result: PreprocessResult) -> bool:
    """Whether a ``success=False`` result is still usable for optimization.

    True only when (a) every recorded error is a budget-limit abort (step/cost
    limit) — never a real tool crash — and (b) the essential artifacts exist on
    disk: a COMMANDMENT.md always, plus a harness file on Path B. This is the
    adapter-level mirror of the orchestrator's salvage so a step_limit overshoot
    cannot abort the whole GEAK run when the harness + COMMANDMENT were produced.
    """
    errors = result.errors or []
    if not errors:
        return False
    if not all(("step_limit" in e) or ("cost_limit" in e) or ("Limits exceeded" in e) for e in errors):
        return False
    commandment_path = result.commandment_path
    if commandment_path is None or not Path(commandment_path).is_file():
        return False
    if result.path_taken == "A":
        return True
    harness_path = result.harness_path
    return harness_path is not None and Path(harness_path).is_file()


def _preprocess_result_to_legacy_context(
    *,
    result: PreprocessResult,
    repo_root: str,
    output_dir: Path,
    kernel_path_input: Path,
    harness: str | None = None,
    eval_command: str | None = None,
    correctness_command: str | list[str] | None = None,
    performance_command: str | list[str] | None = None,
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

    benchmark_baseline_path: str | None = None
    full_benchmark_baseline_path: str | None = None
    if result.baseline is not None and result.baseline.success:
        representative_stdout = _pick_representative_stdout(result.baseline)
        if representative_stdout:
            bb = output_dir / "benchmark_baseline.txt"
            bb.write_text(representative_stdout, encoding="utf-8")
            benchmark_baseline_path = str(bb)
        full_bench_stdout = result.full_benchmark_stdout or representative_stdout
        if full_bench_stdout:
            fbb = output_dir / "full_benchmark_baseline.txt"
            fbb.write_text(full_bench_stdout, encoding="utf-8")
            full_benchmark_baseline_path = str(fbb)

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

    test_command = _extract_test_command(result) or _join_legacy_command(
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
    )
    harness_path = _recover_harness_path(
        result=result,
        harness=harness,
        test_command=test_command,
        repo_root=repo_root,
    )

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
        "harness_path": harness_path,
        "test_command": test_command,
        "harness_results": None,
        "testcase_selection": None,
        "profiling": profiling,
        "baseline_metrics": baseline_metrics or None,
        "baseline_metrics_path": baseline_metrics_path,
        "benchmark_baseline": benchmark_baseline_path,
        "full_benchmark_baseline": full_benchmark_baseline_path,
        "correctness": None,
        "commandment": commandment_text,
        "commandment_path": commandment_path_str,
        "kernel_analysis_md": None,
        "evaluation_contract": None,
        "v3_subagent_runs": list(result.subagent_runs),
        "v3_elapsed_s": result.elapsed_s,
        "v3_path_taken": result.path_taken,
        "path_taken": result.path_taken,
    }
    if result.translation is not None:
        legacy_ctx["v3_translation"] = asdict(result.translation)

    return legacy_ctx


def _pick_representative_stdout(baseline: Any) -> str | None:
    """Return stdout from the baseline run whose latency is closest to the median."""
    if not baseline.raw_outputs or baseline.median_ms is None:
        return None
    best_output: str | None = None
    best_distance = float("inf")
    for run in baseline.raw_outputs:
        lat = run.get("latency_ms")
        stdout = run.get("stdout") or ""
        if lat is None or not stdout.strip():
            continue
        distance = abs(lat - baseline.median_ms)
        if distance < best_distance:
            best_distance = distance
            best_output = stdout
    return best_output


def _project_baseline(result: PreprocessResult) -> dict[str, Any]:
    """Project :class:`BaselineMetrics` into the legacy ``baseline_metrics`` dict shape.

    The legacy preprocess wrote this dict to ``baseline_metrics.json`` and
    consumers (compose, planned-mode orchestrator, mini's heterogeneous
    routing) read fields like ``duration_us`` and ``median_ms``. We map
    what v3 produces; missing legacy fields stay absent rather than
    fabricated.

    When a profile result is also available, enrich the dict with
    ``bottleneck`` / ``top_kernels`` / ``kernel_name`` / ``metrics`` /
    ``observations`` via legacy ``build_baseline_metrics``. Without this
    enrichment, downstream consumers (``inject_pipeline_context``, the
    planner) read empty values where the legacy pipeline used to surface
    the dominant-kernel breakdown.
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
    if result.profile is not None and result.profile.profile:
        try:
            from minisweagent.run.preprocess.baseline import build_baseline_metrics

            metrics = build_baseline_metrics(result.profile.profile, include_all=True)
        except Exception as exc:
            logger.debug("_project_baseline: build_baseline_metrics fallback skipped: %s", exc)
        else:
            for key in (
                "bottleneck",
                "top_kernels",
                "kernel_name",
                "kernel_names",
                "metrics",
                "observations",
            ):
                if metrics.get(key):
                    out.setdefault(key, metrics[key])
    return out


def _join_legacy_command(
    *,
    eval_command: str | None,
    correctness_command: str | list[str] | None,
    performance_command: str | list[str] | None,
) -> str | None:
    """Recover the legacy ``test_command`` surface from v3 call-site kwargs."""

    if eval_command and eval_command.strip():
        return eval_command.strip()

    parts: list[str] = []
    for cmd in (correctness_command, performance_command):
        if cmd is None:
            continue
        if isinstance(cmd, list):
            parts.extend(c.strip() for c in cmd if c and c.strip())
        elif cmd.strip():
            parts.append(cmd.strip())
    return " && ".join(parts) if parts else None


def _recover_harness_path(
    *,
    result: PreprocessResult,
    harness: str | None,
    test_command: str | None,
    repo_root: str,
) -> str:
    """Recover legacy ``harness_path`` for postprocess consumers.

    Path-A commandment rendering can legitimately leave
    ``PreprocessResult.harness_path`` empty, but promoted harness commands
    still need a harness path in ``preprocess_ctx`` so postprocess can build
    ``GEAK_HARNESS`` and profile correctly. This mirrors the legacy
    preprocessor's ``extract_harness_path(test_command)`` fallback.
    """

    if result.harness_path:
        return str(result.harness_path)

    candidate = harness or test_command
    if not candidate:
        return ""
    try:
        from minisweagent.run.preprocess.harness_utils import extract_harness_path

        harness_path = Path(extract_harness_path(candidate)).expanduser()
    except Exception:
        return ""

    if not harness_path.is_absolute():
        harness_path = Path(repo_root).expanduser().resolve() / harness_path
    resolved = harness_path.resolve()
    # Static-validate the recovered path. ``extract_harness_path`` is greedy
    # and will happily pick up things like ``scripts/task_runner.py`` that
    # don't actually expose the GEAK 4-mode CLI; failing the gate here keeps
    # us from threading a bogus harness_path into postprocess where it'd
    # silently break profile / benchmark invocations.
    try:
        from minisweagent.run.preprocess.harness_utils import validate_harness

        valid, messages = validate_harness(str(resolved))
    except Exception:
        valid, messages = True, []
    if not valid:
        logger.warning(
            "_recover_harness_path: rejected %s (validate_harness failed: %s)",
            resolved,
            messages,
        )
        return ""
    return str(resolved)


def _write_benchmark_baseline(result: PreprocessResult, output_dir: Path) -> str | None:
    """Persist the raw v3 benchmark baseline text in legacy artifact files."""

    baseline = result.baseline
    if baseline is None:
        return None
    for raw in baseline.raw_outputs:
        if raw.get("returncode") == 0 and str(raw.get("stdout") or "").strip():
            text = str(raw["stdout"])
            (output_dir / "benchmark_baseline.txt").write_text(text, encoding="utf-8")
            (output_dir / "full_benchmark_baseline.txt").write_text(text, encoding="utf-8")
            return text
    return None


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
