"""Preprocessor: sequential pipeline of existing modules.

Runs resolve-kernel-url -> codebase-context -> test-discovery ->
harness-execution -> kernel-profile -> baseline-metrics -> commandment
in order and returns a context dict for the orchestrator.

Each step calls the *same* Python function that the corresponding CLI
uses, so behaviour is identical whether invoked from here or from the
shell.
"""

from __future__ import annotations

import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from minisweagent.debug_runtime import emit_debug_log
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _ensure_mcp_importable() -> None:
    """Add MCP tool source directories to sys.path if not already present."""
    for sub in (
        "mcp_tools/profiler-mcp/src",
        "mcp_tools/metrix-mcp/src",
        "mcp_tools/automated-test-discovery/src",
    ):
        p = str(_REPO_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


from minisweagent.run.pipeline_helpers import (
    DEFAULT_EVAL_BENCHMARK_ITERATIONS,
    _materialize_validated_harness,
    create_validated_harness,
    execute_harness_validation,
    extract_harness_path,
    run_baseline_profile,
    validate_harness,
)
from minisweagent.run.testcase_cache import (
    build_testcase_cache_key,
    get_testcase_cache_dir,
    get_testcase_cache_entry,
    materialize_cached_harness,
    save_cached_harness,
)
from minisweagent.benchmark_parsing import extract_latency_ms

# ── main entry point ─────────────────────────────────────────────────


def _build_deterministic_test_command(harness_path: str | Path) -> str:
    harness = Path(harness_path).resolve()
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(harness))} --correctness"


def _resolve_deterministic_harness(
    harness_spec: str,
    *,
    kernel_url: str,
    repo_root: str | Path,
    output_dir: Path,
) -> tuple[str, dict[str, Any]]:
    from minisweagent.tools.resolve_kernel_url_impl import (
        is_weblink,
        parse_github_source_url,
        resolve_kernel_url,
    )

    spec = (harness_spec or "").strip()
    if not spec:
        raise RuntimeError("Deterministic harness spec is empty")

    repo_root_path = Path(repo_root).resolve()
    if not is_weblink(spec):
        harness_path = Path(spec).expanduser()
        if not harness_path.is_absolute():
            harness_path = repo_root_path / harness_path
        harness_path = harness_path.resolve()
        if not harness_path.is_file():
            raise RuntimeError(f"Deterministic harness file not found: {harness_path}")
        return str(harness_path), {"source": "local_path", "path": str(harness_path)}

    harness_remote = parse_github_source_url(spec)
    if harness_remote is None:
        raise RuntimeError(f"Unsupported deterministic harness URL: {spec}")

    kernel_remote = parse_github_source_url(kernel_url) if is_weblink(kernel_url) else None
    if kernel_remote and all(
        kernel_remote[key] == harness_remote[key] for key in ("owner", "repo", "ref")
    ):
        harness_path = (repo_root_path / harness_remote["file_path"]).resolve()
        if not harness_path.is_file():
            raise RuntimeError(
                "Deterministic harness is in the same remote repo/ref as the kernel, "
                f"but the file is missing from the fresh clone: {harness_path}"
            )
        return str(harness_path), {"source": "same_fresh_remote_repo", **harness_remote}

    resolved = resolve_kernel_url(spec, clone_into=output_dir / "_deterministic_harness")
    if resolved.get("error"):
        raise RuntimeError(f"Deterministic harness resolve failed: {resolved['error']}")

    harness_repo = Path(resolved["local_repo_path"]).resolve() if resolved.get("local_repo_path") else None
    if harness_repo is not None and harness_repo != repo_root_path:
        raise RuntimeError(
            "Deterministic harness must come from the same remote repo/ref as --kernel-url "
            "so the harness benchmarks the exact fresh source tree being optimized."
        )

    harness_path = Path(resolved["local_file_path"]).resolve()
    if not harness_path.is_file():
        raise RuntimeError(f"Resolved deterministic harness file not found: {harness_path}")
    return str(harness_path), {"source": "fresh_remote_clone", **harness_remote}


_TRUSTED_IRRELEVANT_TOP_TEST_CACHE_SOURCES = {
    "deterministic_harness",
    "focused_test",
    "fallback_focused_test",
    "unit_test_agent",
}


def _common_path_depth(left: str | Path, right: str | Path) -> int:
    left_parts = Path(left).resolve().parts
    right_parts = Path(right).resolve().parts
    depth = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        depth += 1
    return depth


def _focused_harness_candidate(disc_dict: dict[str, Any]) -> tuple[str, str] | None:
    focused = disc_dict.get("focused_test") or {}
    focused_cmd = str(focused.get("focused_command") or "").strip()
    if not focused_cmd:
        return None
    focused_harness = extract_harness_path(focused_cmd)
    if not Path(focused_harness).is_file():
        return None
    return focused_cmd, focused_harness


def _should_skip_cached_harness(manifest: dict[str, Any] | None, disc_dict: dict[str, Any]) -> bool:
    focused = disc_dict.get("focused_test") or {}
    if focused.get("top_test_is_relevant") is not False:
        return False
    source = str((manifest or {}).get("source") or "").strip()
    return source not in _TRUSTED_IRRELEVANT_TOP_TEST_CACHE_SOURCES


def _build_harness_candidates(
    tests: list[dict[str, Any]],
    disc_dict: dict[str, Any],
    kernel_path: str | Path,
) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    focused_candidate = _focused_harness_candidate(disc_dict)
    focused = disc_dict.get("focused_test") or {}
    if focused_candidate is not None:
        focused_cmd, focused_harness = focused_candidate
        candidates.append((focused_cmd, focused_harness, "focused_test"))

    ranked_discovery: list[tuple[tuple[int, int], int, str, str, str]] = []
    kernel_parent = Path(kernel_path).resolve().parent
    focused_harness = focused_candidate[1] if focused_candidate is not None else None
    restrict_to_kernel_tree = focused.get("top_test_is_relevant") is False
    for index, test in enumerate(tests[:5]):
        cmd = test.get("command")
        path = test.get("file")
        if not cmd or not path:
            continue
        harness_path = extract_harness_path(cmd)
        if not Path(harness_path).is_file():
            continue
        candidate_parent = Path(harness_path).resolve().parent
        candidate_in_kernel_tree = candidate_parent == kernel_parent or kernel_parent in candidate_parent.parents
        if restrict_to_kernel_tree and not candidate_in_kernel_tree:
            continue
        same_directory = candidate_parent == kernel_parent
        common_depth = _common_path_depth(candidate_parent, kernel_parent)
        duplicate_focused = focused_harness is not None and Path(harness_path).resolve() == Path(focused_harness).resolve()
        rank = (
            1 if duplicate_focused else 0,
            0 if same_directory else 1,
            -common_depth,
        )
        ranked_discovery.append((rank, index, cmd, harness_path, "discovery_test"))

    for _rank, _index, cmd, harness_path, source in sorted(ranked_discovery, key=lambda item: (item[0], item[1])):
        candidates.append((cmd, harness_path, source))
    return candidates


def _materialize_preprocessor_harness(
    *,
    test_command: str,
    harness_path: str,
    repo_root: str | Path,
    output_dir: Path,
    kernel_path: str | Path,
    gpu_id: int,
    harness_results: list[dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]]]:
    materialized = _materialize_validated_harness(
        test_command=test_command,
        harness_path=harness_path,
        repo_root=Path(repo_root),
        log_dir=output_dir,
        kernel_path=Path(kernel_path),
        gpu_id=gpu_id,
    )
    if materialized is not None:
        return materialized
    return test_command, harness_path, harness_results


def run_preprocessor(
    kernel_url: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    model=None,
    model_factory=None,
    console=None,
    deterministic: bool = False,
    deterministic_harness: str | None = None,
) -> dict[str, Any]:
    """Run all preprocessing steps and return a context dict.

    Parameters
    ----------
    kernel_url:
        GitHub URL or local path to the kernel.
    output_dir:
        Directory to write intermediate artefacts (resolved.json, etc.).
    gpu_id:
        GPU device to use for profiling.
    model:
        LLM model instance for the UnitTestAgent (optional).
    model_factory:
        Callable returning a new model instance (used if model is None).
    console:
        Optional Rich console for progress messages.

    Returns
    -------
    dict with keys:
        resolved, codebase_context_path, discovery, harness_results,
        profiling, baseline_metrics, commandment, test_command,
        kernel_path, repo_root, harness_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    ctx: dict[str, Any] = {}
    deterministic_requested = bool(deterministic or deterministic_harness)
    if deterministic_requested and not deterministic_harness:
        raise RuntimeError("--deterministic currently requires --deterministic-harness")

    # ── 1. resolve-kernel-url ────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 1/7: Resolve kernel URL ---[/bold cyan]"
        if console
        else "--- Step 1/7: Resolve kernel URL ---"
    )

    from minisweagent.tools.resolve_kernel_url_impl import resolve_kernel_url

    resolved = resolve_kernel_url(kernel_url, clone_into=str(output_dir))
    if resolved.get("error"):
        raise RuntimeError(f"resolve-kernel-url failed: {resolved['error']}")

    kernel_path = resolved["local_file_path"]
    repo_root = resolved.get("local_repo_path") or str(Path(kernel_path).parent)
    ctx["resolved"] = resolved
    ctx["kernel_path"] = kernel_path
    ctx["repo_root"] = repo_root

    (output_dir / "resolved.json").write_text(json.dumps(resolved, indent=2, default=str))
    _print(f"  Kernel: {kernel_path}")

    # ── 2. codebase context ──────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 2/7: Codebase context ---[/bold cyan]"
        if console
        else "--- Step 2/7: Codebase context ---"
    )

    from minisweagent.run.codebase_context import generate_codebase_context

    codebase_context_path = generate_codebase_context(
        repo_root=Path(repo_root),
        kernel_path=Path(kernel_path),
        output_dir=output_dir,
    )
    ctx["codebase_context_path"] = str(codebase_context_path)
    _print(f"  CODEBASE_CONTEXT.md written ({codebase_context_path.stat().st_size} bytes)")

    # ── 3. test-discovery (automated_test_discovery MCP) ────────────
    _print("[bold cyan]--- Step 3/7: Test discovery ---[/bold cyan]" if console else "--- Step 3/7: Test discovery ---")

    _ensure_mcp_importable()
    from automated_test_discovery.server import discover as atd_discover

    _discover_fn = getattr(atd_discover, "fn", atd_discover)
    disc_dict = {}  # Initialize to empty dict to avoid NameError if discovery fails
    try:
        disc_dict = _discover_fn(
            kernel_path=kernel_path,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        logger.warning("Test discovery failed: %s", exc)
        _print(f"  [yellow]Warning: Test discovery failed: {exc}[/yellow]" if console else f"  Warning: Test discovery failed: {exc}")
    
    ctx["discovery"] = disc_dict
    (output_dir / "discovery.json").write_text(json.dumps(disc_dict, indent=2, default=str))

    tests = disc_dict.get("tests", [])
    _print(f"  Tests found: {len(tests)}")

    # ── 3b. UnitTestAgent: create a proper test harness ─────────────
    # The MCP discovery finds test files but doesn't create a validated
    # harness with --correctness/--profile modes. The UnitTestAgent is a
    # full LLM agent that can read the kernel, read existing tests, run
    # them, see errors, and iterate until the harness works.
    #
    # After the agent produces a harness we:
    #   1. Statically validate it (argparse, --profile, --correctness)
    #   2. Run it in ALL modes (correctness, profile, benchmark,
    #      full-benchmark) to catch runtime errors early
    # If either step fails we feed errors back to the agent and retry.
    test_command = None
    harness_results: list[dict] | None = None
    selected_harness_source: str | None = None
    _uta_model = model or (model_factory() if model_factory else None)
    testcase_cache_dir = None if deterministic_requested else get_testcase_cache_dir()
    testcase_cache_key = build_testcase_cache_key(kernel_url, kernel_path)
    testcase_cache_entry = (
        get_testcase_cache_entry(testcase_cache_dir, testcase_cache_key)
        if testcase_cache_dir is not None
        else None
    )
    testcase_selection: dict[str, Any] = {
        "cache_key": testcase_cache_key,
        "cache_dir": str(testcase_cache_entry) if testcase_cache_entry else None,
        "reused_cache": False,
        "selected_source": None,
        "saved_cache_manifest": None,
        "deterministic_requested": deterministic_requested,
        "deterministic_harness": deterministic_harness,
    }

    if deterministic_harness:
        deterministic_path, deterministic_meta = _resolve_deterministic_harness(
            deterministic_harness,
            kernel_url=kernel_url,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        ok_static, static_errors = validate_harness(deterministic_path)
        if not ok_static:
            raise RuntimeError(
                "Deterministic harness validation failed: " + "; ".join(static_errors)
            )
        ok_runtime, runtime_errors, candidate_results = execute_harness_validation(
            deterministic_path,
            repo_root=repo_root,
            gpu_id=gpu_id,
        )
        if not ok_runtime:
            raise RuntimeError(
                "Deterministic harness execution failed: " + "; ".join(runtime_errors)
            )
        test_command = _build_deterministic_test_command(deterministic_path)
        harness_results = candidate_results
        ctx["harness_path"] = deterministic_path
        selected_harness_source = "deterministic_harness"
        testcase_selection["selected_source"] = selected_harness_source
        testcase_selection["deterministic_resolution"] = deterministic_meta
        _print(f"  Using deterministic harness: {deterministic_path}")
        for r in harness_results:
            status = "PASS" if r["success"] else "FAIL"
            _print(f"  Harness --{r['mode']}: {status} ({r['duration_s']}s)")
        _print("  Deterministic harness execution: ALL MODES PASSED")

    if testcase_cache_entry is not None:
        try:
            cached = materialize_cached_harness(
                testcase_cache_entry,
                repo_root=repo_root,
                output_dir=output_dir,
                kernel_path=kernel_path,
            )
            if cached:
                candidate_cmd, candidate_harness, _manifest = cached
                if _should_skip_cached_harness(_manifest, disc_dict):
                    testcase_selection["cache_skipped"] = True
                    testcase_selection["cache_skip_reason"] = "focused_test_required_for_irrelevant_top_test"
                    testcase_selection["cache_skipped_source"] = _manifest.get("source")
                else:
                    ok_static, _ = validate_harness(candidate_harness)
                    if ok_static:
                        ok_runtime, _runtime_errors, candidate_results = execute_harness_validation(
                            candidate_harness,
                            repo_root=repo_root,
                            gpu_id=gpu_id,
                        )
                        if ok_runtime:
                            candidate_cmd, candidate_harness, candidate_results = _materialize_preprocessor_harness(
                                test_command=candidate_cmd,
                                harness_path=candidate_harness,
                                repo_root=repo_root,
                                output_dir=output_dir,
                                kernel_path=kernel_path,
                                gpu_id=gpu_id,
                                harness_results=candidate_results,
                            )
                            test_command = candidate_cmd
                            harness_results = candidate_results
                            ctx["harness_path"] = candidate_harness
                            selected_harness_source = "canonical_cache"
                            testcase_selection["reused_cache"] = True
                            testcase_selection["selected_source"] = selected_harness_source
                            _print(f"  Reusing canonical testcase harness: {candidate_harness}")
                            for r in harness_results:
                                status = "PASS" if r["success"] else "FAIL"
                                _print(f"  Harness --{r['mode']}: {status} ({r['duration_s']}s)")
                            _print("  Canonical harness execution: ALL MODES PASSED")
        except Exception as exc:
            testcase_selection["cache_error"] = str(exc)

    _discovery_harness_candidates = _build_harness_candidates(tests, disc_dict, kernel_path)

    _seen_harnesses: set[str] = set()
    if test_command is None:
        for candidate_cmd, candidate_harness, source in _discovery_harness_candidates:
            if candidate_harness in _seen_harnesses:
                continue
            _seen_harnesses.add(candidate_harness)
            try:
                ok_static, static_errors = validate_harness(candidate_harness)
                if not ok_static:
                    continue
                ok_runtime, runtime_errors, candidate_results = execute_harness_validation(
                    candidate_harness,
                    repo_root=repo_root,
                    gpu_id=gpu_id,
                )
                if not ok_runtime:
                    continue

                candidate_cmd, candidate_harness, candidate_results = _materialize_preprocessor_harness(
                    test_command=candidate_cmd,
                    harness_path=candidate_harness,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    kernel_path=kernel_path,
                    gpu_id=gpu_id,
                    harness_results=candidate_results,
                )
                test_command = candidate_cmd
                harness_results = candidate_results
                ctx["harness_path"] = candidate_harness
                selected_harness_source = source
                testcase_selection["selected_source"] = source
                _print(f"  Using discovered harness directly: {candidate_harness}")
                for r in harness_results:
                    status = "PASS" if r["success"] else "FAIL"
                    _print(f"  Harness --{r['mode']}: {status} ({r['duration_s']}s)")
                _print("  Harness execution: ALL MODES PASSED")
                # region agent log
                emit_debug_log(
                    "preprocessor.py:run_preprocessor:harness_fast_path",
                    "Used discovery-provided harness instead of UnitTestAgent",
                    {
                        "kernel_path": kernel_path,
                        "source": source,
                        "harness_path": candidate_harness,
                        "modes": [
                            {
                                "mode": r.get("mode"),
                                "success": bool(r.get("success")),
                                "returncode": r.get("returncode"),
                            }
                            for r in harness_results
                        ],
                    },
                    hypothesis_id="H6",
                )
                # endregion
                break
            except Exception:
                continue

    if test_command is None and _uta_model and repo_root:
        # region agent log
        emit_debug_log(
            "preprocessor.py:run_preprocessor:harness_agent_fallback",
            "Falling back to UnitTestAgent for harness creation",
            {
                "kernel_path": kernel_path,
                "candidate_count": len(_seen_harnesses),
            },
            hypothesis_id="H6",
        )
        # endregion
        _print(
            "[bold cyan]--- Step 3b/3c: UnitTestAgent (harness creation + execution) ---[/bold cyan]"
            if console
            else "--- Step 3b/3c: UnitTestAgent (harness creation + execution) ---"
        )
        try:
            from minisweagent.agents.unit_test_agent import format_discovery_for_agent
            from minisweagent.tools.discovery_types import DiscoveryResult

            disc_result = DiscoveryResult.from_dict(disc_dict, kernel_path)
            discovery_context = format_discovery_for_agent(disc_result)

            if codebase_context_path.exists():
                discovery_context = codebase_context_path.read_text() + "\n\n" + discovery_context

            kernel_name = Path(kernel_path).stem
            discovery_context += (
                "\n\nIMPORTANT: Your TEST_COMMAND must use absolute paths "
                "to the test script (e.g., `python /absolute/path/to/test_harness.py --correctness`). "
                "Do NOT use `cd` in the command. The profiler cannot handle compound shell commands."
            )

            test_command, harness_results = create_validated_harness(
                model=_uta_model,
                repo=Path(repo_root),
                kernel_name=kernel_name,
                log_dir=output_dir,
                kernel_path=Path(kernel_path),
                discovery_context=discovery_context,
                gpu_id=gpu_id,
            )
            selected_harness_source = "unit_test_agent"
            testcase_selection["selected_source"] = selected_harness_source
            _print(f"  UnitTestAgent test_command: {test_command}")
            _print("  Harness static validation: OK")
            for r in harness_results:
                status = "PASS" if r["success"] else "FAIL"
                _print(f"  Harness --{r['mode']}: {status} ({r['duration_s']}s)")
            _print("  Harness execution: ALL MODES PASSED")
        except Exception as exc:
            _print(
                f"  [yellow]UnitTestAgent failed ({exc}), falling back to discovery[/yellow]"
                if console
                else f"  UnitTestAgent failed ({exc}), falling back to discovery"
            )
            logger.warning("UnitTestAgent failed: %s", exc, exc_info=True)
            test_command = None
            harness_results = None

    # Fall back to discovery results if UnitTestAgent didn't produce one.
    # Prefer the focused test (which targets the specific kernel) over
    # the generic test commands (which may be pytest suites without
    # --correctness/--profile support).
    if not test_command:
        focused = disc_dict.get("focused_test") or {}
        focused_cmd = focused.get("focused_command")
        if focused_cmd:
            test_command = focused_cmd
            selected_harness_source = "fallback_focused_test"
            testcase_selection["selected_source"] = selected_harness_source
            _print(f"  Falling back to discovery focused test: {test_command}")
        elif tests:
            test_command = tests[0]["command"]
            selected_harness_source = "fallback_discovery_test"
            testcase_selection["selected_source"] = selected_harness_source
            _print(f"  Falling back to discovery test: {test_command}")

    ctx["test_command"] = test_command
    ctx["harness_results"] = harness_results
    ctx["testcase_selection"] = testcase_selection
    if harness_results:
        (output_dir / "harness_results.json").write_text(
            json.dumps(harness_results, indent=2, default=str)
        )
    if test_command and not ctx.get("harness_path"):
        ctx["harness_path"] = extract_harness_path(test_command)
    if testcase_cache_entry is not None and test_command and harness_results and ctx.get("harness_path"):
        try:
            manifest_path = save_cached_harness(
                testcase_cache_entry,
                kernel_url=kernel_url,
                source=selected_harness_source or "validated_harness",
                test_command=test_command,
                harness_path=ctx["harness_path"],
                repo_root=repo_root,
                output_dir=output_dir,
                kernel_path=kernel_path,
                harness_results=harness_results,
            )
            testcase_selection["saved_cache_manifest"] = str(manifest_path) if manifest_path else None
        except Exception as exc:
            testcase_selection["cache_save_error"] = str(exc)
    testcase_selection["test_command"] = test_command
    testcase_selection["harness_path"] = ctx.get("harness_path")
    (output_dir / "testcase_selection.json").write_text(
        json.dumps(testcase_selection, indent=2, default=str)
    )

    # Collect a canonical benchmark baseline using the same iteration count the
    # orchestrator evaluation will use, so every reported speedup is
    # benchmark-vs-benchmark on the exact same contract.
    benchmark_baseline: str | None = None
    full_benchmark_baseline: str | None = None

    eval_iters = DEFAULT_EVAL_BENCHMARK_ITERATIONS
    harness_path_for_baseline = ctx.get("harness_path") or (
        extract_harness_path(test_command) if test_command else None
    )
    if harness_path_for_baseline and harness_results:
        extra = f"--iterations {eval_iters}"
        _print(
            f"  Re-running all modes with {extra} for baselines..."
        )
        bl_ok, bl_errors, baseline_results = execute_harness_validation(
            harness_path_for_baseline,
            repo_root=repo_root,
            gpu_id=gpu_id,
            benchmark_extra_args=extra,
        )
        for r in baseline_results:
            status = "PASS" if r["success"] else "FAIL"
            _print(f"    --{r['mode']}: {status} ({r['duration_s']}s)")
        if not bl_ok:
            _print(f"  WARNING: baseline re-run had failures: {bl_errors}")
        for r in baseline_results:
            if r["mode"] == "benchmark" and r["success"]:
                benchmark_baseline = r["stdout"]
            if r["mode"] == "full-benchmark" and r["success"]:
                full_benchmark_baseline = r["stdout"]
    elif harness_results:
        for r in harness_results:
            if r["mode"] == "benchmark" and r["success"]:
                benchmark_baseline = r["stdout"]
            if r["mode"] == "full-benchmark" and r["success"]:
                full_benchmark_baseline = r["stdout"]

    canonical_benchmark_baseline = full_benchmark_baseline or benchmark_baseline
    if canonical_benchmark_baseline:
        benchmark_baseline = canonical_benchmark_baseline
        full_benchmark_baseline = canonical_benchmark_baseline
        (output_dir / "benchmark_baseline.txt").write_text(canonical_benchmark_baseline)
        (output_dir / "full_benchmark_baseline.txt").write_text(canonical_benchmark_baseline)

    ctx["benchmark_baseline"] = benchmark_baseline
    ctx["full_benchmark_baseline"] = full_benchmark_baseline

    if test_command:
        _print(f"  Test command: {test_command}")

    # ── 5. kernel-profile (via profiler-mcp) ─────────────────────────
    _print(
        "[bold cyan]--- Step 5/7: Kernel profiling (Metrix instrumented) ---[/bold cyan]"
        if console
        else "--- Step 5/7: Kernel profiling (Metrix instrumented) ---"
    )

    profiling: dict[str, Any] | None = None
    if test_command:
        ctx["harness_path"] = extract_harness_path(test_command)
        (output_dir / "harness_path.txt").write_text(ctx["harness_path"])

        try:
            profiling = run_baseline_profile(test_command, gpu_id=gpu_id)
        except Exception as exc:
            _print(f"  [yellow]Profiling failed: {exc}[/yellow]" if console else f"  Profiling failed: {exc}")
            logger.warning("Profiling failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping profiling (no test command found)")

    ctx["profiling"] = profiling
    if profiling:
        (output_dir / "profile.json").write_text(json.dumps(profiling, indent=2, default=str))
        _print("  Profiling complete")

    # ── 6. baseline-metrics ──────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 6/7: Baseline metrics ---[/bold cyan]" if console else "--- Step 6/7: Baseline metrics ---"
    )

    baseline_metrics: dict[str, Any] | None = None
    if profiling and profiling.get("success", True):
        try:
            from minisweagent.baseline_metrics import build_baseline_metrics

            baseline_metrics = build_baseline_metrics(profiling, include_all=True)
            dur = baseline_metrics.get("duration_us", "?")
            bn = baseline_metrics.get("bottleneck", "?")
            _print(f"  Baseline: {dur} µs, bottleneck={bn}")
        except Exception as exc:
            _print(
                f"  [yellow]Baseline metrics failed: {exc}[/yellow]" if console else f"  Baseline metrics failed: {exc}"
            )
            logger.warning("Baseline metrics failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping baseline metrics (no profiling data)")

    ctx["baseline_metrics"] = baseline_metrics

    # Enrich baseline_metrics with the canonical wall-clock benchmark so all
    # consumers compare benchmark-vs-benchmark instead of mixing Metrix
    # profile durations with wall-clock latencies.
    if baseline_metrics is None:
        baseline_metrics = {}
    bb_path = output_dir / "benchmark_baseline.txt"
    if bb_path.exists():
        import re as _re

        bb_text = bb_path.read_text()
        _bm_val = extract_latency_ms(bb_text)
        if _bm_val is not None:
            baseline_metrics["benchmark_duration_us"] = _bm_val * 1000.0
        _sm = _re.search(r"(\d+)\s+shapes", bb_text, _re.IGNORECASE)
        if _sm:
            baseline_metrics["benchmark_shape_count"] = int(_sm.group(1))
        ctx["baseline_metrics"] = baseline_metrics

    if baseline_metrics:
        (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2, default=str))

    # ── 7. commandment ───────────────────────────────────────────────
    _print("[bold cyan]--- Step 7/7: Commandment ---[/bold cyan]" if console else "--- Step 7/7: Commandment ---")

    commandment: str | None = None
    if test_command:
        try:
            from minisweagent.tools.commandment import generate_commandment

            from minisweagent.tools.discovery_types import _infer_kernel_language

            harness = ctx.get("harness_path") or extract_harness_path(test_command)
            _ktype = (disc_dict.get("kernel") or {}).get("type", "")
            _kl = _infer_kernel_language(Path(kernel_path), _ktype)
            commandment = generate_commandment(
                kernel_path=kernel_path,
                harness_path=harness,
                repo_root=repo_root,
                kernel_language=_kl,
            )
            _print("  COMMANDMENT.md generated")
        except Exception as exc:
            _print(f"  [yellow]Commandment failed: {exc}[/yellow]" if console else f"  Commandment failed: {exc}")
            logger.warning("Commandment generation failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping commandment (no test command)")

    ctx["commandment"] = commandment
    if commandment:
        (output_dir / "COMMANDMENT.md").write_text(commandment)

    # region agent log
    emit_debug_log(
        "preprocessor.py:run_preprocessor:complete",
        "Preprocessor completed with artifact summary",
        {
            "kernel_url": kernel_url,
            "kernel_path": kernel_path,
            "repo_root": repo_root,
            "tests_found": len(tests),
            "has_test_command": bool(test_command),
            "harness_path": ctx.get("harness_path"),
            "harness_modes": (
                {
                    r.get("mode", "?"): {
                        "success": bool(r.get("success")),
                        "returncode": r.get("returncode"),
                    }
                    for r in (harness_results or [])
                }
            ),
            "benchmark_baseline_present": bool(benchmark_baseline),
            "full_benchmark_baseline_present": bool(full_benchmark_baseline),
            "profiling_success": None if profiling is None else bool(profiling.get("success", True)),
            "baseline_bottleneck": (baseline_metrics or {}).get("bottleneck"),
            "baseline_duration_us": (baseline_metrics or {}).get("duration_us"),
            "commandment_present": bool(commandment),
            "artifacts": {
                "resolved.json": (output_dir / "resolved.json").exists(),
                "discovery.json": (output_dir / "discovery.json").exists(),
                "harness_results.json": (output_dir / "harness_results.json").exists(),
                "benchmark_baseline.txt": (output_dir / "benchmark_baseline.txt").exists(),
                "full_benchmark_baseline.txt": (output_dir / "full_benchmark_baseline.txt").exists(),
                "profile.json": (output_dir / "profile.json").exists(),
                "baseline_metrics.json": (output_dir / "baseline_metrics.json").exists(),
                "COMMANDMENT.md": (output_dir / "COMMANDMENT.md").exists(),
            },
        },
        hypothesis_id="H3",
    )
    # endregion

    _print("")
    _print("Preprocessing complete. Artefacts written to: " + str(output_dir))
    return ctx


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI: ``geak-preprocess <url> -o output_dir/``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GEAK preprocessor: resolve → context → discover → harness-exec → profile → baseline → commandment",
    )
    parser.add_argument("url", help="GitHub URL or local path to the kernel")
    parser.add_argument(
        "-o",
        "--output",
        default="geak_preprocess_output",
        help="Output directory for intermediate artefacts (default: geak_preprocess_output)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID for profiling (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name for UnitTestAgent harness creation (uses default if omitted)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require an explicit deterministic harness contract for this preprocessing run.",
    )
    parser.add_argument(
        "--deterministic-harness",
        default=None,
        help="Exact harness URL/path to use. Discovery and UnitTestAgent fallback are disabled.",
    )
    args = parser.parse_args()

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    from minisweagent.run.pipeline_helpers import geak_model_factory

    _model_factory = geak_model_factory(args.model)

    ctx = run_preprocessor(
        args.url,
        Path(args.output),
        gpu_id=args.gpu,
        model_factory=_model_factory,
        console=console,
        deterministic=args.deterministic,
        deterministic_harness=args.deterministic_harness,
    )

    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    main()
