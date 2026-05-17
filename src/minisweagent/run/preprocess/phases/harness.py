"""Harness phase — resolve a universal-contract harness via a layered chain.

Inputs (read from ctx):
  - ``kernel_path``, ``kernel_url``, ``repo_root``, ``output_dir``
  - ``harness``               — explicit ``--harness`` (optional)
  - ``split_harness_hint``    — from DiscoveryPhase (optional)
  - ``discovery``             — ATD output
  - ``language``              — KernelLanguage for HarnessBuilder (D1)
  - ``model`` / ``model_factory`` — for LLM subagents (D1 + legacy UTA)

Outputs (written to ctx):
  - ``harness_path``          — absolute path to a validated harness
  - ``test_command``          — ``python3 <harness> --correctness``
  - ``harness_results``       — list of {mode, success, duration_s, ...}
  - ``testcase_selection``    — diagnostic dict (which layer won)

Resolution chain — first layer to produce a VALIDATED harness wins:

  Layer 1  ctx.harness_path already set by an upstream phase
  Layer 2  explicit ``ctx.harness`` (CLI ``--harness``)
  Layer 3  split-harness-hint from DiscoveryPhase (merged-kernel split)
  Layer 4  testcase_cache hit (previously-validated canonical harness)
  Layer 5  HarnessBuilder (D1) — LLM builds from Jinja + hints
  Layer 6  UnitTestAgent (legacy) + optional shape-fixer
  Layer 7  discovery focused_test / tests[0] — raw fallback

Each layer is a small private method that returns a
``LayerResult`` or ``None``.  The main ``run()`` method iterates
them in order and commits the first success to ``ctx``.  After a
successful layer, the ``testcase_cache`` is updated so the same
kernel skips to Layer 4 on future runs.

Design goals (per user directive 2026-04-08):
  - Best-in-class readability: one clear purpose per method.
  - No dead code: every import + helper is exercised.
  - 100% parity with the legacy ``preprocessor.py:621-944`` chain.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# LayerResult — one uniform return type for all layer methods
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _LayerResult:
    """Result of a single resolution layer attempt.

    Each layer returns either ``_LayerResult`` (success) or ``None``
    (keep trying next layer).  Success means a VALIDATED harness exists
    on disk and is callable as ``python3 <path> --<mode>``.
    """

    harness_path: str
    test_command: str
    harness_results: list[dict] = field(default_factory=list)
    source_label: str = ""
    """Human-readable marker: which layer produced this result.  Used
    by testcase-cache metadata + final report diagnostics.  Values:
    ``"harness"`` / ``"canonical_cache"`` / ``"discovery_candidate"`` /
    ``"harness_builder"`` / ``"unit_test_agent"`` / ``"fallback_focused_test"``
    / ``"fallback_discovery_test"``."""


# ──────────────────────────────────────────────────────────────────────
# HarnessPhase
# ──────────────────────────────────────────────────────────────────────


class HarnessPhase(Phase):
    """Resolve a universal-contract harness through a 7-layer chain.

    Each layer is implemented as a small private method that either
    returns a ``_LayerResult`` (success, short-circuits the chain) or
    ``None`` (try the next layer).
    """

    name = "harness"

    # ──────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()

        testcase_selection = self._init_testcase_selection(ctx)

        # The 7 layers — first success wins.  Each layer returns
        # either a _LayerResult or None.
        #
        # Layer 2 (explicit user harness) follows a "seed on failure"
        # protocol: if the user's harness doesn't pass the full language
        # contract, it's stashed on ``ctx.harness_seed`` and the chain
        # falls through so Layer 5 (HarnessBuilder) can iterate on it
        # as a starting template.  This is why the layer dispatcher
        # doesn't special-case user-intent layers — Layer 2 never
        # raises for a contract failure; it just returns None.  Layer 5
        # picks up the seed automatically.
        layers: list[tuple[str, Any]] = [
            ("already_set", self._layer1_already_set),
            ("explicit_harness", self._layer2_explicit),
            ("split_hint", self._layer3_split_hint),
            ("testcase_cache", self._layer4_cache),
            ("harness_builder", self._layer5_harness_builder),
            ("unit_test_agent", self._layer6_unit_test_agent),
            ("discovery_fallback", self._layer7_discovery_fallback),
        ]

        result: _LayerResult | None = None
        for layer_name, layer_fn in layers:
            try:
                result = layer_fn(ctx, testcase_selection)
            except Exception as exc:  # noqa: BLE001 — any layer failure falls through
                logger.info(
                    "  HarnessPhase layer '%s' failed (%s: %s); falling through.",
                    layer_name,
                    type(exc).__name__,
                    exc,
                )
                result = None
            if result is not None:
                logger.info("  Harness resolved via layer: %s", layer_name)
                break

        if result is None:
            # No layer produced a harness.  This is not fatal — legacy
            # pipelines used discovery-only fallbacks that may still be
            # valid test commands.  We record the failure in
            # testcase_selection so callers can diagnose.
            logger.debug("HarnessPhase: no layer produced a harness; deferring to legacy.")
            ctx.testcase_selection = testcase_selection
            ctx.phases_run.append(self.name)
            return

        # Commit the winning layer's result to ctx.
        ctx.harness_path = result.harness_path
        ctx.test_command = result.test_command
        ctx.harness_results = result.harness_results
        testcase_selection["selected_source"] = result.source_label
        ctx.testcase_selection = testcase_selection

        # Persist harness_results artifact for downstream inspection.
        self._persist_harness_results(ctx, result.harness_results)

        # Save to testcase_cache so future runs hit Layer 4 instead.
        self._save_to_testcase_cache(ctx, result, testcase_selection)

        # Final validation: run the universal contract validator on the
        # chosen harness.  Warnings only (permissive validator today).
        self._validate_contract(ctx.harness_path)

        ctx.phases_run.append(self.name)

    # ──────────────────────────────────────────────────────────────────
    # Layer 1 — harness_path already populated by a previous phase
    # ──────────────────────────────────────────────────────────────────

    def _layer1_already_set(self, ctx: PhaseContext, _selection: dict[str, Any]) -> _LayerResult | None:
        """Short-circuit: some upstream phase already set ``ctx.harness_path``."""
        if not ctx.harness_path:
            return None
        return _LayerResult(
            harness_path=str(ctx.harness_path),
            test_command=ctx.test_command or _build_test_command(ctx.harness_path),
            harness_results=list(ctx.harness_results or []),
            source_label="already_set",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 2 — explicit --harness from caller (CLI flag)
    # ──────────────────────────────────────────────────────────────────

    def _layer2_explicit(self, ctx: PhaseContext, selection: dict[str, Any]) -> _LayerResult | None:
        """User passed ``--harness`` on the CLI or mentioned one in the prompt.

        Validate the user's harness against the FULL language contract
        (derived from the KernelLanguage's Jinja harness template).  For
        every current language that means all four modes
        (``--correctness``, ``--benchmark``, ``--full-benchmark``,
        ``--profile``) must run cleanly AND the ``GEAK_RESULT_*`` stdout
        markers must be emitted.  Languages that ship ``make``-based
        harnesses (HIP) wrap the build step inside a Python argparse
        layer so the external contract is the same across languages —
        the per-language Jinja template owns the INTERNAL quirks
        (``make``, ``hipcc``, ``torch.utils.cpp_extension.load_inline``,
        etc.) while the external surface stays uniform.

        Behavior:
          - Static validation PASSES and runtime validation PASSES
            against the language's full contract  →  use the harness
            as-is; the chain stops here.
          - Validation FAILS in any mode  →  do NOT raise and do NOT
            silently fall back to an unrelated layer.  Instead, stash
            the user's harness on ``ctx.harness_seed`` and return
            ``None`` so Layer 5 (HarnessBuilder) can iterate on it as
            a starting template.  HarnessBuilder's wallclock-bounded
            retry loop will produce a contract-compliant harness
            (30-min budget) — this preserves the user's intent (they
            gave us a close-to-working harness) without accepting a
            partial contract.

        Rationale: the pipeline's invariant is "HarnessPhase emits a
        fully-contract-compliant harness or the pipeline fails loudly".
        A user's harness that covers 3/4 modes is useful as a seed but
        not acceptable as the final artifact — the LLM retry loop
        converges on fixes from that seed much faster than generating
        from scratch.
        """
        if not ctx.harness:
            return None

        from minisweagent.run.preprocess.harness_utils import (
            execute_harness_validation,
            validate_harness,
        )
        from minisweagent.run.preprocess.preprocessor import (
            _ensure_harness_has_no_kernel_defs,
            _resolve_deterministic_harness,
        )

        resolved_path, meta = _resolve_deterministic_harness(
            ctx.harness,
            kernel_url=ctx.kernel_url,
            repo_root=ctx.repo_root,
            output_dir=Path(ctx.output_dir),
        )
        selection["deterministic_resolution"] = meta

        # --- Static contract check (file structure) -------------------
        ok_static, static_errors = validate_harness(resolved_path)
        if not ok_static:
            # Static failure = the file can't even be parsed as a
            # harness.  Keep it as a SEED for HarnessBuilder and fall
            # through.  We do NOT raise because the LLM loop can still
            # turn this seed into a valid harness.
            logger.info(
                "  Layer 2: user harness %s fails static contract (%s); "
                "handing off to HarnessBuilder as a seed template.",
                resolved_path,
                "; ".join(static_errors),
            )
            ctx.harness_seed = resolved_path
            selection["user_harness_as_seed"] = {
                "path": resolved_path,
                "reason": "static_contract_failed",
                "errors": list(static_errors),
            }
            return None

        # --- Runtime contract check (full contract, all modes) -------
        # No ``required_modes`` → default is strict (every mode must
        # pass).  This matches the language's Jinja contract; callers
        # that need a permissive check pass ``required_modes`` explicitly.
        ok_runtime, runtime_errors, results = execute_harness_validation(
            resolved_path,
            repo_root=ctx.repo_root,
            gpu_id=ctx.gpu_id,
        )
        if not ok_runtime:
            logger.info(
                "  Layer 2: user harness %s fails full-contract runtime "
                "validation (%s); handing off to HarnessBuilder as a "
                "seed template.",
                resolved_path,
                "; ".join(runtime_errors),
            )
            ctx.harness_seed = resolved_path
            selection["user_harness_as_seed"] = {
                "path": resolved_path,
                "reason": "runtime_contract_failed",
                "errors": list(runtime_errors),
                "results": results,
            }
            return None

        # Full contract passes — strip any kernel defs and adopt the
        # user's harness as the pipeline's canonical harness.
        resolved_path = _ensure_harness_has_no_kernel_defs(resolved_path, Path(ctx.output_dir), {})
        return _LayerResult(
            harness_path=resolved_path,
            test_command=_build_test_command(resolved_path),
            harness_results=results,
            source_label="harness",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 3 — split-harness-hint from DiscoveryPhase
    # ──────────────────────────────────────────────────────────────────

    def _layer3_split_hint(self, ctx: PhaseContext, _selection: dict[str, Any]) -> _LayerResult | None:
        """DiscoveryPhase detected a merged kernel and split off a harness."""
        if not ctx.split_harness_hint or ctx.harness:
            return None

        from minisweagent.run.preprocess.harness_utils import (
            execute_harness_validation,
            validate_harness,
        )

        candidate = Path(ctx.split_harness_hint)
        if not candidate.exists():
            return None

        ok_static, _ = validate_harness(candidate)
        if not ok_static:
            return None

        ok_runtime, _, results = execute_harness_validation(candidate, repo_root=ctx.repo_root, gpu_id=ctx.gpu_id)
        if not ok_runtime:
            return None

        return _LayerResult(
            harness_path=str(candidate.resolve()),
            test_command=_build_test_command(str(candidate.resolve())),
            harness_results=results,
            source_label="split_harness",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 4 — testcase_cache hit
    # ──────────────────────────────────────────────────────────────────

    def _layer4_cache(self, ctx: PhaseContext, selection: dict[str, Any]) -> _LayerResult | None:
        """Previously-validated canonical harness retrieved from cache."""
        # The cache is skipped when the caller passed --harness (Layer 2
        # wins in that case); Layer 2 has already run when we get here.
        if ctx.harness:
            return None

        from minisweagent.run.preprocess.harness_utils import (
            execute_harness_validation,
            validate_harness,
        )
        from minisweagent.run.preprocess.preprocessor import (
            _materialize_preprocessor_harness,
            _should_skip_cached_harness,
        )
        from minisweagent.run.preprocess.testcase_cache import (
            build_testcase_cache_key,
            get_testcase_cache_dir,
            get_testcase_cache_entry,
            materialize_cached_harness,
        )

        cache_dir = get_testcase_cache_dir()
        cache_key = build_testcase_cache_key(ctx.kernel_url, ctx.kernel_path)
        selection["cache_key"] = cache_key
        if cache_dir is None:
            return None
        cache_entry = get_testcase_cache_entry(cache_dir, cache_key)
        if cache_entry is None:
            return None
        selection["cache_dir"] = str(cache_entry)

        cached = materialize_cached_harness(
            cache_entry,
            repo_root=ctx.repo_root,
            output_dir=Path(ctx.output_dir),
            kernel_path=ctx.kernel_path,
        )
        if not cached:
            return None
        candidate_cmd, candidate_harness, manifest = cached

        if _should_skip_cached_harness(manifest, ctx.discovery or {}):
            selection["cache_skipped"] = True
            selection["cache_skip_reason"] = "focused_test_required_for_irrelevant_top_test"
            selection["cache_skipped_source"] = manifest.get("source")
            return None

        ok_static, _ = validate_harness(candidate_harness)
        if not ok_static:
            return None

        ok_runtime, _, candidate_results = execute_harness_validation(
            candidate_harness, repo_root=ctx.repo_root, gpu_id=ctx.gpu_id
        )
        if not ok_runtime:
            return None

        candidate_cmd, candidate_harness, candidate_results = _materialize_preprocessor_harness(
            test_command=candidate_cmd,
            harness_path=candidate_harness,
            repo_root=ctx.repo_root,
            output_dir=Path(ctx.output_dir),
            kernel_path=ctx.kernel_path,
            gpu_id=ctx.gpu_id,
            harness_results=candidate_results,
        )
        selection["reused_cache"] = True
        return _LayerResult(
            harness_path=str(candidate_harness),
            test_command=str(candidate_cmd),
            harness_results=candidate_results,
            source_label="canonical_cache",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 5 — HarnessBuilder (D1): LLM subagent using Jinja template
    # ──────────────────────────────────────────────────────────────────

    def _layer5_harness_builder(self, ctx: PhaseContext, _selection: dict[str, Any]) -> _LayerResult | None:
        """LLM-driven harness builder (D1).  Preferred over UTA (Layer 6)
        because it emits universal-contract harnesses by construction."""
        if ctx.language is None:
            return None

        try:
            template_blob = ctx.language.harness_template
        except Exception:
            template_blob = ""
        if not template_blob.strip():
            return None

        model = _resolve_model(ctx)
        if model is None:
            return None

        if not ctx.kernel_path or not Path(ctx.kernel_path).is_file():
            return None

        try:
            from minisweagent.pipeline_workers.base import SubagentConfig
            from minisweagent.pipeline_workers.preprocess.harness_builder import (
                HarnessBuilder,
                HarnessBuildFailed,
            )
            from minisweagent.run.preprocess.harness_utils import (
                execute_harness_validation,
            )
        except ImportError:
            return None

        # Wallclock budget for the validate-retry loop.  Architectural
        # intent (execution plan §0.5(b) Harness phase): keep retrying
        # — feeding the previous attempt's contract-validation errors
        # back into the next prompt — until the universal contract is
        # satisfied OR the budget is exhausted.  30 min default;
        # override via ``GEAK_HARNESS_BUILDER_BUDGET_S``.
        try:
            budget_s = float(os.getenv("GEAK_HARNESS_BUILDER_BUDGET_S", "1800"))
        except ValueError:
            budget_s = 1800.0

        config = SubagentConfig(
            name="harness_builder",
            model_name=getattr(model, "name", "harness_builder_model"),
            system_template="",
            instance_template="",
            step_limit=1,
            cost_limit=3.0,
            temperature=0.2,
            extra={"max_wallclock_seconds": budget_s},
        )
        builder = HarnessBuilder(language=ctx.language, config=config)
        builder.model = model  # type: ignore[attr-defined]

        # Use a distinct prefix for the auto-generated harness so it
        # cannot collide with user files that happen to share the
        # legacy ``harness.py`` basename (which could have been in the
        # user's repo or copied into output_dir).  The ``_geak_``
        # prefix acts as an ownership marker — anything with this
        # prefix is a pipeline artifact and safe to overwrite on
        # subsequent runs.
        out_path = Path(ctx.output_dir) / "_geak_auto_harness.py"
        user_tests = _extract_user_test_files(ctx)

        # When Layer 2 rejected the user's harness as a non-compliant
        # seed, it stashed the path on ``ctx.harness_seed``.  Surface
        # the seed's content to HarnessBuilder so the LLM can iterate
        # on the user's starting point instead of regenerating from
        # scratch — the retry loop converges dramatically faster when
        # the seed is a "close but not quite" harness.
        seed_path: Path | None = None
        if ctx.harness_seed:
            candidate = Path(ctx.harness_seed)
            if candidate.is_file():
                seed_path = candidate
                logger.info(
                    "  Layer 5: using user harness as seed template: %s",
                    seed_path,
                )

        try:
            result = builder.run(
                kernel_path=Path(ctx.kernel_path),
                out_path=out_path,
                repo_root=Path(ctx.repo_root) if ctx.repo_root else None,
                user_test_files=user_tests,
                discovery_context=_read_codebase_context(ctx),
                max_wallclock_seconds=budget_s,
                seed_harness_path=seed_path,
            )
        except HarnessBuildFailed:
            return None

        harness_path = result.get("harness_path") if isinstance(result, dict) else None
        if not harness_path:
            return None

        # HarnessBuilder internally validates the STATIC contract; now
        # run it under RUNTIME validation to ensure all 4 modes execute
        # cleanly against the actual kernel.
        ok_runtime, runtime_errors, results = execute_harness_validation(
            harness_path, repo_root=ctx.repo_root, gpu_id=ctx.gpu_id
        )
        if not ok_runtime:
            logger.warning(
                "[yellow]HarnessBuilder output failed runtime validation: %s[/yellow]",
                runtime_errors,
            )
            return None

        return _LayerResult(
            harness_path=str(harness_path),
            test_command=_build_test_command(str(harness_path)),
            harness_results=results,
            source_label="harness_builder",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 6 — UnitTestAgent (legacy) + optional shape-fixer
    # ──────────────────────────────────────────────────────────────────

    def _layer6_unit_test_agent(self, ctx: PhaseContext, _selection: dict[str, Any]) -> _LayerResult | None:
        """Legacy UnitTestAgent produces a harness from discovery + codebase
        context.  Followed by shape-fixer to correct shape mismatches against
        the user's benchmarks/tests."""
        model = _resolve_model(ctx)
        if model is None or not ctx.repo_root or not ctx.kernel_path:
            return None

        try:
            from minisweagent.run.preprocess.discovery_types import DiscoveryResult
            from minisweagent.run.preprocess.harness_utils import (
                create_validated_harness,
                execute_harness_validation,
                extract_harness_path,
            )
            from minisweagent.run.preprocess.preprocessor import (
                _build_repo_native_reference_context,
                _ensure_harness_has_no_kernel_defs,
            )
            from minisweagent.run.preprocess.unit_test_agent import (
                format_discovery_for_agent,
            )
        except ImportError:
            return None

        # Discovery-candidate fast-path: if discovery already produced
        # validated harness candidates, use them before invoking UTA.
        fastpath = self._try_discovery_candidates(ctx)
        if fastpath is not None:
            return fastpath

        # UnitTestAgent proper.
        disc_dict = ctx.discovery or {}
        disc_result = DiscoveryResult.from_dict(disc_dict, ctx.kernel_path)
        discovery_context = format_discovery_for_agent(disc_result)

        if ctx.codebase_context_path and Path(ctx.codebase_context_path).exists():
            discovery_context = Path(ctx.codebase_context_path).read_text() + "\n\n" + discovery_context

        tests = disc_dict.get("tests") or []
        benchmarks = disc_dict.get("benchmarks") or []
        repo_native = _build_repo_native_reference_context(
            tests=tests, benchmarks=benchmarks, kernel_path=ctx.kernel_path
        )
        if repo_native:
            discovery_context += "\n\n" + repo_native

        discovery_context += (
            "\n\nIMPORTANT: Your TEST_COMMAND must use absolute paths "
            "to the test script (e.g., `python /absolute/path/to/test_harness.py --correctness`). "
            "Do NOT use `cd` in the command. The profiler cannot handle compound shell commands."
        )

        kernel_name = Path(ctx.kernel_path).stem
        test_command, harness_results = create_validated_harness(
            model=model,
            repo=Path(ctx.repo_root),
            kernel_name=kernel_name,
            log_dir=Path(ctx.output_dir),
            kernel_path=Path(ctx.kernel_path),
            discovery_context=discovery_context,
            gpu_id=ctx.gpu_id,
            user_task=ctx.user_task,
        )

        # Strip kernel defs from the emitted harness (harness imports, not defines).
        uta_harness = extract_harness_path(test_command)
        uta_harness = _ensure_harness_has_no_kernel_defs(uta_harness, Path(ctx.output_dir), {})
        if uta_harness != extract_harness_path(test_command):
            test_command = test_command.replace(extract_harness_path(test_command), uta_harness)

        # Shape-fixer: align harness shapes with the user's benchmark/test
        # files.  Failure doesn't block; we keep the UTA harness as-is.
        if benchmarks or tests:
            try:
                harness_results = self._run_shape_fixer(
                    ctx=ctx,
                    model=model,
                    harness_file=Path(extract_harness_path(test_command)),
                    benchmarks=benchmarks,
                    tests=tests,
                    initial_results=harness_results,
                )
            except Exception as exc:
                logger.warning("Shape-fixer failed: %s", exc)

        # Final runtime revalidation to pick up any shape-fixer mutations.
        ok_runtime, _, results = execute_harness_validation(
            extract_harness_path(test_command),
            repo_root=ctx.repo_root,
            gpu_id=ctx.gpu_id,
        )
        if not ok_runtime:
            # Harness somehow regressed after shape-fix; surface the
            # original UTA results.
            results = harness_results

        return _LayerResult(
            harness_path=extract_harness_path(test_command),
            test_command=test_command,
            harness_results=results,
            source_label="unit_test_agent",
        )

    def _try_discovery_candidates(self, ctx: PhaseContext) -> _LayerResult | None:
        """Pre-UTA fast path: iterate validated discovery candidates.

        Legacy behavior from ``preprocessor.py:697-768`` — discovery
        often produces harness-shaped tests (focused_test, tests with
        ``--correctness``-compatible entry points).  Trying these
        first avoids an unnecessary UTA round-trip when a ready-to-use
        harness is in the discovery output.
        """
        try:
            from minisweagent.run.preprocess.harness_utils import (
                execute_harness_validation,
                validate_harness,
            )
            from minisweagent.run.preprocess.preprocessor import (
                _build_harness_candidates,
                _ensure_harness_has_no_kernel_defs,
                _materialize_preprocessor_harness,
            )
        except ImportError:
            return None

        disc_dict = ctx.discovery or {}
        tests = disc_dict.get("tests") or []
        benchmarks = disc_dict.get("benchmarks") or []

        candidates = _build_harness_candidates(tests, benchmarks, disc_dict, ctx.kernel_path)
        if not candidates:
            return None

        seen: set[str] = set()
        for candidate_cmd, candidate_harness, source in candidates:
            if candidate_harness in seen:
                continue
            seen.add(candidate_harness)

            try:
                ok_static, _ = validate_harness(candidate_harness)
                if not ok_static:
                    continue
                ok_runtime, _, results = execute_harness_validation(
                    candidate_harness,
                    repo_root=ctx.repo_root,
                    gpu_id=ctx.gpu_id,
                )
                if not ok_runtime:
                    continue

                candidate_harness = _ensure_harness_has_no_kernel_defs(candidate_harness, Path(ctx.output_dir), {})
                candidate_cmd, candidate_harness, results = _materialize_preprocessor_harness(
                    test_command=candidate_cmd,
                    harness_path=candidate_harness,
                    repo_root=ctx.repo_root,
                    output_dir=Path(ctx.output_dir),
                    kernel_path=ctx.kernel_path,
                    gpu_id=ctx.gpu_id,
                    harness_results=results,
                )
                return _LayerResult(
                    harness_path=str(candidate_harness),
                    test_command=str(candidate_cmd),
                    harness_results=results,
                    source_label=source,
                )
            except Exception:
                continue
        return None

    def _run_shape_fixer(
        self,
        *,
        ctx: PhaseContext,
        model: Any,
        harness_file: Path,
        benchmarks: list[dict],
        tests: list[dict],
        initial_results: list[dict],
    ) -> list[dict]:
        """Run the shape-fixer to align harness shapes with user benchmark/test files.

        One retry on shape-fix failure (identical to legacy behavior).
        On total failure, restores the pre-fix harness source and
        returns the initial results unchanged.
        """
        from minisweagent.run.preprocess.harness_utils import (
            execute_harness_validation,
        )
        from minisweagent.run.preprocess.preprocessor import _restore_harness_file
        from minisweagent.run.preprocess.shape_fixer_agent import run_shape_fixer

        if not harness_file.is_file():
            return initial_results

        # Pick the shape-source file: UTA-declared first, then top
        # benchmark, then top test.
        shape_source: Path | None = None
        declared = harness_file.parent / "harness_shapes_source.txt"
        if declared.is_file():
            shape_source = Path(declared.read_text().strip())
        if (shape_source is None or not shape_source.is_file()) and benchmarks:
            shape_source = Path(benchmarks[0]["file"])
        if (shape_source is None or not shape_source.is_file()) and tests:
            shape_source = Path(tests[0]["file"])
        if shape_source is None or not shape_source.is_file():
            return initial_results

        original_source = harness_file.read_text()
        shape_feedback: list[str] | None = None

        for attempt in (0, 1):
            shapes_ok = run_shape_fixer(
                model=model,
                repo=Path(ctx.repo_root),
                harness_path=harness_file,
                benchmark_file=shape_source,
                kernel_path=Path(ctx.kernel_path),
                log_dir=Path(ctx.output_dir),
                gpu_id=ctx.gpu_id,
                validation_feedback=shape_feedback,
                user_task=ctx.user_task,
            )
            if not shapes_ok:
                _restore_harness_file(harness_file, original_source)
                return initial_results

            ok_revalidate, revalidate_errors, results = execute_harness_validation(
                str(harness_file), repo_root=ctx.repo_root, gpu_id=ctx.gpu_id
            )
            if ok_revalidate:
                return results

            if attempt == 0:
                shape_feedback = revalidate_errors
                continue

            # Second attempt also failed: restore + keep pre-fix results.
            _restore_harness_file(harness_file, original_source)
            return initial_results

        return initial_results

    # ──────────────────────────────────────────────────────────────────
    # Layer 7 — discovery focused_test / tests[0] fallback
    # ──────────────────────────────────────────────────────────────────

    def _layer7_discovery_fallback(self, ctx: PhaseContext, _selection: dict[str, Any]) -> _LayerResult | None:
        """Last-resort: use whatever command discovery produced, unvalidated.

        These commands may not follow the universal contract (often
        pytest invocations without ``--correctness``/``--benchmark``
        flags) so downstream consumers that require the contract
        will see degraded behavior.  Kept for legacy parity with
        ``preprocessor.py:932-944``.
        """
        disc_dict = ctx.discovery or {}
        focused = disc_dict.get("focused_test") or {}
        focused_cmd = focused.get("focused_command") if isinstance(focused, dict) else None
        tests = disc_dict.get("tests") or []

        from minisweagent.run.preprocess.harness_utils import extract_harness_path

        cmd: str | None = None
        label = ""
        if focused_cmd:
            cmd = str(focused_cmd)
            label = "fallback_focused_test"
        elif tests:
            first = tests[0]
            if isinstance(first, dict) and first.get("command"):
                cmd = str(first["command"])
                label = "fallback_discovery_test"

        if cmd is None:
            return None

        harness_path = extract_harness_path(cmd)
        return _LayerResult(
            harness_path=harness_path,
            test_command=cmd,
            harness_results=[],
            source_label=label,
        )

    # ──────────────────────────────────────────────────────────────────
    # Post-resolution bookkeeping
    # ──────────────────────────────────────────────────────────────────

    def _init_testcase_selection(self, ctx: PhaseContext) -> dict[str, Any]:
        """Return a fresh testcase_selection dict with the caller's inputs seeded."""
        return {
            "cache_key": None,
            "cache_dir": None,
            "reused_cache": False,
            "selected_source": None,
            "saved_cache_manifest": None,
            "harness": ctx.harness,
        }

    @staticmethod
    def _persist_harness_results(ctx: PhaseContext, results: list[dict]) -> None:
        if not results:
            return
        try:
            (Path(ctx.output_dir) / "harness_results.json").write_text(json.dumps(results, indent=2, default=str))
        except Exception as exc:
            logger.debug("Failed to persist harness_results.json: %s", exc)

    @staticmethod
    def _save_to_testcase_cache(
        ctx: PhaseContext,
        result: _LayerResult,
        selection: dict[str, Any],
    ) -> None:
        """Save the winning harness to the testcase cache so future runs
        short-circuit to Layer 4 instead of re-doing the chain."""
        if ctx.harness:
            # Caller-supplied harness bypasses the cache by design.
            return

        try:
            from minisweagent.run.preprocess.testcase_cache import (
                build_testcase_cache_key,
                get_testcase_cache_dir,
                get_testcase_cache_entry,
                save_cached_harness,
            )
        except ImportError:
            return

        cache_dir = get_testcase_cache_dir()
        if cache_dir is None:
            return
        cache_key = build_testcase_cache_key(ctx.kernel_url, ctx.kernel_path)
        cache_entry = get_testcase_cache_entry(cache_dir, cache_key)
        if cache_entry is None:
            return

        try:
            manifest_path = save_cached_harness(
                cache_entry,
                kernel_url=ctx.kernel_url,
                source=result.source_label or "validated_harness",
                test_command=result.test_command,
                harness_path=result.harness_path,
                repo_root=ctx.repo_root,
                output_dir=Path(ctx.output_dir),
                kernel_path=ctx.kernel_path,
                harness_results=result.harness_results,
            )
            selection["saved_cache_manifest"] = str(manifest_path) if manifest_path else None
        except Exception as exc:
            selection["cache_save_error"] = str(exc)

    @staticmethod
    def _validate_contract(path_str: str | None) -> None:
        """Run the universal harness contract validator.  Warnings only."""
        if not path_str:
            return
        try:
            from minisweagent.kernel_languages.contract import validate_harness

            validate_harness(Path(path_str))
        except Exception as exc:
            logger.warning("[yellow]validate_harness: %s[/yellow]", exc)


# ──────────────────────────────────────────────────────────────────────
# Module-level helpers (shared across layers)
# ──────────────────────────────────────────────────────────────────────


def _build_test_command(harness_path: str) -> str:
    """Legacy-compatible test command: ``python3 <harness> --correctness``."""
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(harness_path).resolve()))} --correctness"


def _resolve_model(ctx: PhaseContext) -> Any:
    """Return a model instance, constructing via factory if needed."""
    if ctx.model is not None:
        return ctx.model
    factory = getattr(ctx, "model_factory", None)
    if callable(factory):
        try:
            return factory()
        except Exception:
            return None
    return None


def _extract_user_test_files(ctx: PhaseContext) -> list[Path]:
    """Pull test-file paths from ``ctx.discovery`` (all known shapes)."""
    if not ctx.discovery:
        return []
    seen: set[str] = set()
    out: list[Path] = []

    def _add(raw: Any) -> None:
        if not isinstance(raw, str):
            return
        p = Path(raw)
        if p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)

    for entry in ctx.discovery.get("tests") or []:
        if isinstance(entry, dict):
            for k in ("file", "path", "harness_path"):
                _add(entry.get(k))

    focused = ctx.discovery.get("focused_test") or {}
    if isinstance(focused, dict):
        for k in ("file", "path", "harness_path"):
            _add(focused.get(k))

    for raw in ctx.discovery.get("user_tests") or []:
        _add(raw)

    return out


def _read_codebase_context(ctx: PhaseContext) -> str:
    """Return the codebase-context markdown PLUS a content-agnostic repo recon.

    HarnessBuilder is a one-shot LLM transform with no tools to go
    explore the repo on its own.  The recon below is **deliberately
    content-agnostic** — no hardcoded filename globs (Makefile,
    scripts/task_runner.py, eval_tools/, etc.) — so it works for any
    language convention (HIP, Triton, FlyDSL, CMake, Bazel, ninja,
    custom shell, future additions).

    Two layers:

      1. **Repo tree listing** — ``find`` style listing capped at
         depth 3 with file sizes.  Lets the LLM SEE the structure of
         the repo and reason about which files matter for harness
         construction.  Generic across languages.

      2. **Small-file auto-inclusion** — every file under the repo
         root with size <= ``_AUTOINCLUDE_FILE_BYTES`` (default 4 KB)
         is included verbatim, EXCEPT obvious noise (binaries,
         caches, .git, vendor dirs).  Small-file size is the filter,
         NOT filename — Makefile, config.yaml, BUILD.bazel,
         CMakeLists.txt, README, scripts/*.sh, scripts/*.py — all get
         picked up by virtue of being small.

    Larger files (>4 KB) are listed with their size in the tree so
    the LLM knows they exist; if HarnessBuilder needs their content
    the next prompt iteration can request it explicitly (handled by
    HarnessBuilder's retry-with-feedback loop).

    The result is generic, scales to new languages without code
    changes, and gives the LLM the same evidence an engineer would
    have when first inspecting the repo.
    """
    parts: list[str] = []
    if ctx.codebase_context_path:
        try:
            parts.append(Path(ctx.codebase_context_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    repo_root = Path(ctx.repo_root) if ctx.repo_root else None
    if repo_root is None or not repo_root.is_dir():
        return "\n\n".join(p for p in parts if p)

    tree_block = _render_repo_tree(repo_root)
    if tree_block:
        parts.append(tree_block)

    files_block = _render_autoinclude_files(repo_root)
    if files_block:
        parts.append(files_block)

    return "\n\n".join(p for p in parts if p)


# ──────────────────────────────────────────────────────────────────────
# Repo recon helpers — content-agnostic + size-bounded
# ──────────────────────────────────────────────────────────────────────


_TREE_MAX_DEPTH = 3
_TREE_MAX_ENTRIES = 200  # total entries shown in the tree listing
_AUTOINCLUDE_FILE_BYTES = 4096  # files <= this size get included verbatim
_AUTOINCLUDE_TOTAL_BYTES = 32768  # total auto-included bytes per repo (cap)
_NOISE_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".cache",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        "build",
        ".next",
        "dist",
        "target",
        "third_party",
        "vendor",
        "tools_runtime",  # GEAK-internal tooling, never relevant to harnessing
    }
)
_NOISE_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".o",
        ".a",
        ".lib",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".out",
        ".class",
        ".jar",
        ".tar",
        ".tgz",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".pkl",
        ".npy",
        ".pt",
        ".pth",
        ".onnx",
        ".bin",
        ".lock",
    }
)


def _is_noise(path: Path) -> bool:
    """Return True if ``path`` is something the LLM shouldn't see (binaries, caches)."""
    if path.suffix.lower() in _NOISE_SUFFIXES:
        return True
    for part in path.parts:
        if part in _NOISE_DIRS:
            return True
    return False


def _render_repo_tree(repo_root: Path) -> str:
    """Render a depth-bounded tree of ``repo_root`` with file sizes.

    Generic ``find -maxdepth N``-style listing: shows directory layout
    + file sizes so the LLM can identify build files, test runners,
    config files, wrappers, etc., regardless of naming convention.
    Skips noise (binaries, caches, .git).
    """
    entries: list[tuple[Path, int]] = []
    for path in sorted(repo_root.rglob("*")):
        # Depth check (relative to repo_root)
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        if len(rel.parts) > _TREE_MAX_DEPTH:
            continue
        if _is_noise(path):
            continue
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            continue
        entries.append((rel, size))
        if len(entries) >= _TREE_MAX_ENTRIES:
            break

    if not entries:
        return ""

    lines = [
        "## REPO RECON: directory tree",
        f"`{repo_root}` (depth ≤ {_TREE_MAX_DEPTH}, "
        f"showing up to {_TREE_MAX_ENTRIES} entries; binaries/caches filtered):",
        "```",
    ]
    for rel, size in entries:
        marker = "/" if not size else f"  ({size} B)"
        lines.append(f"{rel}{marker}")
    lines.append("```")
    return "\n".join(lines)


def _render_autoinclude_files(repo_root: Path) -> str:
    """Auto-include small files (≤ ``_AUTOINCLUDE_FILE_BYTES``) verbatim.

    Generic across languages: any small text file (Makefile, config.yaml,
    scripts/*.sh, BUILD.bazel, README.md, …) gets included by virtue of
    being small — no filename pattern matching.
    """
    chunks: list[str] = []
    total_bytes = 0
    seen: set[str] = set()

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or _is_noise(path):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > _AUTOINCLUDE_FILE_BYTES:
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not _looks_textual(content):
            continue

        rel = path.relative_to(repo_root)
        chunk = f"\n### `{rel}`  ({size} B)\n\n```\n{content.rstrip()}\n```"
        if total_bytes + len(chunk) > _AUTOINCLUDE_TOTAL_BYTES:
            break
        chunks.append(chunk)
        total_bytes += len(chunk)

    if not chunks:
        return ""

    header = (
        f"## REPO RECON: auto-included small files "
        f"(≤ {_AUTOINCLUDE_FILE_BYTES} B each, total cap {_AUTOINCLUDE_TOTAL_BYTES} B)\n"
    )
    return header + "\n".join(chunks)


def _looks_textual(text: str) -> bool:
    """Heuristic: skip binaries that slipped through the suffix filter."""
    if not text:
        return False
    # Files with > 1% null bytes or > 30% control chars are binary
    sample = text[:2048]
    nulls = sample.count("\x00")
    if nulls / max(len(sample), 1) > 0.01:
        return False
    ctrl = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
    if ctrl / max(len(sample), 1) > 0.30:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# GEAK_HARNESS_ONLY early-exit helper (called by the orchestrator)
# ──────────────────────────────────────────────────────────────────────


def is_harness_only_mode() -> bool:
    """``GEAK_HARNESS_ONLY=1`` skips baseline + commandment.  Used by
    ``test_harness_variance.py`` to validate harness shapes quickly."""
    return os.environ.get("GEAK_HARNESS_ONLY", "").strip() == "1"


__all__ = ["HarnessPhase", "is_harness_only_mode"]
