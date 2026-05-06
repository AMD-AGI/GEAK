"""Discovery phase — resolve URL, capture codebase context, run test discovery.

Output: ``ctx.kernel_path``, ``ctx.repo_root``, ``ctx.resolved``,
``ctx.codebase_context_path``, ``ctx.discovery``.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext
from minisweagent.run.preprocess.repo_paths import ensure_preprocess_mcp_importable

logger = logging.getLogger(__name__)


def _ensure_mcp_importable() -> None:
    ensure_preprocess_mcp_importable(
        "mcp_tools/profiler-mcp/src",
        "mcp_tools/metrix-mcp/src",
        "mcp_tools/automated-test-discovery/src",
    )


def _infer_repo_root(kernel_path: str) -> str:
    """Walk up from ``kernel_path`` until a repo marker is found.

    Markers: ``.git``, ``pyproject.toml``, ``setup.py``, ``setup.cfg``.
    Falls back to the kernel's parent directory with a warning.
    """
    p = Path(kernel_path).resolve().parent
    for ancestor in [p, *p.parents]:
        if any((ancestor / marker).exists() for marker in (".git", "pyproject.toml", "setup.py", "setup.cfg")):
            return str(ancestor)
    logger.warning("Could not infer repo_root from %s; using kernel parent dir", kernel_path)
    return str(p)


class DiscoveryPhase(Phase):
    """Step 1 + 2 + 3-discovery of the legacy preprocessor monolith.

    Runs:
      1. ``resolve_kernel_url(kernel_url, ...)`` → ``ctx.resolved`` +
         ``ctx.kernel_path`` + ``ctx.repo_root``.
      2. ``build_codebase_context(...)`` → ``ctx.codebase_context_path``
         (written to ``{output_dir}/CODEBASE_CONTEXT.md``).
      3. ATD MCP ``discover(...)`` → ``ctx.discovery``
         (written to ``{output_dir}/discovery.json``).

    This phase does NOT produce a harness; that's ``HarnessPhase``'s
    concern.  If the kernel file is a "merged" kernel (contains both
    kernel defs and test logic), the split is performed here so
    downstream phases see a clean kernel.
    """

    name = "discovery"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()

        if not ctx.kernel_url:
            raise ValueError("DiscoveryPhase requires ctx.kernel_url to be set")

        output_dir = Path(ctx.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve kernel URL ─────────────────────────────────────────
        from minisweagent.run.preprocess.resolve_kernel_url import resolve_kernel_url

        logger.info("[bold cyan]  ▸ resolve_kernel_url[/bold cyan]")
        resolved = resolve_kernel_url(
            ctx.kernel_url,
            repo=ctx.repo,
            clone_into=str(output_dir),
        )
        if resolved.get("error"):
            raise RuntimeError(f"resolve-kernel-url failed: {resolved['error']}")

        kernel_path = resolved["local_file_path"]
        repo_root = resolved.get("local_repo_path") or _infer_repo_root(kernel_path)
        if not repo_root:
            raise RuntimeError(f"Cannot determine repo_root for kernel: {kernel_path}")

        ctx.resolved = resolved
        ctx.kernel_path = kernel_path
        ctx.repo_root = repo_root

        # If the kernel file is a merged file (kernel defs + test logic
        # in one), split the test logic out so agents only patch the
        # clean kernel.
        #
        # Skip this when the user has explicitly supplied a harness
        # (``ctx.harness`` set by CLI/prompt parser): the user's harness
        # is canonical, and the auto-split would clobber any harness
        # file in ``output_dir`` that happens to share a stem with the
        # split target (e.g. user passes ``test_kernel_harness.py`` AND
        # the kernel's stem is ``kernel`` → split writes to
        # ``output_dir/test_kernel_harness.py``, overwriting the user's
        # file).
        from minisweagent.run.preprocess.harness_utils import (
            detect_and_split_kernel_from_harness,
        )

        if ctx.harness:
            logger.info(
                "  Skipping kernel split (user supplied harness: %s)",
                ctx.harness,
            )
            split_result = None
        else:
            split_result = detect_and_split_kernel_from_harness(kernel_path, output_dir)
        if split_result is not None:
            new_harness, clean_kernel = split_result
            logger.info(
                "  Kernel file was merged — split test logic to %s; kernel stays at %s",
                new_harness,
                clean_kernel,
            )
            ctx.kernel_path = clean_kernel
            # §13.2-A row 6: stash the split harness on a PUBLIC field
            # so HarnessPhase can pick it up when the caller didn't
            # supply an explicit ``--harness``.  The legacy monolith
            # (preprocessor.py:518-523) runs the split harness through
            # validate_harness and, if it passes static validation,
            # treats it as an explicit harness.  That matching behaviour
            # lives in HarnessPhase.
            ctx.split_harness_hint = str(new_harness)

        # 2. Codebase context ───────────────────────────────────────────
        from minisweagent.run.preprocess.codebase_context import (
            generate_codebase_context,
        )

        logger.info("[bold cyan]  ▸ codebase_context[/bold cyan]")
        try:
            codebase_context_path = generate_codebase_context(
                repo_root=Path(ctx.repo_root),
                kernel_path=Path(ctx.kernel_path),
                output_dir=output_dir,
            )
            ctx.codebase_context_path = str(codebase_context_path) if codebase_context_path else None
        except Exception as exc:
            logger.warning("codebase_context failed (non-fatal): %s", exc)
            ctx.codebase_context_path = None

        # 3. Test discovery (ATD MCP) ───────────────────────────────────
        logger.info("[bold cyan]  ▸ test_discovery[/bold cyan]")
        _ensure_mcp_importable()
        try:
            atd_server = importlib.import_module("automated_test_discovery.server")
            atd_discover = atd_server.discover
            discover_fn = getattr(atd_discover, "fn", atd_discover)

            discovery_kwargs: dict[str, Any] = {
                "kernel_path": ctx.kernel_path,
                "output_dir": str(output_dir),
            }
            if ctx.harness:
                discovery_kwargs["harness"] = ctx.harness
                discovery_kwargs["use_llm"] = False

            disc_dict = discover_fn(**discovery_kwargs)
        except Exception as exc:
            logger.warning("[yellow]Test discovery failed: %s[/yellow]", exc)
            disc_dict = {}

        ctx.discovery = disc_dict
        (output_dir / "discovery.json").write_text(json.dumps(disc_dict, indent=2, default=str))

        tests = disc_dict.get("tests", [])
        logger.info("  Tests found: %d", len(tests))

        # Resolve KernelLanguage via the registry so downstream phases
        # (ExplorePhase's Jinja commandment render; future
        # HarnessBuilder template lookup) can read
        # ``ctx.language.<path>`` without re-detecting.  Prefer the
        # discovery-provided kernel.type hint to avoid a file-read
        # when available; fall back to registry.detect_best otherwise.
        try:
            from minisweagent.kernel_languages import registry

            kernel_type_hint = (disc_dict.get("kernel") or {}).get("type")
            resolved_lang = None
            if kernel_type_hint:
                resolved_lang = registry.detect_best_by_name(kernel_type_hint)
            if resolved_lang is None and ctx.kernel_path:
                resolved_lang = registry.detect_best(Path(ctx.kernel_path))
            if resolved_lang is not None:
                ctx.language = resolved_lang
                logger.info("  KernelLanguage resolved: %s", resolved_lang.name)
        except Exception as exc:
            # Language resolution is best-effort — falling back to
            # None makes ExplorePhase use its legacy path.
            logger.debug("KernelLanguage resolution failed (non-fatal): %s", exc)

        ctx.phases_run.append(self.name)


__all__ = ["DiscoveryPhase"]
