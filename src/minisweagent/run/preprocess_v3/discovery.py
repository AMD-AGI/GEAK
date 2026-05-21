"""Legacy discovery wrapper for the v3 preprocess pipeline.

This is the deterministic front half we want to preserve from the legacy
preprocessor:

1. resolve the kernel/repo path,
2. write ``CODEBASE_CONTEXT.md`` using the legacy codebase-context builder,
3. run automated-test-discovery (ATD),
4. write ``discovery.json``,
5. render the same discovery briefing legacy UTA consumed.

The v3 redesign starts *after* this point: the downstream harness agents are
general, with language-specific KB injection. They should not rediscover tests
or invent shape sources when ATD already found authoritative files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.discovery import DiscoveryPhase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveryContext:
    """Artifacts produced by the legacy discovery front half."""

    kernel_path: Path
    repo_root: Path
    codebase_context_path: Path | None = None
    discovery_path: Path | None = None
    discovery: dict[str, Any] = field(default_factory=dict)
    discovery_context_text: str = ""
    codebase_context_text: str = ""
    kernel_language: KernelLanguage | None = None


def run_legacy_discovery(
    *,
    kernel_path: Path,
    repo_root: Path,
    output_dir: Path,
    kernel_language: KernelLanguage | None = None,
    harness: str | None = None,
) -> DiscoveryContext:
    """Run legacy ``DiscoveryPhase`` and format its output for v3 agents."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = PhaseContext(
        kernel_url=str(Path(kernel_path).resolve()),
        output_dir=output_dir,
        repo=Path(repo_root).resolve(),
        harness=harness,
    )
    DiscoveryPhase().run(ctx)

    resolved_kernel_path = Path(ctx.kernel_path or kernel_path).resolve()
    resolved_repo_root = Path(ctx.repo_root or repo_root).resolve()
    discovery = ctx.discovery or {}
    codebase_context_path = Path(ctx.codebase_context_path).resolve() if ctx.codebase_context_path else None
    discovery_path = output_dir / "discovery.json"

    codebase_text = ""
    if codebase_context_path and codebase_context_path.is_file():
        try:
            codebase_text = codebase_context_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("failed to read CODEBASE_CONTEXT.md: %s", exc)

    discovery_context = format_legacy_discovery_context(
        discovery,
        kernel_path=resolved_kernel_path,
    )
    discovery_context_path = output_dir / "DISCOVERY_CONTEXT.md"
    if discovery_context:
        discovery_context_path.write_text(discovery_context, encoding="utf-8")

    return DiscoveryContext(
        kernel_path=resolved_kernel_path,
        repo_root=resolved_repo_root,
        codebase_context_path=codebase_context_path,
        discovery_path=discovery_path if discovery_path.is_file() else None,
        discovery=discovery,
        discovery_context_text=discovery_context,
        codebase_context_text=codebase_text,
        kernel_language=kernel_language,
    )


def format_legacy_discovery_context(
    discovery: dict[str, Any],
    *,
    kernel_path: Path,
) -> str:
    """Render the UTA-facing legacy discovery context from ``discovery.json``."""

    if not discovery:
        return ""

    try:
        from minisweagent.run.preprocess.discovery_types import DiscoveryResult
        from minisweagent.run.preprocess.preprocessor import _build_repo_native_reference_context
        from minisweagent.run.preprocess.unit_test_agent import format_discovery_for_agent

        disc_result = DiscoveryResult.from_dict(discovery, str(kernel_path))
        parts = [
            "# Discovery Context",
            "This block is copied from the legacy preprocessing handoff. It is authoritative for harness generation.",
            format_discovery_for_agent(disc_result),
        ]
        repo_native = _build_repo_native_reference_context(
            tests=discovery.get("tests") or [],
            benchmarks=discovery.get("benchmarks") or [],
            kernel_path=kernel_path,
        )
        if repo_native:
            parts.append(repo_native)
        return "\n\n".join(p for p in parts if p and p.strip()).rstrip() + "\n"
    except Exception as exc:  # noqa: BLE001
        logger.warning("legacy discovery context render failed (non-fatal): %s", exc)
        return ""


__all__ = ["DiscoveryContext", "format_legacy_discovery_context", "run_legacy_discovery"]
