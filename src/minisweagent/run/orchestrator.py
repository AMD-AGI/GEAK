"""Preprocess directory probing utility.

``_probe_preprocess_dir`` reconstructs a ``PreprocessContext`` by scanning
artefact files on disk.  Used by tests and any tool that consumes a
preprocess artefact directory without a ``preprocess_context.json`` manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _probe_preprocess_dir(pp_dir: Path):
    """Backward-compatible fallback: reconstruct PreprocessContext by probing files.

    Retained for tests and for any tool that consumes a preprocess artefact
    directory without a ``preprocess_context.json`` manifest.  Not part of
    a CLI flow anymore.
    """
    from minisweagent.run.pipeline_types import PreprocessContext

    logger.debug("_probe_preprocess_dir: probing %s for preprocessor artefacts.", pp_dir)
    kernel_path = ""
    repo_root = str(pp_dir)
    harness_path = ""

    resolved_path = pp_dir / "resolved.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text())
        kernel_path = resolved.get("local_file_path", "")
        repo_path = resolved.get("local_repo_path")
        if kernel_path:
            kp = Path(kernel_path).resolve()
            git_root = None
            cur = kp if kp.is_dir() else kp.parent
            while cur != cur.parent:
                if (cur / ".git").exists():
                    git_root = cur
                    break
                cur = cur.parent
            if git_root:
                repo_root = str(git_root)
                logger.debug("_probe_preprocess_dir: repo_root from git walk: %s", repo_root)
            elif repo_path:
                repo_root = repo_path
                logger.debug("_probe_preprocess_dir: repo_root from resolved.json: %s", repo_root)
            else:
                repo_root = str(Path(kernel_path).parent)
                logger.debug("_probe_preprocess_dir: repo_root defaulted to kernel parent: %s", repo_root)

    testcase_sel_path = pp_dir / "testcase_selection.json"
    if testcase_sel_path.exists():
        try:
            ts = json.loads(testcase_sel_path.read_text())
            if isinstance(ts, dict):
                harness_path = ts.get("harness_path", "")
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("_probe_preprocess_dir: failed to read testcase_selection.json: %s", exc)

    discovery = None
    discovery_path = pp_dir / "discovery.json"
    if discovery_path.exists():
        try:
            discovery = json.loads(discovery_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("_probe_preprocess_dir: failed to read discovery.json: %s", exc)

    return PreprocessContext(
        kernel_path=kernel_path,
        repo_root=repo_root,
        harness_path=harness_path,
        preprocess_dir=str(pp_dir),
        commandment_path=str(pp_dir / "COMMANDMENT.md") if (pp_dir / "COMMANDMENT.md").exists() else "",
        codebase_context_path=str(pp_dir / "CODEBASE_CONTEXT.md") if (pp_dir / "CODEBASE_CONTEXT.md").exists() else "",
        baseline_metrics_path=str(pp_dir / "baseline_metrics.json")
        if (pp_dir / "baseline_metrics.json").exists()
        else "",
        profiling_result_path=str(pp_dir / "profile.json") if (pp_dir / "profile.json").exists() else "",
        discovery=discovery,
    )
