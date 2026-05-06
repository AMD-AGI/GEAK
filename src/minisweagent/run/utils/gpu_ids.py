"""Parse ``gpu_ids`` strings for CLI, orchestrator dispatch, and tests.

Keeps GPU parsing out of agent packages so ``homogeneous/`` stays a thin
fixed-mode runner + tests rather than owning shared utilities.
"""

from __future__ import annotations


def parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    """Parse a comma/range GPU spec into a non-empty list of device indices.

    Accepts ``"4,5,6,7"``, ``"4-7"``, ``"0,1,4-7"``, ``None`` / ``""``
    (defaults to ``[0]``).  Implementation delegates to
    ``config_editor._parse_gpu_ids_string``.
    """
    from minisweagent.run.utils.config_editor import _parse_gpu_ids_string

    result = _parse_gpu_ids_string(gpu_ids_str)
    return result if result else [0]
