#!/usr/bin/env python3
"""Resolve one stable fair-share group from a managed run's working directory.

The result deliberately skips direction/workspace leaves: a direction is work *for* a kernel, not
an independent tenant. The module is used by both ``gpu_client.py`` and GEAK's
``kernel_workflow/scripts/gpu_lock.sh`` (broker path, GEAK_GPU_BROKER=1) so their fallbacks cannot
drift.

The ``.geak_gpu_group`` marker is OPTIONAL and nothing in GEAK writes it: a campaign that wants an
exact fair-share key writes it at its root (or exports GEAK_GPU_GROUP / GEAK_KERNEL_NAME). Without
it the path-shape fallback below applies.
"""
from __future__ import annotations

import argparse
import os
import re

MARKER_NAME = ".geak_gpu_group"
_DIRECTION_LEAF = re.compile(
    r"^(?:dir_\d+|direction(?:_\d+)?|workspace|ws(?:_.+)?|round_\d+|"
    r"engineer_\d+|arm_\d+|arms|ic_tech_lead|direction_worker|verify|"
    r"integrate|build|source)$"
)


def _clean(value):
    value = str(value or "").strip()
    return value or None


def infer_group(path) -> str | None:
    """Infer a kernel-level group while ignoring known per-direction leaves."""
    parts = [p for p in os.path.abspath(path).split(os.sep) if p]

    # The explicit team wrapper is per-run; the segment below it is the kernel identity.
    for i, segment in enumerate(parts):
        if segment.startswith("team_") and i + 1 < len(parts):
            candidate = parts[i + 1]
            if not _DIRECTION_LEAF.match(candidate):
                return candidate

    # Common experiment roots also make the next non-leaf segment a kernel identity.
    for i, segment in enumerate(parts):
        if segment in ("exp", "eval", "exp_root") or segment.startswith(
                ("exp_", "eval_", "exp_root_")):
            for candidate in parts[i + 1:]:
                if not _DIRECTION_LEAF.match(candidate):
                    return candidate

    # Hand-run trees may have no experiment anchor. Walk up, discarding direction leaves.
    for candidate in reversed(parts):
        if not _DIRECTION_LEAF.match(candidate) and candidate not in ("tmp", "home"):
            return candidate
    return None


def resolve_group(start=None, environ=None, allow_env=True):
    """Return ``(group, source)`` where source is env, marker, fallback, or missing."""
    environ = os.environ if environ is None else environ
    if allow_env:
        explicit = _clean(environ.get("GEAK_GPU_GROUP"))
        if explicit:
            return explicit, "env"
        kernel_name = _clean(environ.get("GEAK_KERNEL_NAME"))
        if kernel_name:
            return kernel_name, "kernel_env"

    current = os.path.abspath(start or os.getcwd())
    while True:
        marker = os.path.join(current, MARKER_NAME)
        try:
            with open(marker) as fh:
                marked = _clean(fh.readline())
            if marked:
                return marked, "marker"
        except OSError:
            pass
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    inferred = infer_group(start or os.getcwd())
    return inferred, "fallback" if inferred else "missing"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve the GPU scheduler fair-share group")
    parser.add_argument("--path", default=os.getcwd())
    parser.add_argument("--ignore-env", action="store_true")
    parser.add_argument("--source", action="store_true", help="print source after the group")
    args = parser.parse_args(argv)
    group, source = resolve_group(args.path, allow_env=not args.ignore_env)
    if group:
        print(f"{group}\t{source}" if args.source else group)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
