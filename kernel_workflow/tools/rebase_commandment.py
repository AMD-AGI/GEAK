#!/usr/bin/env python3
"""Materialize a cached COMMANDMENT with one controlled EVAL_DIR rewrite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


_EVAL_ROOT = re.compile(
    r"(/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*/geak_runs/"
    r"[A-Za-z0-9_.-]+/ws_[A-Za-z0-9_.-]+)"
)
_FIXED_CANDIDATE = re.compile(r"/geak_state/[^\s`\"']+/candidates/")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rebase_commandment(
    source: Path,
    target: Path,
    target_eval: Path,
) -> dict[str, object]:
    source = source.resolve()
    target = target.resolve()
    target_eval = target_eval.resolve()
    if target != target_eval / "COMMANDMENT.md":
        raise ValueError("target must be TARGET_EVAL/COMMANDMENT.md")

    raw = source.read_bytes()
    text = raw.decode("utf-8")
    roots = sorted(set(_EVAL_ROOT.findall(text)))
    if len(roots) != 1:
        raise ValueError(f"expected exactly one cached EVAL_DIR, found {roots}")
    if _FIXED_CANDIDATE.search(text):
        raise ValueError("cached COMMANDMENT contains a fixed candidate path")

    source_eval = roots[0]
    current_eval = str(target_eval)
    materialized = text.replace(source_eval, current_eval)
    if source_eval in materialized:
        raise ValueError("cached EVAL_DIR survived materialization")
    remaining = sorted(set(_EVAL_ROOT.findall(materialized)))
    if remaining != [current_eval]:
        raise ValueError(f"materialized COMMANDMENT has foreign EVAL_DIRs: {remaining}")
    if _FIXED_CANDIDATE.search(materialized):
        raise ValueError("materialized COMMANDMENT contains a fixed candidate path")

    source_normalized = text.replace(source_eval, "<EVAL_DIR>")
    target_normalized = materialized.replace(current_eval, "<EVAL_DIR>")
    if source_normalized != target_normalized:
        raise ValueError("materialization changed content beyond EVAL_DIR")

    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = materialized.encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=".COMMANDMENT.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return {
        "materialized": True,
        "source_eval": source_eval,
        "target_eval": current_eval,
        "source_sha256": _sha(raw),
        "target_sha256": _sha(encoded),
        "normalized_sha256": _sha(source_normalized.encode("utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-eval", type=Path, required=True)
    args = parser.parse_args()
    result = rebase_commandment(args.source, args.target, args.target_eval)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
