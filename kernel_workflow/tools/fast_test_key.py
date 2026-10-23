#!/usr/bin/env python3
"""Compute the path-independent fast-test cache key."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(raw: str) -> bytes:
    value = json.loads(raw)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _tree_digest(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*.py")
        if ".git" not in path.relative_to(root).parts
    )
    if not files:
        raise ValueError(f"no Python files under frozen tree {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def compute_key(
    frozen: Path,
    bench: Path,
    control_json: str,
    guards_json: str,
) -> dict[str, str]:
    frozen_rev = _tree_digest(frozen)
    bench_sha = _sha_bytes(bench.read_bytes())
    control_sha = _sha_bytes(_canonical_json(control_json))
    guards_sha = _sha_bytes(_canonical_json(guards_json))
    key = _sha_bytes(
        f"{frozen_rev}|{bench_sha}|{control_sha}|{guards_sha}".encode("ascii")
    )
    return {
        "key": key,
        "frozen_rev": frozen_rev,
        "bench_sha": bench_sha,
        "control_sha": control_sha,
        "guards_sha": guards_sha,
        "algorithm": "geak-fast-test-key-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--control-json", required=True)
    parser.add_argument("--guards-json", required=True)
    args = parser.parse_args()
    result = compute_key(
        args.frozen.resolve(),
        args.bench.resolve(),
        args.control_json,
        args.guards_json,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
