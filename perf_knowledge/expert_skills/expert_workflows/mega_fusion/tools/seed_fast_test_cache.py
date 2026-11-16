#!/usr/bin/env python3
"""Seed a self-contained Mega fast-test cache from a completed run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fast_test_key import compute_key


def seed_cache(
    source_eval: Path,
    cache_dir: Path,
    frozen: Path,
    bench: Path,
    control_json: str,
    guards_json: str,
) -> dict[str, object]:
    source_eval = source_eval.resolve()
    cache_dir = cache_dir.resolve()
    baseline = source_eval / "baseline_timing.json"
    commandment = source_eval / "COMMANDMENT.md"
    controls = sorted(source_eval.glob("setup_ab_control*.json"))
    missing = [str(path) for path in (baseline, commandment) if not path.is_file()]
    if missing or not controls:
        raise ValueError(f"incomplete source run: missing {missing or 'setup_ab_control*.json'}")

    parsed_controls = [(path, json.loads(path.read_text())) for path in controls]
    completed = [
        (path, value)
        for path, value in parsed_controls
        if value.get("claim_complete") is True
        and value.get("ran") is True
        and value.get("passed") is True
    ]
    if len(completed) != 1:
        raise ValueError(f"expected one completed positive control, found {len(completed)}")
    evidence = Path(str(completed[0][1].get("evidence_manifest", "")))
    if not evidence.is_file():
        raise ValueError(f"positive-control evidence is missing: {evidence}")

    key = compute_key(
        frozen.resolve(), bench.resolve(), control_json, guards_json
    )
    staging = cache_dir.with_name(f".{cache_dir.name}.tmp_{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copy2(baseline, staging / baseline.name)
    shutil.copy2(commandment, staging / commandment.name)
    shutil.copy2(evidence, staging / "setup_ab_control_evidence_manifest.json")

    for path, value in parsed_controls:
        if path == completed[0][0]:
            value["evidence_manifest"] = str(
                cache_dir / "setup_ab_control_evidence_manifest.json"
            )
        (staging / path.name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    key.update(
        {
            "source_eval_dir": str(source_eval),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    (staging / "fast_test_key.json").write_text(
        json.dumps(key, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if cache_dir.exists():
        backup = cache_dir.with_name(
            f"{cache_dir.name}.old_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
        )
        os.replace(cache_dir, backup)
    os.replace(staging, cache_dir)
    return {
        "published": True,
        "key": key["key"],
        "cache_dir": str(cache_dir),
        "source_eval_dir": str(source_eval),
        "files": sorted(path.name for path in cache_dir.iterdir()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--control-json", required=True)
    parser.add_argument("--guards-json", required=True)
    args = parser.parse_args()
    result = seed_cache(
        args.source_eval,
        args.cache_dir,
        args.frozen,
        args.bench,
        args.control_json,
        args.guards_json,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
