#!/usr/bin/env python3
"""Restricted public facade for the canonical optimization journal.

The facade exposes only bounded journal operations.  It defaults every mutable
path to the current work root and rejects path escapes; the richer compatibility
surface in round_record.py remains internal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import round_record


PUBLIC_COMMANDS = ("append", "recall", "summary", "query", "mark-measurement")
PATH_FLAGS = {
    "--ledger", "--out", "--receipt-out", "--recall-receipt", "--measurement-events",
    "--recall-events", "--events",
}


def describe() -> dict:
    return {
        "schema": "recordctl.describe/1",
        "name": "recordctl",
        "classification": "public",
        "commands": list(PUBLIC_COMMANDS),
        "journal_schema": round_record.JOURNAL_SCHEMA,
        "work_kinds": list(round_record.WORK_KINDS),
        "guarantees": [
            "append requires a canonical work_kind envelope",
            "recall always writes a hashed bounded receipt",
            "mutable paths remain inside the current work root",
            "no raw profile body is accepted",
        ],
    }


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv or any(item.startswith(flag + "=") for item in argv)


def _flag_values(argv: list[str], flag: str) -> list[str]:
    values = []
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} needs a value")
            values.append(argv[index + 1])
        elif item.startswith(flag + "="):
            values.append(item.split("=", 1)[1])
    return values


def _confined(raw: str, work: Path) -> None:
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (work / candidate).resolve()
    try:
        resolved.relative_to(work)
    except ValueError as exc:
        raise ValueError(f"public journal path escapes the work root: {raw!r}") from exc


def _prepare(command: str, argv: list[str], work: Path) -> list[str]:
    forwarded = list(argv)
    if command in ("append", "recall", "summary", "query") and not _has_flag(forwarded, "--ledger"):
        forwarded[:0] = ["--ledger", round_record.DEFAULT_JOURNAL]
    if command == "append" and not _has_flag(forwarded, "--work-kind"):
        raise ValueError("public append requires --work-kind")
    if command == "recall":
        if not _has_flag(forwarded, "--receipt-out"):
            receipt = f".kod/recall/{uuid.uuid4().hex}.json"
            forwarded.extend(["--receipt-out", receipt])
        if not _has_flag(forwarded, "--trigger"):
            forwarded.extend(["--trigger", "manual"])
    for flag in PATH_FLAGS:
        for value in _flag_values(forwarded, flag):
            _confined(value, work)
    return [command, *forwarded]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--format", choices=["json"])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("command", nargs="?")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="recordctl-") as tmp:
        work = Path(tmp).resolve()
        prepared = _prepare(
            "append",
            ["--work-kind", "sweep_batch", "--round", "1", "--hypothesis", "h",
             "--prediction", "p", "--change-ref", "work:/patch.diff",
             "--comparator", "golden", "--verdict", "null"],
            work,
        )
        assert prepared[0] == "append" and round_record.DEFAULT_JOURNAL in prepared
        recall = _prepare("recall", ["--trigger", "manual"], work)
        assert "--receipt-out" in recall
        try:
            _prepare("query", ["--ledger", "../escape.jsonl"], work)
        except ValueError:
            pass
        else:
            raise AssertionError("path escape was accepted")
    print("[recordctl] SELFTEST PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.selftest:
        return _selftest()
    if args.command not in PUBLIC_COMMANDS:
        _parser().error(f"command must be one of {PUBLIC_COMMANDS}")
    try:
        forwarded = _prepare(args.command, args.argv, Path.cwd().resolve())
    except ValueError as exc:
        print(f"[recordctl] ERROR: {exc}", file=sys.stderr)
        return 2
    return round_record.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
