#!/usr/bin/env python3
"""Create an identity-bound closure for a preflight debt.

Usage:
  debt.py close --preflight preflight.json --debt-id EVIDENCE-... \
    --source-sha SHA256 --boundary-id ID --owner direction \
    --evidence-ref capture.json --out debt_closure.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "kernel_opt.debt_closure/1"


def _load(path: str):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def close(args: argparse.Namespace) -> dict:
    preflight = _load(args.preflight)
    debt = next((item for item in preflight.get("debts", [])
                 if isinstance(item, dict) and item.get("debt_id") == args.debt_id), None)
    if debt is None:
        raise ValueError("preflight contains no matching debt_id")
    if debt.get("source_sha256") != args.source_sha:
        raise ValueError("source SHA does not match the preflight debt")
    if debt.get("boundary_id") != args.boundary_id:
        raise ValueError("boundary identity does not match the preflight debt")
    if not args.owner or not args.evidence_ref:
        raise ValueError("owner and evidence ref are required")
    return {
        "schema": SCHEMA,
        "debt_id": args.debt_id,
        "preflight_ref": args.preflight,
        "source_sha256": args.source_sha,
        "boundary_id": args.boundary_id,
        "owner": args.owner,
        "evidence_ref": args.evidence_ref,
        "status": "closed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("close")
    command.add_argument("--preflight", required=True)
    command.add_argument("--debt-id", required=True)
    command.add_argument("--source-sha", required=True)
    command.add_argument("--boundary-id", required=True)
    command.add_argument("--owner", required=True)
    command.add_argument("--evidence-ref", required=True)
    command.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        document = close(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[debt] ERROR: {exc}")
        return 2
    Path(args.out).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
