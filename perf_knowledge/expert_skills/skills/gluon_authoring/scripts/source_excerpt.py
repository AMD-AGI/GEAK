#!/usr/bin/env python3
"""Issue bounded source-read tickets and produce audited symbol excerpts.

This tool bounds only its own output.  It is not a sandbox and cannot stop a host,
editor, or agent from invoking a direct file-read operation outside this convention.

Usage:
  source_excerpt.py source-read-request --root DIR --source PATH --tool NAME
      --symbol NAME --reason TEXT --max-window LINES --out TICKET.json
  source_excerpt.py source-excerpt --root DIR --ticket TICKET.json --out EXCERPT.json
      --receipt-out RECEIPTS.jsonl
  source_excerpt.py --describe json|--format=json
  source_excerpt.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from typing import Any


REQUEST_SCHEMA = "kernel_opt.source_read_request/1"
EXCERPT_SCHEMA = "kernel_opt.source_excerpt/1"
RECEIPT_SCHEMA = "kernel_opt.source_read_receipt/1"
MAX_WINDOW = 128
MAX_REASON = 240


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()[:16]


def _nonempty(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty string")
    return value.strip()[:limit]


def _relative_source(root: str, source: str) -> str:
    root_abs = os.path.realpath(root)
    source_abs = os.path.realpath(source if os.path.isabs(source)
                                  else os.path.join(root_abs, source))
    try:
        if os.path.commonpath([root_abs, source_abs]) != root_abs:
            raise ValueError("source must be inside --root")
    except ValueError as exc:
        raise ValueError("source must be inside --root") from exc
    return os.path.relpath(source_abs, root_abs).replace(os.sep, "/")


def source_read_request(*, root: str, source: str, tool: str, symbol: str, reason: str,
                        max_window: int, run_id: str | None = None, track: str | None = None,
                        context_hash: str | None = None) -> dict:
    """Create a portable ticket.  It deliberately contains no source bytes."""
    if not isinstance(max_window, int) or not 1 <= max_window <= MAX_WINDOW:
        raise ValueError(f"max_window must be an integer in [1, {MAX_WINDOW}]")
    ticket = {
        "schema": REQUEST_SCHEMA,
        "tool": _nonempty(tool, 64),
        "source_ref": _relative_source(root, _nonempty(source, 256)),
        "symbol": _nonempty(symbol, 160),
        "reason": _nonempty(reason, MAX_REASON),
        "max_window": max_window,
        "run_id": _nonempty(run_id, 96) if run_id else None,
        "track": _nonempty(track, 160) if track else None,
        "context_hash": _nonempty(context_hash, 96) if context_hash else None,
    }
    ticket = {key: value for key, value in ticket.items() if value is not None}
    ticket["ticket_id"] = "src-" + _sha(ticket)
    return ticket


def _validate_ticket(ticket: Any) -> dict:
    if not isinstance(ticket, dict) or ticket.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"ticket must be a {REQUEST_SCHEMA} object")
    for field, limit in (("tool", 64), ("source_ref", 256), ("symbol", 160),
                         ("reason", MAX_REASON), ("ticket_id", 96)):
        _nonempty(ticket.get(field), limit)
    window = ticket.get("max_window")
    if not isinstance(window, int) or not 1 <= window <= MAX_WINDOW:
        raise ValueError(f"ticket.max_window must be an integer in [1, {MAX_WINDOW}]")
    return ticket


def _child_defs(node: ast.AST) -> list[ast.AST]:
    return [child for child in getattr(node, "body", [])
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _locate_python_symbol(tree: ast.Module, symbol: str) -> ast.AST:
    current: ast.AST = tree
    for part in symbol.split("."):
        if not part.isidentifier():
            raise ValueError(f"symbol component {part!r} is not a Python identifier")
        current = next((child for child in _child_defs(current)
                        if getattr(child, "name", None) == part), None)
        if current is None:
            raise ValueError(f"symbol {symbol!r} was not found")
    return current


def _source_path(root: str, source_ref: str) -> str:
    return os.path.realpath(os.path.join(os.path.realpath(root), source_ref))


def source_excerpt(*, root: str, ticket: dict) -> tuple[dict, dict]:
    """Return at most ticket.max_window source lines plus an independently storable receipt."""
    ticket = _validate_ticket(ticket)
    source_ref = _relative_source(root, ticket["source_ref"])
    path = _source_path(root, source_ref)
    if not path.endswith(".py"):
        raise ValueError("only Python symbol excerpts are supported; refuse unbounded text fallback")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ValueError(f"cannot read requested source: {exc}") from exc
    try:
        node = _locate_python_symbol(ast.parse(text, filename=source_ref), ticket["symbol"])
    except SyntaxError as exc:
        raise ValueError(f"cannot parse requested Python source: {exc.msg}") from exc
    start, end = node.lineno, node.end_lineno
    if not isinstance(end, int) or end < start:
        raise ValueError("symbol has no stable source line range")
    span = end - start + 1
    if span > ticket["max_window"]:
        raise ValueError(f"symbol spans {span} lines, exceeding ticket max_window={ticket['max_window']}")
    lines = text.splitlines()
    spare = ticket["max_window"] - span
    before = min(2, spare // 2, start - 1)
    after = min(2, spare - before, len(lines) - end)
    # If one side hit a file boundary, use the unused window on the other side.
    if before + after < spare:
        after = min(len(lines) - end, spare - before)
    if before + after < spare:
        before = min(start - 1, spare - after)
    line_start, line_end = start - before, end + after
    excerpt_text = "\n".join(lines[line_start - 1:line_end]) + "\n"
    excerpt = {
        "schema": EXCERPT_SCHEMA,
        "ticket_id": ticket["ticket_id"],
        "tool": ticket["tool"],
        "source_ref": source_ref,
        "symbol": ticket["symbol"],
        "reason": ticket["reason"],
        "max_window": ticket["max_window"],
        "returned_window": {"start_line": line_start, "end_line": line_end,
                            "line_count": line_end - line_start + 1},
        "symbol_window": {"start_line": start, "end_line": end},
        "excerpt_hash": hashlib.sha256(excerpt_text.encode("utf-8")).hexdigest()[:16],
        "content": excerpt_text,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "receipt_id": "receipt-" + _sha({"ticket": ticket["ticket_id"],
                                         "excerpt": excerpt["excerpt_hash"]}),
        "category": "re-read",
        "ticket_id": ticket["ticket_id"],
        "tool": ticket["tool"],
        "source_ref": source_ref,
        "symbol": ticket["symbol"],
        "reason_hash": hashlib.sha256(ticket["reason"].encode("utf-8")).hexdigest()[:16],
        "max_window": ticket["max_window"],
        "returned_window": excerpt["returned_window"],
        "returned_bytes": len(excerpt_text.encode("utf-8")),
        "excerpt_hash": excerpt["excerpt_hash"],
        "run_id": ticket.get("run_id"),
        "track": ticket.get("track"),
        "context_hash": ticket.get("context_hash"),
    }
    return excerpt, {key: value for key, value in receipt.items() if value is not None}


def _atomic_write(path: str, doc: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".source_excerpt.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_receipt(path: str, receipt: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")


def _load_ticket(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return _validate_ticket(json.load(fh))


def describe() -> dict:
    return {
        "tool": "source_excerpt",
        "schemas": {"ticket": REQUEST_SCHEMA, "excerpt": EXCERPT_SCHEMA,
                    "receipt": RECEIPT_SCHEMA},
        "commands": ["source-read-request", "source-excerpt"],
        "ticket_required_fields": ["tool", "source_ref", "symbol", "reason", "max_window"],
        "limits": {"max_window_lines": MAX_WINDOW, "max_reason_chars": MAX_REASON},
        "guarantees": [
            "ticketed excerpts only return a requested Python symbol plus bounded context",
            "every excerpt appends a receipt containing bytes and hash",
            "full ticket/excerpt JSON is written only with --out",
        ],
        "non_guarantees": [
            "cannot prevent host/editor/agent direct Read operations outside this tool",
        ],
    }


def _selftest() -> int:
    root = tempfile.mkdtemp(prefix="source_excerpt_selftest_")
    try:
        source = os.path.join(root, "sample.py")
        ticket_path = os.path.join(root, "ticket.json")
        excerpt_path = os.path.join(root, "excerpt.json")
        receipt_path = os.path.join(root, "source_read_receipts.jsonl")
        with open(source, "w", encoding="utf-8") as fh:
            fh.write("# preamble\n\nclass Box:\n    def work(self, x):\n        return x + 1\n\n# tail\n")
        ticket = source_read_request(root=root, source="sample.py", tool="ReadFile",
                                     symbol="Box.work", reason="inspect return path",
                                     max_window=8, run_id="run-a", track="track-a")
        assert set(("tool", "symbol", "reason", "max_window")) <= set(ticket)
        _atomic_write(ticket_path, ticket)
        excerpt, receipt = source_excerpt(root=root, ticket=_load_ticket(ticket_path))
        _atomic_write(excerpt_path, excerpt)
        _append_receipt(receipt_path, receipt)
        assert "def work" in excerpt["content"] and excerpt["returned_window"]["line_count"] <= 8
        assert receipt["category"] == "re-read" and receipt["returned_bytes"] > 0
        with open(receipt_path, encoding="utf-8") as fh:
            assert json.loads(fh.readline())["ticket_id"] == ticket["ticket_id"]
        too_small = dict(ticket, max_window=1)
        try:
            source_excerpt(root=root, ticket=too_small)
            raise AssertionError("oversized symbol was returned")
        except ValueError:
            pass
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print("[source_excerpt] SELFTEST PASS")
    return 0


def main() -> int:
    if sys.argv[1:] in (["--describe", "json"], ["--describe", "--format=json"]):
        print(json.dumps(describe(), ensure_ascii=False, sort_keys=True))
        return 0
    if sys.argv[1:] == ["--selftest"]:
        return _selftest()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    request = sub.add_parser("source-read-request", aliases=["request", "source_read_request"],
                             help="write a bounded source-read ticket")
    request.add_argument("--root", required=True)
    request.add_argument("--source", required=True)
    request.add_argument("--tool", required=True)
    request.add_argument("--symbol", required=True)
    request.add_argument("--reason", required=True)
    request.add_argument("--max-window", type=int, required=True)
    request.add_argument("--run-id")
    request.add_argument("--track")
    request.add_argument("--context-hash")
    request.add_argument("--out", required=True, help="where to write complete ticket JSON")
    excerpt = sub.add_parser("source-excerpt", aliases=["excerpt", "source_excerpt"],
                             help="write a bounded ticketed source excerpt and receipt")
    excerpt.add_argument("--root", required=True)
    excerpt.add_argument("--ticket", required=True)
    excerpt.add_argument("--out", required=True, help="where to write complete excerpt JSON")
    excerpt.add_argument("--receipt-out", required=True, help="append receipt JSONL here")
    sub.add_parser("selftest", help="run no-GPU fixture checks")
    args = parser.parse_args()
    if args.command == "selftest":
        return _selftest()
    try:
        if args.command in ("source-read-request", "request", "source_read_request"):
            ticket = source_read_request(root=args.root, source=args.source, tool=args.tool,
                                         symbol=args.symbol, reason=args.reason,
                                         max_window=args.max_window, run_id=args.run_id,
                                         track=args.track, context_hash=args.context_hash)
            _atomic_write(args.out, ticket)
            print(json.dumps({"ticket_id": ticket["ticket_id"], "symbol": ticket["symbol"],
                              "max_window": ticket["max_window"], "out": args.out},
                             ensure_ascii=False, sort_keys=True))
            return 0
        result, receipt = source_excerpt(root=args.root, ticket=_load_ticket(args.ticket))
        _atomic_write(args.out, result)
        _append_receipt(args.receipt_out, receipt)
        print(json.dumps({"ticket_id": receipt["ticket_id"], "receipt_id": receipt["receipt_id"],
                          "returned_bytes": receipt["returned_bytes"], "out": args.out,
                          "receipt_out": args.receipt_out}, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[source_excerpt] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
