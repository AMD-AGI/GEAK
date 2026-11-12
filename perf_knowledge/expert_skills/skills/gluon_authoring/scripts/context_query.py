#!/usr/bin/env python3
"""Select one bounded slice from a stage_context package.

There is intentionally no "dump" selector.  Callers must request one named slice
or one exact artifact reference; a package is the recall boundary, not a route back
to a full ledger or profile.

Usage:
  context_query.py --context stage_context.json --selector identity [--out FILE]
  context_query.py --context stage_context.json --ref exact/artifact/path [--out FILE]
  context_query.py --describe json|--format=json
  context_query.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import context_contracts as contracts
from stage_context import (
    SCHEMA as CONTEXT_SCHEMA,
    _atomic_write,
    _read_object,
    artifact_roots,
    context_paths,
    dt_now,
    verify_context_document,
)


SCHEMA = "kernel_opt.context_query/1"
SELECTORS = ("current-stage", "identity", "measurement", "obligations", "recall", "actions",
             "artifacts")
MAX_TEXT_LINES = 200


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get("schema") != CONTEXT_SCHEMA:
        raise ValueError(f"{path} is not a {CONTEXT_SCHEMA} package")
    return doc


def query(doc: dict, *, selector: str | None = None, ref: str | None = None) -> dict:
    if bool(selector) == bool(ref):
        raise ValueError("supply exactly one of selector or ref; a full-package query is forbidden")
    base = {"schema": SCHEMA, "context_hash": doc.get("context_hash")}
    if selector:
        selected = {
            "current-stage": {"current_stage": doc.get("current_stage")},
            "identity": {"identity": doc.get("identity") or {}},
            "measurement": {"measurement_summary": doc.get("measurement_summary") or []},
            "obligations": {"obligations": doc.get("obligations") or []},
            "recall": {"recall_summary": doc.get("recall_summary") or {}},
            "actions": {"actions": doc.get("actions") or {}},
            "artifacts": {"artifact_refs": doc.get("artifact_refs") or []},
        }[selector]
        return dict(base, selector=selector, matched=1, result=selected)

    refs = [entry for entry in (doc.get("artifact_refs") or [])
            if isinstance(entry, dict) and entry.get("ref") == ref]
    return dict(base, ref=ref, matched=len(refs), result={"artifact_refs": refs})


def _settle(lease_path: Path, lease: dict) -> None:
    lease["status"] = "settled"
    _atomic_write(str(lease_path), lease)


def _artifact_entry(context: dict, selector: str) -> dict:
    matches = [
        item for item in context.get("artifact_refs") or []
        if isinstance(item, dict) and selector in (item.get("name"), item.get("ref"))
    ]
    if len(matches) != 1:
        raise ValueError(f"artifact selector must match exactly one context ref: {selector!r}")
    if not isinstance(matches[0].get("artifact_ref"), dict):
        raise ValueError(f"artifact {selector!r} has no typed artifact_ref")
    return matches[0]


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must be empty or start with '/'")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdigit():
                raise ValueError(f"JSON Pointer list index is invalid: {token!r}")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"JSON Pointer list index is out of range: {index}")
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise ValueError(f"JSON Pointer member does not exist: {token!r}")
            current = current[token]
        else:
            raise ValueError(f"JSON Pointer descends through a scalar at {token!r}")
    return current


def _markdown_heading(text: str, heading: str) -> str:
    lines = text.splitlines(keepends=True)
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    start = level = None
    for index, line in enumerate(lines):
        match = heading_re.match(line.rstrip("\r\n"))
        if match and match.group(2) == heading:
            start, level = index, len(match.group(1))
            break
    if start is None or level is None:
        raise ValueError(f"Markdown heading not found: {heading!r}")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = heading_re.match(lines[index].rstrip("\r\n"))
        if match and len(match.group(1)) <= level:
            end = index
            break
    return "".join(lines[start:end])


def _python_symbol(text: str, symbol: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"artifact is not valid Python: {exc}") from exc
    candidates: list[tuple[str, ast.AST]] = []

    def walk(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                candidates.append((qualified, node))
                if isinstance(node, ast.ClassDef):
                    walk(node.body, qualified)

    walk(tree.body)
    matches = [node for name, node in candidates if name == symbol or name.endswith(f".{symbol}")]
    if len(matches) != 1:
        raise ValueError(f"Python symbol must match exactly once: {symbol!r}")
    node = matches[0]
    end = getattr(node, "end_lineno", None)
    if not isinstance(end, int):
        raise ValueError(f"Python parser did not report an end line for {symbol!r}")
    return "".join(text.splitlines(keepends=True)[node.lineno - 1:end])


def _text_window(text: str, start: int, count: int) -> str:
    if start < 1 or count < 1 or count > MAX_TEXT_LINES:
        raise ValueError(f"text window requires start>=1 and count in [1, {MAX_TEXT_LINES}]")
    return "".join(text.splitlines(keepends=True)[start - 1:start - 1 + count])


def _bounded(value: Any, max_bytes: int) -> tuple[Any, int, bool]:
    encoded = contracts.canonical_json_bytes(value)
    if len(encoded) <= max_bytes:
        return value, len(encoded), False
    if not isinstance(value, str):
        raise ValueError(
            f"selected structured result is {len(encoded)} bytes; maximum is {max_bytes}"
        )
    raw = value.encode("utf-8")
    clipped = raw[:max_bytes]
    while clipped:
        try:
            text = clipped.decode("utf-8")
            break
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    else:
        text = ""
    while len(contracts.canonical_json_bytes(text)) > max_bytes:
        text = text[:-1]
    return text, len(contracts.canonical_json_bytes(text)), True


def leased_query(
    work: Path,
    pack: Path,
    role: str,
    *,
    selector: str | None = None,
    ref: str | None = None,
    artifact: str | None = None,
    json_pointer: str | None = None,
    markdown_heading: str | None = None,
    python_symbol: str | None = None,
    text_window: tuple[int, int] | None = None,
    max_bytes: int | None = None,
) -> dict:
    """Execute one bounded query while atomically charging its active lease."""
    paths = context_paths(work.resolve())
    roots = artifact_roots(work.resolve(), pack.resolve())
    paths["dir"].mkdir(parents=True, exist_ok=True)
    lock_path = paths["dir"] / "context.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            context = _read_object(paths["context"])
            lease = _read_object(paths["lease"])
            policy = _read_object(paths["policy"])
            try:
                contracts.validate_context_lease(lease)
                context_hash = verify_context_document(context)
                contracts.resolve_artifact_ref(lease["context_ref"], roots)
                contracts.resolve_artifact_ref(lease["policy_ref"], roots)
            except (ValueError, KeyError, contracts.ContextContractError):
                _settle(paths["lease"], lease)
                raise
            if (
                lease.get("status") != "active"
                or lease.get("role") != role
                or lease.get("run_id") != context.get("run_id")
                or lease.get("generation") != context.get("generation")
                or lease.get("stage") != context.get("current_stage")
                or lease.get("context_sha256") != context_hash
            ):
                _settle(paths["lease"], lease)
                raise ValueError("active lease is not bound to this role and exact context")

            # Every query revalidates all context-carried artifacts.  External mutation therefore
            # invalidates the old lease even though no experiment directory is scanned.
            try:
                for item in context.get("artifact_refs") or []:
                    if isinstance(item, dict) and isinstance(item.get("artifact_ref"), dict):
                        contracts.resolve_artifact_ref(item["artifact_ref"], roots)
            except contracts.ContextContractError:
                _settle(paths["lease"], lease)
                raise

            modes = sum(value is not None for value in (
                selector, ref, json_pointer, markdown_heading, python_symbol, text_window
            ))
            if modes != 1:
                raise ValueError("select exactly one context/ref/artifact query mode")
            if selector is not None:
                selected = query(context, selector=selector)
                result: Any = selected["result"]
                query_kind, query_value = "selector", selector
            elif ref is not None:
                selected = query(context, ref=ref)
                result = selected["result"]
                query_kind, query_value = "ref", ref
            else:
                if not artifact:
                    raise ValueError("artifact query modes require --artifact")
                entry = _artifact_entry(context, artifact)
                path = contracts.resolve_artifact_ref(entry["artifact_ref"], roots)
                raw = path.read_bytes()
                if json_pointer is not None:
                    try:
                        source = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"artifact is not valid UTF-8 JSON: {exc}") from exc
                    result = _json_pointer(source, json_pointer)
                    query_kind, query_value = "json_pointer", json_pointer
                else:
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError("text artifact is not UTF-8") from exc
                    if markdown_heading is not None:
                        result = _markdown_heading(text, markdown_heading)
                        query_kind, query_value = "markdown_heading", markdown_heading
                    elif python_symbol is not None:
                        result = _python_symbol(text, python_symbol)
                        query_kind, query_value = "python_symbol", python_symbol
                    else:
                        assert text_window is not None
                        result = _text_window(text, *text_window)
                        query_kind, query_value = "text_window", f"{text_window[0]}:{text_window[1]}"

            requested_bound = max_bytes or policy["max_query_return_bytes"]
            if (
                not isinstance(requested_bound, int)
                or isinstance(requested_bound, bool)
                or requested_bound < 1
                or requested_bound > policy["max_query_return_bytes"]
            ):
                raise ValueError(
                    f"max bytes must be in [1, {policy['max_query_return_bytes']}]"
                )
            result, returned_bytes, truncated = _bounded(result, requested_bound)
            remaining = lease.get("remaining") or {}
            if remaining.get("stage_events", 0) < 1 or remaining.get("returned_bytes", 0) < returned_bytes:
                lease["status"] = "rotation_required"
                _atomic_write(str(paths["lease"]), lease)
                raise ValueError("context lease budget exhausted; rotate before querying again")
            remaining["stage_events"] -= 1
            remaining["returned_bytes"] -= returned_bytes
            if remaining["stage_events"] == 0 or remaining["returned_bytes"] == 0:
                lease["status"] = "rotation_required"
            _atomic_write(str(paths["lease"]), lease)
            receipt = {
                "schema": SCHEMA,
                "timestamp": dt_now(),
                "run_id": context["run_id"],
                "generation": context["generation"],
                "role": role,
                "stage": context["current_stage"],
                "lease_id": lease["lease_id"],
                "context_hash": context_hash,
                "query": {"kind": query_kind, "value": query_value, "artifact": artifact},
                "returned_bytes": returned_bytes,
                "truncated": truncated,
            }
            paths["receipts"].parent.mkdir(parents=True, exist_ok=True)
            with paths["receipts"].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return dict(receipt, result=result)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: str, doc: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".context_query.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def describe() -> dict:
    return {
        "tool": "context_query",
        "schema": SCHEMA,
        "input_schema": CONTEXT_SCHEMA,
        "selectors": list(SELECTORS),
        "query_rule": "exactly one --selector or --ref is required",
        "guarantees": [
            "no default all-context response",
            "--ref matches artifact_refs[].ref exactly",
            "prints a concise receipt unless --out is supplied",
        ],
    }


def _selftest() -> int:
    doc: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA, "context_hash": "ctx-1", "current_stage": "search",
        "identity": {"run_id": "run-a"},
        "measurement_summary": [{"value": 1.1, "unit": "ratio"}],
        "obligations": [{"obligation_id": "profile@g2", "status": "open"}],
        "recall_summary": {"counts": {"stale": 1}},
        "actions": {"allowed": ["measure"], "forbidden": ["raw dump"]},
        "artifact_refs": [{"kind": "canonical", "ref": "canonical.json", "status": "read"}],
    }
    result = query(doc, selector="identity")
    assert result["result"] == {"identity": {"run_id": "run-a"}}
    exact = query(doc, ref="canonical.json")
    assert exact["matched"] == 1 and exact["result"]["artifact_refs"][0]["kind"] == "canonical"
    try:
        query(doc)
        raise AssertionError("unselected full package was permitted")
    except ValueError:
        pass
    print("[context_query] SELFTEST PASS")
    return 0


def main() -> int:
    if sys.argv[1:] in (["--describe", "json"], ["--describe", "--format=json"]):
        print(json.dumps(describe(), ensure_ascii=False, sort_keys=True))
        return 0
    if sys.argv[1:] == ["--selftest"]:
        return _selftest()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", required=True, help="stage_context JSON package")
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--selector", choices=SELECTORS)
    selected.add_argument("--ref", help="exact artifact_refs[].ref value")
    parser.add_argument("--out", help="write the complete selected slice atomically")
    args = parser.parse_args()
    try:
        result = query(_load(args.context), selector=args.selector, ref=args.ref)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[context_query] ERROR: {exc}", file=sys.stderr)
        return 1
    if args.out:
        _atomic_write(args.out, result)
    print(json.dumps({"context_hash": result["context_hash"], "selector": args.selector,
                      "ref": args.ref, "matched": result["matched"], "out": args.out or None},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
