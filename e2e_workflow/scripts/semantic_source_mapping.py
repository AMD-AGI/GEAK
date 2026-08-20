#!/usr/bin/env python3
"""Add source-backed wrapper/launcher candidates to a Shape Capture Plan."""
import argparse
import ast
import json
import os
import re


IGNORED = {
    "kernel", "void", "const", "float", "half", "bfloat", "unsigned",
    "cache", "modifier", "block", "size", "group", "even", "grid",
}


def _source_files(paths):
    files = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            files.append(os.path.abspath(path))
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs[:] = [
                    name for name in dirs
                    if name not in (".git", "__pycache__", "tests")]
                files.extend(
                    os.path.abspath(os.path.join(root, name))
                    for name in names if name.endswith(".py"))
    return sorted(set(files))


def _tokens(symbol):
    value = str(symbol or "")
    candidates = []
    for regex in (
            r"_ZN\d*aiter\d+([A-Za-z_][A-Za-z0-9_]*?)"
            r"(?:_?kernel|ID|I|E)",
            r"aiter::([A-Za-z_][A-Za-z0-9_]*)",
            r"(?:void\s+)?(?:[A-Za-z_][A-Za-z0-9_]*::)+"
            r"([A-Za-z_][A-Za-z0-9_]*)",
            r"^([A-Za-z_][A-Za-z0-9_]*?)(?:_kernel|kernel)(?:_|$)"):
        match = re.search(regex, value)
        if match:
            candidates.append(match.group(1))
    candidates.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", value))
    result = []
    for token in candidates:
        token = re.sub(r"(?:_?kernel)$", "", token).strip("_")
        if (len(token) >= 5 and token.lower() not in IGNORED
                and token not in result):
            result.append(token)
    return result[:12]


def _function_ranges(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inferred_end = max(
                (int(getattr(child, "lineno", node.lineno))
                 for child in ast.walk(node)),
                default=int(node.lineno))
            ranges.append((
                int(node.lineno), int(
                    getattr(node, "end_lineno", None) or inferred_end),
                node.name))
    return ranges


def _enclosing(ranges, line):
    values = [
        item for item in ranges if item[0] <= line <= item[1]]
    return min(values, key=lambda item: item[1] - item[0]) if values else None


def _index(paths):
    index = {}
    for path in _source_files(paths):
        try:
            with open(path, errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        index[path] = {
            "lines": text.splitlines(),
            "functions": _function_ranges(text),
        }
    return index


def map_plan(plan_path, runtime_sources, out_path):
    with open(plan_path) as fh:
        plan = json.load(fh)
    source_index = _index(runtime_sources)
    for target in plan.get("capture_targets", []):
        evidence = []
        matched_token = None
        for token in _tokens(target.get("raw_name")):
            token_re = re.compile(r"\b%s\b" % re.escape(token))
            token_hits = []
            for path, source in source_index.items():
                for number, line in enumerate(source["lines"], 1):
                    if not token_re.search(line):
                        continue
                    enclosing = _enclosing(source["functions"], number)
                    token_hits.append({
                        "token": token,
                        "file": path,
                        "line": number,
                        "enclosing_function": enclosing[2] if enclosing else None,
                        "snippet": line.strip()[:240],
                    })
            if token_hits:
                evidence.extend(token_hits)
                matched_token = token
                break
        target["source_evidence"] = evidence
        if evidence:
            wrappers = sorted({
                "%s:%s" % (item["file"], item["enclosing_function"])
                for item in evidence if item["enclosing_function"]})
            target["candidate_terminal_launcher"] = matched_token
            target["candidate_wrapper"] = (
                wrappers[0] if len(wrappers) == 1 else None)
            target["candidate_op_path"] = None
            target["mapping_cardinality"] = "probe_required"
            target["source_mapping_status"] = (
                "unique_wrapper_candidate" if len(wrappers) == 1
                else "multiple_source_candidates")
        else:
            target["source_mapping_status"] = "not_found"
    plan["runtime_source_roots"] = [
        os.path.abspath(path) for path in runtime_sources]
    plan["source_mapping_summary"] = {
        "target_count": len(plan.get("capture_targets", [])),
        "with_source_candidate": sum(
            1 for target in plan.get("capture_targets", [])
            if target.get("source_evidence")),
        "unique_wrapper_candidates": sum(
            1 for target in plan.get("capture_targets", [])
            if target.get("source_mapping_status") ==
            "unique_wrapper_candidate"),
    }
    with open(out_path, "w") as fh:
        json.dump(plan, fh, indent=2)
    return plan["source_mapping_summary"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = map_plan(
        args.capture_plan, args.runtime_source, args.out)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
