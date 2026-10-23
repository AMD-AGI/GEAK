#!/usr/bin/env python3
"""Generic GPU-free verifier for declarative Expert Skill contracts.

The verifier owns reusable mechanics: source collection, AST-scoped checks,
PlanIR assertions, provenance sealing, and optional reference-copy detection.
Operator-specific facts live in a skill-local ``contract.yaml``.

This tool never imports a GPU runtime and never claims correctness, liveness,
launch count, residency, or performance.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any

import yaml


RESULT_SCHEMA_VERSION = "expert-skill-contract-result-v1"
CONTRACT_SCHEMA_VERSION = "expert-skill-contract-v1"
CHECK_TIERS = ("semantic", "surface")
DEFAULT_CHECK_TIER = "semantic"
NEXT_BLOCKER_DETAIL_LIMIT = 6
NEXT_BLOCKER_DETAIL_CHARS = 240


def load_contract(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".md":
        matches = re.findall(
            r"```expert-skill-contract[^\n]*\n(.*?)\n```",
            text,
            flags=re.S,
        )
        if len(matches) != 1:
            raise ValueError(
                f"skill markdown must contain exactly one embedded contract: {path}"
            )
        text = matches[0]
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"contract must be a mapping: {path}")
    if data.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported contract schema {data.get('schema_version')!r}; "
            f"expected {CONTRACT_SCHEMA_VERSION!r}"
        )
    if not str(data.get("skill_id") or "").strip():
        raise ValueError("contract.skill_id must be non-empty")
    source = data.get("source")
    if not isinstance(source, dict) or not source.get("include"):
        raise ValueError("contract.source.include must be a non-empty list")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("contract.checks must be a non-empty list")
    ids = [str(item.get("id") or "") for item in checks if isinstance(item, dict)]
    if len(ids) != len(checks) or any(not item for item in ids):
        raise ValueError("every contract check must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("contract check ids must be unique")
    bad_tiers = [
        f"{item.get('id')}={item.get('tier')!r}"
        for item in checks
        if "tier" in item and item.get("tier") not in CHECK_TIERS
    ]
    if bad_tiers:
        raise ValueError(
            f"contract check tier must be one of {', '.join(CHECK_TIERS)}: "
            + ", ".join(bad_tiers)
        )
    return data


def collect_files(root: Path, contract: dict[str, Any]) -> dict[str, Path]:
    source = contract.get("source") or {}
    excluded = [str(value) for value in source.get("exclude") or []]
    files: dict[str, Path] = {}
    for pattern in source.get("include") or []:
        for path in sorted(root.glob(str(pattern))):
            if path.is_file():
                name = path.relative_to(root).as_posix()
                if not any(fnmatch.fnmatch(name, pattern) for pattern in excluded):
                    files[name] = path
    return files


def _digest(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None


def tree_digest(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class _RemoveStaticDeadCode(ast.NodeTransformer):
    @staticmethod
    def _static_bool(node: ast.AST) -> bool | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if (
            isinstance(node, ast.Call)
            and _call_name(node) == "const_expr"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, bool)
        ):
            return node.args[0].value
        return None

    def visit_If(self, node: ast.If):  # noqa: N802
        node = self.generic_visit(node)
        selected = self._static_bool(node.test)
        if selected is not None:
            return node.body if selected else node.orelse
        return node

    def visit_While(self, node: ast.While):  # noqa: N802
        node = self.generic_visit(node)
        if self._static_bool(node.test) is False:
            return node.orelse
        return node

    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        return node


def _active_code(node: ast.AST) -> str:
    copied = ast.parse(ast.unparse(node))
    transformed = _RemoveStaticDeadCode().visit(copied)
    return ast.unparse(ast.fix_missing_locations(transformed))


def _parse_files(files: dict[str, Path]) -> tuple[dict[str, ast.AST], dict[str, str]]:
    trees: dict[str, ast.AST] = {}
    errors: dict[str, str] = {}
    for name, path in files.items():
        try:
            trees[name] = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors[name] = str(exc)
    return trees, errors


def _qualified_function_nodes(tree: ast.AST) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                found[name] = node
                visit(node.body, f"{name}.")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(getattr(tree, "body", []))
    return found


def _qualified_functions(tree: ast.AST) -> set[str]:
    return set(_qualified_function_nodes(tree))


def _function_dumps(tree: ast.AST) -> set[str]:
    return {
        ast.dump(node, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _scope_node(tree: ast.AST | None, scope: str | None) -> ast.AST | None:
    if tree is None:
        return None
    if not scope:
        return tree
    nodes = _qualified_function_nodes(tree)
    if scope in nodes:
        return nodes[scope]
    matches = [node for name, node in nodes.items() if re.fullmatch(scope, name)]
    return matches[0] if len(matches) == 1 else None


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ast.unparse(func)


def _normalized_tokens(path: Path) -> tuple[str, ...]:
    keep: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        for token in tokens:
            if token.type in {
                tokenize.ENCODING,
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                continue
            keep.append(token.string)
    except (OSError, UnicodeError, tokenize.TokenError):
        return ()
    return tuple(keep)


def _shingle_similarity(
    left: tuple[str, ...], right: tuple[str, ...], width: int = 7
) -> float:
    def shingles(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
        if len(tokens) < width:
            return {tokens} if tokens else set()
        return {
            tokens[index:index + width]
            for index in range(len(tokens) - width + 1)
        }

    left_set = shingles(left)
    right_set = shingles(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _result(
    passed: bool,
    total: int,
    passed_units: int,
    failures: list[str],
    severity: str,
) -> dict[str, Any]:
    return {
        "pass": bool(passed),
        "severity": severity,
        "units_passed": max(0, int(passed_units)),
        "units_total": max(1, int(total)),
        "ratio": max(0, int(passed_units)) / max(1, int(total)),
        "failures": failures,
    }


def _check_regex(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    name = str(rule.get("file") or "")
    scope = rule.get("scope")
    patterns = [str(value) for value in rule.get("patterns") or []]
    if not patterns:
        return _result(False, 1, 0, ["regex check has no patterns"], severity)
    node = _scope_node(trees.get(name), str(scope) if scope else None)
    if node is None:
        return _result(
            False, len(patterns), 0,
            [f"missing file/scope: {name}:{scope or '<module>'}"], severity,
        )
    text = _active_code(node)
    mode = str(rule.get("match") or "all")
    found = [bool(re.search(pattern, text, re.MULTILINE)) for pattern in patterns]
    if mode == "all":
        failures = [pattern for pattern, hit in zip(patterns, found) if not hit]
        return _result(not failures, len(patterns), sum(found), failures, severity)
    if mode == "any":
        passed = any(found)
        return _result(passed, 1, int(passed), [] if passed else patterns, severity)
    if mode == "none":
        failures = [pattern for pattern, hit in zip(patterns, found) if hit]
        return _result(not failures, len(patterns), len(patterns) - len(failures), failures, severity)
    return _result(False, 1, 0, [f"unsupported regex match mode: {mode}"], severity)


def _check_regex_sequence(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Require related source patterns to occur in declaration order."""
    name = str(rule.get("file") or "")
    scope = rule.get("scope")
    patterns = [str(value) for value in rule.get("patterns") or []]
    if not patterns:
        return _result(False, 1, 0, ["regex_sequence has no patterns"], severity)
    node = _scope_node(trees.get(name), str(scope) if scope else None)
    if node is None:
        return _result(
            False, len(patterns), 0,
            [f"missing file/scope: {name}:{scope or '<module>'}"], severity,
        )
    text = _active_code(node)
    cursor = 0
    passed = 0
    failures: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text[cursor:], re.MULTILINE)
        if match is None:
            failures.append(f"after offset {cursor}: {pattern}")
            break
        cursor += match.end()
        passed += 1
    if failures:
        failures.extend(patterns[passed + 1:])
    return _result(not failures, len(patterns), passed, failures, severity)


def _check_call_keywords(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    name = str(rule.get("file") or "")
    scope = rule.get("scope")
    required = {
        str(key): str(value)
        for key, value in (rule.get("keywords") or {}).items()
    }
    total = max(1, len(required))
    node = _scope_node(trees.get(name), str(scope) if scope else None)
    if node is None:
        return _result(
            False, total, 0,
            [f"missing file/scope: {name}:{scope or '<module>'}"], severity,
        )
    call_pattern = str(rule.get("call") or ".*")
    calls = [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call) and re.fullmatch(call_pattern, _call_name(item))
    ]
    if not calls:
        return _result(False, total, 0, [f"no call matching {call_pattern}"], severity)
    policy = str(rule.get("policy") or "any")

    def failures_for(call: ast.Call) -> list[str]:
        values = {
            str(keyword.arg): ast.unparse(keyword.value)
            for keyword in call.keywords
            if keyword.arg is not None
        }
        return [
            f"{key}={values.get(key, '<missing>')} !~ {pattern}"
            for key, pattern in required.items()
            if key not in values or not re.search(pattern, values[key])
        ]

    failures_by_call = [failures_for(call) for call in calls]
    if policy == "any":
        best = min(failures_by_call, key=len)
        passed = not best
        return _result(
            passed, total, len(required) - len(best),
            [] if passed else best, severity,
        )
    if policy == "every":
        failures = [
            f"call line {getattr(call, 'lineno', 0)}: {failure}"
            for call, call_failures in zip(calls, failures_by_call)
            for failure in call_failures
        ]
        failed_keywords = {
            key for key in required
            if any(
                failure.startswith(f"{key}=")
                for call_failures in failures_by_call
                for failure in call_failures
            )
        }
        return _result(
            not failures, total, len(required) - len(failed_keywords), failures, severity
        )
    return _result(False, 1, 0, [f"unsupported call policy: {policy}"], severity)


def _check_selected_branch_calls(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Check calls made directly by one selected control-flow branch.

    Calls nested under another branch or loop are deliberately excluded. This
    lets a contract distinguish an active fused arm from diagnostic fallbacks
    retained below it without treating mere token presence as reachability.
    """
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    required = [str(value) for value in rule.get("required_calls") or []]
    forbidden = [str(value) for value in rule.get("forbidden_calls") or []]
    require_return = bool(rule.get("require_return", False))
    total = max(1, len(required) + len(forbidden) + int(require_return))
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, total, 0, [f"missing function scope: {name}:{scope}"], severity
        )
    condition = str(rule.get("condition") or "")
    arm = str(rule.get("arm") or "body")
    if not condition or arm not in {"body", "orelse"}:
        return _result(
            False, total, 0,
            ["selected_branch_calls requires condition and arm=body|orelse"],
            severity,
        )
    branches = [
        item for item in ast.walk(node)
        if isinstance(item, ast.If) and re.search(condition, ast.unparse(item.test))
    ]
    if len(branches) != 1:
        return _result(
            False, total, 0,
            [f"expected one branch matching {condition!r}, found {len(branches)}"],
            severity,
        )
    statements = getattr(branches[0], arm)

    class DirectVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[ast.Call] = []

        def visit_Call(self, item: ast.Call) -> None:
            self.calls.append(item)
            self.generic_visit(item)

        def visit_If(self, item: ast.If) -> None:
            return

        def visit_For(self, item: ast.For) -> None:
            return

        visit_AsyncFor = visit_For

        def visit_While(self, item: ast.While) -> None:
            return

        def visit_Try(self, item: ast.Try) -> None:
            return

        def visit_With(self, item: ast.With) -> None:
            return

        visit_AsyncWith = visit_With

        def visit_Match(self, item: ast.Match) -> None:
            return

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

    visitor = DirectVisitor()
    for statement in statements:
        visitor.visit(statement)
    call_names = [_call_name(call) for call in visitor.calls]
    failures = [
        f"selected branch has no direct call matching {pattern}"
        for pattern in required
        if not any(re.fullmatch(pattern, called) for called in call_names)
    ]
    failures.extend(
        f"selected branch directly calls forbidden {called}"
        for called in call_names
        if any(re.fullmatch(pattern, called) for pattern in forbidden)
    )
    direct_return = any(isinstance(statement, ast.Return) for statement in statements)
    if require_return and not direct_return:
        failures.append("selected branch has no direct return")
    failed_required = sum(
        not any(re.fullmatch(pattern, called) for called in call_names)
        for pattern in required
    )
    failed_forbidden = sum(
        any(re.fullmatch(pattern, called) for called in call_names)
        for pattern in forbidden
    )
    passed = total - failed_required - failed_forbidden - int(
        require_return and not direct_return
    )
    return _result(not failures, total, passed, failures, severity)


def _check_function_shape(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Require a callable ABI of sufficient width whose parameters are consumed."""
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, 2, 0, [f"missing function scope: {name}:{scope}"], severity
        )
    parameters = [
        argument.arg
        for argument in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
    ]
    if node.args.vararg:
        parameters.append(node.args.vararg.arg)
    if node.args.kwarg:
        parameters.append(node.args.kwarg.arg)
    ignored = {str(value) for value in rule.get("ignore_parameters") or ["self"]}
    considered = [parameter for parameter in parameters if parameter not in ignored]
    loaded = {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }
    minimum_parameters = int(rule.get("min_parameters") or 0)
    minimum_loaded = int(rule.get("min_loaded_parameters") or 0)
    failures = []
    if len(considered) < minimum_parameters:
        failures.append(
            f"function has {len(considered)} parameters, requires {minimum_parameters}"
        )
    loaded_count = len(set(considered) & loaded)
    if loaded_count < minimum_loaded:
        failures.append(
            f"function loads {loaded_count} parameters, requires {minimum_loaded}"
        )
    return _result(
        not failures, 2,
        int(len(considered) >= minimum_parameters) + int(loaded_count >= minimum_loaded),
        failures, severity,
    )


def _check_tuple_bindings(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Require host argument bundles to contain distinct, non-placeholder sources."""
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    bindings = [item for item in rule.get("bindings") or [] if isinstance(item, dict)]
    total = max(1, len(bindings) * 3)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, total, 0, [f"missing function scope: {name}:{scope}"], severity
        )
    tuples: dict[str, list[str]] = {}
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or not isinstance(item.value, (ast.Tuple, ast.List)):
            continue
        values = [ast.unparse(value) for value in item.value.elts]
        for target in item.targets:
            tuples[ast.unparse(target)] = values
    failures = []
    passed = 0
    for binding in bindings:
        target = str(binding.get("target") or "")
        values = tuples.get(target)
        minimum = int(binding.get("min_elements") or 1)
        minimum_distinct = int(binding.get("min_distinct") or minimum)
        forbidden = [str(value) for value in binding.get("forbidden") or []]
        if values is None or len(values) < minimum:
            failures.append(
                f"{target} has {0 if values is None else len(values)} elements, requires {minimum}"
            )
        else:
            passed += 1
        if values is None or len(set(values)) < minimum_distinct:
            failures.append(
                f"{target} has {0 if values is None else len(set(values))} distinct sources, "
                f"requires {minimum_distinct}"
            )
        else:
            passed += 1
        placeholders = [
            value for value in (values or [])
            if any(re.fullmatch(pattern, value) for pattern in forbidden)
        ]
        if placeholders:
            failures.append(f"{target} contains placeholder sources: {placeholders}")
        else:
            passed += 1
    return _result(not failures, total, passed, failures, severity)


def _check_bundle_flow(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Prove producer outputs populate a bundle expanded into the launch sink."""
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, 3, 0, [f"missing function scope: {name}:{scope}"], severity
        )
    producer = str(rule.get("producer_call") or "")
    outputs = [str(value) for value in rule.get("outputs") or []]
    bundle = str(rule.get("bundle_target") or "")
    sink = str(rule.get("sink_call") or "")
    keys = {str(key): str(value) for key, value in (rule.get("keys") or {}).items()}
    failures = []
    passed = 0

    producer_assignments = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or not isinstance(item.value, ast.Call):
            continue
        if not re.fullmatch(producer, _call_name(item.value)) or len(item.targets) != 1:
            continue
        target = item.targets[0]
        if isinstance(target, (ast.Tuple, ast.List)):
            producer_assignments.append([ast.unparse(value) for value in target.elts])
    if outputs and outputs in producer_assignments:
        passed += 1
    else:
        failures.append(f"producer {producer} does not unpack exactly into {outputs}")

    bundle_values = None
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or len(item.targets) != 1:
            continue
        if ast.unparse(item.targets[0]) != bundle or not isinstance(item.value, ast.Dict):
            continue
        bundle_values = {
            str(key.value): ast.unparse(value)
            for key, value in zip(item.value.keys, item.value.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    bad_keys = [
        f"{key}={('<missing>' if bundle_values is None else bundle_values.get(key, '<missing>'))}"
        for key, value in keys.items()
        if bundle_values is None or bundle_values.get(key) != value
    ]
    if not bad_keys:
        passed += 1
    else:
        failures.append(f"bundle {bundle} has wrong producer bindings: {bad_keys}")

    sink_expands = any(
        isinstance(item, ast.Call)
        and re.fullmatch(sink, _call_name(item))
        and any(
            keyword.arg is None and ast.unparse(keyword.value) == bundle
            for keyword in item.keywords
        )
        for item in ast.walk(node)
    )
    if sink_expands:
        passed += 1
    else:
        failures.append(f"sink {sink} does not expand **{bundle}")
    return _result(not failures, 3, passed, failures, severity)


def _check_call_arity(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Require a selected call to forward a minimum ABI, including expansions."""
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    if node is None:
        return _result(
            False, 2, 0, [f"missing file/scope: {name}:{scope}"], severity
        )
    call_pattern = str(rule.get("call") or "")
    calls = [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call) and re.fullmatch(call_pattern, _call_name(item))
    ]
    if len(calls) != 1:
        return _result(
            False, 2, 0,
            [f"expected one call matching {call_pattern!r}, found {len(calls)}"],
            severity,
        )
    call = calls[0]
    minimum_arguments = int(rule.get("min_arguments") or 0)
    minimum_starred = int(rule.get("min_starred") or 0)
    argument_count = len(call.args) + len(call.keywords)
    starred_count = sum(isinstance(argument, ast.Starred) for argument in call.args)
    starred_count += sum(keyword.arg is None for keyword in call.keywords)
    failures = []
    if argument_count < minimum_arguments:
        failures.append(
            f"call forwards {argument_count} argument nodes, requires {minimum_arguments}"
        )
    if starred_count < minimum_starred:
        failures.append(
            f"call has {starred_count} argument expansion(s), requires {minimum_starred}"
        )
    return _result(
        not failures, 2,
        int(argument_count >= minimum_arguments) + int(starred_count >= minimum_starred),
        failures, severity,
    )


def _check_nontrivial_function(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Reject identity adapters and require the declared numerical callees."""
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    required_calls = [str(value) for value in rule.get("required_calls") or []]
    total = max(1, len(required_calls) + 2)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, total, 0, [f"missing function scope: {name}:{scope}"], severity
        )
    minimum_statements = int(rule.get("min_statements") or 2)
    statements = [
        item for item in ast.walk(node)
        if isinstance(item, ast.stmt)
        and not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    calls = [
        _call_name(item) for item in ast.walk(node) if isinstance(item, ast.Call)
    ]
    parameters = {
        argument.arg
        for argument in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
    }

    def is_identity_return(item: ast.Return) -> bool:
        value = item.value
        if value is None or isinstance(value, ast.Constant):
            return True
        values = value.elts if isinstance(value, (ast.Tuple, ast.List)) else [value]
        return bool(values) and all(
            isinstance(element, ast.Name) and element.id in parameters
            for element in values
        )

    returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
    non_identity = (
        (not returns and bool(rule.get("allow_no_return", False)))
        or any(not is_identity_return(item) for item in returns)
    )
    failures = []
    if len(statements) < minimum_statements:
        failures.append(
            f"function has {len(statements)} executable statements, requires {minimum_statements}"
        )
    if not non_identity:
        failures.append("function only returns None/constants/input identities")
    for pattern in required_calls:
        if not any(re.fullmatch(pattern, called) for called in calls):
            failures.append(f"function has no call matching {pattern}")
    passed = (
        int(len(statements) >= minimum_statements)
        + int(non_identity)
        + sum(
            any(re.fullmatch(pattern, called) for called in calls)
            for pattern in required_calls
        )
    )
    return _result(not failures, total, passed, failures, severity)


def _check_forbid_methods(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    name = str(rule.get("file") or "")
    scope = rule.get("scope")
    node = _scope_node(trees.get(name), str(scope) if scope else None)
    if node is None:
        return _result(False, 1, 0, [f"missing file/scope: {name}:{scope or '<module>'}"], severity)
    methods = {str(value) for value in rule.get("methods") or []}
    receivers = [str(value) for value in rule.get("receivers") or [".*"]]
    failures = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr not in methods:
            continue
        receiver = ast.unparse(item.func.value)
        if any(re.search(pattern, receiver) for pattern in receivers):
            failures.append(
                f"line {getattr(item, 'lineno', 0)}: {receiver}.{item.func.attr}()"
            )
    return _result(not failures, 1, int(not failures), failures, severity)


def _check_assignment_value(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Relate an assignment target to its value without operator-specific code.

    Plain regex checks can independently find a target name and an unrelated
    expression elsewhere in a function. This check keeps the relation local to
    one AST assignment so declarative contracts can reject dead or miswired
    shape/capacity variables.
    """
    name = str(rule.get("file") or "")
    scope = rule.get("scope")
    node = _scope_node(trees.get(name), str(scope) if scope else None)
    if node is None:
        return _result(
            False, 1, 0,
            [f"missing file/scope: {name}:{scope or '<module>'}"], severity,
        )
    target_pattern = str(rule.get("target") or "")
    value_pattern = str(rule.get("value") or "")
    if not target_pattern or not value_pattern:
        return _result(
            False, 1, 0,
            ["assignment_value requires target and value patterns"], severity,
        )

    assignments: list[tuple[str, str, int]] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Assign):
            for target in item.targets:
                assignments.append(
                    (ast.unparse(target), ast.unparse(item.value), getattr(item, "lineno", 0))
                )
        elif isinstance(item, ast.AnnAssign) and item.value is not None:
            assignments.append(
                (ast.unparse(item.target), ast.unparse(item.value), getattr(item, "lineno", 0))
            )
        elif isinstance(item, ast.NamedExpr):
            assignments.append(
                (ast.unparse(item.target), ast.unparse(item.value), getattr(item, "lineno", 0))
            )

    targeted = [
        (target, value, line)
        for target, value, line in assignments
        if re.fullmatch(target_pattern, target)
    ]
    matching = [
        (target, value, line)
        for target, value, line in targeted
        if re.search(value_pattern, value)
    ]
    mode = str(rule.get("match") or "any")
    if mode == "any":
        passed = bool(matching)
        failures = [] if passed else [
            f"no assignment {target_pattern!r} has value matching {value_pattern!r}"
        ]
    elif mode == "every":
        passed = bool(targeted) and len(matching) == len(targeted)
        failures = [] if passed else [
            f"line {line}: {target}={value} !~ {value_pattern}"
            for target, value, line in targeted
            if not re.search(value_pattern, value)
        ]
        if not targeted:
            failures.append(f"no assignment target matching {target_pattern!r}")
    elif mode == "none":
        passed = not matching
        failures = [
            f"line {line}: forbidden {target}={value} matches {value_pattern}"
            for target, value, line in matching
        ]
    else:
        return _result(
            False, 1, 0, [f"unsupported assignment_value mode: {mode}"], severity
        )
    return _result(passed, 1, int(passed), failures, severity)


def _check_parameter_loads(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    groups = rule.get("parameters") or []
    node = _scope_node(trees.get(name), scope)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, max(1, len(groups)), 0,
            [f"missing function scope: {name}:{scope}"], severity,
        )
    minimum = max(1, int(rule.get("min_loads") or 1))
    load_counts: dict[str, int] = {}
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load):
            load_counts[item.id] = load_counts.get(item.id, 0) + 1
    failures = []
    for group in groups:
        alternatives = [str(group)] if isinstance(group, str) else [str(value) for value in group]
        if not any(load_counts.get(parameter, 0) >= minimum for parameter in alternatives):
            failures.append(
                f"none of {alternatives} has at least {minimum} runtime load(s)"
            )
    return _result(
        not failures, max(1, len(groups)), max(0, len(groups) - len(failures)),
        failures, severity,
    )


def _check_publication_placement(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Reject completion RMWs anywhere in a selected hot loop.

    Local helper calls are followed transitively so renaming or wrapping the
    publication operation does not evade the placement relation. An owner-only
    store may remain in the loop; an RMW is legal only after the hot loop.
    """
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    node = _scope_node(trees.get(name), scope)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, 3, 0,
            [f"missing function scope: {name}:{scope}"], severity,
        )
    active = ast.parse(_active_code(node))
    node = active.body[0] if active.body else node
    loop_test = str(rule.get("loop_test") or "")
    loop_contains = str(rule.get("loop_contains") or "")
    compute_call = str(rule.get("compute_call") or "")
    address = str(rule.get("address") or "")
    rmw_calls = [str(value) for value in rule.get("rmw_calls") or []]
    store_calls = [str(value) for value in rule.get("store_calls") or []]
    if not loop_test or not compute_call or not address or not rmw_calls:
        return _result(
            False, 3, 0,
            ["publication_placement requires loop_test, compute_call, address and rmw_calls"],
            severity,
        )

    helper_nodes = {
        item.name: item for item in ast.walk(node)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item is not node
    }

    def direct_kind(call: ast.Call) -> str:
        text = ast.unparse(call)
        if not re.search(address, text):
            return ""
        called = _call_name(call)
        if any(re.fullmatch(pattern, called) for pattern in rmw_calls):
            return "rmw"
        if any(re.fullmatch(pattern, called) for pattern in store_calls):
            return "store"
        return ""

    effects: dict[str, set[str]] = {}
    for helper_name, helper in helper_nodes.items():
        effects[helper_name] = {
            kind for item in ast.walk(helper)
            if isinstance(item, ast.Call) and (kind := direct_kind(item))
        }
    changed = True
    while changed:
        changed = False
        for helper_name, helper in helper_nodes.items():
            inherited = set().union(*(
                effects.get(_call_name(item), set())
                for item in ast.walk(helper) if isinstance(item, ast.Call)
            ))
            if not inherited.issubset(effects[helper_name]):
                effects[helper_name].update(inherited)
                changed = True

    class ScopedCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loops: list[ast.While] = []
            self.calls: list[ast.Call] = []

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            if item is node:
                self.generic_visit(item)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_While(self, item: ast.While) -> None:
            if re.search(loop_test, ast.unparse(item.test)):
                self.loops.append(item)
            self.generic_visit(item)

        def visit_Call(self, item: ast.Call) -> None:
            self.calls.append(item)
            self.generic_visit(item)

    collector = ScopedCollector()
    collector.visit(node)
    hot_loops = [
        loop for loop in collector.loops
        if not loop_contains or re.search(loop_contains, ast.unparse(loop))
    ]
    failures: list[str] = []
    if len(hot_loops) != 1:
        failures.append(f"expected one hot loop matching {loop_test}, found {len(hot_loops)}")
        return _result(False, 3, 0, failures, severity)
    hot_loop = hot_loops[0]
    loop_start = getattr(hot_loop, "lineno", 0)
    loop_end = getattr(hot_loop, "end_lineno", loop_start)
    loop_calls = [
        call for call in collector.calls
        if loop_start <= getattr(call, "lineno", 0) <= loop_end
    ]
    compute_lines = [
        getattr(call, "lineno", 0) for call in loop_calls
        if re.fullmatch(compute_call, _call_name(call))
    ]
    if not compute_lines:
        failures.append(f"hot loop has no compute call matching {compute_call}")
    first_compute = min(compute_lines) if compute_lines else loop_end + 1

    def invocation_kinds(call: ast.Call) -> set[str]:
        direct = direct_kind(call)
        return ({direct} if direct else set()) | effects.get(_call_name(call), set())

    loop_publications = [
        (getattr(call, "lineno", 0), invocation_kinds(call))
        for call in loop_calls if invocation_kinds(call)
    ]
    post_publications = [
        (getattr(call, "lineno", 0), invocation_kinds(call))
        for call in collector.calls
        if getattr(call, "lineno", 0) > loop_end and invocation_kinds(call)
    ]
    mechanism_present = any(effects.values()) or any(
        direct_kind(call) for call in collector.calls
    )
    unsafe_lines = [
        line for line, kinds in loop_publications if "rmw" in kinds
    ]
    legal_publication = bool(post_publications) or any(
        "store" in kinds
        for line, kinds in loop_publications
    )
    if not mechanism_present:
        failures.append(f"no completion publication tied to address {address}")
    if unsafe_lines:
        failures.append(f"completion RMW occurs inside hot loop at lines {unsafe_lines}")
    if not legal_publication:
        failures.append("no legal owner-store or post-loop publication")
    passed_units = int(bool(compute_lines)) + int(mechanism_present) + int(
        legal_publication and not unsafe_lines
    )
    return _result(not failures, 3, passed_units, failures, severity)


def _check_owned_completion_protocol(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    """Prove an owner-only completion store and reject completion-address RMWs.

    The rule is intentionally declarative: operator names, helper names,
    expressions and primitive names all come from the contract. Local callees
    are followed transitively for the negative RMW proof.
    """
    name = str(rule.get("file") or "")
    scope = str(rule.get("scope") or "")
    root = _scope_node(trees.get(name), scope)
    if not isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _result(
            False, 9, 0, [f"missing function scope: {name}:{scope}"], severity
        )
    active = ast.parse(_active_code(root))
    root = active.body[0] if active.body else root
    wrappers = {str(value) for value in rule.get("normalize_wrappers") or []}

    class StripWrappers(ast.NodeTransformer):
        def visit_Call(self, item: ast.Call):  # noqa: N802
            item = self.generic_visit(item)
            if (
                _call_name(item) in wrappers
                and len(item.args) == 1
                and not item.keywords
            ):
                return item.args[0]
            return item

    def expr(item: ast.AST) -> str:
        copied = ast.parse(ast.unparse(item), mode="eval").body
        normalized = StripWrappers().visit(copied)
        return ast.unparse(ast.fix_missing_locations(normalized))

    def matches(item: ast.AST, pattern: Any) -> bool:
        return bool(re.fullmatch(str(pattern or ""), expr(item)))

    def call_matches(call: ast.Call, pattern: Any) -> bool:
        value = str(pattern or "")
        return bool(
            re.fullmatch(value, _call_name(call))
            or re.fullmatch(value, ast.unparse(call.func))
        )

    def function_parameters(
        item: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        return [
            argument.arg
            for argument in (
                list(item.args.posonlyargs)
                + list(item.args.args)
                + list(item.args.kwonlyargs)
            )
        ]

    class Collector(ast.NodeVisitor):
        def __init__(self, owner: ast.AST, skip_nested: bool = True) -> None:
            self.owner = owner
            self.skip_nested = skip_nested
            self.nodes: list[ast.AST] = []

        def generic_visit(self, item: ast.AST) -> None:
            self.nodes.append(item)
            super().generic_visit(item)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            self.nodes.append(item)
            if item is self.owner or not self.skip_nested:
                super().generic_visit(item)

        visit_AsyncFunctionDef = visit_FunctionDef

    def scoped_nodes(owner: ast.AST, skip_nested: bool = True) -> list[ast.AST]:
        collector = Collector(owner, skip_nested)
        collector.visit(owner)
        return collector.nodes

    all_nodes = scoped_nodes(root, skip_nested=False)
    helpers = {
        item.name: item
        for item in all_nodes
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item is not root
    }
    failures: list[str] = []
    passed = 0

    region_cfg = rule.get("region") or {}
    region_guard = str(region_cfg.get("guard") or "")
    regions = [
        item for item in scoped_nodes(root)
        if isinstance(item, ast.If) and matches(item.test, region_guard)
    ]
    if len(regions) != 1:
        failures.append(
            f"expected one selected region matching {region_guard!r}, found {len(regions)}"
        )
        return _result(False, 9, passed, failures, severity)
    region = regions[0]
    arm = str(region_cfg.get("arm") or "body")
    if arm not in {"body", "orelse"}:
        return _result(
            False, 9, passed, [f"unsupported selected region arm: {arm}"], severity
        )
    region_scope = ast.Module(
        body=list(getattr(region, arm)), type_ignores=[]
    )
    loop_cfg = region_cfg.get("loop") or {}
    loop_test = str(loop_cfg.get("test") or "")
    loop_contains = str(loop_cfg.get("contains") or "")
    loops = [
        item for item in scoped_nodes(region_scope)
        if isinstance(item, ast.While)
        and matches(item.test, loop_test)
        and (
            not loop_contains
            or re.search(loop_contains, ast.unparse(item))
        )
    ]
    if len(loops) != 1:
        failures.append(
            f"expected one selected loop matching {loop_test!r}, found {len(loops)}"
        )
        return _result(False, 9, passed, failures, severity)
    loop = loops[0]
    passed += 1

    claim = rule.get("claim") or {}
    handoff = rule.get("handoff") or {}
    region_nodes = scoped_nodes(region_scope)

    def assignments_to(nodes: list[ast.AST], pattern: Any) -> list[ast.AST]:
        return [
            item for item in nodes
            if isinstance(item, (ast.Assign, ast.AnnAssign))
            and any(
                matches(target, pattern)
                for target in (
                    item.targets if isinstance(item, ast.Assign) else [item.target]
                )
            )
        ]

    extent_definitions = assignments_to(
        region_nodes, claim.get("extent_target")
    )
    extent_assignments = [
        item for item in extent_definitions
        if matches(item.value, claim.get("extent_value"))
    ]
    root_nodes = scoped_nodes(root)
    source_extent_definitions = assignments_to(
        root_nodes, claim.get("source_extent_target")
    )
    source_extent_assignments = [
        item for item in source_extent_definitions
        if matches(item.value, claim.get("source_extent_value"))
    ]
    if (
        len(extent_assignments) == len(source_extent_assignments) == 1
        and len(extent_definitions) == len(source_extent_definitions) == 1
    ):
        passed += 1
    else:
        failures.append("source and fused m-tile extent relation is missing or ambiguous")

    claim_helper_name = str(claim.get("helper") or "")
    claim_helper = helpers.get(claim_helper_name)
    claim_helper_defs = [
        item for item in all_nodes
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == claim_helper_name
    ]
    claim_rebindings = assignments_to(region_nodes, claim_helper_name)
    claim_ok = (
        isinstance(claim_helper, (ast.FunctionDef, ast.AsyncFunctionDef))
        and len(claim_helper_defs) == 1 and not claim_rebindings
    )
    claim_nodes = scoped_nodes(claim_helper) if claim_ok else []

    class GuardedCalls(ast.NodeVisitor):
        def __init__(self) -> None:
            self.guards: list[ast.AST] = []
            self.calls: list[tuple[ast.Call, tuple[ast.AST, ...]]] = []

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_If(self, item: ast.If) -> None:
            self.guards.append(item.test)
            for statement in item.body:
                self.visit(statement)
            self.guards.pop()
            for statement in item.orelse:
                self.visit(statement)

        def visit_Call(self, item: ast.Call) -> None:
            self.calls.append((item, tuple(self.guards)))
            self.generic_visit(item)

    guarded_calls = GuardedCalls()
    for statement in loop.body:
        guarded_calls.visit(statement)
    claim_invocations = [
        (call, guards)
        for call, guards in guarded_calls.calls
        if call_matches(call, claim_helper_name)
    ]
    claim_invoked_by_owner = bool(claim_invocations) and all(
        any(matches(guard, claim.get("owner_test")) for guard in guards)
        for _, guards in claim_invocations
    )

    def writes_handoff(call: ast.Call) -> bool:
        return (
            call_matches(call, handoff.get("store_call"))
            and len(call.args) >= 2
            and matches(call.args[1], handoff.get("scratch"))
        )

    def path_write_states(
        statements: list[ast.stmt],
        incoming: list[tuple[bool, bool]] | None = None,
        helper_is_write: bool = False,
    ) -> list[tuple[bool, bool]]:
        states = list(incoming or [(False, True)])
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            next_states: list[tuple[bool, bool]] = []
            for written, active_path in states:
                if not active_path:
                    next_states.append((written, False))
                elif isinstance(statement, ast.If):
                    next_states.extend(path_write_states(
                        statement.body, [(written, True)], helper_is_write
                    ))
                    next_states.extend(
                        path_write_states(
                            statement.orelse,
                            [(written, True)],
                            helper_is_write,
                        )
                        if statement.orelse else [(written, True)]
                    )
                elif isinstance(
                    statement, (ast.Break, ast.Continue, ast.Return, ast.Raise)
                ):
                    next_states.append((written, False))
                else:
                    call = (
                        statement.value
                        if isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        else None
                    )
                    emits = bool(call) and (
                        writes_handoff(call)
                        or (
                            helper_is_write
                            and call_matches(call, claim_helper_name)
                        )
                    )
                    next_states.append((written or emits, True))
            states = next_states
        return states

    claim_helper_writes_all_paths = (
        claim_ok
        and all(written for written, _ in path_write_states(claim_helper.body))
    )
    owner_ifs = [
        statement for statement in loop.body
        if isinstance(statement, ast.If)
        and matches(statement.test, claim.get("owner_test"))
    ]
    owner_index = loop.body.index(owner_ifs[0]) if len(owner_ifs) == 1 else -1
    owner_reachable = owner_index >= 0 and not any(
        isinstance(item, (ast.Break, ast.Continue, ast.Return, ast.Raise))
        for statement in loop.body[:owner_index]
        for item in ast.walk(statement)
    )
    owner_writes_all_paths = (
        len(owner_ifs) == 1
        and owner_reachable
        and claim_helper_writes_all_paths
        and all(
            written for written, _ in path_write_states(
                owner_ifs[0].body, helper_is_write=True
            )
        )
    )
    local_definitions = assignments_to(claim_nodes, claim.get("local_target"))
    local_assignments = [
        item for item in local_definitions
        if matches(item.value, claim.get("local_value") or ".*")
    ]
    local_ok = False
    if len(local_assignments) == 1:
        atomic_calls = [
            item for item in ast.walk(local_assignments[0].value)
            if isinstance(item, ast.Call)
            and call_matches(item, claim.get("atomic_call"))
        ]
        local_ok = len(atomic_calls) == 1 and len(atomic_calls[0].args) >= 2
        if local_ok:
            local_ok = (
                matches(atomic_calls[0].args[0], claim.get("head_address"))
                and matches(atomic_calls[0].args[1], claim.get("increment"))
            )
    index_definitions = assignments_to(claim_nodes, claim.get("index_target"))
    index_assignments = [
        item for item in index_definitions
        if matches(item.value, claim.get("index_value"))
    ]
    bounds = [
        item for item in claim_nodes
        if isinstance(item, ast.If) and matches(item.test, claim.get("bound"))
    ]
    if (
        claim_ok and claim_invoked_by_owner and local_ok
        and owner_writes_all_paths
        and len(local_definitions) == len(local_assignments) == 1
        and len(index_definitions) == len(index_assignments) == 1
        and len(bounds) == 1
    ):
        passed += 1
    else:
        failures.append(
            "m-tile claim relation failed: "
            f"helper={claim_ok}, owner_call={claim_invoked_by_owner}, "
            f"all_paths={owner_writes_all_paths}, atomic={local_ok}, "
            f"index={len(index_assignments)}, bound={len(bounds)}"
        )

    handoff_ok = False
    if len(bounds) == 1:
        stores = [
            statement.value for statement in bounds[0].body
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and call_matches(statement.value, handoff.get("store_call"))
            and len(statement.value.args) >= 2
            and matches(statement.value.args[1], handoff.get("scratch"))
        ]
        expected = handoff.get("values") or []
        handoff_ok = len(stores) == 1 and any(
            isinstance(call, ast.Call)
            and call_matches(call, handoff.get("vector_call"))
            and call.args
            and isinstance(call.args[0], (ast.List, ast.Tuple))
            and len(call.args[0].elts) == len(expected)
            and all(
                matches(value, pattern)
                for value, pattern in zip(call.args[0].elts, expected)
            )
            for call in ast.walk(stores[0].args[0])
        )
    view_definitions = assignments_to(region_nodes, handoff.get("view_target"))
    view_assignments = [
        item for item in view_definitions
        if matches(item.value, handoff.get("view_value"))
    ]
    load_definitions = assignments_to(
        scoped_nodes(loop), handoff.get("load_target")
    )
    load_assignments = [
        item for item in loop.body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and matches(item.targets[0], handoff.get("load_target"))
        and matches(item.value, handoff.get("load_value"))
    ]
    unit_definitions = assignments_to(
        scoped_nodes(loop), handoff.get("unit_target")
    )
    unit_assignments = [
        item for item in loop.body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and matches(item.targets[0], handoff.get("unit_target"))
        and matches(item.value, handoff.get("unit_value"))
    ]
    claim_statement_indexes = [
        index for index, statement in enumerate(loop.body)
        if any(
            isinstance(item, ast.Call) and call_matches(item, claim_helper_name)
            for item in ast.walk(statement)
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    ]
    load_indexes = [
        index for index, statement in enumerate(loop.body)
        if statement in load_assignments
    ]
    barrier_between = (
        len(claim_statement_indexes) == 1 and len(load_indexes) == 1
        and claim_statement_indexes[0] < load_indexes[0]
        and any(
            isinstance(loop.body[index], ast.Expr)
            and isinstance(loop.body[index].value, ast.Call)
            and call_matches(loop.body[index].value, handoff.get("barrier_call"))
            for index in range(claim_statement_indexes[0] + 1, load_indexes[0])
        )
    )
    if (
        handoff_ok and len(view_definitions) == len(view_assignments) == 1
        and len(load_definitions) == len(load_assignments) == 1
        and len(unit_definitions) == len(unit_assignments) == 1
        and barrier_between
    ):
        passed += 1
    else:
        failures.append(
            "claimed m-tile handoff failed: "
            f"store={handoff_ok}, view={len(view_assignments)}, "
            f"load={len(load_assignments)}, unit={len(unit_assignments)}, "
            f"barrier={barrier_between}"
        )

    stripes = rule.get("stripes") or {}
    mode_ifs = [
        item for item in scoped_nodes(loop)
        if isinstance(item, ast.If) and matches(item.test, stripes.get("mode_test"))
    ]
    stripe_loop: ast.For | None = None
    compute_ok = False
    publish_ok = False
    if len(mode_ifs) == 1:
        for_candidates = [
            item for item in mode_ifs[0].body
            if isinstance(item, ast.For)
            and matches(item.target, stripes.get("iterator"))
            and isinstance(item.iter, ast.Call)
            and call_matches(item.iter, stripes.get("range_call"))
            and len(item.iter.args) == 1
            and matches(item.iter.args[0], stripes.get("count"))
        ]
        if len(for_candidates) == 1:
            stripe_loop = for_candidates[0]
            compute_calls = [
                statement.value for statement in stripe_loop.body
                if isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and call_matches(statement.value, stripes.get("compute_call"))
            ]
            compute_ok = (
                not stripe_loop.orelse
                and len(stripe_loop.body) == 1
                and len(compute_calls) == 1
                and len(compute_calls[0].args) == 1
                and matches(compute_calls[0].args[0], stripes.get("compute_index"))
            )
            publication = rule.get("publication") or {}
            later = mode_ifs[0].body[mode_ifs[0].body.index(stripe_loop) + 1:]
            publish_calls = [
                statement.value for statement in later
                if isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and call_matches(statement.value, publication.get("helper"))
            ]
            all_publish_calls = [
                item for item in scoped_nodes(region_scope)
                if isinstance(item, ast.Call)
                and call_matches(item, publication.get("helper"))
            ]
            publish_ok = (
                len(later) == 1
                and len(publish_calls) == 1
                and len(all_publish_calls) == 1
                and len(publish_calls[0].args) == 1
                and matches(publish_calls[0].args[0], publication.get("argument"))
                and not mode_ifs[0].orelse
            )
    if compute_ok:
        passed += 1
    else:
        failures.append("one owner claim does not compute every configured N stripe")
    if publish_ok:
        passed += 1
    else:
        failures.append("completion publication does not occur once after all stripe compute")

    publication = rule.get("publication") or {}
    publish_helper_name = str(publication.get("helper") or "")
    publish_helper = helpers.get(publish_helper_name)
    publish_helper_defs = [
        item for item in all_nodes
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == publish_helper_name
    ]
    publish_rebindings = assignments_to(region_nodes, publish_helper_name)
    protocol_ok = False
    if (
        isinstance(publish_helper, (ast.FunctionDef, ast.AsyncFunctionDef))
        and len(publish_helper_defs) == 1 and not publish_rebindings
    ):
        statements = publish_helper.body
        helper_params = function_parameters(publish_helper)
        wait_indexes = [
            index for index, statement in enumerate(statements)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and call_matches(statement.value, publication.get("wait_call"))
            and len(statement.value.args) == 1
            and matches(statement.value.args[0], publication.get("wait_argument"))
        ]
        barrier_indexes = [
            index for index, statement in enumerate(statements)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and call_matches(statement.value, publication.get("barrier_call"))
        ]
        owner_ifs = [
            (index, statement) for index, statement in enumerate(statements)
            if isinstance(statement, ast.If)
            and matches(statement.test, publication.get("owner_test"))
        ]
        if len(wait_indexes) == len(barrier_indexes) == len(owner_ifs) == 1:
            owner_index, owner_if = owner_ifs[0]
            stores = [
                statement.value for statement in owner_if.body
                if isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and call_matches(statement.value, publication.get("store_call"))
            ]
            all_stores = [
                item for item in ast.walk(publish_helper)
                if isinstance(item, ast.Call)
                and call_matches(item, publication.get("store_call"))
            ]
            if (
                len(stores) == len(all_stores) == 1
                and len(owner_if.body) == 1 and not owner_if.orelse
                and len(stores[0].args) >= 3
            ):
                address = stores[0].args[0]
                address_ok = (
                    isinstance(address, ast.Call)
                    and call_matches(address, publication.get("address_call"))
                    and len(address.args) == 1
                    and matches(address.args[0], publication.get("address_argument"))
                )
                protocol_ok = (
                    bool(helper_params)
                    and len(statements) == 3
                    and matches(
                        ast.Name(id=helper_params[0], ctx=ast.Load()),
                        publication.get("address_argument"),
                    )
                    and wait_indexes[0] < barrier_indexes[0] < owner_index
                    and address_ok
                    and matches(stores[0].args[1], publication.get("store_offset"))
                    and matches(stores[0].args[2], publication.get("store_value"))
                )
    if protocol_ok:
        passed += 1
    else:
        failures.append("waitcnt/barrier/thread-owner/system-store protocol relation failed")

    rmw_calls = [str(value) for value in publication.get("forbidden_rmw_calls") or []]
    address_call = str(publication.get("address_call") or "")
    direct_address_patterns = [
        str(value)
        for value in publication.get("direct_address_patterns") or []
    ]
    address_base_patterns = [
        str(value) for value in publication.get("address_base_patterns") or []
    ]
    address_index_patterns = [
        str(value) for value in publication.get("address_index_patterns") or []
    ]

    def names_in(item: ast.AST) -> set[str]:
        return {
            child.id for child in ast.walk(item)
            if isinstance(child, ast.Name)
        }

    address_producers: set[str] = set()
    address_callable_aliases: set[str] = set()

    def contains_address_producer(item: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and (
                call_matches(child, address_call)
                or _call_name(child) in address_producers
                or _call_name(child) in address_callable_aliases
            )
            for child in ast.walk(item)
        ) or any(
            re.search(pattern, expr(item)) for pattern in direct_address_patterns
        )

    changed = True
    while changed:
        changed = False
        for helper_name, helper in helpers.items():
            returns = [
                item.value for item in ast.walk(helper)
                if isinstance(item, ast.Return) and item.value is not None
            ]
            if (
                helper_name not in address_producers
                and any(contains_address_producer(value) for value in returns)
            ):
                address_producers.add(helper_name)
                changed = True

    changed = True
    while changed:
        changed = False
        for item in all_nodes:
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value_name = (
                item.value.id if isinstance(item.value, ast.Name)
                else item.value.attr if isinstance(item.value, ast.Attribute)
                else ""
            )
            if (
                target.id not in address_callable_aliases
                and (
                    re.fullmatch(address_call, value_name)
                    or value_name in address_producers
                    or value_name in address_callable_aliases
                )
            ):
                address_callable_aliases.add(target.id)
                changed = True

    def has_address_call(item: ast.AST) -> bool:
        return contains_address_producer(item)

    def tainted_names(helper: ast.AST) -> set[str]:
        tainted: set[str] = set()
        base_tainted: set[str] = set()
        index_tainted: set[str] = set()
        changed_taint = True
        while changed_taint:
            changed_taint = False
            for item in ast.walk(helper):
                if not isinstance(item, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                value_names = names_in(item.value)
                base_value = (
                    any(re.search(pattern, expr(item.value)) for pattern in address_base_patterns)
                    or bool(value_names & base_tainted)
                )
                index_value = (
                    any(re.search(pattern, expr(item.value)) for pattern in address_index_patterns)
                    or bool(value_names & index_tainted)
                )
                completion_value = (
                    has_address_call(item.value)
                    or bool(value_names & tainted)
                    or (base_value and index_value)
                    or (
                        bool(value_names & base_tainted)
                        and bool(value_names & index_tainted)
                    )
                )
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    for state, applies in (
                        (base_tainted, base_value),
                        (index_tainted, index_value),
                        (tainted, completion_value),
                    ):
                        if applies and target.id not in state:
                            state.add(target.id)
                            changed_taint = True
        return tainted

    def expanded_names(
        helper: ast.AST,
        item: ast.AST,
    ) -> set[str]:
        dependencies: dict[str, set[str]] = {}
        for assignment in ast.walk(helper):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    dependencies[target.id] = names_in(assignment.value)
        found = names_in(item)
        changed_names = True
        while changed_names:
            changed_names = False
            inherited = set().union(*(
                dependencies.get(value, set()) for value in found
            ))
            if not inherited.issubset(found):
                found.update(inherited)
                changed_names = True
        return found

    def callable_aliases(owner: ast.AST, patterns: list[str]) -> set[str]:
        aliases: set[str] = set()
        changed_aliases = True
        while changed_aliases:
            changed_aliases = False
            for item in ast.walk(owner):
                if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                    continue
                target = item.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                value_name = (
                    item.value.id if isinstance(item.value, ast.Name)
                    else item.value.attr if isinstance(item.value, ast.Attribute)
                    else ""
                )
                if (
                    target.id not in aliases
                    and (
                        any(re.fullmatch(pattern, value_name) for pattern in patterns)
                        or value_name in aliases
                    )
                ):
                    aliases.add(target.id)
                    changed_aliases = True
        return aliases

    def call_argument(
        call: ast.Call,
        callee: ast.FunctionDef | ast.AsyncFunctionDef,
        index: int,
    ) -> ast.AST | None:
        if index < len(call.args):
            return call.args[index]
        params = function_parameters(callee)
        if index >= len(params):
            return None
        return next(
            (
                keyword.value for keyword in call.keywords
                if keyword.arg == params[index]
            ),
            None,
        )

    sink_params: dict[str, set[int]] = {helper: set() for helper in helpers}
    unsafe_helpers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for helper_name, helper in helpers.items():
            params = function_parameters(helper)
            tainted = tainted_names(helper)
            helper_rmw_aliases = callable_aliases(helper, rmw_calls)
            for call in (
                item for item in ast.walk(helper) if isinstance(item, ast.Call)
            ):
                if (
                    any(call_matches(call, pattern) for pattern in rmw_calls)
                    or _call_name(call) in helper_rmw_aliases
                ):
                    for argument in [
                        *call.args, *(keyword.value for keyword in call.keywords)
                    ]:
                        argument_names = expanded_names(helper, argument)
                        if has_address_call(argument) or argument_names & tainted:
                            unsafe_helpers.add(helper_name)
                        for index, parameter in enumerate(params):
                            if parameter in argument_names and index not in sink_params[helper_name]:
                                sink_params[helper_name].add(index)
                                changed = True
                callee = _call_name(call)
                for sink_index in sink_params.get(callee, set()):
                    argument = call_argument(call, helpers[callee], sink_index)
                    if argument is None:
                        continue
                    argument_names = expanded_names(helper, argument)
                    if has_address_call(argument) or argument_names & tainted:
                        unsafe_helpers.add(helper_name)
                    for index, parameter in enumerate(params):
                        if parameter in argument_names and index not in sink_params[helper_name]:
                            sink_params[helper_name].add(index)
                            changed = True

    def direct_calls(owner: ast.AST) -> set[str]:
        return {
            _call_name(item) for item in scoped_nodes(owner)
            if isinstance(item, ast.Call) and _call_name(item)
        }

    reachable: set[str] = set()
    pending = list(direct_calls(region_scope))
    while pending:
        callee = pending.pop()
        if callee in reachable or callee not in helpers:
            continue
        reachable.add(callee)
        pending.extend(direct_calls(helpers[callee]))

    region_unsafe = False
    region_tainted = tainted_names(region_scope)
    region_rmw_aliases = callable_aliases(region_scope, rmw_calls)
    for call in (
        item for item in scoped_nodes(region_scope) if isinstance(item, ast.Call)
    ):
        if (
            any(call_matches(call, pattern) for pattern in rmw_calls)
            or _call_name(call) in region_rmw_aliases
        ):
            region_unsafe |= any(
                has_address_call(argument)
                or expanded_names(region_scope, argument) & region_tainted
                for argument in [
                    *call.args, *(keyword.value for keyword in call.keywords)
                ]
            )
        for sink_index in sink_params.get(_call_name(call), set()):
            argument = call_argument(
                call, helpers[_call_name(call)], sink_index
            )
            if argument is not None:
                region_unsafe |= (
                    has_address_call(argument)
                    or bool(expanded_names(region_scope, argument) & region_tainted)
                )
    extra_completion_store = False
    store_pattern = publication.get("store_call")
    for owner_name, owner in [
        ("<region>", region_scope),
        *((name, helpers[name]) for name in reachable if name != publish_helper_name),
    ]:
        owner_tainted = tainted_names(owner)
        store_aliases = callable_aliases(owner, [str(store_pattern or "")])
        for call in (
            item for item in scoped_nodes(owner) if isinstance(item, ast.Call)
        ):
            if not (
                call_matches(call, store_pattern)
                or _call_name(call) in store_aliases
            ) or not call.args:
                continue
            address = call.args[0]
            if (
                has_address_call(address)
                or expanded_names(owner, address) & owner_tainted
            ):
                extra_completion_store = True
                break
        if extra_completion_store:
            break
    rmw_safe = (
        not region_unsafe
        and not (reachable & unsafe_helpers)
        and not extra_completion_store
    )
    if rmw_safe:
        passed += 1
    else:
        failures.append(
            "extra completion write or atomic RMW is reachable from the selected fused region"
        )

    forbidden_calls = [
        str(value) for value in publication.get("forbidden_calls") or []
    ]
    selected_nodes = [
        region_scope,
        *(helpers[name] for name in reachable),
    ]
    helper_forbidden = any(
        any(
            isinstance(item, ast.Call)
            and any(call_matches(item, pattern) for pattern in forbidden_calls)
            for item in scoped_nodes(owner)
        )
        for owner in selected_nodes
    )
    if not helper_forbidden:
        passed += 1
    else:
        failures.append("publication helper contains a forbidden per-item fence/call")

    return _result(not failures, 9, passed, failures, severity)


class _SemanticModel:
    """Small fail-closed AST/structured-CFG/def-use model.

    The model deliberately understands Python structure, not operator names.
    Names, calls, expressions, and protocol facts are supplied by a contract.
    """

    def __init__(
        self,
        trees: dict[str, ast.AST],
        wrappers: list[str] | None = None,
    ) -> None:
        self.trees = trees
        self.wrappers = {
            part
            for value in wrappers or []
            for part in (value, value.rsplit(".", 1)[-1])
        }

    def scope(self, file_name: str, scope: str | None) -> ast.AST | None:
        return _scope_node(self.trees.get(file_name), scope)

    def text(self, node: ast.AST) -> str:
        copied = ast.parse(ast.unparse(node), mode="eval").body

        class Strip(ast.NodeTransformer):
            def __init__(self, wrappers: set[str]) -> None:
                self.wrappers = wrappers

            def visit_Call(self, item: ast.Call):  # noqa: N802
                item = self.generic_visit(item)
                if (
                    _call_name(item) in self.wrappers
                    and len(item.args) == 1
                    and not item.keywords
                ):
                    return item.args[0]
                return item

        return ast.unparse(ast.fix_missing_locations(Strip(self.wrappers).visit(copied)))

    def matches(self, node: ast.AST, pattern: Any, *, search: bool = False) -> bool:
        value = self.text(node)
        return bool(
            re.search(str(pattern or ""), value)
            if search
            else re.fullmatch(str(pattern or ""), value)
        )

    @staticmethod
    def _targets(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.Assign):
            return list(node.targets)
        if isinstance(node, ast.AnnAssign):
            return [node.target]
        if isinstance(node, ast.AugAssign):
            return [node.target]
        if isinstance(node, ast.NamedExpr):
            return [node.target]
        return []

    @staticmethod
    def _value(node: ast.AST) -> ast.AST | None:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            return node.value
        if isinstance(node, ast.AugAssign):
            return ast.BinOp(
                left=node.target, op=node.op, right=node.value
            )
        return None

    def parents(self, root: ast.AST) -> dict[ast.AST, ast.AST]:
        return {
            child: parent
            for parent in ast.walk(root)
            for child in ast.iter_child_nodes(parent)
        }

    def ancestors(self, root: ast.AST, node: ast.AST) -> list[ast.AST]:
        parents = self.parents(root)
        result: list[ast.AST] = []
        while node in parents:
            node = parents[node]
            result.append(node)
        return result

    def _in_arm(
        self, root: ast.AST, node: ast.AST, branch: ast.If, arm: str
    ) -> bool:
        parents = self.parents(root)
        current = node
        while current in parents and parents[current] is not branch:
            current = parents[current]
        return current in getattr(branch, arm, [])

    def guarded(
        self,
        root: ast.AST,
        node: ast.AST,
        pattern: str,
        arm: str = "body",
    ) -> bool:
        return any(
            isinstance(parent, ast.If)
            and self.matches(parent.test, pattern)
            and self._in_arm(root, node, parent, arm)
            for parent in self.ancestors(root, node)
        )

    def select(self, root: ast.AST, spec: dict[str, Any]) -> list[ast.AST]:
        kind = str(spec.get("node") or "")
        selected: list[ast.AST] = []
        for item in ast.walk(root):
            if kind == "function":
                hit = isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                hit = hit and bool(
                    re.fullmatch(str(spec.get("name") or ".*"), item.name)
                )
                if hit and spec.get("decorators") is not None:
                    decorators = [ast.unparse(value) for value in item.decorator_list]
                    hit = all(
                        any(re.fullmatch(str(pattern), value) for value in decorators)
                        for pattern in spec.get("decorators") or []
                    )
            elif kind == "while":
                hit = isinstance(item, ast.While) and self.matches(
                    item.test, spec.get("test")
                )
            elif kind == "if":
                hit = isinstance(item, ast.If) and self.matches(
                    item.test, spec.get("test")
                )
            elif kind == "assert":
                hit = isinstance(item, ast.Assert) and self.matches(
                    item.test, spec.get("test")
                )
            elif kind == "for":
                hit = (
                    isinstance(item, ast.For)
                    and self.matches(item.target, spec.get("target"))
                    and self.matches(item.iter, spec.get("iter"))
                )
            elif kind == "assign":
                hit = bool(self._targets(item)) and any(
                    self.matches(target, spec.get("target"))
                    for target in self._targets(item)
                )
                if hit and spec.get("value") is not None:
                    value = self._value(item)
                    hit = value is not None and self.matches(
                        value, spec.get("value"), search=bool(spec.get("search"))
                    )
            elif kind == "call":
                hit = isinstance(item, ast.Call) and bool(
                    re.fullmatch(
                        str(spec.get("call") or ".*"),
                        _call_name(item),
                    )
                    or re.fullmatch(
                        str(spec.get("call") or ".*"),
                        ast.unparse(item.func),
                    )
                )
                if hit and spec.get("args") is not None:
                    patterns = list(spec.get("args") or [])
                    hit = len(item.args) >= len(patterns) and all(
                        self.matches(value, pattern)
                        for value, pattern in zip(item.args, patterns)
                    )
                if hit and spec.get("arg_count") is not None:
                    hit = len(item.args) == int(spec["arg_count"])
                if hit and spec.get("keywords") is not None:
                    values = {
                        keyword.arg: keyword.value
                        for keyword in item.keywords
                        if keyword.arg is not None
                    }
                    hit = all(
                        key in values and self.matches(values[key], pattern)
                        for key, pattern in (spec.get("keywords") or {}).items()
                    )
                if hit and spec.get("star_keywords") is not None:
                    hit = (
                        sum(keyword.arg is None for keyword in item.keywords)
                        == int(spec["star_keywords"])
                    )
            elif kind == "return":
                hit = (
                    isinstance(item, ast.Return)
                    and item.value is not None
                    and self.matches(item.value, spec.get("value"))
                )
            elif kind == "attribute":
                hit = isinstance(item, ast.Attribute) and self.matches(
                    item, spec.get("value")
                )
            else:
                hit = False
            if hit and spec.get("guard") is not None:
                hit = self.guarded(
                    root,
                    item,
                    str(spec["guard"]),
                    str(spec.get("guard_arm") or "body"),
                )
            if hit and spec.get("guards") is not None:
                hit = all(
                    self.guarded(
                        root,
                        item,
                        str(guard.get("test") or ""),
                        str(guard.get("arm") or "body"),
                    )
                    for guard in spec.get("guards") or []
                )
            if hit:
                selected.append(item)
        return selected

    def definitions(
        self, root: ast.AST, name: str, before: int | None = None
    ) -> list[ast.AST]:
        found = []
        for item in ast.walk(root):
            if before is not None and getattr(item, "lineno", 0) >= before:
                continue
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in self._targets(item)
            ):
                found.append(item)
        return found

    def depends_on(
        self,
        root: ast.AST,
        value: ast.AST,
        source: str,
        *,
        before: int | None = None,
        seen: set[str] | None = None,
    ) -> bool:
        """Conservative unique-definition def-use query."""
        if self.matches(value, source) or re.search(source, self.text(value)):
            return True
        seen = set(seen or ())
        for name in {
            item.id for item in ast.walk(value) if isinstance(item, ast.Name)
        }:
            if name in seen:
                continue
            definitions = self.definitions(
                root, name, before=before or getattr(value, "lineno", None)
            )
            if len(definitions) != 1:
                continue
            rhs = self._value(definitions[0])
            if rhs is not None and self.depends_on(
                root,
                rhs,
                source,
                before=getattr(definitions[0], "lineno", None),
                seen=seen | {name},
            ):
                return True
        return False

    @staticmethod
    def direct_statement_ancestor(root: ast.AST, node: ast.AST) -> ast.stmt | None:
        parents = {
            child: parent
            for parent in ast.walk(root)
            for child in ast.iter_child_nodes(parent)
        }
        current = node
        while current in parents and current is not root:
            if isinstance(current, ast.stmt):
                return current
            current = parents[current]
        return current if isinstance(current, ast.stmt) else None

    def structurally_precedes(
        self, root: ast.AST, first: ast.AST, second: ast.AST
    ) -> bool:
        """Require lexical order on a common structured-CFG arm."""
        parents = self.parents(root)

        def chain(item: ast.AST) -> list[ast.AST]:
            result = [item]
            while item in parents:
                item = parents[item]
                result.append(item)
            return result

        left, right = chain(first), chain(second)
        common = next((item for item in left if item in set(right)), None)
        if common is None:
            return False
        l_child = next((item for item in left if parents.get(item) is common), None)
        r_child = next((item for item in right if parents.get(item) is common), None)
        for field in ("body", "orelse", "finalbody"):
            arm = getattr(common, field, None)
            if isinstance(arm, list) and l_child in arm and r_child in arm:
                return arm.index(l_child) < arm.index(r_child)
        return getattr(first, "lineno", 0) < getattr(second, "lineno", 0)


_CONST_UNKNOWN = object()


def _constant_expr(node: ast.AST, environment: dict[str, Any]) -> Any:
    """Evaluate a deliberately small, side-effect-free Python expression subset."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment.get(node.id, _CONST_UNKNOWN)
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_constant_expr(value, environment) for value in node.elts]
        return (
            _CONST_UNKNOWN
            if any(value is _CONST_UNKNOWN for value in values)
            else tuple(values)
        )
    if isinstance(node, ast.Dict):
        keys = [_constant_expr(value, environment) for value in node.keys]
        values = [_constant_expr(value, environment) for value in node.values]
        if any(value is _CONST_UNKNOWN for value in [*keys, *values]):
            return _CONST_UNKNOWN
        return dict(zip(keys, values))
    if isinstance(node, ast.Subscript):
        value = _constant_expr(node.value, environment)
        index = _constant_expr(node.slice, environment)
        if value is _CONST_UNKNOWN or index is _CONST_UNKNOWN:
            return _CONST_UNKNOWN
        try:
            return value[index]
        except (KeyError, IndexError, TypeError):
            return _CONST_UNKNOWN
    if isinstance(node, ast.UnaryOp):
        value = _constant_expr(node.operand, environment)
        if value is _CONST_UNKNOWN:
            return value
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        return _CONST_UNKNOWN
    if isinstance(node, ast.BinOp):
        left = _constant_expr(node.left, environment)
        right = _constant_expr(node.right, environment)
        if left is _CONST_UNKNOWN or right is _CONST_UNKNOWN:
            return _CONST_UNKNOWN
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.BitAnd: lambda: left & right,
            ast.BitOr: lambda: left | right,
        }
        operation = operations.get(type(node.op))
        try:
            return operation() if operation else _CONST_UNKNOWN
        except (TypeError, ValueError, ZeroDivisionError):
            return _CONST_UNKNOWN
    if isinstance(node, ast.BoolOp):
        values = [_constant_expr(value, environment) for value in node.values]
        if any(value is _CONST_UNKNOWN for value in values):
            return _CONST_UNKNOWN
        return (
            all(values) if isinstance(node.op, ast.And) else any(values)
        )
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _constant_expr(node.left, environment)
        right = _constant_expr(node.comparators[0], environment)
        if left is _CONST_UNKNOWN or right is _CONST_UNKNOWN:
            return _CONST_UNKNOWN
        comparisons = {
            ast.Eq: lambda: left == right,
            ast.NotEq: lambda: left != right,
            ast.Lt: lambda: left < right,
            ast.LtE: lambda: left <= right,
            ast.Gt: lambda: left > right,
            ast.GtE: lambda: left >= right,
            ast.In: lambda: left in right,
            ast.NotIn: lambda: left not in right,
        }
        comparison = comparisons.get(type(node.ops[0]))
        try:
            return comparison() if comparison else _CONST_UNKNOWN
        except TypeError:
            return _CONST_UNKNOWN
    if isinstance(node, ast.IfExp):
        test = _constant_expr(node.test, environment)
        if test is _CONST_UNKNOWN:
            return test
        return _constant_expr(node.body if test else node.orelse, environment)
    if isinstance(node, ast.Call):
        called = _call_name(node)
        args = [_constant_expr(value, environment) for value in node.args]
        if any(value is _CONST_UNKNOWN for value in args):
            return _CONST_UNKNOWN
        functions = {
            "int": int,
            "bool": bool,
            "tuple": tuple,
            "max": max,
            "min": min,
            "len": len,
        }
        function = functions.get(called)
        if function is None:
            return _CONST_UNKNOWN
        try:
            return function(*args)
        except (TypeError, ValueError):
            return _CONST_UNKNOWN
    return _CONST_UNKNOWN


def _is_statically_hashable(
    model: _SemanticModel,
    root: ast.AST,
    node: ast.AST,
    before: int | None = None,
    seen: set[str] | None = None,
) -> bool:
    if isinstance(node, ast.Constant):
        try:
            hash(node.value)
            return True
        except TypeError:
            return False
    if isinstance(node, ast.Tuple):
        return all(
            _is_statically_hashable(model, root, value, before, seen)
            for value in node.elts
        )
    if isinstance(node, ast.IfExp):
        return _is_statically_hashable(
            model, root, node.body, before, seen
        ) and _is_statically_hashable(model, root, node.orelse, before, seen)
    if isinstance(node, ast.Name):
        seen = set(seen or ())
        if node.id in seen:
            return False
        definitions = model.definitions(root, node.id, before=before)
        if len(definitions) != 1:
            return False
        value = model._value(definitions[0])
        return value is not None and _is_statically_hashable(
            model,
            root,
            value,
            getattr(definitions[0], "lineno", None),
            seen | {node.id},
        )
    if isinstance(node, ast.Call):
        called = _call_name(node)
        if called in {"int", "bool", "str", "float"}:
            return True
        if called == "tuple" and len(node.args) == 1:
            argument = node.args[0]
            if (
                isinstance(argument, ast.Call)
                and _call_name(argument) == "sorted"
                and len(argument.args) == 1
                and isinstance(argument.args[0], ast.Call)
                and isinstance(argument.args[0].func, ast.Attribute)
                and argument.args[0].func.attr == "items"
            ):
                return True
            return _is_statically_hashable(model, root, argument, before, seen)
    return False


def _definite_self_attributes(
    tree: ast.AST,
    class_name: str,
    entry_name: str,
) -> set[str]:
    functions = _qualified_function_nodes(tree)
    class_functions = {
        name.rsplit(".", 1)[-1]: node
        for name, node in functions.items()
        if name.startswith(f"{class_name}.")
        and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    visiting: set[str] = set()

    def analyze(name: str) -> set[str]:
        if name in visiting or name not in class_functions:
            return set()
        visiting.add(name)
        function = class_functions[name]

        mappings: dict[str, set[str]] = {}
        for statement in function.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Dict)
            ):
                mappings[statement.targets[0].id] = {
                    str(key.value)
                    for key in statement.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }

        def sequence(statements: list[ast.stmt], incoming: set[str]) -> set[str]:
            definite = set(incoming)
            for statement in statements:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    definite.update(
                        target.attr
                        for target in targets
                        if isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    )
                elif isinstance(statement, ast.If):
                    body = sequence(statement.body, definite)
                    other = sequence(statement.orelse, definite)
                    definite = body & other
                elif isinstance(statement, ast.For):
                    mapping_name = (
                        statement.iter.func.value.id
                        if isinstance(statement.iter, ast.Call)
                        and isinstance(statement.iter.func, ast.Attribute)
                        and statement.iter.func.attr == "items"
                        and isinstance(statement.iter.func.value, ast.Name)
                        else ""
                    )
                    has_dynamic_setattr = any(
                        isinstance(item, ast.Call)
                        and _call_name(item) == "setattr"
                        and len(item.args) >= 2
                        and isinstance(item.args[0], ast.Name)
                        and item.args[0].id == "self"
                        for item in ast.walk(statement)
                    )
                    if has_dynamic_setattr:
                        definite.update(mappings.get(mapping_name, set()))
                elif isinstance(statement, ast.Expr) and isinstance(
                    statement.value, ast.Call
                ):
                    call = statement.value
                    if (
                        isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "self"
                    ):
                        definite.update(analyze(call.func.attr))
                    elif (
                        _call_name(call) == "setattr"
                        and len(call.args) >= 2
                        and isinstance(call.args[0], ast.Name)
                        and call.args[0].id == "self"
                        and isinstance(call.args[1], ast.Constant)
                        and isinstance(call.args[1].value, str)
                    ):
                        definite.add(call.args[1].value)
            return definite

        result = sequence(function.body, set())
        visiting.remove(name)
        return result

    return analyze(entry_name)


def _constant_function_return(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    inputs: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    environment = dict(inputs)

    def bind(target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            environment[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple):
            if len(target.elts) == len(value):
                for child, child_value in zip(target.elts, value):
                    bind(child, child_value)

    def execute(statements: list[ast.stmt]) -> ast.Return | None:
        for statement in statements:
            if isinstance(statement, ast.Assign):
                value = _constant_expr(statement.value, environment)
                for target in statement.targets:
                    bind(target, value)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                bind(statement.target, _constant_expr(statement.value, environment))
            elif isinstance(statement, ast.If):
                test = _constant_expr(statement.test, environment)
                if test is _CONST_UNKNOWN:
                    return None
                returned = execute(statement.body if test else statement.orelse)
                if returned is not None:
                    return returned
            elif isinstance(statement, ast.Return):
                return statement
        return None

    returned = execute(function.body)
    if returned is None or not isinstance(returned.value, ast.Call):
        return None
    values = {
        str(keyword.arg): _constant_expr(keyword.value, environment)
        for keyword in returned.value.keywords
        if keyword.arg is not None
    }
    return _call_name(returned.value), values


def _semantic_relation_failures(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
) -> tuple[int, int, list[str]]:
    """Evaluate declarative AST/CFG/def-use relations."""
    model = _SemanticModel(
        trees, [str(value) for value in rule.get("normalize_wrappers") or []]
    )
    default_file = str(rule.get("file") or "")
    default_scope = str(rule.get("scope") or "")
    relations = [item for item in rule.get("relations") or [] if isinstance(item, dict)]
    failures: list[str] = []
    passed = 0

    def context(config: dict[str, Any]) -> tuple[ast.AST | None, str, str]:
        file_name = str(config.get("file") or default_file)
        scope = (
            str(config.get("scope") or "")
            if "scope" in config
            else default_scope
        )
        root = model.scope(file_name, scope)
        if root is not None and config.get("active"):
            parsed = ast.parse(_active_code(root))
            if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
                root = parsed.body[0] if parsed.body else None
            else:
                root = parsed
        return root, file_name, scope

    def one(
        root: ast.AST | None,
        selector: dict[str, Any],
        label: str,
    ) -> ast.AST | None:
        found = model.select(root, selector) if root is not None else []
        if len(found) != 1:
            failures.append(f"{label}: expected one node, found {len(found)}")
            return None
        return found[0]

    for index, relation in enumerate(relations):
        label = str(relation.get("id") or f"relation_{index + 1}")
        kind = str(relation.get("op") or "")
        before_failures = len(failures)
        if kind == "any_of":
            failures.extend(_any_of_failures(rule, trees, relation, label))
            if len(failures) == before_failures:
                passed += 1
            continue
        root, file_name, scope = context(relation)
        if root is None:
            failures.append(f"{label}: missing scope {file_name}:{scope}")
        elif kind == "count":
            found = model.select(root, relation.get("select") or {})
            minimum = int(relation.get("min", relation.get("count", 1)))
            maximum = int(relation.get("max", relation.get("count", minimum)))
            if not minimum <= len(found) <= maximum:
                failures.append(
                    f"{label}: node count {len(found)} outside [{minimum},{maximum}]"
                )
        elif kind == "constant_return":
            if not isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
                failures.append(f"{label}: selected scope is not a function")
            else:
                evaluated = _constant_function_return(
                    root, dict(relation.get("inputs") or {})
                )
                expected_call = str(relation.get("call") or "")
                expected = dict(relation.get("keywords") or {})
                if evaluated is None:
                    failures.append(f"{label}: constant path is ambiguous")
                else:
                    called, values = evaluated
                    bad = [
                        key for key, value in expected.items()
                        if values.get(str(key), _CONST_UNKNOWN) != value
                    ]
                    if (expected_call and not re.fullmatch(expected_call, called)) or bad:
                        failures.append(
                            f"{label}: return={called} values={values}, "
                            f"wrong={bad}"
                        )
        elif kind == "profile_value":
            selected = one(
                root, relation.get("select") or {}, f"{label}.value"
            )
            value = model._value(selected) if selected is not None else None
            actual = (
                _constant_expr(value, dict(relation.get("inputs") or {}))
                if value is not None
                else _CONST_UNKNOWN
            )
            expected = relation.get("value", _CONST_UNKNOWN)
            maximum = relation.get("max", _CONST_UNKNOWN)
            if (
                actual is _CONST_UNKNOWN
                or (expected is not _CONST_UNKNOWN and actual != expected)
                or (
                    maximum is not _CONST_UNKNOWN
                    and (not isinstance(actual, (int, float)) or actual > maximum)
                )
            ):
                failures.append(
                    f"{label}: profile value={actual!r}, "
                    f"expected={expected!r}, max={maximum!r}"
                )
        elif kind == "profile_group_segment":
            inputs = dict(relation.get("inputs") or {})

            def selected_value(key: str) -> Any:
                selected = one(
                    root,
                    relation.get(key) or {},
                    f"{label}.{key}",
                )
                value = model._value(selected) if selected is not None else None
                return (
                    _constant_expr(value, inputs)
                    if value is not None else _CONST_UNKNOWN
                )

            base = selected_value("base")
            role = selected_value("role")
            additive = selected_value("additive")
            halves = int(relation.get("role_multiplier", 1))
            limit = int(relation.get("limit", 0))
            if any(
                value is _CONST_UNKNOWN for value in (base, role, additive)
            ):
                failures.append(f"{label}: profile components are ambiguous")
            else:
                total = max(base, role * halves) + additive
                expected = relation.get("expected")
                if total > limit or (expected is not None and total != int(expected)):
                    failures.append(
                        f"{label}: group_segment={total}, expected={expected}, "
                        f"limit={limit}"
                    )
        elif kind == "within_count":
            region = one(root, relation.get("region") or {}, f"{label}.region")
            if region is not None:
                selected_region: ast.AST = region
                if isinstance(region, ast.If) and relation.get("region_arm"):
                    selected_region = ast.Module(
                        body=list(getattr(region, str(relation["region_arm"]))),
                        type_ignores=[],
                    )
                found = model.select(selected_region, relation.get("select") or {})
                minimum = int(relation.get("min", relation.get("count", 1)))
                maximum = int(relation.get("max", relation.get("count", minimum)))
                if not minimum <= len(found) <= maximum:
                    failures.append(
                        f"{label}: nested node count {len(found)} outside "
                        f"[{minimum},{maximum}]"
                    )
        elif kind == "ordered":
            nodes = [
                one(root, selector, f"{label}[{part}]")
                for part, selector in enumerate(relation.get("selectors") or [])
            ]
            if all(nodes) and not all(
                model.structurally_precedes(root, left, right)
                for left, right in zip(nodes, nodes[1:])
            ):
                failures.append(f"{label}: nodes do not dominate in declared order")
        elif kind == "body_prefix":
            region = one(root, relation.get("region") or {}, f"{label}.region")
            body = getattr(region, "body", None) if region is not None else None
            expected = relation.get("select") or {}
            if not isinstance(body, list) or not body:
                failures.append(f"{label}: selected region has no body")
            elif not model.select(
                ast.Module(body=[body[0]], type_ignores=[]), expected
            ):
                failures.append(
                    f"{label}: first statement does not match required prefix"
                )
        elif kind == "all_precede":
            first = model.select(root, relation.get("first") or {})
            second = model.select(root, relation.get("second") or {})
            if not first or not second:
                failures.append(
                    f"{label}: missing predecessor ({len(first)}) or sink ({len(second)})"
                )
            elif not all(
                any(model.structurally_precedes(root, before, after) for before in first)
                for after in second
            ):
                failures.append(f"{label}: a sink is not dominated by a predecessor")
        elif kind == "guarded":
            subjects = model.select(root, relation.get("subject") or {})
            if not subjects:
                failures.append(f"{label}: no subject nodes")
            elif not all(
                model.guarded(
                    root,
                    item,
                    str(relation.get("guard") or ""),
                    str(relation.get("arm") or "body"),
                )
                for item in subjects
            ):
                failures.append(f"{label}: not every subject is in the selected branch")
        elif kind == "not_guarded":
            subjects = model.select(root, relation.get("subject") or {})
            if not subjects:
                failures.append(f"{label}: no subject nodes")
            elif any(
                model.guarded(
                    root,
                    item,
                    str(relation.get("guard") or ""),
                    str(relation.get("arm") or "body"),
                )
                for item in subjects
            ):
                failures.append(f"{label}: a subject is under the forbidden branch")
        elif kind == "enclosed_by":
            subjects = model.select(root, relation.get("subject") or {})
            containers = model.select(root, relation.get("container") or {})
            if not subjects or not containers:
                failures.append(
                    f"{label}: missing subjects ({len(subjects)}) or "
                    f"containers ({len(containers)})"
                )
            elif not all(
                any(container in model.ancestors(root, item) for container in containers)
                for item in subjects
            ):
                failures.append(f"{label}: a subject escapes its required container")
        elif kind == "ancestor_count":
            subjects = model.select(root, relation.get("subject") or {})
            containers = model.select(root, relation.get("container") or {})
            expected = int(relation.get("count", 1))
            bad = [
                getattr(item, "lineno", 0)
                for item in subjects
                if sum(
                    container in model.ancestors(root, item)
                    for container in containers
                )
                != expected
            ]
            if not subjects or bad:
                failures.append(
                    f"{label}: subjects={len(subjects)}, bad_lines={bad}, "
                    f"expected_ancestors={expected}"
                )
        elif kind == "forbidden":
            region = one(root, relation.get("region") or {}, f"{label}.region")
            if region is not None:
                selected_region: ast.AST = region
                if isinstance(region, ast.If) and relation.get("region_arm"):
                    selected_region = ast.Module(
                        body=list(
                            getattr(region, str(relation.get("region_arm")))
                        ),
                        type_ignores=[],
                    )
                found = model.select(selected_region, relation.get("select") or {})
                if found:
                    failures.append(f"{label}: found {len(found)} forbidden node(s)")
        elif kind == "forbidden_each":
            regions = model.select(root, relation.get("regions") or {})
            if not regions:
                failures.append(f"{label}: no selected regions")
            for region in regions:
                selected_region: ast.AST = region
                if isinstance(region, ast.If):
                    selected_region = ast.Module(
                        body=list(
                            getattr(region, str(relation.get("region_arm") or "body"))
                        ),
                        type_ignores=[],
                    )
                found = model.select(selected_region, relation.get("select") or {})
                if found:
                    failures.append(
                        f"{label}: region line {getattr(region, 'lineno', 0)} "
                        f"has {len(found)} forbidden node(s)"
                    )
        elif kind == "depends":
            sink = one(root, relation.get("sink") or {}, f"{label}.sink")
            value: ast.AST | None = None
            if isinstance(sink, ast.Call):
                if "arg" in relation:
                    arg = int(relation["arg"])
                    value = sink.args[arg] if arg < len(sink.args) else None
                elif "star_arg" in relation:
                    starred = [
                        item.value for item in sink.args
                        if isinstance(item, ast.Starred)
                    ]
                    position = int(relation["star_arg"])
                    value = starred[position] if position < len(starred) else None
                else:
                    keyword = str(relation.get("keyword") or "")
                    if relation.get("star_keyword"):
                        value = next(
                            (item.value for item in sink.keywords if item.arg is None),
                            None,
                        )
                    else:
                        value = next(
                            (
                                item.value for item in sink.keywords
                                if item.arg == keyword
                            ),
                            None,
                        )
            elif sink is not None:
                value = model._value(sink)
            if value is None or not model.depends_on(
                root,
                value,
                str(relation.get("source") or ""),
                before=getattr(sink, "lineno", None) if sink else None,
            ):
                failures.append(f"{label}: sink does not depend on declared source")
        elif kind == "same_value":
            left = one(root, relation.get("left") or {}, f"{label}.left")
            right = one(root, relation.get("right") or {}, f"{label}.right")
            if left is not None and right is not None:
                left_value = model._value(left) or left
                right_value = model._value(right) or right
                if model.text(left_value) != model.text(right_value):
                    failures.append(f"{label}: selected expressions differ")
        elif kind == "same_argument":
            left = one(root, relation.get("left") or {}, f"{label}.left")
            right = one(root, relation.get("right") or {}, f"{label}.right")
            position = int(relation.get("arg", 0))
            if isinstance(left, ast.Call) and isinstance(right, ast.Call):
                if (
                    len(left.args) <= position
                    or len(right.args) <= position
                    or model.text(left.args[position]) != model.text(right.args[position])
                ):
                    failures.append(f"{label}: selected call arguments differ")
        elif kind == "different_value":
            left = one(root, relation.get("left") or {}, f"{label}.left")
            right = one(root, relation.get("right") or {}, f"{label}.right")
            if left is not None and right is not None:
                left_value = model._value(left) or left
                right_value = model._value(right) or right
                if model.text(left_value) == model.text(right_value):
                    failures.append(f"{label}: selected expressions unexpectedly match")
        elif kind == "different_call_values":
            call = one(root, relation.get("call") or {}, f"{label}.call")
            if isinstance(call, ast.Call):
                values = {
                    keyword.arg: model.text(keyword.value)
                    for keyword in call.keywords
                    if keyword.arg is not None
                }
                keys = [str(value) for value in relation.get("keywords") or []]
                selected = [values.get(key) for key in keys]
                if None in selected or len(set(selected)) != len(selected):
                    failures.append(
                        f"{label}: keyword values are missing or aliased: {selected}"
                    )
        elif kind == "all_calls_match":
            calls = model.select(
                root, {"node": "call", "call": relation.get("call") or ".*"}
            )
            minimum = int(relation.get("min", 1))
            bad: list[str] = []
            for call in calls:
                assert isinstance(call, ast.Call)
                values = {
                    keyword.arg: keyword.value
                    for keyword in call.keywords
                    if keyword.arg is not None
                }
                for key, pattern in (relation.get("keywords") or {}).items():
                    if key not in values or not model.matches(values[key], pattern):
                        bad.append(f"line {getattr(call, 'lineno', 0)}: {key}")
                for position, pattern in enumerate(relation.get("args") or []):
                    if position >= len(call.args) or not model.matches(
                        call.args[position], pattern
                    ):
                        bad.append(f"line {getattr(call, 'lineno', 0)}: arg {position}")
            if len(calls) < minimum or bad:
                failures.append(
                    f"{label}: found {len(calls)} calls (minimum {minimum}); "
                    f"mismatches={bad}"
                )
        elif kind == "parameters_used":
            if not isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
                failures.append(f"{label}: selected scope is not a function")
            else:
                patterns = [
                    str(value) for value in relation.get("parameters") or [".*"]
                ]
                parameters = [
                    argument.arg
                    for argument in (
                        list(root.args.posonlyargs)
                        + list(root.args.args)
                        + list(root.args.kwonlyargs)
                    )
                    if any(re.fullmatch(pattern, argument.arg) for pattern in patterns)
                ]
                loads = {
                    item.id
                    for item in ast.walk(root)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                }
                missing = [value for value in parameters if value not in loads]
                minimum = int(relation.get("min_parameters", 1))
                if len(parameters) < minimum or missing:
                    failures.append(
                        f"{label}: matched={parameters}, unused={missing}, "
                        f"minimum={minimum}"
                    )
        elif kind == "function_parameters":
            function = one(
                root, relation.get("function") or {}, f"{label}.function"
            )
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [
                    argument.arg
                    for argument in (
                        list(function.args.posonlyargs)
                        + list(function.args.args)
                        + list(function.args.kwonlyargs)
                    )
                ]
                required = [str(value) for value in relation.get("required") or []]
                forbidden = [str(value) for value in relation.get("forbidden") or []]
                missing = [
                    pattern for pattern in required
                    if not any(re.fullmatch(pattern, value) for value in parameters)
                ]
                invalid = [
                    value for value in parameters
                    if any(re.fullmatch(pattern, value) for pattern in forbidden)
                ]
                if missing or invalid:
                    failures.append(
                        f"{label}: missing={missing}, forbidden={invalid}"
                    )
        elif kind == "definite_attributes":
            tree = model.trees.get(file_name)
            class_name = str(relation.get("class") or "")
            entry = str(relation.get("entry") or "__init__")
            required = [str(value) for value in relation.get("attributes") or []]
            definite = (
                _definite_self_attributes(tree, class_name, entry)
                if tree is not None else set()
            )
            missing = [value for value in required if value not in definite]
            if missing:
                failures.append(
                    f"{label}: attributes not definitely initialized: {missing}"
                )
        elif kind == "cached_call_hashable":
            call = one(root, relation.get("call") or {}, f"{label}.call")
            bad: list[str] = []
            if isinstance(call, ast.Call):
                values = {
                    keyword.arg: keyword.value
                    for keyword in call.keywords
                    if keyword.arg is not None
                }
                for keyword in relation.get("keywords") or []:
                    value = values.get(str(keyword))
                    if value is None or not _is_statically_hashable(
                        model, root, value, getattr(call, "lineno", None)
                    ):
                        bad.append(str(keyword))
            if bad:
                failures.append(
                    f"{label}: cached call has non-canonical values: {bad}"
                )
        elif kind == "tuple_elements":
            target = str(relation.get("target") or "")
            candidates = model.select(
                root, {"node": "assign", "target": target}
            )
            values: list[ast.AST] | None = None
            if len(candidates) == 1:
                rhs = model._value(candidates[0])
                if isinstance(rhs, (ast.Tuple, ast.List)):
                    values = list(rhs.elts)
            expected = [str(value) for value in relation.get("values") or []]
            if values is None:
                failures.append(f"{label}: tuple {target!r} is missing or ambiguous")
            else:
                exact_length = int(relation.get("length", len(expected) or len(values)))
                bad = [
                    position
                    for position, pattern in enumerate(expected)
                    if position >= len(values)
                    or not model.matches(values[position], pattern)
                ]
                rendered = [model.text(value) for value in values]
                minimum_distinct = int(
                    relation.get("min_distinct", exact_length)
                )
                if (
                    len(values) != exact_length
                    or bad
                    or len(set(rendered)) < minimum_distinct
                ):
                    failures.append(
                        f"{label}: length={len(values)}/{exact_length}, "
                        f"bad_positions={bad}, distinct={len(set(rendered))}/"
                        f"{minimum_distinct}"
                    )
        elif kind == "mapping_entries":
            selected = one(
                root, relation.get("mapping") or {}, f"{label}.mapping"
            )
            rhs = model._value(selected) if selected is not None else None
            entries: dict[str, ast.AST] = {}
            if isinstance(rhs, ast.Dict):
                entries = {
                    str(key.value): value
                    for key, value in zip(rhs.keys, rhs.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
            elif isinstance(rhs, ast.Call) and _call_name(rhs) == "dict":
                entries = {
                    str(item.arg): item.value
                    for item in rhs.keywords
                    if item.arg is not None
                }
            expected = {
                str(key): str(value)
                for key, value in (relation.get("entries") or {}).items()
            }
            bad = [
                key for key, pattern in expected.items()
                if key not in entries or not model.matches(entries[key], pattern)
            ]
            forbidden = [str(value) for value in relation.get("forbidden") or []]
            invalid = [
                key for key, value in entries.items()
                if any(model.matches(value, pattern) for pattern in forbidden)
            ]
            if bad or invalid:
                failures.append(
                    f"{label}: wrong entries={bad}, forbidden entries={invalid}"
                )
        elif kind == "starred_arguments":
            call = one(root, relation.get("call") or {}, f"{label}.call")
            if isinstance(call, ast.Call):
                actual = [
                    model.text(item.value)
                    for item in call.args if isinstance(item, ast.Starred)
                ]
                expected = [str(value) for value in relation.get("values") or []]
                if len(actual) != len(expected) or any(
                    not re.fullmatch(pattern, value)
                    for pattern, value in zip(expected, actual)
                ):
                    failures.append(
                        f"{label}: starred arguments {actual} != {expected}"
                    )
        elif kind == "exclusive_effect":
            nodes = model.select(root, relation.get("select") or {})
            if len(nodes) == 1:
                pass
            elif len(nodes) == 2:
                left_ifs = [
                    item for item in model.ancestors(root, nodes[0])
                    if isinstance(item, ast.If)
                ]
                common = next(
                    (
                        item for item in left_ifs
                        if item in model.ancestors(root, nodes[1])
                    ),
                    None,
                )
                if common is None or (
                    model._in_arm(root, nodes[0], common, "body")
                    == model._in_arm(root, nodes[1], common, "body")
                ):
                    failures.append(
                        f"{label}: effects are not mutually exclusive"
                    )
            else:
                failures.append(
                    f"{label}: expected one effect or one exclusive pair, "
                    f"found {len(nodes)}"
                )
        elif kind == "all_assignments_classified":
            assignments = model.select(root, relation.get("assignments") or {})
            classifiers = list(relation.get("classifiers") or [])
            if not assignments:
                failures.append(f"{label}: no assignments to classify")
            for assignment in assignments:
                if not any(
                    model.guarded(
                        root,
                        assignment,
                        str(classifier.get("guard") or ""),
                        str(classifier.get("arm") or "body"),
                    )
                    for classifier in classifiers
                ):
                    failures.append(
                        f"{label}: assignment line {getattr(assignment, 'lineno', 0)} "
                        "is unclassified"
                    )
        elif kind == "all_nodes_classified":
            nodes = model.select(root, relation.get("nodes") or {})
            classifiers = list(relation.get("classifiers") or [])
            if not nodes:
                failures.append(f"{label}: no nodes to classify")
            for item in nodes:
                if not any(
                    model.guarded(
                        root,
                        item,
                        str(classifier.get("guard") or ""),
                        str(classifier.get("arm") or "body"),
                    )
                    for classifier in classifiers
                ):
                    failures.append(
                        f"{label}: node line {getattr(item, 'lineno', 0)} is unclassified"
                    )
        elif kind == "branch_partition":
            nodes = model.select(root, relation.get("nodes") or {})
            if not nodes:
                failures.append(f"{label}: no branch nodes")
            signatures: list[tuple[tuple[str, str], ...]] = []
            for item in nodes:
                guards = [
                    (
                        model.text(parent.test),
                        "body" if model._in_arm(root, item, parent, "body") else "orelse",
                    )
                    for parent in model.ancestors(root, item)
                    if isinstance(parent, ast.If)
                ]
                if not guards:
                    failures.append(
                        f"{label}: node line {getattr(item, 'lineno', 0)} "
                        "is unconditional"
                    )
                signatures.append(tuple(sorted(guards)))
            if len(signatures) != len(set(signatures)):
                failures.append(f"{label}: duplicate branch path")
        elif kind == "callable_identity":
            factory = one(
                root,
                relation.get("factory_call") or {},
                f"{label}.factory_call",
            )
            produced = ""
            statement = (
                model.direct_statement_ancestor(root, factory)
                if factory is not None else None
            )
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                produced = ast.unparse(statement.targets[0])
            if not produced:
                failures.append(f"{label}: factory result is not uniquely bound")
            for consumer in relation.get("consumers") or []:
                croot, cfile, cscope = context(consumer)
                calls = (
                    model.select(croot, consumer.get("call") or {})
                    if croot is not None else []
                )
                if len(calls) != 1:
                    failures.append(
                        f"{label}: consumer {cfile}:{cscope} found {len(calls)} calls"
                    )
                    continue
                source = str(consumer.get("source") or produced)
                if source and not (
                    re.search(source, ast.unparse(calls[0].func))
                    or model.depends_on(
                        croot,
                        calls[0].func,
                        source,
                        before=getattr(calls[0], "lineno", None),
                    )
                ):
                    failures.append(
                        f"{label}: consumer {cfile}:{cscope} does not use factory identity"
                    )
        elif kind == "branch_calls":
            branch = one(root, relation.get("branch") or {}, f"{label}.branch")
            if isinstance(branch, ast.If):
                expected = [
                    str(value) for value in relation.get("body_calls") or []
                ]
                body_text = ast.Module(body=branch.body, type_ignores=[])
                calls = [
                    model.text(item.args[int(relation.get("arg", 1))])
                    for item in ast.walk(body_text)
                    if isinstance(item, ast.Call)
                    and re.fullmatch(
                        str(relation.get("call") or ".*"), _call_name(item)
                    )
                    and len(item.args) > int(relation.get("arg", 1))
                ]
                if sorted(calls) != sorted(expected):
                    failures.append(
                        f"{label}: selected branch calls {calls}, expected {expected}"
                    )
        else:
            failures.append(f"{label}: unsupported semantic relation op {kind!r}")
        if len(failures) == before_failures:
            passed += 1
    return max(1, len(relations)), passed, failures


def _any_of_failures(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    relation: dict[str, Any],
    label: str,
) -> list[str]:
    """Hold when every relation of at least one declared alternative holds.

    An alternative is one relation, a list of relations, or a mapping with
    ``relations`` and an optional ``id``; its relations form a conjunction.
    ``file``/``scope`` on the ``any_of`` relation default its alternatives.
    """
    alternatives = relation.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return [f"{label}: any_of requires a non-empty alternatives list"]
    attempts: list[str] = []
    for position, alternative in enumerate(alternatives):
        name = str(position)
        group: Any = alternative
        if isinstance(alternative, dict) and "relations" in alternative:
            name = str(alternative.get("id") or position)
            group = alternative.get("relations")
        elif isinstance(alternative, dict):
            group = [alternative]
        members = (
            [item for item in group if isinstance(item, dict)]
            if isinstance(group, list) else []
        )
        if not members:
            attempts.append(f"[{name}] declares no relations")
            continue
        sub_rule = {**rule, "relations": members}
        for key in ("file", "scope"):
            if key in relation:
                sub_rule[key] = relation[key]
        _, _, sub_failures = _semantic_relation_failures(sub_rule, trees)
        if not sub_failures:
            return []
        attempts.append(f"[{name}] " + "; ".join(sub_failures))
    return [f"{label}: no alternative holds: " + " | ".join(attempts)]


def _check_semantic_relations(
    rule: dict[str, Any],
    trees: dict[str, ast.AST],
    severity: str,
) -> dict[str, Any]:
    total, passed, failures = _semantic_relation_failures(rule, trees)
    return _result(not failures, total, passed, failures, severity)


def _check_structured_cfg(rule, trees, severity):
    return _check_semantic_relations(rule, trees, severity)


def _check_value_flow(rule, trees, severity):
    return _check_semantic_relations(rule, trees, severity)


def _check_shared_callable(rule, trees, severity):
    return _check_semantic_relations(rule, trees, severity)


def _check_collective_protocol(rule, trees, severity):
    return _check_semantic_relations(rule, trees, severity)


def _check_arithmetic_pipeline(rule, trees, severity):
    return _check_semantic_relations(rule, trees, severity)


def evaluate_checks(
    contract: dict[str, Any],
    trees: dict[str, ast.AST],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    evaluators = {
        "regex": _check_regex,
        "regex_sequence": _check_regex_sequence,
        "call_keywords": _check_call_keywords,
        "selected_branch_calls": _check_selected_branch_calls,
        "function_shape": _check_function_shape,
        "tuple_bindings": _check_tuple_bindings,
        "bundle_flow": _check_bundle_flow,
        "call_arity": _check_call_arity,
        "nontrivial_function": _check_nontrivial_function,
        "forbid_methods": _check_forbid_methods,
        "assignment_value": _check_assignment_value,
        "parameter_loads": _check_parameter_loads,
        "publication_placement": _check_publication_placement,
        "owned_completion_protocol": _check_owned_completion_protocol,
        "structured_cfg": _check_structured_cfg,
        "value_flow": _check_value_flow,
        "shared_callable": _check_shared_callable,
        "collective_protocol": _check_collective_protocol,
        "arithmetic_pipeline": _check_arithmetic_pipeline,
    }
    for rule in contract.get("checks") or []:
        check_id = str(rule["id"])
        severity = str(rule.get("severity") or "required")
        category = str(rule.get("category") or "semantic")
        tier = str(rule.get("tier") or DEFAULT_CHECK_TIER)
        kind = str(rule.get("kind") or "regex")
        evaluator = evaluators.get(kind)
        if evaluator is None:
            results[check_id] = _result(
                False, 1, 0, [f"unsupported check kind: {kind}"], severity
            )
        else:
            results[check_id] = evaluator(rule, trees, severity)
        results[check_id]["category"] = category
        results[check_id]["tier"] = tier if tier in CHECK_TIERS else DEFAULT_CHECK_TIER
    return results


_MISSING = object()


def _get_path(value: Any, path: str) -> Any:
    current = value
    for component in str(path).split("."):
        selected = re.fullmatch(r"([^\[]+)\[id=([^\]]+)\]", component)
        key = selected.group(1) if selected else component
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
        if selected:
            if not isinstance(current, list):
                return _MISSING
            matches = [
                item for item in current
                if isinstance(item, dict)
                and str(item.get("id") or "") == selected.group(2)
            ]
            if len(matches) != 1:
                return _MISSING
            current = matches[0]
    return current


def _eval_expr(expr: Any, plan: dict[str, Any]) -> Any:
    if isinstance(expr, dict):
        if set(expr) == {"path"}:
            return _get_path(plan, str(expr["path"]))
        op = str(expr.get("op") or "")
        args = [_eval_expr(item, plan) for item in expr.get("args") or []]
        if any(item is _MISSING for item in args):
            return _MISSING
        if op == "add":
            return sum(args)
        if op == "multiply":
            result = 1
            for item in args:
                result *= item
            return result
        if op == "max":
            return max(args)
        if op == "min":
            return min(args)
        raise ValueError(f"unsupported plan expression op: {op}")
    return expr


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if actual is _MISSING or expected is _MISSING:
        return False
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "gte":
        return actual >= expected
    if op == "lte":
        return actual <= expected
    if op == "gt":
        return actual > expected
    if op == "lt":
        return actual < expected
    if op == "truthy":
        return bool(actual)
    if op == "nonempty":
        return hasattr(actual, "__len__") and len(actual) > 0
    raise ValueError(f"unsupported plan assertion op: {op}")


def validate_plan(
    contract: dict[str, Any],
    plan: dict[str, Any] | None,
) -> tuple[bool | None, list[str], dict[str, bool]]:
    spec = contract.get("plan") or {}
    assertions = spec.get("assertions") or []
    if plan is None:
        if spec.get("required"):
            return False, ["plan is required by the Expert Skill contract"], {}
        return None, [], {}
    errors: list[str] = []
    results: dict[str, bool] = {}
    for assertion in assertions:
        assertion_id = str(assertion.get("id") or assertion.get("path") or "assertion")
        op = str(assertion.get("op") or "eq")
        actual = _eval_expr(
            assertion.get("left", {"path": assertion.get("path")}), plan
        )
        expected = _eval_expr(
            assertion.get("right", assertion.get("value")), plan
        )
        try:
            passed = _compare(actual, op, expected)
        except (TypeError, ValueError) as exc:
            passed = False
            errors.append(f"{assertion_id}: comparison failed: {exc}")
        results[assertion_id] = passed
        if not passed and not any(error.startswith(f"{assertion_id}:") for error in errors):
            errors.append(
                f"{assertion_id}: actual={actual!r} op={op} expected={expected!r}"
            )
    return not errors, errors, results


def _input_evidence(
    contract: dict[str, Any],
    baseline: Path,
    candidate: Path,
    reference: Path | None,
) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    symlinks: list[str] = []
    hardlinks: list[str] = []
    required = [str(value) for value in (contract.get("source") or {}).get("required") or []]
    for label, root in (("baseline", baseline), ("candidate", candidate)):
        if not root.is_dir():
            errors.append(f"{label} root is not a directory: {root}")
            continue
        for name in required:
            if not (root / name).is_file():
                errors.append(f"{label} missing required file: {name}")
    if reference is not None:
        if not reference.is_dir():
            errors.append(f"reference root is not a directory: {reference}")
        else:
            for name in required:
                if not (reference / name).is_file():
                    errors.append(f"reference missing required file: {name}")

    reference_files = collect_files(reference, contract) if reference and reference.is_dir() else {}
    candidate_files = collect_files(candidate, contract) if candidate.is_dir() else {}
    for name, path in candidate_files.items():
        if path.is_symlink():
            symlinks.append(name)
        ref = reference_files.get(name)
        if ref and path.resolve() != ref.resolve():
            try:
                if (
                    path.stat().st_dev == ref.stat().st_dev
                    and path.stat().st_ino == ref.stat().st_ino
                ):
                    hardlinks.append(name)
            except OSError:
                pass

    pins = contract.get("pins") or {}
    if pins.get("enforce"):
        baseline_files = collect_files(baseline, contract) if baseline.is_dir() else {}
        baseline_pin = str(pins.get("baseline_tree_sha256") or "")
        reference_pin = str(pins.get("reference_tree_sha256") or "")
        if baseline_pin and baseline_files and tree_digest(baseline_files) != baseline_pin:
            errors.append(
                f"baseline digest mismatch: {tree_digest(baseline_files)} != {baseline_pin}"
            )
        if reference is not None and reference_pin and reference_files:
            if tree_digest(reference_files) != reference_pin:
                errors.append(
                    f"reference digest mismatch: {tree_digest(reference_files)} != {reference_pin}"
                )
    return errors, sorted(symlinks), sorted(hardlinks)


def _next_blocker_summary(
    check_results: dict[str, dict[str, Any]],
    semantic_failures: list[str],
    surface_failures: list[str],
    plan_errors: list[str],
) -> str:
    """One line naming every failing required check, semantic tier first."""
    parts: list[str] = []
    failing = semantic_failures + surface_failures
    if failing:
        noun = "check" if len(failing) == 1 else "checks"
        parts.append(
            f"{len(failing)} required {noun} failing "
            f"({len(semantic_failures)} semantic, {len(surface_failures)} surface)."
        )
        groups = [
            f"{tier}: {', '.join(ids)}"
            for tier, ids in (("semantic", semantic_failures), ("surface", surface_failures))
            if ids
        ]
        parts.append("; ".join(groups) + ".")
        messages = [
            f"{check_id}: {failure}"
            for check_id in failing
            for failure in check_results[check_id]["failures"]
        ]
        shown = [
            message if len(message) <= NEXT_BLOCKER_DETAIL_CHARS
            else message[:NEXT_BLOCKER_DETAIL_CHARS - 3] + "..."
            for message in messages[:NEXT_BLOCKER_DETAIL_LIMIT]
        ]
        if shown:
            omitted = len(messages) - len(shown)
            parts.append(
                "details: " + "; ".join(shown)
                + (f"; (+{omitted} more unit failures)" if omitted else "")
            )
    if plan_errors:
        parts.append(f"plan errors ({len(plan_errors)}): " + "; ".join(plan_errors))
    return " ".join(parts)


def evaluate(
    contract: dict[str, Any],
    baseline: Path,
    candidate: Path,
    reference: Path | None = None,
    authoring_manifest: dict[str, Any] | None = None,
    plan_ir: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_errors, candidate_symlinks, reference_hardlinks = _input_evidence(
        contract, baseline, candidate, reference
    )
    base_files = collect_files(baseline, contract) if baseline.is_dir() else {}
    cand_files = collect_files(candidate, contract) if candidate.is_dir() else {}
    ref_files = (
        collect_files(reference, contract)
        if reference is not None and reference.is_dir()
        else {}
    )
    base_trees, base_parse_errors = _parse_files(base_files)
    cand_trees, cand_parse_errors = _parse_files(cand_files)
    ref_trees, ref_parse_errors = _parse_files(ref_files)

    check_results = evaluate_checks(contract, cand_trees)
    plan_consistent, plan_errors, plan_results = validate_plan(contract, plan_ir)
    required_check_failures = [
        check_id for check_id, result in check_results.items()
        if result["severity"] == "required" and not result["pass"]
    ]
    forbidden_markers = [str(value) for value in contract.get("forbidden_markers") or []]
    candidate_code = "\n".join(_active_code(tree) for tree in cand_trees.values())
    forbidden_present = [
        marker for marker in forbidden_markers if marker in candidate_code
    ]

    all_names = sorted(set(base_files) | set(cand_files) | set(ref_files))
    changed_by_candidate = [
        name for name in all_names
        if _digest(base_files.get(name)) != _digest(cand_files.get(name))
    ]
    changed_by_reference = [
        name for name in all_names
        if ref_files and _digest(base_files.get(name)) != _digest(ref_files.get(name))
    ]
    exact_reference_files = [
        name for name in all_names
        if ref_files and _digest(ref_files.get(name)) == _digest(cand_files.get(name))
    ]

    required_symbols: dict[str, list[str]] = {}
    missing_symbols: dict[str, list[str]] = {}
    if ref_trees:
        for name, ref_tree in ref_trees.items():
            base_symbols = _qualified_functions(base_trees[name]) if name in base_trees else set()
            new_symbols = sorted(_qualified_functions(ref_tree) - base_symbols)
            if not new_symbols:
                continue
            required_symbols[name] = new_symbols
            candidate_symbols = _qualified_functions(cand_trees[name]) if name in cand_trees else set()
            missing = sorted(set(new_symbols) - candidate_symbols)
            if missing:
                missing_symbols[name] = missing

    core_files = [
        str(value) for value in (contract.get("source") or {}).get("core_identity") or []
    ]
    token_similarity: dict[str, float] = {}
    baseline_similarity: dict[str, float] = {}
    for name in core_files:
        if name not in ref_files or name not in cand_files:
            continue
        reference_tokens = _normalized_tokens(ref_files[name])
        candidate_tokens = _normalized_tokens(cand_files[name])
        baseline_tokens = _normalized_tokens(base_files[name]) if name in base_files else ()
        token_similarity[name] = _shingle_similarity(reference_tokens, candidate_tokens)
        baseline_similarity[name] = _shingle_similarity(reference_tokens, baseline_tokens)
    similarity_mean = (
        sum(token_similarity.values()) / len(token_similarity)
        if token_similarity else 0.0
    )
    baseline_similarity_mean = (
        sum(baseline_similarity.values()) / len(baseline_similarity)
        if baseline_similarity else 0.0
    )

    core_exact = sorted(set(core_files) & set(exact_reference_files))
    core_ast_exact = sorted(
        name for name in core_files
        if name in ref_trees
        and name in cand_trees
        and ast.dump(ref_trees[name], include_attributes=False)
        == ast.dump(cand_trees[name], include_attributes=False)
    )
    ref_function_dumps = set().union(
        *(_function_dumps(tree) for tree in ref_trees.values())
    ) if ref_trees else set()
    base_function_dumps = set().union(
        *(_function_dumps(tree) for tree in base_trees.values())
    ) if base_trees else set()
    cand_function_dumps = set().union(
        *(_function_dumps(tree) for tree in cand_trees.values())
    ) if cand_trees else set()
    reference_exclusive_functions = ref_function_dumps - base_function_dumps
    cloned_reference_functions = reference_exclusive_functions & cand_function_dumps
    clone_ratio = (
        len(cloned_reference_functions) / len(reference_exclusive_functions)
        if reference_exclusive_functions else 0.0
    )
    copy_policy = contract.get("copy_detection") or {}
    exact_threshold = float(copy_policy.get("exact_similarity") or 0.985)
    suspect_threshold = float(copy_policy.get("suspect_similarity") or 0.90)
    clone_exact = float(copy_policy.get("exact_function_ratio") or 0.80)
    clone_suspect = float(copy_policy.get("suspect_function_ratio") or 0.50)
    reference_copy_detected = bool(ref_files) and (
        set(core_files).issubset(exact_reference_files)
        or set(core_files).issubset(core_ast_exact)
        or (
            len(token_similarity) == len(core_files)
            and similarity_mean >= exact_threshold
            and similarity_mean >= baseline_similarity_mean + 0.02
        )
        or (
            len(reference_exclusive_functions) >= 3
            and clone_ratio >= clone_exact
        )
    )
    reference_copy_suspected = bool(ref_files) and not reference_copy_detected and (
        (
            len(token_similarity) == len(core_files)
            and similarity_mean >= suspect_threshold
            and similarity_mean >= baseline_similarity_mean + 0.05
        )
        or (
            len(reference_exclusive_functions) >= 4
            and clone_ratio >= clone_suspect
        )
    )

    current_digest = tree_digest(cand_files) if cand_files else ""
    manifest = authoring_manifest if isinstance(authoring_manifest, dict) else {}
    manifest_digest = str(manifest.get("candidate_tree_digest") or "")
    sealed = manifest.get("sealed_before_reference_comparison")
    reference_hidden = manifest.get("reference_exposed_during_authoring")
    digest_matches = bool(manifest_digest) and manifest_digest == current_digest
    require_manifest = bool((contract.get("provenance") or {}).get("require_manifest", True))
    provenance_attestation_valid = (
        (not require_manifest or (
            sealed is True
            and (reference is None or reference_hidden is False)
            and digest_matches
        ))
        and not candidate_symlinks
        and not reference_hardlinks
    )

    structural_compatible = (
        not input_errors
        and not cand_parse_errors
        and not ref_parse_errors
        and not required_check_failures
        and not forbidden_present
        and plan_consistent is not False
    )
    structural_exact = bool(ref_files) and (
        not input_errors
        and not base_parse_errors
        and not cand_parse_errors
        and not ref_parse_errors
        and set(all_names).issubset(exact_reference_files)
    )
    independent_structure_pass = (
        structural_compatible
        and not reference_copy_detected
        and not reference_copy_suspected
        and not candidate_symlinks
        and not reference_hardlinks
    )
    capability_eligible = independent_structure_pass and provenance_attestation_valid
    verdict = (
        "calibration_only"
        if structural_exact and reference_copy_detected
        else "compatible"
        if structural_compatible
        else "incomplete"
    )
    if reference_copy_detected:
        provenance_status = "copy_detected"
    elif reference_copy_suspected:
        provenance_status = "copy_suspected"
    elif provenance_attestation_valid:
        provenance_status = "attested_unverified"
    else:
        provenance_status = "unverified"

    units_passed = sum(result["units_passed"] for result in check_results.values())
    units_total = sum(result["units_total"] for result in check_results.values())
    semantic_failures = [
        check_id for check_id in required_check_failures
        if check_results[check_id].get("tier", DEFAULT_CHECK_TIER) == "semantic"
    ]
    surface_failures = [
        check_id for check_id in required_check_failures
        if check_results[check_id].get("tier", DEFAULT_CHECK_TIER) == "surface"
    ]
    contract_failures = [
        {
            "id": check_id,
            "category": result.get("category", "semantic"),
            "tier": result.get("tier", DEFAULT_CHECK_TIER),
            "severity": result["severity"],
            "messages": list(result["failures"]),
        }
        for check_id, result in check_results.items()
        if result["severity"] == "required" and not result["pass"]
    ]
    contract_failures.extend({
        "id": error.split(":", 1)[0],
        "category": "plan",
        "tier": DEFAULT_CHECK_TIER,
        "severity": "required",
        "messages": [error],
    } for error in plan_errors)
    next_blocker = _next_blocker_summary(
        check_results, semantic_failures, surface_failures, plan_errors
    )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "contract_schema_version": contract.get("schema_version"),
        "skill_id": contract.get("skill_id"),
        "contract_revision": contract.get("revision", ""),
        "contract_sha256": hashlib.sha256(
            yaml.safe_dump(contract, sort_keys=True).encode()
        ).hexdigest(),
        "verdict": verdict,
        "structural_exact": structural_exact,
        "structural_compatible": structural_compatible,
        "independent_structure_pass": independent_structure_pass,
        "capability_eligible": capability_eligible,
        "reference_copy_detected": reference_copy_detected,
        "reference_copy_suspected": reference_copy_suspected,
        "provenance_status": provenance_status,
        "provenance_attestation_valid": provenance_attestation_valid,
        "candidate_tree_digest": current_digest,
        "manifest_tree_digest": manifest_digest,
        "sealed_before_reference_comparison": sealed,
        "reference_exposed_during_authoring": reference_hidden,
        "input_errors": input_errors,
        "candidate_symlinks": candidate_symlinks,
        "reference_hardlinks": reference_hardlinks,
        "plan_consistent": plan_consistent,
        "plan_consistency_errors": plan_errors,
        "plan_assertions": plan_results,
        "semantic_features_passed": units_passed,
        "semantic_features_total": units_total,
        "semantic_feature_ratio": units_passed / units_total if units_total else 0.0,
        "checks": check_results,
        "contract_failures": contract_failures,
        "failed_required_checks": required_check_failures,
        "semantic_failures": semantic_failures,
        "surface_failures": surface_failures,
        "forbidden_markers_present": forbidden_present,
        "baseline_tree_digest": tree_digest(base_files) if base_files else "",
        "reference_tree_digest": tree_digest(ref_files) if ref_files else "",
        "reference_revision": (contract.get("pins") or {}).get("reference_revision", ""),
        "candidate_changed_files": changed_by_candidate,
        "reference_changed_files": changed_by_reference,
        "required_changed_files_missing": sorted(
            set(changed_by_reference) - set(changed_by_candidate)
        ),
        "exact_reference_files": exact_reference_files,
        "core_reference_identity_files": core_exact,
        "core_reference_ast_identity_files": core_ast_exact,
        "core_token_similarity": token_similarity,
        "core_token_similarity_mean": similarity_mean,
        "baseline_to_reference_similarity_mean": baseline_similarity_mean,
        "reference_exclusive_function_count": len(reference_exclusive_functions),
        "cloned_reference_function_count": len(cloned_reference_functions),
        "cloned_reference_function_ratio": clone_ratio,
        "required_new_symbols": required_symbols,
        "missing_new_symbols": missing_symbols,
        "parse_errors": {
            "baseline": base_parse_errors,
            "candidate": cand_parse_errors,
            "reference": ref_parse_errors,
        },
        "hardware_verified": False,
        "gpu_executed": False,
        "device_jit_verified": False,
        "path_activation_verified": False,
        "launch_count_verified": False,
        "accuracy_verified": False,
        "numeric_accuracy_verified": False,
        "graph_liveness_verified": False,
        "residency_verified": False,
        "distributed_memory_order_verified": False,
        "performance_verified": False,
        "next_blocker": next_blocker,
        "note": (
            "Declarative source/PlanIR preflight only. A passing contract cannot prove "
            "device compilation, numerical correctness, launch count, distributed liveness, "
            "residency, or performance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--authoring-manifest", type=Path)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--digest-only", action="store_true")
    parser.add_argument(
        "--require",
        choices=("report", "compatible", "independent", "exact"),
        default="report",
    )
    args = parser.parse_args()
    contract = load_contract(args.contract.resolve())
    if args.digest_only:
        files = collect_files(args.candidate.resolve(), contract)
        print(tree_digest(files))
        return 0
    if args.baseline is None:
        parser.error("--baseline is required unless --digest-only is used")
    manifest = (
        json.loads(args.authoring_manifest.read_text())
        if args.authoring_manifest else None
    )
    plan_ir = json.loads(args.plan_json.read_text()) if args.plan_json else None
    result = evaluate(
        contract=contract,
        baseline=args.baseline.resolve(),
        candidate=args.candidate.resolve(),
        reference=args.reference.resolve() if args.reference else None,
        authoring_manifest=manifest,
        plan_ir=plan_ir,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    if args.require == "exact":
        return 0 if result["structural_exact"] else 1
    if args.require == "compatible":
        return 0 if result["structural_compatible"] else 1
    if args.require == "independent":
        return 0 if result["capability_eligible"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
