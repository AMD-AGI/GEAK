#!/usr/bin/env python3
"""Ground-truth shape extractor for GEAK v3 preprocess test runner.

Reads ``scripts/task_runner.py`` from each AKA HIP kernel task directory
and extracts the canonical ``TEST_SHAPES`` list used by ``--correctness``
and ``--performance`` modes. The list is the *oracle* the runner diffs
the harness-generated shape list against (Path B coverage assertion).

The extractor is deliberately AST-based (not ``eval``-based) so that:

* It refuses to execute arbitrary code from the AKA repo (a string-eval
  approach would happily import torch + run device code at parse time).
* It works whether the file is importable in the current environment or
  not (some AKA tasks import torch/triton/etc. unconditionally at module
  scope; an importing oracle would either need those deps or skip).

Public API (kept tiny — the runner imports just two functions):

* :func:`extract_test_shapes` — given a task directory, returns the
  ``TEST_SHAPES`` list as a list of tuples (or empty list when the file
  is missing / the assignment can't be parsed).
* :func:`load_oracle_for_kernel` — convenience wrapper used by the
  runner; takes a kernel YAML entry and returns ``{shapes, label,
  source_path}``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


def _to_python_value(node: ast.AST) -> Any:
    """Coerce a literal-only AST node tree into a Python value.

    Supports ints, floats, strings, bools, ``None``, lists, and tuples
    (the only constructs that appear in the AKA ``TEST_SHAPES``
    declarations we care about). Anything else raises ``ValueError`` —
    we never want this oracle to run user code, even by accident.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _to_python_value(node.operand)
        if not isinstance(inner, (int, float)):
            raise ValueError(f"unary minus on non-numeric: {inner!r}")
        return -inner
    if isinstance(node, ast.Tuple):
        return tuple(_to_python_value(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_to_python_value(elt) for elt in node.elts]
    raise ValueError(f"Unsupported AST node for literal extraction: {type(node).__name__}")


def _find_assignment(tree: ast.Module, name: str) -> ast.AST | None:
    """Find the first top-level ``name = <literal>`` assignment in ``tree``.

    AKA's ``TEST_SHAPES`` lives at module scope; a simple top-level walk
    is enough. We deliberately don't recurse into functions / classes —
    if a runner moves the constant into a function body, this oracle
    should fail loudly so the test plan author knows to update.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
                return node.value
    return None


def extract_test_shapes(task_dir: str | Path) -> list[tuple]:
    """Extract ``TEST_SHAPES`` from ``<task_dir>/scripts/task_runner.py``.

    Returns the list of shape tuples on success, an empty list on any
    failure (missing file, missing assignment, parse error). The runner
    treats an empty oracle as "no ground truth available" and downgrades
    the coverage assertion to a warning.
    """
    task_path = Path(task_dir)
    runner_py = task_path / "scripts" / "task_runner.py"
    if not runner_py.is_file():
        return []
    try:
        tree = ast.parse(runner_py.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    value_node = _find_assignment(tree, "TEST_SHAPES")
    if value_node is None:
        return []
    try:
        value = _to_python_value(value_node)
    except ValueError:
        return []

    if not isinstance(value, list):
        return []
    out: list[tuple] = []
    for item in value:
        if isinstance(item, tuple):
            out.append(item)
        elif isinstance(item, list):
            out.append(tuple(item))
        else:
            out.append((item,))
    return out


def shape_signature(shape: tuple) -> tuple:
    """Reduce a shape tuple to its integer dim-prefix for set-membership checks.

    The AKA ``TEST_SHAPES`` rows mix integer dims with kernel-specific
    floats (e.g. ``ball_query``'s ``max_radius``) and ints (``nsample``).
    The harness-generator may legitimately emit only the integer dims
    (B, N, M) and drop the kernel-config columns. Matching the integer
    prefix is a relaxed but principled containment check: if the harness
    covers ``(2, 256, 32)`` we treat that as covering the oracle row
    ``(2, 256, 32, 1.0, 5)``.

    The signature is the longest leading run of ``int`` values. If the
    row is all ints, the signature is the whole row.
    """
    sig: list[int] = []
    for elem in shape:
        if isinstance(elem, bool):
            break
        if isinstance(elem, int):
            sig.append(elem)
        else:
            break
    return tuple(sig)


def load_oracle_for_kernel(kernel_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve a kernel YAML entry to its ground-truth shapes.

    ``kernel_cfg`` keys consumed:

    * ``repo_path`` — root of AgentKernelArena (or per-task root).
    * ``task_dir_relative`` — relative path from ``repo_path`` to the
      kernel's task directory (``tasks/hip2hip/others/<name>``).
    * ``ground_truth_extractor`` — currently only ``aka_others`` is
      supported; reserved for ``aka_gpumode`` / ``aiter`` later.

    Returns a dict ``{shapes, signatures, label, source_path}`` with the
    raw shape tuples + their integer-prefix signatures + a human-readable
    label + the absolute path of the source file the shapes came from.
    """
    extractor = kernel_cfg.get("ground_truth_extractor", "aka_others")
    repo_path = Path(kernel_cfg["repo_path"]).expanduser().resolve()
    task_dir_rel = kernel_cfg.get("task_dir_relative") or kernel_cfg.get("task_dir") or ""
    task_dir = (repo_path / task_dir_rel).resolve() if task_dir_rel else repo_path

    if extractor != "aka_others":
        return {
            "shapes": [],
            "signatures": [],
            "label": f"<unsupported extractor: {extractor}>",
            "source_path": "",
        }

    shapes = extract_test_shapes(task_dir)
    sigs = [shape_signature(s) for s in shapes]
    return {
        "shapes": [list(s) for s in shapes],
        "signatures": [list(s) for s in sigs],
        "label": kernel_cfg.get("name", task_dir.name),
        "source_path": str(task_dir / "scripts" / "task_runner.py"),
    }


def _cli() -> int:
    """Tiny CLI for spot-checking: ``oracle.py <task_dir>`` prints JSON."""
    parser = argparse.ArgumentParser(description="Extract TEST_SHAPES from an AKA task directory.")
    parser.add_argument("task_dir", help="Path to the AKA task directory (contains scripts/task_runner.py).")
    args = parser.parse_args()

    shapes = extract_test_shapes(args.task_dir)
    sigs = [shape_signature(s) for s in shapes]
    print(json.dumps({"shapes": [list(s) for s in shapes], "signatures": [list(s) for s in sigs]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
