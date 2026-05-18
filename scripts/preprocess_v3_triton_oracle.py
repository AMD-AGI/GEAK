#!/usr/bin/env python3
"""Ground-truth shape extractor for GEAK v3 preprocess Triton sweep.

Triton equivalent of :mod:`preprocess_v3_oracle` (which serves the HIP
sweep). Three extractor strategies are supported, mapped one-to-one to
the three Triton kernels in scope:

* ``aiter_parametrize`` — read an aiter ``op_tests/triton_tests/...``
  pytest file, find a target test function, decode its
  ``@pytest.mark.parametrize`` stack into a list of shape dicts. Used
  for ``aiter_topk`` (direct test of the kernel) and ``aiter_mla_decode``
  (indirect — reuses ``test_op_fwd_rope_integration`` from the rope
  test file as the shape source per the test plan).
* ``aka_module_constant`` — read an AKA-style ``test_kernel_harness.py``
  and extract a module-scope shape constant list (e.g. ``ALL_SHAPES``,
  ``HARNESS_SHAPES``). Used for ``aka_gemm_a16wfp4``.

Both strategies are AST-only — we never ``exec`` / ``import`` the
target file. The aiter parametrize decorator stack can mix two forms:

* Single-value form: ``@pytest.mark.parametrize("dtype", [torch.float32])``
* Multi-value form:  ``@pytest.mark.parametrize("B, H, S", [(...), ...])``

Both are flattened to a list of ``dict[name -> value]`` and then the
Cartesian product across decorators is taken. ``torch.*`` constants
(e.g. ``torch.bfloat16``) are kept as the string ``"torch.bfloat16"``
so the oracle JSON stays purely textual.

Public API (kept tiny — the runner imports just two functions):

* :func:`extract_oracle_shapes` — given a kernel YAML entry, return a
  ``list[tuple]`` of shape signatures (integer prefix, signature-style,
  matching the HIP oracle's :func:`shape_signature`).
* :func:`load_oracle_for_kernel` — convenience wrapper used by the
  runner; returns ``{shapes, signatures, label, source_path}``.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Literal coercion — pure-AST, never executes user code.
# ---------------------------------------------------------------------------


def _to_literal(node: ast.AST) -> Any:
    """Best-effort projection of an AST node to a JSON-friendly Python value.

    Supports ints, floats, strings, bools, ``None``, lists, tuples, and
    dotted attribute chains (``torch.bfloat16`` becomes the string
    ``"torch.bfloat16"``). Anything else returns the sentinel string
    ``"<unparsed:NodeName>"`` so the caller can decide whether to keep
    the row or drop it.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _to_literal(node.operand)
        if isinstance(inner, (int, float)):
            return -inner
        return f"<unparsed:UnaryOp({inner!r})>"
    if isinstance(node, ast.Tuple):
        return tuple(_to_literal(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_to_literal(e) for e in node.elts]
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        chain: list[str] = []
        cursor: ast.AST = node
        while isinstance(cursor, ast.Attribute):
            chain.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            chain.append(cursor.id)
            return ".".join(reversed(chain))
        return f"<unparsed:Attribute>"
    if isinstance(node, ast.Call):
        # Common case: ``torch.tensor(...)`` or ``torch.float32``. Surface the
        # callee chain so the harness comparison stays meaningful.
        func = _to_literal(node.func)
        return f"<call:{func}>"
    return f"<unparsed:{type(node).__name__}>"


def _resolve_name(tree: ast.Module, name: str) -> Any:
    """Resolve a top-level ``name = <literal>`` assignment to its value.

    Returns ``None`` if the name isn't bound at module scope to a
    literal we can decode. Used so aiter parametrize decorators can
    reference module-scope lists (``BATCH_SIZES = [...]``, etc.).
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _to_literal(node.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == name
                and node.value is not None
            ):
                return _to_literal(node.value)
    return None


# ---------------------------------------------------------------------------
# aiter parametrize-stack extractor
# ---------------------------------------------------------------------------


def _split_param_names(spec: str) -> list[str]:
    """Split a parametrize first-arg string into a list of names.

    Handles both ``"dtype"`` (single name) and ``"B, H, S, kv_lora_rank"``
    (multi-name, comma-separated). Whitespace tolerant.
    """
    return [s.strip() for s in spec.split(",") if s.strip()]


def _extract_parametrize_decorators(
    func_node: ast.FunctionDef, tree: ast.Module
) -> list[tuple[list[str], list[Any]]]:
    """Pull every ``pytest.mark.parametrize`` decorator off ``func_node``.

    Returns a list of ``(names, values_list)`` tuples in *decorator
    order* (top to bottom in source, which is how pytest itself
    interprets them). ``values_list`` is the decoded second argument,
    possibly resolved via :func:`_resolve_name` if it was a name
    reference. Decorators that aren't ``parametrize`` (e.g.
    ``pytest.mark.skip``) are ignored.
    """
    out: list[tuple[list[str], list[Any]]] = []
    for dec in func_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        # Match pytest.mark.parametrize  AND  bare parametrize alias.
        is_param = False
        if isinstance(func, ast.Attribute) and func.attr == "parametrize":
            is_param = True
        elif isinstance(func, ast.Name) and func.id == "parametrize":
            is_param = True
        if not is_param or len(dec.args) < 2:
            continue
        spec_node, values_node = dec.args[0], dec.args[1]
        if isinstance(spec_node, ast.Constant) and isinstance(spec_node.value, str):
            names = _split_param_names(spec_node.value)
        else:
            continue
        # Values may be an inline list/tuple, or a Name that resolves to one.
        if isinstance(values_node, ast.Name):
            resolved = _resolve_name(tree, values_node.id)
            if resolved is None:
                continue
            values = resolved if isinstance(resolved, (list, tuple)) else [resolved]
        else:
            values = _to_literal(values_node)
            if not isinstance(values, (list, tuple)):
                continue
        out.append((names, list(values)))
    return out


def _row_to_dict(names: list[str], row: Any) -> dict[str, Any]:
    """Map a parametrize row to a ``{name: value}`` dict.

    Single-name decorators bind ``names = ["x"]`` and ``row = <scalar>``.
    Multi-name decorators bind ``names = ["B", "H", ...]`` and
    ``row = (b, h, ...)`` (or a list).
    """
    if len(names) == 1:
        return {names[0]: row}
    if isinstance(row, (list, tuple)) and len(row) == len(names):
        return dict(zip(names, row, strict=True))
    return {}


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def extract_aiter_parametrize(
    test_file: str | Path, test_function: str
) -> list[dict[str, Any]]:
    """Return the cross-product of ``parametrize`` rows for one test func.

    Each output element is a dict ``{param_name: value}``. The set of
    keys per element is determined by the union of names across every
    decorator on the function, so a 4-decorator stack of
    ``(BATCH_SIZES) x (DIM2) x (K) x (FLOAT_DTYPES)`` yields
    ``len(BATCH_SIZES) * len(DIM2) * len(K) * len(FLOAT_DTYPES)`` dicts
    each with keys ``{batch_size, hiddensize, topk, dtype}``.
    """
    path = Path(test_file)
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    func = _find_function(tree, test_function)
    if func is None:
        return []
    decorators = _extract_parametrize_decorators(func, tree)
    if not decorators:
        return []

    per_decorator_dicts: list[list[dict[str, Any]]] = []
    for names, rows in decorators:
        rendered: list[dict[str, Any]] = []
        for row in rows:
            d = _row_to_dict(names, row)
            if d:
                rendered.append(d)
        if rendered:
            per_decorator_dicts.append(rendered)

    if not per_decorator_dicts:
        return []
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*per_decorator_dicts):
        merged: dict[str, Any] = {}
        for d in combo:
            merged.update(d)
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# AKA module-constant extractor (test_kernel_harness.py shape list).
# ---------------------------------------------------------------------------


_AKA_SHAPE_NAMES = (
    "ALL_SHAPES",
    "HARNESS_SHAPES",
    "TEST_SHAPES",
    "BENCHMARK_SHAPES",
    "SHAPES",
    "DEFAULT_SHAPES",
)


def extract_aka_shape_list(
    harness_path: str | Path, prefer: tuple[str, ...] = _AKA_SHAPE_NAMES
) -> list[tuple]:
    """Return the first module-scope shape list found in ``harness_path``.

    Tries the names in ``prefer`` order. The matched assignment must
    decode to a list-of-tuples (or list-of-lists, which we coerce to
    tuples). Returns an empty list on failure so the runner can treat
    "no oracle" as a soft warning rather than a hard fail.
    """
    p = Path(harness_path)
    if not p.is_file():
        return []
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    for name in prefer:
        value = _resolve_name(tree, name)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            continue
        out: list[tuple] = []
        for item in value:
            if isinstance(item, tuple):
                out.append(item)
            elif isinstance(item, list):
                out.append(tuple(item))
        if out:
            return out
    return []


# ---------------------------------------------------------------------------
# Shape signatures — match the HIP oracle so the runner's coverage check
# code can stay identical.
# ---------------------------------------------------------------------------


def shape_signature(shape: tuple) -> tuple:
    """Reduce a shape tuple to its integer dim-prefix for membership checks.

    Identical contract to :func:`preprocess_v3_oracle.shape_signature`:
    walk left-to-right, collect a contiguous prefix of *int* elements
    (excluding ``bool`` since ``bool <: int`` in Python and we treat
    booleans as "non-dim" config flags). Floats / strings / etc.
    terminate the prefix.
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


# ---------------------------------------------------------------------------
# Per-kernel rendering: dict-of-params -> tuple of dim ints.
# ---------------------------------------------------------------------------


def _render_topk_shape(row: dict[str, Any]) -> tuple[int, ...]:
    """``aiter_topk`` shape = (batch_size, hiddensize, topk).

    ``largest`` and ``dtype`` aren't dim ints; they're configuration so
    we drop them from the signature. ``shape_signature`` would also drop
    them if we kept them in the suffix, but constructing the dim tuple
    here keeps the harness-vs-oracle comparison crisper.
    """
    b = row.get("batch_size")
    n = row.get("hiddensize")
    k = row.get("topk")
    if not (isinstance(b, int) and isinstance(n, int) and isinstance(k, int)):
        return ()
    return (b, n, k)


def _render_mla_decode_shape(row: dict[str, Any]) -> tuple[int, ...]:
    """``aiter_mla_decode`` shape = (B, H, S, head_dim).

    The rope test file's tuples are
    ``(B, H, S, kv_lora_rank, qk_rope_head_dim, rotary_dim)``. The plain
    mla_decode signature exposes a single ``head_dim`` (= kv_lora_rank +
    qk_rope_head_dim), so we fold those two columns together per the
    test plan's instructions and drop ``rotary_dim`` (rope-only).
    """
    b = row.get("B")
    h = row.get("H")
    s = row.get("S")
    kvl = row.get("kv_lora_rank")
    qkr = row.get("qk_rope_head_dim")
    if not all(isinstance(v, int) for v in (b, h, s, kvl, qkr)):
        return ()
    return (b, h, s, kvl + qkr)


def _render_aka_gemm_shape(shape: tuple) -> tuple[int, ...]:
    """``aka_gemm_a16wfp4`` shape = (M, N, K), already integer-only."""
    if not all(isinstance(v, int) for v in shape):
        return ()
    return tuple(shape)


# ---------------------------------------------------------------------------
# Top-level oracle loader.
# ---------------------------------------------------------------------------


def extract_oracle_shapes(kernel_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve a kernel YAML entry to its ground-truth shapes + signatures.

    Dispatch on ``ground_truth_extractor``:

    * ``aiter_parametrize`` — keys consumed:
        - ``shape_source`` — absolute path to the aiter test file
        - ``shape_function`` — name of the test function to read
        - ``shape_renderer`` — one of ``topk``, ``mla_decode``
    * ``aka_module_constant`` — keys consumed:
        - ``shape_source`` — absolute path to test_kernel_harness.py
        - ``shape_renderer`` — currently only ``aka_gemm``
        - optional ``shape_list_name`` — override the auto-discovered list
    """
    extractor = kernel_cfg.get("ground_truth_extractor", "")
    src = kernel_cfg.get("shape_source")
    renderer = kernel_cfg.get("shape_renderer")
    if not src:
        return _empty_oracle(kernel_cfg, "missing shape_source")

    if extractor == "aiter_parametrize":
        func = kernel_cfg.get("shape_function")
        if not func:
            return _empty_oracle(kernel_cfg, "missing shape_function")
        rows = extract_aiter_parametrize(src, func)
        if renderer == "topk":
            rendered = [_render_topk_shape(r) for r in rows]
        elif renderer == "mla_decode":
            rendered = [_render_mla_decode_shape(r) for r in rows]
        else:
            return _empty_oracle(kernel_cfg, f"unknown renderer: {renderer!r}")
        # Drop empty tuples (rows we couldn't render to all-ints) and
        # uniqueify so a 4-decorator cross-product with redundant dtype
        # rows collapses to its dim-distinct entries.
        uniq: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for r in rendered:
            if r and r not in seen:
                uniq.append(r)
                seen.add(r)
        sigs = [shape_signature(r) for r in uniq]
        return {
            "shapes": [list(s) for s in uniq],
            "signatures": [list(s) for s in sigs],
            "label": kernel_cfg.get("name", "<aiter>"),
            "source_path": str(src),
            "row_count": len(rows),
            "rendered_count": len(uniq),
        }

    if extractor == "aka_module_constant":
        preferred = (
            (kernel_cfg["shape_list_name"],) if kernel_cfg.get("shape_list_name") else _AKA_SHAPE_NAMES
        )
        raw_shapes = extract_aka_shape_list(src, preferred)
        if renderer == "aka_gemm":
            rendered = [_render_aka_gemm_shape(s) for s in raw_shapes]
        else:
            rendered = [tuple(s) for s in raw_shapes]
        uniq: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for r in rendered:
            if r and r not in seen:
                uniq.append(r)
                seen.add(r)
        sigs = [shape_signature(r) for r in uniq]
        return {
            "shapes": [list(s) for s in uniq],
            "signatures": [list(s) for s in sigs],
            "label": kernel_cfg.get("name", "<aka>"),
            "source_path": str(src),
            "row_count": len(raw_shapes),
            "rendered_count": len(uniq),
        }

    return _empty_oracle(kernel_cfg, f"unknown extractor: {extractor!r}")


def _empty_oracle(kernel_cfg: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "shapes": [],
        "signatures": [],
        "label": kernel_cfg.get("name", "<unknown>"),
        "source_path": str(kernel_cfg.get("shape_source", "")),
        "row_count": 0,
        "rendered_count": 0,
        "error": reason,
    }


def load_oracle_for_kernel(kernel_cfg: dict[str, Any]) -> dict[str, Any]:
    """Public entry-point used by ``preprocess_v3_triton_runner``."""
    return extract_oracle_shapes(kernel_cfg)


# ---------------------------------------------------------------------------
# CLI for spot-checking.
# ---------------------------------------------------------------------------


def _cli() -> int:
    """``oracle.py --kernel aiter_topk`` (or supply a YAML kernel entry as JSON)."""
    parser = argparse.ArgumentParser(
        description="Extract oracle shapes for a Triton kernel."
    )
    parser.add_argument(
        "--kernel-json",
        type=str,
        required=True,
        help="JSON-encoded kernel YAML entry (or @path/to/file.json).",
    )
    args = parser.parse_args()
    raw = args.kernel_json
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    kernel_cfg = json.loads(raw)
    oracle = load_oracle_for_kernel(kernel_cfg)
    print(json.dumps(oracle, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
