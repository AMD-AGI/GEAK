"""Shape-string / dtype helpers shared by the deterministic harness synthesizer.

TraceLens captures per-argument shapes as strings like ``"(1073,3072) fp8"`` or
``"(24960,) int"``. To build a runnable harness from a callable + these captured
shapes, we need (1) a dtype-token -> ``torch.dtype`` map and (2) a parser that
turns the captured string into ``(tuple_of_dims, torch.dtype)``. This module is
intentionally tiny, op-agnostic, and free of any kernel/model-specific logic: it
is the only new low-level primitive the synthesizer depends on.

Kept dependency-light (only ``torch``) so it imports cleanly in preprocess.
"""
from __future__ import annotations

import re
from typing import Any

# Canonical TraceLens / aiter dtype tokens -> torch dtype attribute names.
# Resolved lazily (getattr on torch) so this module imports even if a given
# torch build lacks an exotic fp8 variant; unknown tokens fall back to bf16.
_DTYPE_TOKEN_TO_TORCH_ATTR: dict[str, str] = {
    "fp8": "float8_e4m3fnuz",        # AMD/CDNA fnuz is the on-box fp8; matches aiter
    "fp8_e4m3": "float8_e4m3fn",
    "fp8_e4m3fnuz": "float8_e4m3fnuz",
    "fp8_e5m2": "float8_e5m2",
    "fp8_e5m2fnuz": "float8_e5m2fnuz",
    "float8": "float8_e4m3fnuz",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
    "fp64": "float64",
    "float64": "float64",
    "int": "int32",
    "int32": "int32",
    "int64": "int64",
    "long": "int64",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}

_INT_TOKENS = {"int", "int32", "int64", "long", "int16", "int8", "uint8", "bool"}

# ``(1073,3072) fp8`` / ``(24960,) int`` / ``() ScalarList`` / ``(64, 12, 128) bf16``
_SHAPE_RE = re.compile(r"\(\s*([0-9,\s]*)\)\s*([A-Za-z0-9_]*)")


def torch_dtype_attr(token: str) -> str:
    """Return the ``torch.<attr>`` name for a TraceLens dtype token (bf16 default)."""
    return _DTYPE_TOKEN_TO_TORCH_ATTR.get((token or "").strip().lower(), "bfloat16")


def is_int_token(token: str) -> bool:
    """True when the dtype token denotes an integer/index tensor."""
    return (token or "").strip().lower() in _INT_TOKENS


def parse_shape_token(raw: str) -> tuple[tuple[int, ...], str] | None:
    """Parse one captured arg string into ``(dims, dtype_token)``.

    ``"(1073,3072) fp8"`` -> ``((1073, 3072), "fp8")``;
    ``"(24960,) int"`` -> ``((24960,), "int")``;
    ``"() ScalarList"`` / unparenthesized scalars -> ``((), token)``.
    Returns ``None`` when the string carries no parseable shape.
    """
    if not raw or not isinstance(raw, str):
        return None
    m = _SHAPE_RE.search(raw)
    if not m:
        return None
    dims_str, dtype_token = m.group(1), m.group(2) or "bf16"
    dims = tuple(int(x) for x in dims_str.split(",") if x.strip())
    return dims, dtype_token


def normalize_shape_entry(entry: Any) -> str | None:
    """Coerce a candidate ``input_shapes`` entry to its raw ``"(...) dtype"`` string.

    Accepts the dict form ``{"shape": "(...) fp8", ...}`` (TraceLens) or a bare
    string. Returns ``None`` for anything else so callers can skip it.
    """
    if isinstance(entry, dict):
        v = entry.get("shape") or entry.get("shapes")
        return v if isinstance(v, str) else None
    if isinstance(entry, str):
        return entry
    return None


def render_tensor_expr(dims: tuple[int, ...], dtype_token: str) -> str:
    """Return a Python expression building one input tensor on cuda.

    Integer/index tensors use ``randint`` bounded by the dim extent (a generic,
    op-agnostic best-effort for index args; real routing indices need trace-time
    capture, the documented follow-up). Float tensors use ``randn``; fp8 is built
    in bf16 then cast (``torch.randn`` does not support fp8 directly).
    """
    attr = torch_dtype_attr(dtype_token)
    # Always emit a valid Python tuple literal: 1-D needs a trailing comma
    # ("(24960,)"), 0-D is "()".
    if not dims:
        shape_lit = "()"
    elif len(dims) == 1:
        shape_lit = f"({dims[0]},)"
    else:
        shape_lit = "(" + ", ".join(str(d) for d in dims) + ")"
    if is_int_token(dtype_token):
        hi = max(dims) if dims else 1
        hi = hi if hi > 1 else 2
        return f"torch.randint(0, {hi}, {shape_lit}, dtype=torch.{attr}, device='cuda')"
    if attr.startswith("float8"):
        return f"torch.randn({shape_lit}, dtype=torch.bfloat16, device='cuda').to(torch.{attr})"
    return f"torch.randn({shape_lit}, dtype=torch.{attr}, device='cuda')"


def build_inputs_exprs(input_shapes: Any) -> list[str] | None:
    """Turn a candidate ``input_shapes`` list into per-arg tensor expressions.

    Returns ``None`` when nothing parseable is present (caller falls back to the
    LLM harness generator).
    """
    if not isinstance(input_shapes, (list, tuple)) or not input_shapes:
        return None
    exprs: list[str] = []
    for entry in input_shapes:
        raw = normalize_shape_entry(entry)
        if raw is None:
            continue
        parsed = parse_shape_token(raw)
        if parsed is None:
            continue
        dims, dtype_token = parsed
        exprs.append(render_tensor_expr(dims, dtype_token))
    return exprs or None
