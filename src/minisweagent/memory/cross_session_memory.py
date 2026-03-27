"""Cross-session memory stubs — full implementation in future PR."""

from __future__ import annotations


def classify_kernel_category(kernel_path: str) -> str:
    """Classify kernel into a category from its path."""
    path_lower = kernel_path.lower()
    for tag in ("gemm", "matmul", "mm"):
        if tag in path_lower:
            return "gemm"
    for tag in ("attention", "atten", "mla", "sdpa"):
        if tag in path_lower:
            return "attention"
    for tag in ("norm", "rms", "layernorm"):
        if tag in path_lower:
            return "normalization"
    for tag in ("moe", "expert"):
        if tag in path_lower:
            return "moe"
    for tag in ("rope", "rotary"):
        if tag in path_lower:
            return "positional_encoding"
    return "unknown"
