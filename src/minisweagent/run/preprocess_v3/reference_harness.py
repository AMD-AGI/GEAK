"""Deterministic harness synthesis from a reference callable + traced shapes.

When Hyperloom dispatches a traced kernel it hands GEAK everything the universal
harness contract (``kernel_languages/*/harness.j2``) needs *without any LLM
authoring*: a real importable launcher callable (``module:func``, e.g.
``aiter.ops.moe_op:ck_moe_stage1_fwd``) and the EXACT per-argument shapes/dtypes
the kernel saw during serving. For kernels whose source is a CK/.cu template,
discovery extracts no callable function, so the LLM harness-generator otherwise
burns the whole preprocess budget trying to compile/wire the ``.cu`` from scratch.

This module short-circuits that: given the callable + shapes it writes a
contract-conformant harness directly (a sibling of the existing deterministic
``_try_synthesize_shell_contract_harness`` precedent). It is **general** — keyed
only on "callable + shapes", a property of every traced hot kernel — and strictly
**additive**: it returns ``None`` on any miss so the LLM generator runs unchanged.

Two correctness invariants it enforces (else a harness is faithful in name only):

1. Golden != the patched op. ``_ref`` must NOT call the same ``entry_point`` GEAK
   rewrites (a self-compare is a tautology). We snapshot the ORIGINAL op's output
   to ``golden.pt`` at synthesis time (before any worktree/patch) and ``_ref``
   replays it.
2. aiter worktree routing. aiter is an editable install resolving to the SOURCE
   repo; a harness that imports aiter without routing ``AITER_META_DIR`` +
   ``AITER_JIT_DIR`` to ``$GEAK_WORK_DIR`` silently measures the UNPATCHED baseline
   (the classic ~1.00x bug). The emitted harness sets both before ``import aiter``.

Tier-2 fidelity (documented caveat): inputs are reconstructed from traced
shapes/dtypes (random for floats, bounded ``randint`` for index tensors). For ops
whose correctness depends on *valid* routing indices the snapshot/compare may not
be representative; in that case correctness simply fails and the caller falls back
to the LLM generator. Trace-time real-input capture is the follow-up that removes
this caveat.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from minisweagent.kernel_languages._io_dtypes import build_inputs_exprs

logger = logging.getLogger(__name__)

# ``aiter/ops/moe_op.py(522): ck_moe_stage1_fwd`` | ``aiter.ops.moe_op:func`` |
# ``pkg/mod.py:func`` — extract (dotted_module, func).
_ENTRY_RE = re.compile(
    r"(?P<mod>[\w./]+?)(?:\.py)?\((?:\d+)\)\s*:\s*(?P<fn>[A-Za-z_]\w*)"  # path(line): fn
    r"|(?P<mod2>[\w.]+?)\s*:\s*(?P<fn2>[A-Za-z_]\w*)"                      # mod:fn / path:fn
)


def parse_reference_entry_point(entry: Any) -> tuple[str, str] | None:
    """Parse a launcher entry_point into ``(dotted_module, func_name)``.

    Accepts the TraceLens launcher dict (``{'entry_point': '...'}``), its
    stringified form, or a bare ``"module:func"`` / ``"path(line): func"``
    string. Returns ``None`` when no importable ``module:func`` can be derived
    (e.g. the only frame is a generic dispatcher like ``torch/_ops.py:__call__``).
    """
    if isinstance(entry, dict):
        entry = entry.get("entry_point") or ""
    if not isinstance(entry, str) or not entry.strip():
        return None
    m = _ENTRY_RE.search(entry)
    if not m:
        return None
    mod = m.group("mod") or m.group("mod2") or ""
    fn = m.group("fn") or m.group("fn2") or ""
    if not mod or not fn:
        return None
    # Normalize a path-like module ("aiter/ops/moe_op") to dotted ("aiter.ops.moe_op").
    mod = mod.strip().removesuffix(".py").strip("/").replace("/", ".")
    # Reject generic dispatch frames that are not the real op (no rewritable body).
    if mod in {"torch._ops", "torch", "builtins"} or fn in {"__call__", "run"}:
        return None
    return mod, fn


def _is_importable(module: str, func: str) -> bool:
    """True when ``from <module> import <func>`` yields a callable. Never raises."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        logger.debug("reference_harness: import %s failed: %s", module, exc)
        return False
    obj = getattr(mod, func, None)
    return callable(obj)


def _aiter_routing_prelude(module: str) -> str:
    """Env-routing prelude emitted BEFORE ``import aiter`` for aiter ops.

    Routes both ``AITER_META_DIR`` (source) and ``AITER_JIT_DIR`` (build output)
    to ``$GEAK_WORK_DIR`` so the JIT compiles the PATCHED worktree kernel, not the
    baseline source (the ~1.00x worktree-bypass bug). No-op for non-aiter modules.
    """
    if not module.startswith("aiter"):
        return ""
    return (
        "import os as _os\n"
        "_wd = _os.environ.get('GEAK_WORK_DIR') or _os.environ.get('GEAK_REPO_ROOT')\n"
        "if _wd:\n"
        "    _os.environ.setdefault('AITER_META_DIR', _os.path.join(_wd, 'aiter_meta'))\n"
        "    _os.environ['AITER_JIT_DIR'] = _os.path.join(_wd, '_geak_aiter_jit')\n"
    )


_HARNESS_TEMPLATE = '''\
"""Auto-synthesized GEAK harness (deterministic reference-callable contract).

Built from the launcher callable + TraceLens-captured shapes — NOT LLM-authored.
Golden reference is a pre-patch snapshot of the original op (golden.pt); the
candidate is the (possibly patched) op imported live. See reference_harness.py.
"""
import argparse
import time
{aiter_prelude}
import torch

from {module} import {func} as _candidate

_GOLDEN_PATH = {golden_path!r}


def _build_inputs():
    _inputs = [
{inputs_block}
    ]
    return tuple(_inputs)


def _ref():
    # Golden = the ORIGINAL op output captured before any patch (no self-compare).
    _golden = torch.load(_GOLDEN_PATH)
    def _f(*_a, **_k):
        return _golden
    return _f


def _run_correctness() -> bool:
    _inputs = _build_inputs()
    ref_out = _ref()(*_inputs)
    candidate_out = _candidate(*_inputs)
    ok = bool(torch.allclose(candidate_out.float(), ref_out.float(), atol={atol}, rtol={atol}))
    print("OK" if ok else "FAIL")
    return ok


def _time_n(fn, args, n: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[n // 2]


def _run_benchmark() -> None:
    _inputs = _build_inputs()
    ms = _time_n(_candidate, _inputs)
    print(f"GEAK_RESULT_LATENCY_MS={{ms:.6f}}")


def _run_full_benchmark() -> None:
    _inputs = _build_inputs()
    cand_ms = _time_n(_candidate, _inputs)
    ref_out = _ref()(*_inputs)
    cand_out = _candidate(*_inputs)
    ok = bool(torch.allclose(cand_out.float(), ref_out.float(), atol={atol}, rtol={atol}))
    print(f"GEAK_RESULT_LATENCY_MS={{cand_ms:.6f}}")
    print(f"GEAK_RESULT_SPEEDUP={{1.0:.4f}}")
    print("OK" if ok else "FAIL")


def _run_profile() -> None:
    _inputs = _build_inputs()
    torch.cuda.synchronize()
    for _ in range(3):
        _candidate(*_inputs)
    torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--correctness", action="store_true")
    m.add_argument("--benchmark", action="store_true")
    m.add_argument("--full-benchmark", action="store_true")
    m.add_argument("--profile", action="store_true")
    a = p.parse_args()
    if a.correctness:
        raise SystemExit(0 if _run_correctness() else 1)
    if a.benchmark:
        _run_benchmark()
    elif a.full_benchmark:
        _run_full_benchmark()
    elif a.profile:
        _run_profile()


if __name__ == "__main__":
    main()
'''


def synthesize_reference_harness(
    *,
    reference_entry_point: Any,
    input_shapes: Any,
    output_dir: Path,
    atol: float = 0.05,
) -> str | None:
    """Write a contract-conformant harness from a callable + traced shapes.

    Returns the harness path on success, or ``None`` on any miss (unimportable
    callable, unparseable shapes, golden snapshot failure) so the caller falls
    back to the LLM harness generator. Never raises.
    """
    parsed = parse_reference_entry_point(reference_entry_point)
    if parsed is None:
        logger.info("reference_harness: no importable module:func from %r; skip", reference_entry_point)
        return None
    module, func = parsed
    if not _is_importable(module, func):
        logger.info("reference_harness: %s:%s not importable/callable; skip", module, func)
        return None
    input_exprs = build_inputs_exprs(input_shapes)
    if not input_exprs:
        logger.info("reference_harness: no parseable input_shapes; skip")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    golden_path = output_dir / "golden.pt"

    # Snapshot the ORIGINAL op output NOW (synthesis time = pre-patch). This is the
    # faithful golden; doing it here (not inside the harness) is what avoids the
    # self-compare tautology once GEAK patches the op in a worktree.
    try:
        import torch  # local import; preprocess may run before torch is needed elsewhere

        local: dict[str, Any] = {"torch": torch}
        inputs = tuple(eval(e, {"torch": torch}, local) for e in input_exprs)  # noqa: S307 — our own exprs
        import importlib

        op = getattr(importlib.import_module(module), func)
        with torch.no_grad():
            golden = op(*inputs)
        if not isinstance(golden, torch.Tensor):
            logger.info("reference_harness: op %s:%s returned non-tensor %s; skip", module, func, type(golden))
            return None
        torch.save(golden.detach().cpu(), golden_path)
    except Exception as exc:  # noqa: BLE001 — any failure -> fall back to LLM generator
        logger.info("reference_harness: golden snapshot for %s:%s failed (%s); skip", module, func, exc)
        return None

    inputs_block = "\n".join(f"        {e}," for e in input_exprs)
    harness_src = _HARNESS_TEMPLATE.format(
        aiter_prelude=_aiter_routing_prelude(module),
        module=module,
        func=func,
        golden_path=str(golden_path),
        inputs_block=inputs_block,
        atol=atol,
    )
    harness_path = output_dir / "harness.py"
    harness_path.write_text(harness_src, encoding="utf-8")
    # Pin the production-shape contract marker downstream shape-fixers key on.
    try:
        (output_dir / "harness_shapes_source.txt").write_text("user_task:production", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        "reference_harness: synthesized %s for %s:%s (%d inputs, golden=%s)",
        harness_path, module, func, len(input_exprs), golden_path,
    )
    return str(harness_path)
