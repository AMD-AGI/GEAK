#!/usr/bin/env python3
"""Deterministically resolve the LIVE GEMM rebind seam by (dtype/quant, GPU arch, backend).

WHY THIS EXISTS
---------------
The kernel_extractor used to HARDCODE `target_callable = aiter.tuned_gemm:gemm_a16w16` for every dense
GEMM head. That specific symbol is NOT the seam the live vLLM Linear reaches. The real live entry is the
OUTER dispatcher `vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl`, which routes internally:

    if use_aiter_triton_gemm(...):  aiter.ops.triton.gemm_a16w16   # off when is_fp8_fnuz() (e.g. MI300)
    if is_tgemm_enabled():          aiter.tuned_gemm.tgemm.mm       # gfx950-only (is_linear_enabled & on_gfx950)
    <skinny wvSplitK/LLMM1 paths>                                   # VLLM_ROCM_USE_SKINNY_GEMM
    else:                           torch.nn.functional.linear      # hipBLASLt

So on gfx942 (MI300X) BOTH aiter legs are gated OFF (`use_aiter_triton_gemm` by `is_fp8_fnuz()`,
`is_tgemm_enabled` by `on_gfx950()`) and the live path is hipBLASLt. Binding an authored candidate onto
`aiter.tuned_gemm:gemm_a16w16` on gfx942/bf16 therefore rebinds a DEAD seam: the live server never calls
it (`engagement_hits=0`, `rebound=0`), so even a numerically-correct, isolated-faster candidate produces
a 0.0% e2e delta and is rejected — exactly the h0 `dense_gemm_prefill_largeM` no-engagement failure.

IMPORTANT — this does NOT remove or bypass the aiter GEMM-tune lever. `rocm_unquantized_gemm_impl` is the
OUTER dispatcher that SUBSUMES aiter tuned_gemm (`tgemm.mm`) when `is_tgemm_enabled()` (gfx950); pointing
the rebind seam here INCLUDES the aiter path, it does not delete it. Separately, the aiter per-shape GEMM
DB tune (gradlib → `bf16_tuned_gemm.csv`) is a distinct Tier-A / config lever: it is probed independently
by `op_bench._aiter_gemm` (which scans aiter directly, NOT `meta.target_callable`) and deployed via env +
tuned CSV. This resolver only chooses the AUTHORED-kernel rebind seam and leaves that lever untouched.

WHAT THIS DOES
--------------
Resolve the seam the live server ACTUALLY calls for this (dtype, arch, backend) — the stable OUTER leaf,
so the rebind engages regardless of the internal aiter/triton/skinny/hipBLASLt dispatch AND stays correct
across arches (it captures aiter tuned_gemm on gfx950 and hipBLASLt on gfx942):

  op_kind=gemm, backend=vllm:
    quantized (fp8/...)      -> the quant method's linear-apply seam if resolvable, else '' + note
                                (the extractor then greps the live fp8 path; we never guess a dead seam).
    unquantized bf16/fp16:
      ROCm (gfx*)            -> vllm.model_executor.layers.utils:rocm_unquantized_gemm_impl
                                signature (x, weight, bias=None) -> x @ weightᵀ  (weight=[N,K]).
                                The OUTER dispatcher the Linear forward reaches on ALL gfx; rebinding it
                                engages on every arch and subsumes the aiter tuned_gemm leg.
      CUDA                    -> torch.nn.functional:linear

Operand convention (transpose_b=True, B=[N,K], out=A @ Bᵀ) is IDENTICAL to the old aiter seam, so no
operand rebuild is needed downstream — only the seam module:attr changes.

Resolution is LIVE-PROBE-FIRST (import vLLM, confirm the symbol exists so we never emit a dead symbol),
with a static arch fallback when vLLM is unimportable. The resolver cannot start the server, so it also
emits an `engagement_probe` reminder: the Integrator must confirm `engagement_hits > 0` before spending
authoring budget.
"""
import argparse
import json
import os
import sys

RESOLVER_VERSION = 1

# The stable ROCm unquantized-linear leaf (matches the harness fn(A, B, bias) -> out contract).
_ROCM_UNQUANT_SEAM = "vllm.model_executor.layers.utils:rocm_unquantized_gemm_impl"
_CUDA_UNQUANT_SEAM = "torch.nn.functional:linear"


def _detect_arch(explicit=""):
    """Lowercased GPU arch string, e.g. 'gfx942'. Prefer the explicit value; else probe torch."""
    a = (explicit or "").strip().lower()
    if a:
        return a
    try:
        import torch  # local import: keep the module cheap to import for tests
        return (torch.cuda.get_device_properties(0).gcnArchName or "").lower()
    except Exception:
        return ""


def _is_rocm(arch):
    return arch.startswith("gfx") or "rocm" in arch


def _importable(spec):
    """True if 'module:attr' resolves to a real attribute (so we never emit a dead symbol)."""
    try:
        mod, attr = spec.split(":", 1)
        import importlib
        return getattr(importlib.import_module(mod), attr, None) is not None
    except Exception:
        return False


def _emit(baseline, target, *, confidence, notes, transpose_b=True, quant="none", arch="", backend=""):
    return {
        "resolver_version": RESOLVER_VERSION,
        "op_kind": "gemm",
        "backend": backend,
        "gpu_arch": arch,
        "quant": quant,
        "baseline_callable": baseline,
        "target_callable": target,
        "transpose_b": bool(transpose_b),
        "seam_confidence": confidence,          # high | medium | low
        "seam_note": " ".join(notes),
        # The resolver cannot run the server; the Integrator MUST verify the seam engages.
        "engagement_probe": ("Before authoring: rebind this seam with a call counter and run a few live "
                             "forwards; engagement_hits==0 means the seam is dead on this arch/dtype — "
                             "switch seam or skip the head (do NOT spend authoring budget)."),
    }


def resolve_gemm_seam(regime, gpu_arch="", backend=""):
    """Return the arch/dtype-correct GEMM seam for the live serving path. Pure w.r.t. args; the only
    side effect is an OPTIONAL vLLM import used to confirm the symbol exists."""
    regime = regime or {}
    quant = ((regime.get("quant") or {}).get("method") or "none") or "none"
    backend = (backend or regime.get("backend") or "").strip().lower()
    arch = _detect_arch(gpu_arch)
    notes = []

    # Non-vLLM backends (sglang/atom) have their own Linear seams — out of scope for this resolver.
    if backend and backend != "vllm":
        notes.append(f"backend={backend}: non-vLLM GEMM seam resolution is out of scope; "
                     "returning '' so the extractor greps the live path.")
        return _emit("", "", confidence="low", notes=notes, quant=quant, arch=arch, backend=backend)

    # Quantized GEMM: the seam is the quant method's apply path, NOT the unquantized leaf. We do not
    # guess it here (a wrong fp8 seam is another dead rebind) — leave it to the extractor's live grep.
    if quant not in ("none", "", None):
        notes.append(f"quant={quant}: unquantized-leaf seam does NOT serve quantized Linear; "
                     "returning '' so the extractor resolves the live fp8/quant apply seam.")
        return _emit("", "", confidence="low", notes=notes, quant=quant, arch=arch, backend=backend)

    # Unquantized bf16/fp16.
    if _is_rocm(arch):
        if _importable(_ROCM_UNQUANT_SEAM):
            notes.append("unquantized bf16/fp16 on ROCm: the live vLLM Linear reaches the OUTER "
                         "dispatcher rocm_unquantized_gemm_impl, which routes to aiter tuned_gemm "
                         "(tgemm.mm, gfx950-only), aiter triton (off when is_fp8_fnuz, e.g. MI300), "
                         "skinny, or hipBLASLt. Rebinding this leaf engages on all arch and SUBSUMES "
                         "aiter tuned_gemm (does not remove it). The old aiter.tuned_gemm:gemm_a16w16 "
                         "seam is dead on gfx942 (both aiter legs gated off).")
            return _emit(_ROCM_UNQUANT_SEAM, _ROCM_UNQUANT_SEAM, confidence="high", notes=notes,
                         quant=quant, arch=arch, backend=backend)
        notes.append("vLLM utils not importable; static ROCm fallback to rocm_unquantized_gemm_impl "
                     "(verify it exists in the serving venv before authoring).")
        return _emit(_ROCM_UNQUANT_SEAM, _ROCM_UNQUANT_SEAM, confidence="medium", notes=notes,
                     quant=quant, arch=arch, backend=backend)

    # CUDA (or unknown non-gfx): the unquantized Linear reaches F.linear.
    notes.append("unquantized bf16/fp16 on CUDA/unknown-arch: the live Linear reaches F.linear.")
    return _emit(_CUDA_UNQUANT_SEAM, _CUDA_UNQUANT_SEAM, confidence=("high" if arch else "medium"),
                 notes=notes, quant=quant, arch=arch, backend=backend)


def main():
    ap = argparse.ArgumentParser(description="Resolve the live GEMM rebind seam by dtype/arch/backend.")
    ap.add_argument("--regime", default="", help="path to regime.json (parse_regime.py output); reads "
                                                 "regime.quant.method and regime.backend")
    ap.add_argument("--op-kind", default="gemm", help="only 'gemm' is handled; other kinds pass through "
                                                      "with an empty seam + note")
    ap.add_argument("--gpu-arch", default=os.environ.get("GPU_ARCH", ""),
                    help="e.g. gfx942; auto-detected via torch when omitted")
    ap.add_argument("--backend", default=os.environ.get("BACKEND", ""),
                    help="vllm|sglang|atom; falls back to regime.backend")
    ap.add_argument("--out", default="", help="write seam json here (also printed to stdout)")
    args = ap.parse_args()

    regime = {}
    if args.regime and os.path.isfile(args.regime):
        try:
            regime = json.load(open(args.regime))
        except Exception as e:
            print(f"WARN: could not read regime {args.regime}: {e!r}", file=sys.stderr)

    if args.op_kind != "gemm":
        out = {"resolver_version": RESOLVER_VERSION, "op_kind": args.op_kind,
               "baseline_callable": "", "target_callable": "", "seam_confidence": "low",
               "seam_note": f"op_kind={args.op_kind}: resolver only handles gemm; the extractor resolves "
                            "the attn/moe seam from the captured forward."}
    else:
        out = resolve_gemm_seam(regime, gpu_arch=args.gpu_arch, backend=args.backend)

    js = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(js)
    print(js)


if __name__ == "__main__":
    main()
