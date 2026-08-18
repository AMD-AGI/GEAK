#!/usr/bin/env python3
"""GPU end-to-end smoke for the closed-loop Gluon recovery (gfx950 / CUDA box only).

Runs the full loop on a small plain matmul:
  plain @triton.jit -> compile + run (ref) + dump plain.ttgir
  -> recover concrete layouts (ttgir_to_gluon)              [asserts non-empty + key layouts]
  -> official translator -> @gluon.jit anchor -> import + run [asserts correctness == plain]
  -> recompile anchor -> dump anchor.ttgir -> layout-equivalence vs plain (informational:
     the translator re-infers layouts via AutoLayout, so this may differ -- which is exactly
     why we recover concrete layouts to inject; recover_gluon.assemble_anchor emits them).

Requires triton + torch + a GPU. Invoked by smoke_test_recover.sh when available.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl

import ttgir_to_gluon as t2g
import recover_gluon as rg


@triton.jit
def plain_matmul(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    pm = pid % npm
    pn = pid // npm
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    ap = a_ptr + (om[:, None] * sam + ok[None, :] * sak)
    bp = b_ptr + (ok[:, None] * sbk + on[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BK), num_stages=2):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK * sak
        bp += BK * sbk
    cm = pm * BM + tl.arange(0, BM)
    cn = pn * BN + tl.arange(0, BN)
    tl.store(c_ptr + scm * cm[:, None] + scn * cn[None, :], acc)


def _dump_ttgir(run_fn) -> str:
    """Run a kernel under a fresh cache and return the newest .ttgir text."""
    with tempfile.TemporaryDirectory() as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        run_fn()
        torch.cuda.synchronize()
        files = sorted(glob.glob(os.path.join(cache, "*", "*.ttgir")), key=os.path.getmtime)
        if not files:
            raise RuntimeError("no .ttgir produced in cache")
        return open(files[-1]).read()


def main() -> int:
    dev = "cuda"
    M = N = K = 256
    BM = BN = 128
    BK = 64
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, N, device=dev, dtype=torch.float16)
    ref = torch.empty(M, N, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    args = (M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            ref.stride(0), ref.stride(1), BM, BN, BK)

    plain_ttgir = _dump_ttgir(lambda: plain_matmul[grid](a, b, ref, *args))
    torch.cuda.synchronize()

    # 1. layout recovery must produce something sensible.
    layouts = t2g.parse_layouts(plain_ttgir)
    kinds = {l.kind for l in layouts}
    assert layouts and ("blocked" in kinds), f"no layouts recovered (kinds={kinds})"
    assert any(l.kind == "amd_mfma" for l in layouts) or any(l.kind == "linear" for l in layouts) \
        or "dot_op" in kinds, "no MMA/dot layout recovered"
    print(f"[gpu-smoke] recovered {len(layouts)} layouts: {sorted(kinds)}")

    # 2. official translator -> runnable Gluon anchor -> correctness == plain.
    try:
        from triton.tools.triton_to_gluon_translator.translator import convert_triton_to_gluon
        from triton.tools.triton_to_gluon_translator.target import TranslatorTarget
        from triton.language.target_info import current_target
    except Exception as e:  # noqa: BLE001
        print(f"[gpu-smoke] translator unavailable ({e!r}); skipping anchor compile.")
        print("SMOKE-GPU PASS (recovery only)")
        return 0

    t = current_target()
    target = TranslatorTarget(f"sm{t.arch}" if t.backend == "cuda" else t.arch)
    src = convert_triton_to_gluon([plain_matmul], target=target)
    with tempfile.TemporaryDirectory() as d:
        modpath = os.path.join(d, "anchor_kernel.py")
        open(modpath, "w").write(src)
        spec = importlib.util.spec_from_file_location("anchor_kernel", modpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["anchor_kernel"] = mod
        spec.loader.exec_module(mod)
        gk = getattr(mod, "plain_matmul")

        out = torch.empty(M, N, device=dev, dtype=torch.float32)
        out_args = (M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    out.stride(0), out.stride(1), BM, BN, BK)
        anchor_ttgir = _dump_ttgir(lambda: gk[grid](a, b, out, *out_args))
        torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("[gpu-smoke] translator anchor correctness == plain: PASS")

    # 3. layout-equivalence (informational: translator re-infers via AutoLayout).
    ok, report = rg.verify_equivalence(plain_ttgir, anchor_ttgir)
    print(report)
    print(f"[gpu-smoke] layout-equivalence(translator-anchor, plain) = {ok} "
          f"(differences are expected from AutoLayout re-inference; inject the recovered "
          f"layouts for an equivalent anchor)")

    print("SMOKE-GPU PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
