#!/usr/bin/env python3
"""Explicit gfx950 layout validation; run under GEAK's GPU lock in the workload env.

core: native FP16/BF16/MXFP4 GEMMs and trivial-swizzle copy-back controls.
staging: candidate nontrivial swizzle/padding copy-back, NOT a supported-layout list.
Writes per-case artifacts and status. No timing, downloads, compiler patches or profiler.
Exit 0: all cases pass; 1: failure; 3: incomplete/unsupported cases without failures.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def cases(suite):
    if suite == "core":
        for dtype, tile, warps, store, shape in itertools.product(
            ("float16", "bfloat16", "mxfp4"), (16, 32), ((2, 2), (1, 4)),
            ("mfma", "blocked"), ((32, 32, 128), (31, 19, 35))
        ):
            yield dict(kind="gemm", dtype=dtype, tile=tile, warps=warps,
                       store=store, shape=shape, bad_scale=False)
        yield dict(kind="gemm", dtype="mxfp4", tile=16, warps=(2, 2),
                   store="mfma", shape=(31, 19, 35), bad_scale=True)
    families = ("trivial",) if suite == "core" else ("swizzle_narrow", "swizzle_wide", "padding")
    for dtype, operand, family, mode in itertools.product(
        ("uint8", "float16", "bfloat16"), (0, 1), families, ("sync", "buffer", "global")
    ):
        yield dict(kind="copy", dtype=dtype, operand=operand, family=family, mode=mode)


def gpu_case(case, out):
    # Imported only by an isolated worker, so an LLVM abort cannot erase other results.
    import torch
    import triton
    from triton.experimental import gluon as g
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.amd import cdna4

    if not any(os.environ.get(k) for k in
               ("GEAK_GPU_LOCK_HELD", "GEAK_GPU_LOCKED_SET", "GEAK_GPU_LEASE_ACTIVE")):
        raise RuntimeError("Run under kernel_workflow/scripts/gpu_lock.sh; propagate ownership into containers")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx950":
        raise RuntimeError(f"gfx950 required, got {target}")
    env = dict(triton=triton.__version__, torch=torch.__version__, hip=torch.version.hip,
               target=str(target), device=str(torch.cuda.get_device_properties(0)))
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    torch.manual_seed(1729)
    torch.set_num_threads(1)
    result = dict(case=case, stage="construct", status="error")

    @g.jit
    def matrix(A, B, SA, SB, C, D, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
               BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
               ML: gl.constexpr, MX: gl.constexpr, STORE: gl.constexpr, BAD: gl.constexpr):
        AL: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        BL: gl.constexpr = gl.BlockedLayout([4, 1], [16, 4], [1, 4], [0, 1])
        AD: gl.constexpr = gl.DotOperandLayout(0, ML, 16 if MX else 8)
        BD: gl.constexpr = gl.DotOperandLayout(1, ML, 16 if MX else 8)
        PK: gl.constexpr = (K + 1) // 2 if MX else K
        SK: gl.constexpr = (K + 31) // 32
        am = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, AL))
        bn = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, BL))
        cm = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, ML))
        cn = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, ML))
        acc = gl.load(C + cm[:, None] * N + cn[None, :],
                      (cm[:, None] < M) & (cn[None, :] < N), 0)
        for start in range(gl.cdiv(K, BK)):
            WIDTH: gl.constexpr = BK // 2 if MX else BK
            ak = start * WIDTH + gl.arange(0, WIDTH, layout=gl.SliceLayout(0, AL))
            bk = start * WIDTH + gl.arange(0, WIDTH, layout=gl.SliceLayout(1, BL))
            av = gl.convert_layout(gl.load(A + am[:, None] * PK + ak[None, :],
                                          (am[:, None] < M) & (ak[None, :] < PK), 0), AD)
            bv = gl.convert_layout(gl.load(B + bn[None, :] * PK + bk[:, None],
                                          (bn[None, :] < N) & (bk[:, None] < PK), 0), BD)
            if MX:
                ASL: gl.constexpr = cdna4.get_mfma_scale_layout(AD, [BM, BK // 32])
                BSL: gl.constexpr = cdna4.get_mfma_scale_layout(BD, [BN, BK // 32])
                sm = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, ASL))
                sn = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, BSL))
                ka = start * (BK // 32) + gl.arange(0, BK // 32, layout=gl.SliceLayout(0, ASL))
                kb = start * (BK // 32) + gl.arange(0, BK // 32, layout=gl.SliceLayout(0, BSL))
                sa = gl.load(SA + sm[:, None] * SK + ka[None, :],
                             (sm[:, None] < M) & (ka[None, :] < SK), 127)
                sb = gl.load(SB + sn[:, None] * SK + kb[None, :],
                             (sn[:, None] < N) & (kb[None, :] < SK), 127)
                if BAD:
                    sa = gl.full([BM, BK // 32], 127, gl.uint8, ASL)
                    sb = gl.full([BN, BK // 32], 127, gl.uint8, BSL)
                acc = cdna4.mfma_scaled(av, sa, "e2m1", bv, sb, "e2m1", acc)
            else:
                acc = cdna4.mfma(av, bv, acc)
        if STORE == "blocked":
            OL: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
            acc = gl.convert_layout(acc, OL)
            cm = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, OL))
            cn = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, OL))
        gl.store(D + cm[:, None] * N + cn[None, :], acc,
                 (cm[:, None] < M) & (cn[None, :] < N))

    barrier = getattr(gl, "barrier", None) or gl.thread_barrier

    @g.jit
    def copy_back(X, Y, R: gl.constexpr, C: gl.constexpr, P: gl.constexpr,
                  S: gl.constexpr, MODE: gl.constexpr, OP: gl.constexpr):
        r = gl.arange(0, R, layout=gl.SliceLayout(1, P))
        c = gl.arange(0, C, layout=gl.SliceLayout(0, P))
        offsets = r[:, None] * C + c[None, :] if OP == 0 else r[:, None] + c[None, :] * R
        shared = gl.allocate_shared_memory(X.dtype.element_ty, [R, C], S)
        if MODE == "sync":
            shared.store(gl.load(X + offsets))
            barrier()
        else:
            if MODE == "buffer":
                cdna4.async_copy.buffer_load_to_shared(shared, X, offsets)
            else:
                cdna4.async_copy.global_load_to_shared(shared, X + offsets)
            cdna4.async_copy.commit_group()
            cdna4.async_copy.wait_group(0)
            barrier()
        gl.store(Y + offsets, shared.load(P))

    try:
        if case["kind"] == "gemm":
            M, N, K = case["shape"]
            tile, warps = case["tile"], list(case["warps"])
            mx = case["dtype"] == "mxfp4"
            ik = (128 if tile == 16 else 64) if mx else (32 if tile == 16 else 16)
            parent = gl.amd.AMDMFMALayout(version=4, instr_shape=[tile, tile, ik],
                                         transposed=True, warps_per_cta=warps)
            bm, bn, bk = tile * warps[0], tile * warps[1], 128 if mx else 64
            scales = [torch.randint(126, 129, (rows, (K + 31) // 32), dtype=torch.uint8)
                      for rows in (M, N)]
            if mx:
                codebook = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
                bits = [torch.randint(0, 16, (rows, 2 * ((K + 1) // 2)), dtype=torch.uint8) for rows in (M, N)]
                for b in bits:
                    b[:, K:] = 0
                data = [b[:, ::2] | (b[:, 1::2] << 4) for b in bits]
                decoded = [codebook[b[:, :K].long()] * torch.pow(2., s.double() - 127).repeat_interleave(32, 1)[:, :K]
                           for b, s in zip(bits, scales)]
            else:
                data = [(torch.randint(-8, 9, (rows, K)).float() / 8).to(getattr(torch, case["dtype"])) for rows in (M, N)]
                decoded = [x.double() for x in data]
            c = torch.randint(-8, 9, (M, N)).float() / 32
            ref = (decoded[0] @ decoded[1].T + c.double()).float()
            tensors = [t.cuda() for t in (*data, *scales, c)]
            actual = torch.full((M, N), float("nan"), device="cuda")
            result["layouts"] = dict(parent=repr(parent), block=[bm, bn, bk], k_width=16 if mx else 8)
            result["stage"] = "compile_or_launch"
            compiled = matrix[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
                *tensors, actual, M, N, K, bm, bn, bk, parent, mx, case["store"], case["bad_scale"], num_warps=4)
            expected_opcode = f"v_mfma_scale_f32_{tile}x{tile}x{ik}_f8f6f4" if mx else f"v_mfma_f32_{tile}x{tile}x{ik}"
            result["expected_opcode_found"] = expected_opcode in compiled.asm["amdgcn"]
        else:
            dtype, op = case["dtype"], case["operand"]
            width = 16 if dtype == "uint8" else 8
            rows, cols = (64 if dtype == "uint8" else 32), 64
            spt, threads, warps, order = [1, width], [width, 64 // width], [4, 1], [1, 0]
            if op == 1:
                rows, cols = cols, rows
                spt, threads, warps, order = spt[::-1], threads[::-1], warps[::-1], order[::-1]
            producer = gl.BlockedLayout(spt, threads, warps, order)
            family = case["family"]
            if family == "padding":
                helper = getattr(cdna4, "compute_efficient_padded_shared_layout", None)
                if helper is None:
                    result.update(status="unsupported", reason="padding helper absent")
                    return result
                parent = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128 if dtype == "uint8" else 32],
                                             transposed=True, warps_per_cta=[2, 2])
                shared = helper(gl.DotOperandLayout(op, parent, width), [rows, cols], getattr(gl, dtype), True)
            else:
                shared = gl.SwizzledSharedLayout(width if family == "swizzle_wide" else 1,
                                                 1, 1 if family == "trivial" else 4, order)
            result["layouts"] = dict(producer=repr(producer), shared=repr(shared), shape=[rows, cols],
                                      bytes_per_lane=16, operand=op)
            if shared is None:
                result.update(status="unsupported", reason="padding helper returned None")
                return result
            # Nonconstant data; uint8 collisions cannot be eliminated, so use two random patterns.
            compiled = None
            errors = []
            for seed in (123, 456):
                torch.manual_seed(seed)
                ref = torch.randint(0, 256, (rows * cols,), dtype=torch.int32).to(getattr(torch, dtype))
                x = ref.cuda()
                actual = torch.full_like(x, 17)
                result["stage"] = "compile_or_launch"
                compiled = copy_back[(1,)](x, actual, rows, cols, producer, shared, case["mode"], op, num_warps=4)
                result["stage"] = "correctness"
                errors.append(int((actual.cpu() != ref).sum()))
            result["mismatches_per_seed"] = errors
        result["stage"] = "correctness"
        torch.cuda.synchronize()
        for ext in ("ttgir", "llir", "amdgcn"):
            (out / f"kernel.{ext}").write_text(compiled.asm[ext])
        (out / "kernel.json").write_text(json.dumps(compiled.metadata._asdict(), indent=2, default=str))
        if case["kind"] == "gemm":
            equal = torch.allclose(actual.cpu(), ref, rtol=1e-5, atol=1e-5)
            result["max_abs"] = float((actual.cpu() - ref).abs().max())
            result["status"] = ("negative_control_detected" if not equal else "negative_control_missed") if case["bad_scale"] else ("pass" if equal else "numerical_failure")
            if not result["expected_opcode_found"]:
                result["status"] = "wrong_instruction"
        else:
            result["status"] = "pass" if not any(errors) else "numerical_failure"
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "staging"), default="core")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=120, help="per-case worker timeout, seconds")
    args = parser.parse_args()
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.out.resolve() / "cache"))
    if args.case:
        result = gpu_case(json.loads(args.case), args.out)
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        return 0 if result["status"] in ("pass", "negative_control_detected") else 3 if result["status"] == "unsupported" else 1
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(cases(args.suite)):
        path = args.out / f"case-{index:03d}"
        path.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--out", str(path), "--case", json.dumps(case)]
        try:
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
            (path / "worker.log").write_text(p.stdout)
            row = json.loads((path / "result.json").read_text()) if (path / "result.json").exists() else dict(case=case, status="worker_failure", exit=p.returncode)
        except subprocess.TimeoutExpired as exc:
            (path / "worker.log").write_bytes(exc.stdout or b"")
            row = dict(case=case, status="timeout")
        row["artifact_dir"] = path.name
        rows.append(row)
        print(f"{index:03d} {row['status']} {case}", flush=True)
        summary = dict(suite=args.suite, counts=dict(Counter(r["status"] for r in rows)), cases=rows,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), timing="not_run")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    good = {"pass", "negative_control_detected"}
    return 1 if any(r["status"] not in good | {"unsupported"} for r in rows) else 3 if any(r["status"] == "unsupported" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
