#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured ceilings for gfx1151: read bandwidth, WMMA peaks, and the clock they ran at.

Sections (argv[1], comma-separated; default all):
  bw      read-only bandwidth: DRAM (cold, 178 / 89 MB) and LLC-resident (warm, 24 MB)
  peak    WMMA peaks: Triton tl.dot int8->int32 and fp16/bf16->fp32 sweeps, torch.mm, torch._int_mm
  clock   engine clock and package power, idle and under a sustained fp16 GEMM

Timings are medians, GPU-event timed. "Cold" means a READ-based last-level-cache evict before
every call, timed per call: a write-based evict leaves dirty lines whose write-back is charged to
the timed kernel, which under-reports the ceiling on this part. Warm (LLC-resident) reads are
~30 us per call, so they are timed as batches of back-to-back calls. The read kernel writes one
int32 per program (<0.1% of the bytes read), so its bandwidth counts reads only. Prints one JSON
object per result.

Env: PEAK_SIZES (default "3072,8192"), PEAK_DTYPES (default "int8,fp16,bf16").
"""
import functools
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0)
SECTIONS = (sys.argv[1] if len(sys.argv) > 1 else "bw,peak,clock").split(",")
_EVICT = None


def evict():
    global _EVICT
    if _EVICT is None:
        _EVICT = torch.empty(512 * 2**20 // 2, dtype=torch.float16, device=DEV).normal_()
    _EVICT.sum()


def emit(rec):
    print(json.dumps(rec), flush=True)


def bench(fn, iters=30, warmup=5, cold=True, batch=1):
    """Median ms per call. `batch` > 1 times that many back-to-back calls per sample (warm only)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cold:
            evict()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(batch):
            fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / batch)
    med = statistics.median(ts)
    return med, (max(ts) - min(ts)) / med


@triton.jit
def _read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    base = pid * BLOCK * ITERS
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def section_bw():
    for mb, cold in ((178, True), (89, True), (24, False)):
        n = mb * 2**20 // 4
        x = torch.randint(-2**30, 2**30, (n,), dtype=torch.int32, device=DEV)
        best = None
        for block, iters, nw in ((1024, 8, 4), (2048, 4, 8), (4096, 4, 8), (1024, 16, 4)):
            nprog = triton.cdiv(n, block * iters)
            out = torch.empty(nprog, dtype=torch.int32, device=DEV)
            ms, spread = bench(functools.partial(_read_kernel[(nprog,)], x, out, n, BLOCK=block,
                                                 ITERS=iters, num_warps=nw),
                               cold=cold, batch=1 if cold else 20)
            gbs = n * 4 / ms / 1e6
            if best is None or gbs > best[0]:
                best = (gbs, ms, spread, [block, iters, nw])
        emit({"section": "bw", "MB": mb, "cold": cold, "GBps": round(best[0], 1),
              "ms": round(best[1], 4), "spread": round(best[2], 4), "cfg": best[3]})
        torch.cuda.empty_cache()


@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
               IS_INT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    gsz = GROUP_M * npn
    gid = pid // gsz
    fm = gid * GROUP_M
    gm = min(npm - fm, GROUP_M)
    pm = fm + (pid % gsz) % gm
    pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    if IS_INT:
        acc_i = tl.zeros((BM, BN), dtype=tl.int32)
        for _ in range(0, K, BK):
            acc_i = tl.dot(tl.load(a), tl.load(b), acc_i, out_dtype=tl.int32)
            a += BK * sak
            b += BK * sbk
        acc = acc_i.to(tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc = tl.dot(tl.load(a), tl.load(b), acc)
            a += BK * sak
            b += BK * sbk
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(c_ptr.dtype.element_ty))


def _tops(s, ms):
    return round(2 * s**3 / ms / 1e9, 2)


def section_peak():
    sizes = [int(v) for v in os.environ.get("PEAK_SIZES", "3072,8192").split(",")]
    for s in sizes:
        for dt in os.environ.get("PEAK_DTYPES", "int8,fp16,bf16").split(","):
            if dt == "int8":
                a = torch.randint(-127, 128, (s, s), dtype=torch.int8, device=DEV)
                b = torch.randint(-127, 128, (s, s), dtype=torch.int8, device=DEV)
                c = torch.empty((s, s), dtype=torch.float32, device=DEV)
            else:
                fdt = torch.bfloat16 if dt == "bf16" else torch.float16
                a = torch.randn((s, s), dtype=fdt, device=DEV)
                b = torch.randn((s, s), dtype=fdt, device=DEV)
                c = torch.empty((s, s), dtype=fdt, device=DEV)
            best = None
            for bm, bn in ((128, 128), (256, 128), (128, 256)):
                for bk in ((64, 128) if dt == "int8" else (32, 64)):
                    for nw in (4, 8):
                        for ns in (1, 2):
                            grid = (triton.cdiv(s, bm) * triton.cdiv(s, bn),)
                            f = functools.partial(
                                _mm_kernel[grid], a, b, c, s, s, s, a.stride(0), a.stride(1),
                                b.stride(0), b.stride(1), c.stride(0), c.stride(1), IS_INT=(dt == "int8"),
                                BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=nw, num_stages=ns)
                            try:
                                ms, spread = bench(f, iters=10, warmup=3, cold=False)
                            except Exception:
                                continue
                            if best is None or ms < best[0]:
                                best = (ms, spread, [bm, bn, bk, nw, ns])
            if best:
                emit({"section": "peak", "impl": "triton", "dtype": dt, "size": s,
                      "TOPS": _tops(s, best[0]), "spread": round(best[1], 4), "cfg": best[2]})
            if dt in ("fp16", "bf16"):
                ms, spread = bench(functools.partial(torch.mm, a, b, out=c), iters=10, warmup=3,
                                   cold=False)
                emit({"section": "peak", "impl": "torch.mm", "dtype": dt, "size": s,
                      "TOPS": _tops(s, ms), "spread": round(spread, 4)})
            else:
                try:
                    ms, spread = bench(functools.partial(torch._int_mm, a, b), iters=10, warmup=3,
                                       cold=False)
                    emit({"section": "peak", "impl": "torch._int_mm", "dtype": "int8", "size": s,
                          "TOPS": _tops(s, ms), "spread": round(spread, 4)})
                except RuntimeError as ex:
                    emit({"section": "peak", "impl": "torch._int_mm", "size": s, "error": str(ex)[:200]})
            torch.cuda.empty_cache()


def _smi():
    """sclk (MHz) and package power (W) as rocm-smi reports them; None when it cannot."""
    try:
        out = subprocess.run(["rocm-smi", "--showclocks", "--showpower"], capture_output=True,
                             text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    sclk = power = None
    for line in out.splitlines():
        low = line.lower()
        if "sclk" in low and "mhz" in low and sclk is None:
            num = [t for t in low.replace("(", " ").replace(")", " ").split() if t.endswith("mhz")]
            if num:
                sclk = float(num[-1][:-3])
        if "power" in low and "w" in low and power is None:
            for tok in low.replace(":", " ").split():
                try:
                    power = float(tok.rstrip("w"))
                    break
                except ValueError:
                    continue
    return sclk, power


def section_clock():
    sclk, power = _smi()
    emit({"section": "clock", "state": "idle", "sclk_MHz": sclk, "power_W": power})
    s = 8192
    x = torch.randn((s, s), dtype=torch.float16, device=DEV)
    y = torch.randn((s, s), dtype=torch.float16, device=DEV)
    stop = threading.Event()
    done = []

    def load():
        while not stop.is_set():
            torch.mm(x, y)
            torch.cuda.synchronize()
            done.append(1)

    t = threading.Thread(target=load)
    t0 = time.time()
    t.start()
    samples = []
    for _ in range(10):
        time.sleep(1.5)
        samples.append(_smi())
    stop.set()
    t.join()
    elapsed = time.time() - t0
    clocks = [v[0] for v in samples if v[0] is not None]
    powers = [v[1] for v in samples if v[1] is not None]
    emit({"section": "clock", "state": "sustained torch.mm fp16 8192^3",
          "seconds": round(elapsed, 1), "TFLOPS": round(len(done) * 2 * s**3 / elapsed / 1e12, 2),
          "sclk_MHz_median": statistics.median(clocks) if clocks else None,
          "power_W_median": statistics.median(powers) if powers else None})


if __name__ == "__main__":
    print(json.dumps({"device": torch.cuda.get_device_properties(0).gcnArchName,
                      "torch": torch.__version__, "triton": triton.__version__, "hip": torch.version.hip}))
    for name in SECTIONS:
        {"bw": section_bw, "peak": section_peak, "clock": section_clock}[name]()
