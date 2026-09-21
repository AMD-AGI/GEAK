#!/usr/bin/env python3
"""Does `tl.dot` emit real WMMA on this part, and how does it compare to the vendor path?

Backs two claims in amd_rdna.md: that the matrix path is WMMA (section 2/6) and
the 2048^3 entry of the vendor-comparison table (section 5).

The comparison is the delicate part, and an earlier version of this script got
it wrong in four ways that all flattered Triton: the vendor side allocated a
fresh output every iteration while Triton wrote into a pre-allocated buffer, the
vendor side got no warm-up at all, the two sides ran in sequence rather than
interleaved, and the result was labelled "hipBLASLt" without checking. This
version pre-allocates and warms up both sides, times with GPU events, interleaves
the rounds and takes medians, and reports the BLAS backend it actually observed
rather than assuming one.

Exits non-zero if the ISA check or the correctness check fails, so it cannot
report a green run while measuring a broken kernel.
"""
import re
import statistics
import sys

import torch
import triton
import triton.language as tl

DEV = "cuda"
M = N = K = 2048
BM = BN = 128
BK = 64
REL_TOL = 2e-2          # fp16 inputs, fp32 accumulate, fp16 store
ROUNDS = 40


@triton.jit
def mm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        av = tl.load(a + om[:, None] * K + ok[None, :])
        bv = tl.load(b + ok[:, None] * N + on[None, :])
        acc += tl.dot(av, bv)
    tl.store(c + om[:, None] * N + on[None, :], acc.to(tl.float16))


def _event_ms(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main():
    if not torch.cuda.is_available():
        print("error: no GPU")
        return 1

    arch = torch.cuda.get_device_properties(0).gcnArchName
    a = torch.randn(M, K, device=DEV, dtype=torch.float16)
    b = torch.randn(K, N, device=DEV, dtype=torch.float16)
    c_triton = torch.empty(M, N, device=DEV, dtype=torch.float16)
    c_vendor = torch.empty(M, N, device=DEV, dtype=torch.float16)   # same courtesy as Triton
    grid = (M // BM, N // BN)

    failures = []

    # ---- ISA: what did tl.dot actually compile to? ------------------------
    handle = mm[grid](a, b, c_triton, M, N, K, BM=BM, BN=BN, BK=BK)
    torch.cuda.synchronize()
    asm = handle.asm.get("amdgcn", "")
    wmma = len(re.findall(r"v_wmma\w*", asm))
    mfma = len(re.findall(r"v_mfma\w*", asm))
    fma = len(re.findall(r"v_fma[c_]?_f\d+", asm))
    opcodes = sorted(set(re.findall(r"v_wmma\w+", asm)))
    print("arch: %s" % arch)
    print("ISA: v_wmma=%d  v_mfma=%d  scalar v_fma=%d" % (wmma, mfma, fma))
    print("WMMA opcodes: %s" % (opcodes[:4] or "(none)"))
    if wmma == 0:
        failures.append("tl.dot emitted NO v_wmma -- the matrix path claim is unsupported here")
    if mfma != 0:
        failures.append("tl.dot emitted %d v_mfma on an RDNA part -- unexpected" % mfma)

    # ---- correctness ------------------------------------------------------
    ref = a.float() @ b.float()
    rel = ((c_triton.float() - ref).abs().max() / ref.abs().max()).item()
    print("correctness: max relative error %.2e (tolerance %.0e)" % (rel, REL_TOL))
    if not (rel < REL_TOL):
        failures.append("Triton kernel is wrong: relative error %.2e >= %.0e" % (rel, REL_TOL))

    # ---- which BLAS is the vendor path, actually? -------------------------
    backend = "unknown"
    try:
        backend = str(torch.backends.cuda.preferred_blas_library())
    except Exception:
        pass
    print("torch BLAS backend as reported: %s (hip %s)" % (backend, torch.version.hip))

    # ---- throughput: both pre-allocated, both warmed, interleaved, median --
    def run_triton():
        mm[grid](a, b, c_triton, M, N, K, BM=BM, BN=BN, BK=BK)

    def run_vendor():
        torch.mm(a, b, out=c_vendor)          # out= so neither side pays allocation

    for _ in range(10):                        # warm BOTH, not just Triton
        run_triton()
        run_vendor()
    torch.cuda.synchronize()

    t_ms, v_ms = [], []
    for _ in range(ROUNDS):                    # interleaved, so drift hits both
        t_ms.append(_event_ms(run_triton))
        v_ms.append(_event_ms(run_vendor))

    t = statistics.median(t_ms)
    v = statistics.median(v_ms)
    flops = 2.0 * M * N * K
    print("GEMM %dx%dx%d fp16, median of %d interleaved rounds (GPU events):" % (M, N, K, ROUNDS))
    print("  triton  %8.3f ms  %6.2f TFLOP/s   (spread %.1f%%)"
          % (t, flops / (t * 1e-3) / 1e12, 100 * (max(t_ms) - min(t_ms)) / t))
    print("  torch   %8.3f ms  %6.2f TFLOP/s   (spread %.1f%%)"
          % (v, flops / (v * 1e-3) / 1e12, 100 * (max(v_ms) - min(v_ms)) / v))
    print("  triton / torch = %.2fx" % (v / t))

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
