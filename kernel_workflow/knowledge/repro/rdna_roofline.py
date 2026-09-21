#!/usr/bin/env python3
"""Measured per-kernel roofline efficiency for RDNA — the piece amd_rdna.md is missing.

`perf_knowledge/profiling/kernel_roofline.md` collapses a roofline into one number
per kernel:

    attainable   = min(Peak_Compute[dtype], AI * Peak_BW)
    Roofline Eff = Achieved / attainable

It is scoped `gens: [gfx942, gfx950]` and `run_roofline.py` refuses RDNA outright,
because it drives `rocprof-compute --roof-only`, whose roofline mode does not
support gfx10/11/12. That refusal is correct. But the *analysis* is still wanted
on RDNA, and every term can be had another way:

  * Peak_BW   -- measured here, and there are TWO of them on an APU with a 32 MB
                 last-level cache. Which one applies depends on whether the
                 kernel's working set fits. amd_rdna.md section 3 has these.
  * Peak_Compute -- amd_rdna.md has NO compute roofline at all today. That is the
                 real gap: without it, a statement like "Triton is 0.86x torch"
                 cannot say how far either sits from the hardware.
  * Achieved  -- ordinary timing; no rocprof-compute needed.

Three disciplines are taken verbatim from that doc because they are what keep the
number honest:

  1. **Report both denominators, always.** Empirical and datasheet disagree and
     neither is authoritative alone.
  2. **Efficiency > 100% against the empirical peak is not a good kernel, it is
     proof the denominator is wrong.** Reported as a failed measurement.
  3. **Pick the compute peak by the matrix instruction the kernel actually
     issues**, not by the dtype of its inputs. On RDNA that means WMMA; there is
     no MFMA here, so a CDNA-derived peak table must not be reused.
"""
import re
import statistics as st
import sys

import torch
import triton
import triton.language as tl

DEV = "cuda"
ROUNDS = 11

# gfx1151 / Radeon 8060S datasheet, dense, no sparsity.
# 40 CU * 64 lanes * 2 (FMA) * 2.9 GHz = 14.85 TFLOP/s fp32; WMMA bf16/fp16 is 4x.
DATASHEET = {
    "bf16": 59.4e12,
    "fp16": 59.4e12,
    "fp32": 14.85e12,
}
# amd_rdna.md section 3, measured by repro/bw.py (read-read-write).
BW_LLC_BYTES_S = 790e9
BW_DRAM_BYTES_S = 230e9
LLC_BYTES = 32 * 1024 * 1024


@triton.jit
def _mm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        acc += tl.dot(tl.load(a + om[:, None] * K + ok[None, :]),
                      tl.load(b + ok[:, None] * N + on[None, :]))
    tl.store(c + om[:, None] * N + on[None, :], acc.to(tl.float16))


def _ms(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def time_median(fn, rounds=ROUNDS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return st.median([_ms(fn) for _ in range(rounds)])


def measure(M, N, K, dtype=torch.float16):
    """Return achieved TFLOP/s for torch.mm and a naive Triton tl.dot, same shape."""
    a = torch.randn(M, K, device=DEV, dtype=dtype)
    b = torch.randn(K, N, device=DEV, dtype=dtype)
    ct = torch.empty(M, N, device=DEV, dtype=dtype)
    cv = torch.empty(M, N, device=DEV, dtype=dtype)
    BM = BN = 128 if min(M, N) >= 128 else 64
    BK = 64 if K >= 64 else 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    flops = 2.0 * M * N * K
    t = time_median(lambda: _mm[grid](a, b, ct, M, N, K, BM=BM, BN=BN, BK=BK))
    v = time_median(lambda: torch.mm(a, b, out=cv))
    return flops / (t * 1e-3) / 1e12, flops / (v * 1e-3) / 1e12


def isa_check():
    """Confirm the matrix path is WMMA -- a CDNA peak table must not be reused here."""
    a = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    b = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    c = torch.empty(256, 256, device=DEV, dtype=torch.float16)
    h = _mm[(2, 2)](a, b, c, 256, 256, 256, BM=128, BN=128, BK=64)
    torch.cuda.synchronize()
    asm = h.asm.get("amdgcn", "")
    return len(re.findall(r"v_wmma\w*", asm)), len(re.findall(r"v_mfma\w*", asm))


def main():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print("arch: %s" % arch)
    if not arch.startswith(("gfx10", "gfx11", "gfx12")):
        print("This tool is for RDNA. On CDNA use run_roofline.py (rocprof-compute).")
        return 2

    wmma, mfma = isa_check()
    print("matrix path: v_wmma=%d  v_mfma=%d  -> %s" % (
        wmma, mfma, "WMMA (RDNA)" if wmma and not mfma else "UNEXPECTED"))
    if mfma or not wmma:
        print("FAIL: matrix path is not WMMA; the peak table below does not apply.")
        return 1

    # ---- empirical compute peak ------------------------------------------
    # Sweep square shapes and keep the best TFLOP/s anyone reached. This is a
    # floor on the true peak, not the peak itself: if no kernel here saturates
    # the WMMA units the empirical number is low, which is exactly why the
    # datasheet column has to be printed next to it.
    print("\nsweeping for an empirical fp16 WMMA peak (best achieved, both impls):")
    best, best_at = 0.0, None
    for S in (512, 1024, 2048, 3072, 4096):
        ws = 3 * S * S * 2
        tri, tor = measure(S, S, S)
        fits = ws <= LLC_BYTES
        print("  %4d^3  working set %6.1f MB %-9s triton %6.2f  torch %6.2f TFLOP/s"
              % (S, ws / 1e6, "(LLC)" if fits else "(DRAM)", tri, tor))
        for v, who in ((tri, "triton"), (tor, "torch")):
            if v > best:
                best, best_at = v, "%s @ %d^3" % (who, S)
    print("  empirical peak  = %.2f TFLOP/s  (%s)" % (best, best_at))
    print("  datasheet peak  = %.2f TFLOP/s  (fp16/bf16 WMMA, 40 CU x 2.9 GHz)"
          % (DATASHEET["fp16"] / 1e12))
    print("  empirical / datasheet = %.1f%%  <- if far below 100%%, no kernel here"
          " saturated the WMMA units; treat the empirical column as a floor"
          % (100 * best * 1e12 / DATASHEET["fp16"]))

    # ---- per-kernel efficiency at the shape section 5 argues about --------
    S = 2048
    ws = 3 * S * S * 2
    bw = BW_LLC_BYTES_S if ws <= LLC_BYTES else BW_DRAM_BYTES_S
    regime = "LLC-resident" if ws <= LLC_BYTES else "DRAM-streaming"
    flops = 2.0 * S ** 3
    ai = flops / ws
    tri, tor = measure(S, S, S)

    print("\nper-kernel roofline efficiency, %d^3 fp16" % S)
    print("  working set %.1f MB -> %s, Peak_BW = %.0f GB/s" % (ws / 1e6, regime, bw / 1e9))
    print("  arithmetic intensity = %.0f FLOP/byte" % ai)
    mem_roof = ai * bw
    print("  memory roof   = AI x Peak_BW = %.1f TFLOP/s" % (mem_roof / 1e12))
    print("  %-14s %9s %9s %9s" % ("", "achieved", "eff(emp)", "eff(spec)"))
    bad = []
    for name, got in (("triton", tri), ("torch.mm", tor)):
        att_emp = min(best * 1e12, mem_roof)
        att_spec = min(DATASHEET["fp16"], mem_roof)
        e_emp = 100 * got * 1e12 / att_emp
        e_spec = 100 * got * 1e12 / att_spec
        print("  %-14s %7.2f T %8.1f%% %8.1f%%" % (name, got, e_emp, e_spec))
        if e_emp > 100.5:
            bad.append("%s reads %.1f%% of the EMPIRICAL peak; >100%% is proof the "
                       "denominator is wrong, not a fast kernel" % (name, e_emp))
    bound = "compute" if min(best * 1e12, DATASHEET["fp16"]) < mem_roof else "memory"
    print("  -> %s-bound at this shape (memory roof is %.0fx the compute roof)"
          % (bound, mem_roof / DATASHEET["fp16"]))

    if bad:
        print("\nFAILED MEASUREMENT:")
        for b in bad:
            print("  - " + b)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
