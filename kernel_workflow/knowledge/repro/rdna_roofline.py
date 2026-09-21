#!/usr/bin/env python3
"""Measured roofline terms for RDNA — calibration artifact, not yet a kernel scorer.

`perf_knowledge/profiling/kernel_roofline.md` scores a kernel as

    attainable   = min(Peak_Compute[dtype], AI * Peak_BW)
    Roofline Eff = Achieved / attainable

and `run_roofline.py` refuses RDNA because it drives `rocprof-compute --roof-only`,
whose roofline mode does not support gfx10/11/12. That refusal is correct. This
measures the same terms another way.

WHAT THIS IS NOT
----------------
It is **not** ready to feed an automatic keep/reject decision, and it says so at
the end of every run. Three reasons, all of which are the tool's fault rather
than the hardware's:

  * The empirical peak is the max of a sweep, which is a **winner's curse**
    estimator -- biased high and unstable. Repeated sweeps here move the peak by
    ~1.4% and move the *winning shape* between 1024/3072/4096, so a single number
    would be fiction. Reported as a range, with the winning shape listed.
  * The **hierarchical** roofline is not implemented. Choosing one bandwidth by
    "does the total working set fit in 32 MB" is a simplification; the honest
    form needs per-level traffic,
        attainable = min(Pcompute, AI_DRAM x BW_DRAM, AI_LLC x BW_LLC)
    with bytes measured at each level (GL2C_HIT / GL2C_MISS are collectable on
    this part, so this is buildable -- it just is not built). Until then the tool
    prints *both* memory roofs and the AI window where the choice actually
    changes the verdict.
  * The empirical winner's ISA is verified only when it can be: `tl.dot` is
    disassembled directly, a vendor kernel is identified by its Tensile name via
    rocprofv3 if available, and otherwise the peak is labelled ISA-UNVERIFIED
    rather than quietly trusted.

Disciplines kept from kernel_roofline.md: report both denominators; >100% of the
empirical peak is a broken denominator, not a fast kernel (and this exits
non-zero on it, which that doc mandates and its own driver does not implement);
pick the compute peak by the matrix instruction actually issued.
"""
import os
import re
import statistics as st
import subprocess
import sys

import torch
import triton
import triton.language as tl

DEV = "cuda"
ROUNDS = 11
SWEEP_REPEATS = 3

# gfx1151 / Radeon 8060S datasheet, dense, no sparsity:
# 40 CU * 64 lanes * 2 (FMA) * 2.9 GHz = 14.85 TFLOP/s fp32; WMMA fp16/bf16 = 4x.
DATASHEET_FP16 = 59.4e12
# amd_rdna.md section 3, repro/bw.py, read-read-write.
BW_LLC, BW_DRAM = 790e9, 230e9
LLC_BYTES = 32 * 1024 * 1024
SHAPES = (512, 1024, 2048, 3072, 4096)


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


def measure_pair(M, N, K):
    """Time both implementations INTERLEAVED, with a correctness gate on each.

    Interleaving is not cosmetic: clocks drift on this part and a blocked
    A-then-B layout aliases that drift onto the comparison. wmma_check.py was
    fixed for exactly this and the first version of this script reintroduced it.
    """
    a = torch.randn(M, K, device=DEV, dtype=torch.float16)
    b = torch.randn(K, N, device=DEV, dtype=torch.float16)
    ct = torch.empty(M, N, device=DEV, dtype=torch.float16)
    cv = torch.empty(M, N, device=DEV, dtype=torch.float16)
    BM = BN = 128 if min(M, N) >= 128 else 64
    BK = 64 if K >= 64 else 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    run_t = lambda: _mm[grid](a, b, ct, M, N, K, BM=BM, BN=BN, BK=BK)
    run_v = lambda: torch.mm(a, b, out=cv)

    for _ in range(5):                       # warm BOTH
        run_t(); run_v()
    torch.cuda.synchronize()

    # correctness gate -- a fast wrong kernel must not set the peak
    ref = (a.float() @ b.float())
    scale = ref.abs().max().clamp(min=1e-6)
    errs = {n: ((c.float() - ref).abs().max() / scale).item()
            for n, c in (("triton", ct), ("torch", cv))}

    ts, vs = [], []
    for _ in range(ROUNDS):                  # interleaved
        ts.append(_ms(run_t)); vs.append(_ms(run_v))
    f = 2.0 * M * N * K
    return (f / (st.median(ts) * 1e-3) / 1e12,
            f / (st.median(vs) * 1e-3) / 1e12, errs)


def isa_of_tl_dot():
    a = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    b = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    c = torch.empty(256, 256, device=DEV, dtype=torch.float16)
    h = _mm[(2, 2)](a, b, c, 256, 256, 256, BM=128, BN=128, BK=64)
    torch.cuda.synchronize()
    asm = h.asm.get("amdgcn", "")
    return len(re.findall(r"v_wmma\w*", asm)), len(re.findall(r"v_mfma\w*", asm))


def vendor_kernel_name():
    """Identify the vendor GEMM kernel by its Tensile name, via rocprofv3.

    Returns (name, verdict). Tensile encodes the matrix-instruction tile in the
    name (e.g. `..._MI16x16x16x1_...`), which is what lets the compute peak be
    attributed to a matrix path rather than assumed.
    """
    probe = ("import torch;a=torch.randn(1024,1024,device='cuda',dtype=torch.float16);"
             "b=torch.randn(1024,1024,device='cuda',dtype=torch.float16);"
             "c=torch.empty(1024,1024,device='cuda',dtype=torch.float16);"
             "[torch.mm(a,b,out=c) for _ in range(3)];torch.cuda.synchronize()")
    out = "/tmp/_rdna_roof_prof"
    r = subprocess.run(
        f"rocprofv3 --kernel-trace --stats --output-format csv -d {out} -o t -- python3 -c \"{probe}\"",
        shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return None, "rocprofv3 unavailable or failed"
    names = []
    for root, _, files in os.walk(out):
        for fn in files:
            if fn.endswith(".csv"):
                try:
                    txt = open(os.path.join(root, fn), encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                names += re.findall(r"(Cijk_[A-Za-z0-9_]+)", txt)
    if not names:
        return None, "no Tensile kernel name found in the trace"
    name = max(set(names), key=names.count)
    m = re.search(r"_MI(\d+x\d+x\d+)", name)
    return name, ("matrix-instruction tile %s in the kernel name" % m.group(1) if m
                  else "no _MI tile token in the name; matrix path NOT established")


def main():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print("arch: %s" % arch)
    if not arch.startswith(("gfx10", "gfx11", "gfx12")):
        print("This tool is for RDNA. On CDNA use run_roofline.py.")
        return 2

    # environment -- this APU shares its power budget with the CPU, so a noisy
    # box or a powersave governor silently changes every number below.
    gov = ""
    try:
        gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        gov = "unknown"
    try:
        la = open("/proc/loadavg").read().split()[0]
    except OSError:
        la = "?"
    print("environment: cpufreq governor=%s  loadavg=%s" % (gov, la))
    if gov != "performance":
        print("  WARNING: governor is not 'performance'; clock ramp widens every spread below.")

    wmma, mfma = isa_of_tl_dot()
    print("tl.dot matrix path: v_wmma=%d v_mfma=%d -> %s"
          % (wmma, mfma, "WMMA" if wmma and not mfma else "UNEXPECTED"))
    if mfma or not wmma:
        print("FAIL: not a WMMA path; the RDNA peak table does not apply.")
        return 1

    # ---- sweep, repeated, because max-of-a-sweep is a biased estimator ----
    print("\nsweep x%d (interleaved timing, correctness-gated), fp16 TFLOP/s:" % SWEEP_REPEATS)
    peaks, winners, worst_err = [], [], 0.0
    for rep in range(SWEEP_REPEATS):
        best, best_at = 0.0, None
        row = []
        for S in SHAPES:
            tri, tor, errs = measure_pair(S, S, S)
            worst_err = max(worst_err, max(errs.values()))
            row.append("%d:%.1f/%.1f" % (S, tri, tor))
            for v, who in ((tri, "triton"), (tor, "torch")):
                if v > best:
                    best, best_at = v, "%s@%d^3" % (who, S)
        peaks.append(best); winners.append(best_at)
        print("  rep%d  %s   peak %.2f (%s)" % (rep, "  ".join(row), best, best_at))

    print("  correctness: worst relative error across all shapes = %.2e" % worst_err)
    if worst_err >= 2e-2:
        print("  FAIL: a timed kernel is numerically wrong; its speed is meaningless.")
        return 1

    lo, hi, med = min(peaks), max(peaks), st.median(peaks)
    stable = len(set(winners)) == 1
    print("\nempirical peak  : %.2f - %.2f TFLOP/s (median %.2f), spread %.1f%%"
          % (lo, hi, med, 100 * (hi - lo) / med))
    print("  winning shape : %s%s" % (", ".join(winners),
          "" if stable else "   <- UNSTABLE; the peak is a sweep artifact, not a property"))
    print("datasheet peak  : %.2f TFLOP/s (fp16/bf16 WMMA)" % (DATASHEET_FP16 / 1e12))
    print("empirical/datasheet = %.1f%% -- nothing here saturates the WMMA units, so the"
          " empirical column is a FLOOR on the true peak" % (100 * med * 1e12 / DATASHEET_FP16))

    name, verdict = vendor_kernel_name()
    print("vendor kernel   : %s" % (name or "not identified"))
    print("  ISA evidence  : %s" % verdict)
    if any("torch" in w for w in winners) and (name is None or "_MI" not in (name or "")):
        print("  NOTE: the empirical peak comes from the vendor path and its matrix instruction is"
              " NOT established -> treat that peak as ISA-UNVERIFIED.")

    # ---- one scored shape, with the bracket stated as a bracket ----------
    S = 2048
    ws = 3 * S * S * 2
    ai = (2.0 * S ** 3) / ws
    tri, tor, _ = measure_pair(S, S, S)
    roof_llc, roof_dram = ai * BW_LLC, ai * BW_DRAM
    print("\nscored shape %d^3 fp16: working set %.1f MB, AI %.0f FLOP/byte" % (S, ws / 1e6, ai))
    print("  memory roof   LLC %.0f TFLOP/s | DRAM %.0f TFLOP/s | compute %.1f TFLOP/s"
          % (roof_llc / 1e12, roof_dram / 1e12, DATASHEET_FP16 / 1e12))
    flip_hi = DATASHEET_FP16 / BW_DRAM
    flip_lo = DATASHEET_FP16 / BW_LLC
    print("  both memory roofs exceed the compute roof, so this shape is compute-bound EITHER WAY;")
    print("  the LLC-vs-DRAM choice only changes the verdict for AI in %.0f-%.0f FLOP/byte."
          % (flip_lo, flip_hi))

    print("  %-10s %9s   %s" % ("", "achieved", "true efficiency is bracketed by:"))
    bad = []
    for nm, got in (("triton", tri), ("torch.mm", tor)):
        e_ds = 100 * got * 1e12 / DATASHEET_FP16
        e_emp = 100 * got / med
        print("  %-10s %7.2f T   %.1f%% (vs datasheet) .. %.1f%% (vs empirical)" % (nm, got, e_ds, e_emp))
        if e_emp > 100.5:
            bad.append("%s reads %.1f%% of the empirical peak -- the denominator is wrong" % (nm, e_emp))

    print("\nNOT READY for an automatic keep/reject gate:")
    print("  - empirical peak is a max-of-sweep (winner's curse)%s"
          % ("" if stable else ", and the winning shape is unstable"))
    print("  - hierarchical roofline not implemented: needs per-level bytes")
    print("    attainable = min(Pcompute, AI_DRAM x BW_DRAM, AI_LLC x BW_LLC)")
    print("  - use these as calibration constants; score kernels only once the above land")

    if bad:
        print("\nFAILED MEASUREMENT:")
        for b in bad:
            print("  - " + b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
