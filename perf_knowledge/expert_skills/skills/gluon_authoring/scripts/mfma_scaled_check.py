"""Single-tile numerics + ISA check for gl.amd.cdna4.mfma_scaled (gfx950, Triton >= 3.8).

Sweeps operand k_width x {scales None, real E8M0 via get_mfma_scale_layout} x format pair x instr_shape
and prints max-relative error against an fp32 reference plus the v_mfma op actually emitted. Use it
before carrying a k_width into a kernel: a wrong value compiles and runs and returns wrong numbers.
Run on one GPU:  python3 mfma_scaled_check.py
"""
import argparse


def _parse_args(argv=None):
    # Parsed BEFORE torch/triton are imported and before anything compiles or launches, so
    # --help is instant, needs no GPU and touches none.
    ap = argparse.ArgumentParser(
        prog="mfma_scaled_check.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Single-tile numerics + ISA check for gl.amd.cdna4.mfma_scaled.\n\n"
            "Sweeps operand k_width (4..64) x scales {None, real E8M0 via get_mfma_scale_layout} x\n"
            "format pair {e4m3, e2m1} x instr_shape {16x16x128, 32x32x64} on one tile, and prints per\n"
            "row the max relative error vs an fp32 reference for each k_width (COMPILE_ERR when it\n"
            "does not compile) plus the v_mfma ops actually emitted. Use it before carrying a\n"
            "k_width into a kernel: a wrong value compiles, runs and returns wrong numbers."),
        epilog=("Requires a gfx950 (CDNA4) GPU and Triton >= 3.8 with Gluon; it launches small kernels\n"
                "on the current device (pick it with HIP_VISIBLE_DEVICES). No options: the sweep is\n"
                "fixed (one compile + launch per cell)."))
    return ap.parse_args(argv)


if __name__ == "__main__":
    _parse_args()

import sys, itertools, torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

@gluon.jit
def kern(a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr,
         M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, KA: gl.constexpr, KB: gl.constexpr,
         KW: gl.constexpr, FA: gl.constexpr, FB: gl.constexpr, USE_S: gl.constexpr,
         IM: gl.constexpr, IN: gl.constexpr, IK: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr):
    NW: gl.constexpr = WM * WN
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[IM, IN, IK], transposed=True,
                                              warps_per_cta=[WM, WN])
    blk: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [NW, 1], [1, 0])
    da: gl.constexpr = gl.DotOperandLayout(0, mfma, KW)
    db: gl.constexpr = gl.DotOperandLayout(1, mfma, KW)
    om = gl.arange(0, M, layout=gl.SliceLayout(1, blk))
    oka = gl.arange(0, KA, layout=gl.SliceLayout(0, blk))
    a = gl.load(a_ptr + om[:, None] * KA + oka[None, :])
    okb = gl.arange(0, KB, layout=gl.SliceLayout(1, blk))
    on = gl.arange(0, N, layout=gl.SliceLayout(0, blk))
    b = gl.load(b_ptr + okb[:, None] * N + on[None, :])
    a = gl.convert_layout(a, da)
    b = gl.convert_layout(b, db)
    acc = gl.zeros([M, N], gl.float32, mfma)
    if USE_S:
        KS: gl.constexpr = K // 32
        oks = gl.arange(0, KS, layout=gl.SliceLayout(0, blk))
        sa = gl.load(as_ptr + om[:, None] * KS + oks[None, :])
        onr = gl.arange(0, N, layout=gl.SliceLayout(1, blk))
        sb = gl.load(bs_ptr + onr[:, None] * KS + oks[None, :])
        sa = gl.convert_layout(sa, gl.amd.cdna4.get_mfma_scale_layout(da, [M, KS]))
        sb = gl.convert_layout(sb, gl.amd.cdna4.get_mfma_scale_layout(db, [N, KS]))
        acc = gl.amd.cdna4.mfma_scaled(a, sa, FA, b, sb, FB, acc)
    else:
        acc = gl.amd.cdna4.mfma_scaled(a, None, FA, b, None, FB, acc)
    acc = gl.convert_layout(acc, blk)
    on2 = gl.arange(0, N, layout=gl.SliceLayout(0, blk))
    gl.store(c_ptr + om[:, None] * N + on2[None, :], acc)

def make(fmt, rows, K, g, kdim_first):
    # returns (stored tensor, fp32 values [rows,K] in logical (row, k) orientation)
    if fmt == "e2m1":
        codes = torch.randint(0, 16, (rows, K), generator=g)
        vals = FP4[codes]
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)  # low nibble = even k
        st = packed if not kdim_first else packed.t().contiguous()
    else:
        vals = torch.randn(rows, K, generator=g).to(torch.float8_e4m3fn).float()
        st = vals.to(torch.float8_e4m3fn)
        st = st if not kdim_first else st.t().contiguous()
    return st.cuda(), vals

def run(fa, fb, use_s, instr, M, N, K, WM, WN, kw, asm=False):
    g = torch.Generator().manual_seed(0)
    a, av = make(fa, M, K, g, False)
    b, bv = make(fb, N, K, g, True)  # stored [K(packed), N]
    sa = torch.randint(124, 131, (M, K // 32), generator=g).to(torch.uint8)
    sb = torch.randint(124, 131, (N, K // 32), generator=g).to(torch.uint8)
    if use_s:
        av = av * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
        bv = bv * torch.exp2(sb.float() - 127).repeat_interleave(32, 1)
    ref = av @ bv.t()
    c = torch.empty(M, N, device="cuda")
    KA = a.shape[1]; KB = b.shape[0]
    try:
        h = kern[(1,)](a, b, sa.cuda(), sb.cuda(), c, M, N, K, KA, KB, kw, fa, fb, use_s,
                       instr[0], instr[1], instr[2], WM, WN, num_warps=WM * WN)
    except Exception as e:
        return f"COMPILE_ERR {type(e).__name__}: {str(e).splitlines()[0][:90] if str(e) else ''}", None
    torch.cuda.synchronize()
    err = ((c.cpu() - ref).abs().max() / ref.abs().max()).item()
    a_s = h.asm["amdgcn"]
    mf = sorted(set(l.split()[0] for l in a_s.splitlines() if l.strip().startswith("v_mfma")))
    return f"{err:.2e}", mf

if __name__ == "__main__":
    cfgs = [("e4m3", "e4m3", [16, 16, 128], 32, 32, 256, 1, 1), ("e4m3", "e4m3", [16, 16, 128], 64, 64, 256, 2, 2),
            ("e2m1", "e2m1", [16, 16, 128], 32, 32, 256, 1, 1), ("e2m1", "e2m1", [16, 16, 128], 64, 64, 256, 2, 2),
            ("e4m3", "e2m1", [16, 16, 128], 32, 32, 256, 1, 1),
            ("e4m3", "e4m3", [32, 32, 64], 64, 64, 256, 2, 2), ("e2m1", "e2m1", [32, 32, 64], 64, 64, 256, 2, 2),
            ("e4m3", "e2m1", [32, 32, 64], 64, 64, 256, 2, 2)]
    for (fa, fb, ins, M, N, K, WM, WN) in cfgs:
        for use_s in (False, True):
            row = []
            mfs = set()
            for kw in (4, 8, 16, 32, 64):
                e, mf = run(fa, fb, use_s, ins, M, N, K, WM, WN, kw)
                row.append(f"kw{kw}={e}")
                if mf: mfs |= set(mf)
            print(f"{fa}x{fb} instr={ins} tile={M}x{N}x{K} warps=[{WM},{WN}] scales={'E8M0' if use_s else 'None'}: " + " ".join(row), "|", ",".join(sorted(mfs)), flush=True)
