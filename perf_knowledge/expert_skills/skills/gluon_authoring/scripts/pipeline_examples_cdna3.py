"""Runnable authored-overlap examples for CDNA3 (gfx942), with their own numerics check.

These are the code blocks in `references/gluon/pipeline-reference.md ## Authored overlap`,
kept executable so the claims there can be re-checked on your own box instead of trusted.
Run it and compare the printed op counts with the ones quoted in that section.

The async multi-buffer path is deliberately absent here, but NOT because it is unavailable.
`cdna4.async_copy` does lower on gfx942 -- at 32 bits per thread, not the 128 this file's
layouts are built for. An earlier version of this docstring recorded it as "does not lower",
from a probe whose layout was at fault rather than the arch; see
`references/gluon/pipeline-reference.md ## Authored overlap (no compiler patch)` for the
correction and the width table. What belongs here is the sync staging plus the
`warp_pipeline_stage` hint; for the async forms at their own width, run
`pipeline_examples_cdna4.py`.

**Written against the 3.6.0 Gluon surface so a version result cannot be misread as an arch
one.** Two spellings moved between minors, and an earlier version of this file used only the
3.7 form — on a 3.6.0 box every case then failed at once, which reads as "CDNA3 cannot do any
of this" when the arch was never reached:
  * the barrier was renamed: 3.6.0 has `gl.thread_barrier`, 3.7.0 has `gl.barrier`
  * `gl.zeros(..., layout=)` is a GluonJITFunction, so the layout must survive `_flatten_ir`,
    which the layout classes do not implement -> use the `gl.full` builtin instead
  * `gl.amd.warp_pipeline_stage` arrives with the 3.7 warp pipeline, so B1 is reported as
    unavailable on 3.6.0 rather than run and counted as a failure

**Known limitation of the numerics check here: every iteration reads the SAME tile**, so
`correct=True` proves the arithmetic and NOT the buffer index. A5 and A2 are rings, and reading
the wrong stage or overwriting a buffer early still sums to the same answer under this input.
Treat their `correct` column as a smoke test, and read the op census for the staging claim.
`pipeline_examples_cdna4.py` uses `[ITERS, M, N]` with one distinct tile per iteration, which is
what makes a ring falsifiable; porting that shape back here changes A2's per-tensor-lead
reference and has not been done.

No GPU? `--compile-only` (automatic when torch sees no device) AOT-compiles every case for
`--arch` (default gfx942) on the CPU and prints the same op census; `correct` is skipped.

    python3 pipeline_examples_cdna3.py                 # GPU: launch, numerics + census
    python3 pipeline_examples_cdna3.py --compile-only  # CPU: census only (gfx942 AOT)
"""
import argparse
import re

import torch
import triton
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

# Covers [64, 16] against the [32, 32] tile in __main__, i.e. it repeats. That is harmless for
# these SYNC cases -- the numerics are unaffected and only the op counts you read move -- but do
# NOT carry this layout onto an async path: direct-to-LDS needs each lane to make exactly ONE
# access of a native width, and a layout covering more than the tile makes two. That failure
# wears the same `failed to translate module to LLVM IR` text as a genuinely absent op, and has
# been recorded as "the arch refuses async copy" once already. `pipeline_examples_cdna4.py`
# uses [1,4],[8,8],[4,1], which covers [32, 32] exactly; the difference is deliberate.
BLK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0], [])
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

# 3.6.0 spells it thread_barrier; 3.7.0 renamed it to barrier.
_barrier = getattr(gl, "thread_barrier", None) or gl.barrier


@gluon.jit
def a5_sync_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """A5 -- the CDNA3 base: two LDS buffers, alternate, barrier between phases."""
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    s.index(0).store(gl.load(inp + o))                 # prologue fills buffer 0
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            s.index(nxt).store(gl.load(inp + o))       # stage i+1 while consuming i
        acc += s.index(cur).load(BLK)
        _barrier()
    gl.store(out + o, acc)


@gluon.jit
def a2_per_tensor_depth(out, big, small, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """A2 -- two tensors staged at DIFFERENT lead distances in one loop.

    `big` is double-buffered and prefetched one iteration ahead; `small` is refreshed in the
    same iteration it is consumed. That split is the thing the auto-pipeliner cannot express:
    it assigns one stage schedule to the whole loop.
    """
    sb = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    ss = gl.allocate_shared_memory(gl.float32, [1, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    sb.index(0).store(gl.load(big + o))                # big leads by one iteration
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            sb.index(nxt).store(gl.load(big + o))
        ss.index(0).store(gl.load(small + o))          # small has no lead at all
        _barrier()
        acc += sb.index(cur).load(BLK) * ss.index(0).load(BLK)
    gl.store(out + o, acc)


HALF: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [])


@gluon.jit
def a4_subbuffer_slice(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """A4 -- stage one buffer, then consume it as two independent half-slices.

    Each half is a descriptor in its own right, so the halves can be filled or drained on
    different schedules; here they are just read back separately to show the split is real.
    """
    s = gl.allocate_shared_memory(gl.float32, [M, N], SH)
    rows_hi = gl.arange(0, M // 2, layout=gl.SliceLayout(1, HALF))[:, None]
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, HALF))[None, :]
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    o_hi = rows_hi * N + cols
    o_lo = (rows_hi + M // 2) * N + cols
    for _i in range(ITERS):
        _barrier()
        s.store(gl.load(inp + o))
        _barrier()
        top = s.slice(0, M // 2, 0).load(HALF)          # rows [0, M/2)
        bot = s.slice(M // 2, M // 2, 0).load(HALF)     # rows [M/2, M)
        gl.store(out + o_hi, top * ITERS)
        gl.store(out + o_lo, bot * ITERS)


@gluon.jit
def b1_warp_pipeline_stage(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """B1 -- cluster markers. A HINT: it reorders, it does not stage anything.

    NOT a template. The priorities below are backwards from the production rule -- a real
    warp-pipeline kernel gives the MEMORY stage the higher value, because both clusters issue
    VALU and a compute stage that outranks memory starves the other group's address updates.
    This probe is kept as measured (it has a published census); copy the rule from
    references/tile-programming/warp-pipeline.md instead.
    """
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    for _i in range(ITERS):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            v = gl.load(inp + o)
        with gl.amd.warp_pipeline_stage("compute", priority=3):
            acc += v * 2.0
    gl.store(out + o, acc)


def _cases(big=None, small=None, it=4):
    """(tag, kernel, inputs, reference). Inputs/reference are None in compile-only mode."""
    g = big is not None
    cases = [
        ("A5 sync double buffer", a5_sync_double_buffer, (big,), big * it if g else None),
        ("A2 per-tensor depth", a2_per_tensor_depth, (big, small), big * small * it if g else None),
        ("A4 sub-buffer slice", a4_subbuffer_slice, (big,), big * it if g else None),
        ("B1 warp_pipeline_stage", b1_warp_pipeline_stage, (big,), big * 2 * it if g else None),
    ]
    # B1 needs a builtin that arrives with the 3.7 warp pipeline. Report it as absent on this
    # MINOR rather than running it and printing an exception, which reads as an arch verdict.
    if not hasattr(getattr(gl, "amd", None), "warp_pipeline_stage"):
        cases = [c for c in cases if not c[0].startswith("B1")]
        print(f"  {'B1 warp_pipeline_stage':24} SKIPPED -- gl.amd.warp_pipeline_stage is absent "
              f"on triton {triton.__version__} (arrives with the 3.7 warp pipeline). "
              f"This is a VERSION result, not a CDNA3 one.")
    return cases


def _census_line(tag, correct, asm):
    return (f"  {tag:24} OK correct={correct} ds_write={len(re.findall('ds_write', asm)):2d} "
            f"ds_read={len(re.findall('ds_read', asm)):2d} "
            f"s_setprio={len(re.findall('s_setprio', asm)):2d}")


def _fail(tag, e):
    lines = str(e).splitlines()
    errs = [ln.strip() for ln in lines if "error:" in ln]
    print(f"  {tag:24} {type(e).__name__}: {(errs[0] if errs else lines[-1] if lines else '')[:150]}")


def aot_compile(fn, arch, num_warps, n_ptrs, constexprs):
    """CPU-only AOT compile of a @gluon.jit kernel: `n_ptrs` fp32 pointers, then constexprs."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.experimental.gluon._runtime import GluonASTSource
    target = GPUTarget("hip", arch, 64)
    backend = make_backend(target)
    sig = {nm: ("*fp32" if k < n_ptrs else "constexpr") for k, nm in enumerate(fn.arg_names)}
    cx = {(n_ptrs + k,): v for k, v in enumerate(constexprs)}
    attrs = {(k,): backend.parse_attr("D") for k in range(n_ptrs)}   # 16 B-aligned, as torch
    return triton.compile(GluonASTSource(fn, sig, cx, attrs), target=target,
                          options={"num_warps": num_warps})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compile-only", action="store_true",
                    help="AOT-compile every case on the CPU and print the op census; no launch, "
                         "no numerics (automatic when torch sees no GPU)")
    ap.add_argument("--arch", default="gfx942", help="target for --compile-only (default gfx942)")
    args = ap.parse_args(argv)
    M = N = 32
    IT = 4
    if args.compile_only or not torch.cuda.is_available():
        print(f"[{args.arch} {triton.__version__}]  COMPILE-ONLY (CPU AOT, no launch, numerics skipped)")
        for tag, fn, ins, _ in _cases(it=IT):
            triton.knobs.cache.dir = f"/tmp/ex_{tag.split()[0]}_{triton.__version__}_aot_{args.arch}"
            try:
                k = aot_compile(fn, args.arch, 4, 1 + len(ins), (M, N, IT))
                print(_census_line(tag, "n/a", k.asm["amdgcn"]))
            except Exception as e:  # noqa: BLE001
                _fail(tag, e)
        return
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    big = torch.rand(M * N, device="cuda", dtype=torch.float32) + 1
    small = torch.rand(M * N, device="cuda", dtype=torch.float32) + 1
    print(f"[{arch} {triton.__version__}]")
    for tag, fn, ins, ref in _cases(big, small, IT):
        o = torch.zeros(M * N, device="cuda", dtype=torch.float32)
        triton.knobs.cache.dir = f"/tmp/ex_{tag.split()[0]}_{triton.__version__}"
        try:
            h = fn[(1,)](o, *ins, M, N, IT, num_warps=4)
            torch.cuda.synchronize()
            ok = torch.allclose(o, ref, rtol=1e-5, atol=1e-4)
            print(_census_line(tag, ok, h.asm["amdgcn"]))
        except Exception as e:  # noqa: BLE001 -- reporting which example fails to lower IS the point
            _fail(tag, e)


if __name__ == "__main__":
    main()
