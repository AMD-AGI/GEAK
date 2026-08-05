"""Runnable authored-overlap examples for CDNA4 (gfx950), with their own numerics check.

The CDNA4 counterpart to `pipeline_examples_cdna3.py`, which omits the async multi-buffer path
because gfx942 encodes only the 32-bit direct-to-LDS width and the wider CDNA4 chunks fail there.
This file settles what is reachable on gfx950, and it reports the asm evidence rather than
asserting it. It is also the only one of the pair that runs on triton 3.6.0 (see the surface notes
at the bottom of this docstring) -- so a "CDNA3 examples all fail" result on a 3.6 box is a
version result, not an arch one.

The falsifiable signature of a real async copy: data goes global -> LDS without passing through
registers, so the staging `ds_write` disappears. A sync-staged loop cannot do that. So:

    sync  staging  =>  ds_write > 0   (the register round trip is the staging)
    async staging  =>  ds_write == 0  AND a direct-to-LDS load appears in the asm

**What decides whether these lower is the per-lane access width, and it is easy to get wrong
by accident.** Direct-to-LDS exists only at the widths the hardware encodes -- `hw_constants.json`
`direct_to_lds_bit_widths`, which is `[128, 32]` on gfx950 and `[32]` on gfx942. So each lane must
contribute exactly one 4-byte or one 16-byte access, and BOTH entry points then lower on stock
`gluon_to_ttgir` with no pass spliced in. 8 B/lane and 32 B/lane fail, and so does any layout
whose per-lane contribution is the right size but *split across repetitions* -- which is what an
earlier version of this file shipped: `BLK` covered `[64, 16]` while the tile was `[32, 32]`, so
the layout repeated twice in N and every lane made two accesses instead of one. That failed, and
was misread as the architecture refusing async copy. The layout below covers the tile exactly.
`add_coalesce_async_copy` (`patch_async_reinject.py`) is what rescues the *non-native* patterns,
which is what a coalescing pass is for -- it is not what makes async copy work at all.

Ported to the triton 3.6.0 Gluon surface, which differs from the surface the CDNA3 examples
were authored against:
  * the barrier was renamed: 3.6.0 has `gl.thread_barrier`, 3.7.0 has `gl.barrier`
  * `gl.zeros(..., layout=)` is a GluonJITFunction, so the layout must survive `_flatten_ir`,
    which the layout classes do not implement -> use the `gl.full` builtin instead
  * `gl.amd.warp_pipeline_stage` is absent on 3.6.0 (it arrives with the 3.7 warp pipeline)

    python3 pipeline_examples_cdna4.py
"""
import os
import re

import torch
import triton
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna4 import async_copy

# Covers the [32, 32] tile in main() EXACTLY: [1*8*4, 4*8*1] = [32, 32], so each lane holds one
# 4-element fp32 run = 16 B = one `dwordx4`, the CDNA4 direct-to-LDS width. Change either factor
# and the async cases stop lowering -- see the width rule in the module docstring.
BLK: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [])
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

# 3.6.0 spells it thread_barrier; 3.7.0 renamed it to barrier.
_barrier = getattr(gl, "thread_barrier", None) or gl.barrier


@gluon.jit
def _offsets(M: gl.constexpr, N: gl.constexpr):
    return (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
            + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])


@gluon.jit
def a5_sync_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """A5 -- the sync baseline, carried over from CDNA3. Staging costs a register round trip."""
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = _offsets(M, N)
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    s.index(0).store(gl.load(inp + o))
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            s.index(nxt).store(gl.load(inp + o))
        acc += s.index(cur).load(BLK)
        _barrier()
    gl.store(out + o, acc)


@gluon.jit
def c1_async_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C1 -- async double buffer via global_load_to_shared. No register round trip.

    This is the form CDNA3 cannot express at all: on gfx942 this op fails the pass manager.
    """
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = _offsets(M, N)
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.global_load_to_shared(s.index(0), inp + o)
    async_copy.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            async_copy.global_load_to_shared(s.index(nxt), inp + o)
        async_copy.commit_group()
        async_copy.wait_group(1)          # let the i+1 copy stay in flight
        _barrier()
        acc += s.index(cur).load(BLK)
    gl.store(out + o, acc)


@gluon.jit
def c2_async_buffer_load(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C2 -- same depth, but scalar base + int32 offsets (buffer_load_to_shared).

    Separate lowering path from C1; on gfx942 this one fails LLVM translation instead.
    """
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = _offsets(M, N)
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.buffer_load_to_shared(s.index(0), inp, o)
    async_copy.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            async_copy.buffer_load_to_shared(s.index(nxt), inp, o)
        async_copy.commit_group()
        async_copy.wait_group(1)
        _barrier()
        acc += s.index(cur).load(BLK)
    gl.store(out + o, acc)


@gluon.jit
def c3_async_relaxed_read(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C3 -- C1 plus load_shared_relaxed, which drops the redundant wait before the LDS read."""
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = _offsets(M, N)
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.global_load_to_shared(s.index(0), inp + o)
    async_copy.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            async_copy.global_load_to_shared(s.index(nxt), inp + o)
        async_copy.commit_group()
        async_copy.wait_group(1)
        _barrier()
        acc += async_copy.load_shared_relaxed(s.index(cur), BLK)
    gl.store(out + o, acc)


@gluon.jit
def c4_async_depth3(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C4 -- three buffers, two copies in flight. Depth beyond 2 is where the 160 KiB matters.

    On gfx942 the per-workgroup LDS ceiling is 64 KiB, so deep multi-buffering of real tiles
    runs out of LDS before it runs out of latency to hide; that ceiling is 160 KiB here.
    """
    s = gl.allocate_shared_memory(gl.float32, [3, M, N], SH)
    o = _offsets(M, N)
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.global_load_to_shared(s.index(0), inp + o)
    async_copy.commit_group()
    async_copy.global_load_to_shared(s.index(1), inp + o)
    async_copy.commit_group()
    for i in range(ITERS):
        if i + 2 < ITERS:
            async_copy.global_load_to_shared(s.index((i + 2) % 3), inp + o)
        async_copy.commit_group()
        async_copy.wait_group(2)          # two groups may stay outstanding
        _barrier()
        acc += s.index(i % 3).load(BLK)
    gl.store(out + o, acc)


CASES = [
    ("A5 sync double buffer", a5_sync_double_buffer),
    ("C1 async global->LDS", c1_async_double_buffer),
    ("C2 async buffer->LDS", c2_async_buffer_load),
    ("C3 async + relaxed read", c3_async_relaxed_read),
    ("C4 async depth-3", c4_async_depth3),
]


def main():
    M = N = 32
    IT = 4
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    inp = torch.rand(M * N, device="cuda", dtype=torch.float32) + 1
    ref = inp * IT
    print(f"[{arch} {triton.__version__}]  LDS/workgroup="
          f"{torch.cuda.get_device_properties(0).shared_memory_per_block}")
    print(f"  {'case':26} {'result':8} {'correct':8} "
          f"{'ds_write':9}{'ds_read':8}{'g->lds':8}{'b->lds':8}{'vmcnt':7}")
    for tag, fn in CASES:
        out = torch.zeros(M * N, device="cuda", dtype=torch.float32)
        arm = os.environ.get("TRITON_GLUON_ASYNC", "0")
        triton.knobs.cache.dir = f"/tmp/c4ex_{tag.split()[0]}_{triton.__version__}_a{arm}"
        try:
            h = fn[(1,)](out, inp, M, N, IT, num_warps=4)
            torch.cuda.synchronize()
            asm = h.asm["amdgcn"]
            ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-4)

            def n(p, asm=asm):
                return len(re.findall(p, asm))

            print(f"  {tag:26} {'OK':8} {ok!s:8} "
                  f"{n('ds_write'):<9}{n('ds_read'):<8}"
                  f"{n(r'global_load_lds'):<8}{n(r'buffer_load_dword.*lds|buffer_load.*lds'):<8}"
                  f"{n('vmcnt'):<7}")
        except Exception as e:  # noqa: BLE001 -- which form fails to lower IS the measurement
            msg = str(e).splitlines()
            print(f"  {tag:26} {type(e).__name__}: {msg[-1][:70] if msg else ''}")


if __name__ == "__main__":
    main()
