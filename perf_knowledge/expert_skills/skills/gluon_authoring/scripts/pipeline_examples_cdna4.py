"""Runnable authored-overlap examples for CDNA4 (gfx950), with their own numerics check.

The CDNA4 counterpart to `pipeline_examples_cdna3.py`, which omits the async multi-buffer path
because gfx942 encodes only the 32-bit direct-to-LDS width and the wider CDNA4 chunks fail there.
This file settles what is reachable on gfx950, and it reports the asm evidence rather than
asserting it.

The falsifiable signature of a real async copy: data goes global -> LDS without passing through
registers, so the staging `ds_write` disappears. A sync-staged loop cannot do that. So:

    sync  staging  =>  ds_write > 0   (the register round trip is the staging)
    async staging  =>  ds_write == 0  AND a direct-to-LDS load appears in the asm

That is why A5 is carried over from the CDNA3 file unchanged: it is the control. A run where
every case reports `ds_write > 0` has measured a sync fallback, not an async path.

**Each iteration reads a DIFFERENT tile, and that is what makes the ring falsifiable.** The
input is `[ITERS, M, N]` and iteration `i` consumes tile `i`, so the reference is the sum over
tiles rather than one tile times `ITERS`. Feeding every stage the same tile -- which an earlier
version of these examples did -- makes the numerics pass whether or not the buffer index is
right: reading the wrong stage, or overwriting a buffer before it is consumed, both still sum
to the same answer. Multi-buffering is the only thing these cases exist to demonstrate, so the
data has to be able to catch a broken one.

**What decides whether these lower is the per-lane access width, and it is easy to get wrong
by accident.** Direct-to-LDS exists only at the widths the hardware encodes --
`perf_knowledge/hardware/data/hw_constants.json` `direct_to_lds_bit_widths`, which is `[128, 32]` on gfx950 and
`[32]` on gfx942. So each lane must contribute exactly one 4-byte or one 16-byte access, and
BOTH entry points then lower on stock `gluon_to_ttgir` with no pass spliced in. 8 B/lane and
32 B/lane fail, and so does any layout whose per-lane contribution is the right size but
*split across repetitions*: a `BlockedLayout` covering `[64, 16]` on a `[32, 32]` tile repeats
twice in N, so every lane makes two accesses instead of one. That failure was once misread as
the architecture refusing async copy. `BLK` below covers the tile exactly; the trap is written
up in `references/pitfalls/platform-known-issues.md`.

`add_coalesce_async_copy` is what rescues the *non-native* patterns -- that is what a coalescing
pass is for, and it is NOT what makes async copy work at all. On this pack it is reached through
`gluon_swp.py` (which pairs it with the pipeliner exactly as plain's `make_ttgir` does), never by
patching an installed `compiler.py`. (GEAK also ships `patch_async_reinject.py`, which splices
the same pass into stock `gluon_to_ttgir` behind `TRITON_GLUON_ASYNC=1` -- the one route for a
hand-authored async body that has no pipeliner to pair it with. The cache dir below is keyed on
that arm, so a patched-and-armed run never serves an unarmed run's binary.)

Written against the 3.6.0 Gluon surface, same as the CDNA3 file, so a version failure cannot be
misread as an arch one:
  * the barrier was renamed: 3.6.0 has `gl.thread_barrier`, 3.7.0 has `gl.barrier`
  * `gl.zeros(..., layout=)` is a GluonJITFunction, so the layout must survive `_flatten_ir`,
    which the layout classes do not implement -> use the `gl.full` builtin instead
  * the index expression is written inline in every kernel rather than factored into a nested
    `@gluon.jit` helper: the helper reads better and adds a nested-call surface these examples
    do not need to be testing on the oldest minor they claim

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
def a5_sync_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """A5 -- the sync baseline, carried over from CDNA3. Staging costs a register round trip."""
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    s.index(0).store(gl.load(inp + o))                      # tile 0
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            s.index(nxt).store(gl.load(inp + (i + 1) * M * N + o))
        acc += s.index(cur).load(BLK)
        _barrier()
    gl.store(out + o, acc)


@gluon.jit
def c1_async_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C1 -- async double buffer via global_load_to_shared. No register round trip.

    This is the form CDNA3 cannot express at all: on gfx942 this op fails the pass manager.
    """
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.global_load_to_shared(s.index(0), inp + o)   # tile 0
    async_copy.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            async_copy.global_load_to_shared(s.index(nxt), inp + (i + 1) * M * N + o)
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
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.buffer_load_to_shared(s.index(0), inp, o)    # tile 0
    async_copy.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            async_copy.buffer_load_to_shared(s.index(nxt), inp, o + (i + 1) * M * N)
        async_copy.commit_group()
        async_copy.wait_group(1)
        _barrier()
        acc += s.index(cur).load(BLK)
    gl.store(out + o, acc)


@gluon.jit
def c3_async_relaxed_read(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C3 -- C1 plus load_shared_relaxed, which drops the redundant wait before the LDS read."""
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.global_load_to_shared(s.index(0), inp + o)   # tile 0
    async_copy.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            async_copy.global_load_to_shared(s.index(nxt), inp + (i + 1) * M * N + o)
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
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    async_copy.global_load_to_shared(s.index(0), inp + o)               # tile 0
    async_copy.commit_group()
    async_copy.global_load_to_shared(s.index(1), inp + M * N + o)       # tile 1
    async_copy.commit_group()
    for i in range(ITERS):
        if i + 2 < ITERS:
            async_copy.global_load_to_shared(s.index((i + 2) % 3), inp + (i + 2) * M * N + o)
        async_copy.commit_group()
        async_copy.wait_group(2)          # two groups may stay outstanding
        _barrier()
        acc += s.index(i % 3).load(BLK)
    gl.store(out + o, acc)


@gluon.jit
def c5_warp_pipeline(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    """C5 -- the marker path, wired correctly. STRUCTURE demo, not a performance claim.

    Every other case here authors the overlap inside one wave. This one asks the compiler for
    the WAVE-LEVEL schedule instead: two groups of four warps kept a stage apart, so one group's
    memory stage runs under the other's compute stage.

    Read it for the four things that are easy to get wrong, all of which are structural rather
    than arithmetic (`../references/tile-programming/warp-pipeline.md`):

      1. the memory stage carries the HIGHER priority (1 vs 0) -- backwards collapses the overlap,
         because a compute stage that outranks memory starves the other group's address updates;
      2. `wait_group` sits BETWEEN the two `with` blocks -- inside either one fails conversion;
      3. the loop is a dynamic `range` -- `static_range` produces no `scf.for` and the pass then
         silently does nothing;
      4. the loop covers ITERS-1 tiles because the prologue already issued one, so the last tile
         is drained by an epilogue OUTSIDE any stage.

    What this case does NOT demonstrate is a speedup, and the reason is instructive: there is no
    MFMA here, so by the first applicability gate this loop is not a candidate at all -- a
    wave-level schedule has no matrix pipe to keep busy. It is here to show the shape compiles,
    lowers, and computes the right answer with the priorities the right way round. Judge the
    technique on a compute-bound kernel, never on this one.
    """
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)

    # Prologue: one tile in flight before the pipelined loop starts.
    async_copy.global_load_to_shared(s.index(0), inp + o)
    async_copy.commit_group()

    # ITERS-1 iterations: the prologue took one tile, the epilogue takes the last.
    for i in range(ITERS - 1):
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            async_copy.global_load_to_shared(s.index((i + 1) % 2), inp + (i + 1) * M * N + o)
            async_copy.commit_group()
        async_copy.wait_group(1)                 # between the stages, never inside one
        _barrier()
        with gl.amd.warp_pipeline_stage("compute", priority=0):
            acc += s.index(i % 2).load(BLK)

    # Epilogue: drain the tile the loop did not cover, outside any stage.
    async_copy.wait_group(0)
    _barrier()
    acc += s.index((ITERS - 1) % 2).load(BLK)
    gl.store(out + o, acc)


# (tag, kernel, num_warps). C5 needs 8: the conversion splits a workgroup into two groups of
# one wave per SIMD, so 4 warps gives it one group and nothing to phase against.
CASES = [
    ("A5 sync double buffer", a5_sync_double_buffer, 4),
    ("C1 async global->LDS", c1_async_double_buffer, 4),
    ("C2 async buffer->LDS", c2_async_buffer_load, 4),
    ("C3 async + relaxed read", c3_async_relaxed_read, 4),
    ("C4 async depth-3", c4_async_depth3, 4),
    ("C5 warp_pipeline_stage", c5_warp_pipeline, 8),
]


def main():
    M = N = 32
    IT = 4
    props = torch.cuda.get_device_properties(0)
    arch = props.gcnArchName.split(":")[0]
    # ITERS distinct tiles, so consuming the wrong stage cannot land on the right answer.
    inp = torch.rand(IT * M * N, device="cuda", dtype=torch.float32) + 1
    ref = inp.view(IT, M * N).sum(0)
    print(f"[{arch} {triton.__version__}]  LDS/workgroup={props.shared_memory_per_block}")
    if not arch.startswith("gfx950"):
        # Say it once, up front. The C cases are expected to fail off CDNA4, and a reader who
        # skips the header reads five failures as "async copy is broken" rather than
        # "this is the wrong arch for this file".
        print(f"  NOTE: this file is the CDNA4 (gfx950) half of the pair and you are on {arch}. "
              f"The C* cases are EXPECTED to fail here -- gfx942 encodes only the 32-bit "
              f"direct-to-LDS width. Run pipeline_examples_cdna3.py for this target.")
    print(f"  {'case':26} {'result':8} {'correct':8} "
          f"{'ds_write':9}{'ds_read':8}{'g->lds':8}{'b->lds':8}{'vmcnt':7}{'setprio':8}")
    for tag, fn, nw in CASES:
        out = torch.zeros(M * N, device="cuda", dtype=torch.float32)
        # One cache dir per case: two variants differing only by a constexpr share a Triton
        # cache entry, and the second then silently runs the first's binary.
        arm = os.environ.get("TRITON_GLUON_ASYNC", "0")
        triton.knobs.cache.dir = f"/tmp/c4ex_{tag.split()[0]}_{triton.__version__}_a{arm}"
        try:
            h = fn[(1,)](out, inp, M, N, IT, num_warps=nw)
            torch.cuda.synchronize()
            asm = h.asm["amdgcn"]
            ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-4)

            def n(p, asm=asm):
                return len(re.findall(p, asm))

            print(f"  {tag:26} {'OK':8} {ok!s:8} "
                  f"{n('ds_write'):<9}{n('ds_read'):<8}"
                  f"{n(r'global_load_lds'):<8}{n(r'buffer_load_dword.*lds|buffer_load.*lds'):<8}"
                  f"{n('vmcnt'):<7}{n('s_setprio'):<8}")
            if fn is c5_warp_pipeline and n('s_setprio') == 0:
                # The marker path is version-gated, not probe-gated, and it fails QUIETLY:
                # on a build without `add_warp_pipeline` the `with` blocks compile away to
                # nothing and the kernel still produces the right answer.
                print(f"  {'':26} ^ no s_setprio -- the markers did NOT take effect on this "
                      f"build (warp_pipeline_stage is absent before 3.7)")
            if not ok:
                # correct=False on these is a RING bug, not a tolerance question: every stage
                # carries a different tile, so a wrong buffer index changes the sum outright.
                print(f"  {'':26} ^ wrong SUM -- the stage index or the wait depth is wrong, "
                      f"not the arithmetic")
        except Exception as e:  # noqa: BLE001 -- which form fails to lower IS the measurement
            msg = str(e).splitlines()
            print(f"  {tag:26} {type(e).__name__}: {msg[-1][:70] if msg else ''}")


if __name__ == "__main__":
    main()
