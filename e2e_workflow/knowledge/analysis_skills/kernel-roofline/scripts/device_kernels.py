"""Skill-owned conditioning/marker kernels. Imported only in GPU execution."""
import triton
import triton.language as tl


@triton.jit
def _krf_read_evict(src, dst, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + offsets, mask=offsets < n, other=0.)
    tl.store(dst + tl.program_id(0), tl.sum(values, 0))


@triton.jit
def _krf_write_evict(src, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(src + offsets, 1., mask=offsets < n)


@triton.jit
def _krf_begin_marker(dst):
    tl.store(dst, 1)


@triton.jit
def _krf_end_marker(dst):
    tl.store(dst, 2)


@triton.jit
def _krf_probe_noop(dst):
    # Keep a real dispatched kernel without charging it as a mathematical bound.
    pass
