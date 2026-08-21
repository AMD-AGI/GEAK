#!/usr/bin/env python3
"""What is the fastest a kernel on this GPU can read HBM?

    HIP_VISIBLE_DEVICES=1 python3 /work/analysis/bench_hbm_ceiling.py

The MoE conclusion in FINDINGS.md rests on `moe1` being bandwidth-bound, and that
claim is only as good as the ceiling it is compared against.  `torch.sum` reads at
3.70 TB/s and a `copy_` moves 4.99 TB/s, but neither is a bandwidth-saturating
kernel -- the reduction is limited by its own tree and the copy pays for writes.

This measures a pure streaming read: one block per chunk, 128-bit vector loads,
accumulate in registers, one store per block.  That is the number `moe1` should be
judged against, because `moe1` is also read-dominated (it streams expert weights and
writes a comparatively tiny activation).
"""
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _read(x_ptr, out_ptr, n_elem, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = pid * BLOCK * UNROLL
    for u in tl.static_range(UNROLL):
        off = base + u * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + off, mask=off < n_elem, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters / 1e3          # seconds


def main():
    torch.cuda.set_device(0)
    # 2 GiB, not more: Triton indexes in int32 here, and 4 GiB of bf16 is 2^31
    # elements -- one past the end of the range, which faults rather than wrapping
    # visibly.
    gib = 2
    n = gib * 1024**3 // 2                          # bf16 elements
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    nbytes = n * 2

    rows = []
    for BLOCK, UNROLL in ((1024, 8), (1024, 16), (2048, 8), (512, 16)):
        grid = (triton.cdiv(n, BLOCK * UNROLL),)
        out = torch.empty(grid[0], dtype=torch.float32, device="cuda")
        fn = lambda: _read[grid](x, out, n, BLOCK=BLOCK, UNROLL=UNROLL)  # noqa: E731
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  skip BLOCK={BLOCK} UNROLL={UNROLL}: {type(e).__name__} {e}")
            continue
        reps = [timeit(fn) for _ in range(5)]
        sec = statistics.median(reps)
        rows.append((nbytes / sec / 1e12, BLOCK, UNROLL, sec))

    y = torch.empty(n // 2, dtype=torch.bfloat16, device="cuda")
    csec = statistics.median([timeit(lambda: y.copy_(x[: n // 2])) for _ in range(5)])
    ssec = statistics.median([timeit(lambda: torch.sum(x)) for _ in range(5)])

    print(f"# {gib} GiB bf16 on {torch.cuda.get_device_name(0)}")
    for tbs, BLOCK, UNROLL, sec in sorted(rows, reverse=True):
        print(f"  triton stream-read  BLOCK={BLOCK:>5} UNROLL={UNROLL:>3}  "
              f"{tbs:6.2f} TB/s  ({sec * 1e6:8.1f} us)")
    print(f"  torch copy_ (r+w)   {nbytes / csec / 1e12:6.2f} TB/s")
    print(f"  torch.sum   (read)  {nbytes / ssec / 1e12:6.2f} TB/s")


if __name__ == "__main__":
    main()
