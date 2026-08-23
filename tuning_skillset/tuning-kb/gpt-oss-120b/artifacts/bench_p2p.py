#!/usr/bin/env python3
"""What is the honest GPU0<->GPU1 link bandwidth on this box?

The EXTEND profile says quickreduce twoshot costs ~948 us for the prefill
all-reduce. That tensor is 16384 x 2880 bf16 = 94.4 MB, and a 2-rank twoshot
all-reduce (reduce-scatter + all-gather) moves N bytes out of each rank, so the
implied link rate is ~99.6 GB/s.

Before trying to beat that, find out whether ~100 GB/s IS the link. If it is,
the all-reduce is at the ceiling and the 25.69% of prefill it costs is not
recoverable by swapping algorithms -- which is a negative result worth having
before spending restarts on it.

No distributed runtime needed: a cross-device .copy_() goes over the same fabric.
"""
import statistics
import torch

SIZES_MB = [1, 4, 16, 94.37, 256]
ITERS = 50


def time_copy(dst, src, iters=ITERS):
    for _ in range(10):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    ts = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        dst.copy_(src, non_blocking=True)
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) / 1e3)
    return statistics.median(ts)


def main():
    n = torch.cuda.device_count()
    print(f"visible devices: {n}")
    for i in range(min(n, 2)):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} {p.total_memory/2**30:.0f} GiB "
              f"{p.multi_processor_count} CUs")
    if n < 2:
        print("need 2 devices"); return

    print(f"\ncan_device_access_peer(0,1) = {torch.cuda.can_device_access_peer(0, 1)}")
    print(f"\n{'MB':>8} {'p2p 0->1':>12} {'p2p 1->0':>12} {'local d2d 0':>13}")
    for mb in SIZES_MB:
        nel = int(mb * 2**20) // 2
        a0 = torch.empty(nel, dtype=torch.bfloat16, device="cuda:0")
        b1 = torch.empty(nel, dtype=torch.bfloat16, device="cuda:1")
        a1 = torch.empty(nel, dtype=torch.bfloat16, device="cuda:1")
        b0 = torch.empty(nel, dtype=torch.bfloat16, device="cuda:0")
        c0 = torch.empty(nel, dtype=torch.bfloat16, device="cuda:0")
        by = nel * 2
        t01 = time_copy(b1, a0)
        t10 = time_copy(b0, a1)
        tll = time_copy(c0, a0)
        # local d2d moves the bytes twice (read + write); p2p is one-way over the link
        print(f"{mb:8.2f} {by/t01/1e9:9.1f} GB/s {by/t10/1e9:9.1f} GB/s "
              f"{2*by/tll/1e9:10.1f} GB/s")
        del a0, b1, a1, b0, c0
        torch.cuda.empty_cache()

    print("\nreference points:")
    print("  quickreduce twoshot, 94.4 MB, from the EXTEND profile: 948 us")
    print("    -> a 2-rank twoshot moves N bytes out of each rank, so 94.4MB/948us")
    print("       = 99.6 GB/s of link traffic.")


if __name__ == "__main__":
    main()
