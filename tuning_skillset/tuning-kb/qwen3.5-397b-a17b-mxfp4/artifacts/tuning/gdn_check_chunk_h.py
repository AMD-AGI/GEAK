#!/usr/bin/env python3
"""Check that the tile chosen by gdn_chunk_h_launch_config() produces the same
output as the shipped 32/4/2 tile.

BV tiles the V dimension, which the kernel never reduces over -- each i_v
workgroup owns a disjoint slice of V and both dots (b_w @ h^T reducing over K,
and k @ b_v reducing over BT) have extents that do not depend on BV. So the
expected result is bit-identical, not merely close, and anything less means the
tile is being applied somewhere it changes the math. num_warps/num_stages
likewise only change scheduling for these dot shapes.

Runs both tiles over the shapes the heuristic switches between, on this model's
GDN dims (Hg=4, H=16, K=V=128) and on a second head config so the check is not
just one shape. Compares h, v_new AND the in-place updated recurrent state.

    python3 analysis/gdn/check_chunk_h.py
"""
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch
import triton

from sglang.kernels.ops.attention.fla import chunk_delta_h as CD

DEV = "cuda:3"
torch.cuda.set_device(DEV)


def make(nseq, isl, hg, h, k, v, seed):
    torch.manual_seed(seed)
    T = nseq * isl
    return dict(
        k=torch.randn(1, T, hg, k, dtype=torch.bfloat16, device=DEV) * 0.1,
        w=torch.randn(1, T, h, k, dtype=torch.bfloat16, device=DEV) * 0.1,
        u=torch.randn(1, T, h, v, dtype=torch.bfloat16, device=DEV) * 0.1,
        g=-torch.rand(1, T, h, dtype=torch.float32, device=DEV) * 0.01,
        initial_state=torch.randn(nseq + 2, h, v, k, dtype=torch.float32,
                                  device=DEV) * 0.01,
        initial_state_indices=torch.arange(nseq, dtype=torch.int32, device=DEV),
        cu_seqlens=torch.tensor([i * isl for i in range(nseq + 1)],
                                dtype=torch.int32, device=DEV),
    )


def run_with(tile, args):
    """Force a specific (BV, warps, stages) and return every output tensor."""
    saved = CD.gdn_chunk_h_launch_config
    CD.gdn_chunk_h_launch_config = lambda V, N, H: tile
    try:
        a = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in args.items()}
        h, v_new = CD.chunk_gated_delta_rule_fwd_h(**a)
        return h, v_new, a["initial_state"]
    finally:
        CD.gdn_chunk_h_launch_config = saved


SHAPES = [
    # (nseq, isl, Hg, H, K, V)   -- production GDN shard is Hg=4 H=16 K=V=128
    (1, 8192, 4, 16, 128, 128),
    (2, 8192, 4, 16, 128, 128),
    (3, 8192, 4, 16, 128, 128),
    (4, 8192, 4, 16, 128, 128),
    (2, 1024, 4, 16, 128, 128),
    (1, 512, 4, 16, 128, 128),
    (2, 2048, 2, 8, 128, 128),   # a second head config
    (2, 2048, 8, 8, 64, 64),     # K,V below the 64-block boundary
]

SHIPPED = (32, 4, 2)
ok = True
print(f"{'shape':>34} {'chosen tile':>14} {'h':>10} {'v_new':>10} {'state':>10}")
for i, (nseq, isl, hg, h_, k_, v_) in enumerate(SHAPES):
    args = make(nseq, isl, hg, h_, k_, v_, seed=i)
    tile = CD.gdn_chunk_h_launch_config(v_, nseq, h_)
    a = run_with(SHIPPED, args)
    b = run_with(tile, args)
    eq = [x.equal(y) for x, y in zip(a, b)]
    ok &= all(eq)
    grid = triton.cdiv(v_, tile[0]) * nseq * h_
    print(f"N={nseq} T={nseq*isl:<6} Hg={hg} H={h_} K={k_} V={v_}"
          f"  BV={tile[0]},w{tile[1]},s{tile[2]} g={grid:<5}"
          + "".join(f"{'EQUAL' if e else 'DIFFER':>11}" for e in eq))

print("\nbit-identical everywhere:", ok)

# --- and confirm the edited file actually delivers the speedup the sweep found,
# --- measured through the real entry point rather than by forcing triton.Config.
from torch.profiler import ProfilerActivity, profile


def time_tile(tile, args, iters=30, warmup=8):
    saved = CD.gdn_chunk_h_launch_config
    CD.gdn_chunk_h_launch_config = lambda V, N, H: tile
    try:
        for _ in range(warmup):
            CD.chunk_gated_delta_rule_fwd_h(**args)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                CD.chunk_gated_delta_rule_fwd_h(**args)
            torch.cuda.synchronize()
        for ev in prof.key_averages():
            if "chunk_gated_delta_rule_fwd_kernel_h" in ev.key and ev.count:
                return ev.self_device_time_total / ev.count
    finally:
        CD.gdn_chunk_h_launch_config = saved
    return float("nan")


print(f"\n{'N':>3} {'shipped 32/4/2':>15} {'heuristic':>18} {'speedup':>9}")
for nseq in (1, 2, 3, 4, 8):
    args = make(nseq, 8192, 4, 16, 128, 128, seed=100 + nseq)
    tile = CD.gdn_chunk_h_launch_config(128, nseq, 16)
    a = time_tile(SHIPPED, args)
    b = time_tile(tile, args)
    print(f"{nseq:3d} {a:12.2f} us  {b:9.2f} us ({tile[0]}/{tile[1]}/{tile[2]})"
          f" {a/b:8.3f}x")

sys.exit(0 if ok else 1)
