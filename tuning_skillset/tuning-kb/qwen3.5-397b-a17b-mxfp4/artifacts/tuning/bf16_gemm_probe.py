import torch, os, json
from torch.profiler import profile, ProfilerActivity
import aiter
from aiter.tuned_gemm import gemm_a16w16
from aiter.tuned_gemm import get_GEMM_A16W16_config

# (N, K, per-decode-step count per rank, label)
SHAPES = [
    (5120, 4096, 45, "GDN in_proj_qkvz"),
    (32,   4096, 45, "GDN in_proj_ba"),
    (4096, 2048, 60, "out_proj / o_proj"),
    (4608, 4096, 15, "full-attn qkv_proj"),
    (512,  4096, 120, "router gate + shared gate_up"),
    (4096, 256,  60, "shared_expert down_proj"),
    (1,    4096, 60, "shared_expert_gate"),
]
M = 64
out = []
for N, K, cnt, label in SHAPES:
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    cfg = get_GEMM_A16W16_config(M=M, N=N, K=K, bias=False, dtype="torch.bfloat16",
                                 otype="torch.bfloat16", scaleAB=False, bpreshuffle=False)
    for _ in range(20):
        gemm_a16w16(a, b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(50):
            gemm_a16w16(a, b)
        torch.cuda.synchronize()
    ev = sorted([e for e in p.key_averages() if e.device_time_total > 0],
                key=lambda e: -e.device_time_total)
    us = sum(e.device_time_total for e in ev) / 50.0
    names = [e.key for e in ev]
    wbytes = N * K * 2
    out.append(dict(N=N, K=K, count=cnt, label=label, libtype=cfg["libtype"],
                    us=round(us, 3), gbs=round(wbytes / us / 1e3, 1),
                    per_step_us=round(us * cnt, 1), kernels=names[:2]))
    print(f"N={N:6d} K={K:5d} x{cnt:4d}  {label:30s} {cfg['libtype']:8s} "
          f"{us:7.2f}us {wbytes/us/1e3:7.0f}GB/s  step={us*cnt:8.1f}us  {names[0][:80]}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "probe_baseline.json"), "w"), indent=1)
print("\ntotal bf16 dense per decode step per rank: %.1f us" % sum(o["per_step_us"] for o in out))
