#!/usr/bin/env python3
"""Measured HBM read / write roof for this GPU: the fastest of several stream kernels.

Usage: python3 hbm_stream.py [--mb 256,1024,4096] [--json out.json]
Run under gpu_lock on an idle GPU. Kernels: torch sum (read), torch fill (write), and a HIP
non-temporal dwordx4 read stream (load_inline; skipped if hipcc is unavailable). Each is timed with
events, median of 20 after warmup, on a buffer larger than the last-level cache. Reports TB/s per kernel
and size, and `read_tbs` / `write_tbs` = the best of each. A roof below what another probe reaches is
not a roof: always report the best you have, and add your own kernel's fastest pure stream if faster.
"""
import argparse, json, os, statistics
import torch

HIP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
typedef int v4i __attribute__((ext_vector_type(4)));
__global__ void nt_read(const v4i* __restrict__ p, long n, int* __restrict__ out) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x, stride = (long)gridDim.x * blockDim.x;
  int acc = 0;
  for (; i < n; i += stride) { v4i v = __builtin_nontemporal_load(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x7fffffff) out[0] = acc;   // keep the loads alive
}
void run(torch::Tensor buf, torch::Tensor out, int blocks) {
  long n = buf.numel() * buf.element_size() / 16;
  nt_read<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const v4i*)buf.data_ptr(), n, (int*)out.data_ptr());
}
"""


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e-3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", default="512,2048,4096")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    hip = None
    os.environ.setdefault("PYTORCH_ROCM_ARCH", torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    try:
        from torch.utils.cpp_extension import load_inline
        hip = load_inline("geak_hbm_stream", cpp_sources="#include <torch/extension.h>\nvoid run(torch::Tensor buf, torch::Tensor out, int blocks);", cuda_sources=HIP_SRC, functions=["run"],
                          extra_cuda_cflags=["-O3"], verbose=False)
    except Exception as e:  # noqa: BLE001
        err = " | ".join(l.strip() for l in str(e).splitlines() if "error" in l)[:700]
        print(f"HIP stream unavailable ({type(e).__name__}: {err}); torch kernels only")
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    rows, best_r, best_w = [], 0.0, 0.0
    for mb in [int(x) for x in a.mb.split(",")]:
        n = mb * (1 << 20)
        buf = torch.empty(n // 2, dtype=torch.bfloat16, device="cuda").uniform_()
        out = torch.zeros(1, dtype=torch.int32, device="cuda")
        cand = {"torch_sum_read": (lambda: buf.sum(), "read"), "torch_fill_write": (lambda: buf.fill_(1.0), "write")}
        if hip:
            for k in (4, 8, 16):
                cand[f"hip_nt_read_{k}xCU"] = ((lambda k=k: hip.run(buf, out, cus * k)), "read")
        for name, (fn, kind) in cand.items():
            tbs = n / timed(fn) / 1e12
            rows.append({"kernel": name, "mb": mb, "kind": kind, "tbs": round(tbs, 3)})
            if kind == "read":
                best_r = max(best_r, tbs)
            else:
                best_w = max(best_w, tbs)
        del buf
    for r in rows:
        print(f"{r['kernel']:22s} {r['mb']:6d} MB  {r['tbs']:.3f} TB/s")
    res = {"gpu": torch.cuda.get_device_name(0), "cus": cus, "read_tbs": round(best_r, 3),
           "write_tbs": round(best_w, 3), "rows": rows}
    print(f"READ_ROOF_TBS={res['read_tbs']} WRITE_ROOF_TBS={res['write_tbs']}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
