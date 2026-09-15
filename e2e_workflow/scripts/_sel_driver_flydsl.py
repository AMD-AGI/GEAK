"""Standalone selection driver: certify aiter.ops.flydsl.gemm_kernels:flydsl_hgemm as the
deepest live launcher of the profiled device kernel
hgemm_bf16_16x64x64x8_SPK1_W1x1x2_BLDS1_TN_AS1_BIAS_0.

Drives the REAL aiter.tuned_gemm dispatch chain (TunedGemm.mm -> gemm_a16w16 -> solMap['flydsl']
= flydsl_gemm -> flydsl_hgemm) for the qkv_proj family (N=5120,K=2880) at decode M in {64,1},
which the shipped gptoss tuned CSV routes to exactly this FlyDSL kernel
(flydsl_gemm8_..._t16x64x64_split_k1_block_m_warp1_block_n_warp1_block_k_warp2...).

Env expected (set by launcher): CAPTURE_TARGET, CAPTURE_OUT, CAPTURE_MAX, GEAK_SELECTION_TRACE,
CAPTURE_BYTE_BUDGET, CAPTURE_CASE_BYTE_LIMIT, CAPTURE_PERSIST_POLICY.
"""
import os
import sys

import torch

import aiter
import aiter.tuned_gemm as tg
import aiter.ops.flydsl.gemm_kernels as fk

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# 1) capture hook installs on import (env-driven) -> wraps flydsl_hgemm (CAPTURE_TARGET).
import capture_shapes  # noqa: E402  (self-installs from env)
import seam_trace       # noqa: E402

# The catalog kernelName that round-trips to the profiled device symbol
# hgemm_bf16_16x64x64x8_SPK1_W1x1x2_BLDS1_TN_AS1_BIAS_0 (gptoss_bf16_tuned_gemm.csv line 103,
# qkv_proj N5120 K2880 M64 decode).
FLYDSL_KERNEL_NAME = (
    "flydsl_gemm8_abf16_wbf16_bf16_t16x64x64_split_k1_block_m_warp1_block_n_warp1_"
    "block_k_warp2_async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950"
)


def _forced_flydsl_config(M, N, K, bias, dtype, otype, scaleAB=False, bpreshuffle=False):
    # Force the exact FlyDSL routing so flydsl_hgemm launches the profiled device kernel
    # regardless of whether the merged AITER_CONFIG_GEMM_BF16 DB has the gptoss rows loaded.
    return {
        "libtype": "flydsl",
        "solidx": 3289,
        "splitK": 1,
        "kernelName": FLYDSL_KERNEL_NAME,
    }


def main():
    dev = "cuda"
    # 2) install profiler markers on every safe candidate (deepest first). Capture already wrapped
    #    flydsl_hgemm, so the marker composes over the capture wrapper.
    candidates = [
        "aiter.ops.flydsl.gemm_kernels:flydsl_hgemm",   # target (deepest safe interceptable launcher)
        "aiter.tuned_gemm:flydsl_gemm",                  # source-evidence inner (solMap ref: never fires)
        "aiter.tuned_gemm:gemm_a16w16",                  # op_seam
        "aiter.tuned_gemm:TunedGemm.mm",                 # dispatcher
        "sglang.srt.layers.quantization.unquant:UnquantizedLinearMethod.apply",  # outer wrapper
    ]
    installed = []
    for c in candidates:
        try:
            seam_trace.install(c)
            installed.append(c)
        except Exception as exc:
            sys.stderr.write(f"[driver] marker install skipped for {c}: {exc!r}\n")
    sys.stderr.write(f"[driver] installed markers: {installed}\n")

    # 3) force FlyDSL routing for the driven shapes.
    tg.get_GEMM_A16W16_config = _forced_flydsl_config

    N, K = 5120, 2880  # qkv_proj family
    torch.manual_seed(0)
    for M in (64, 1):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
        bias = torch.randn(N, dtype=torch.bfloat16, device=dev)
        # warmup / JIT compile (also captured/traced; fine)
        out = tg.tgemm.mm(A, B, bias=bias)
        torch.cuda.synchronize()
        # timed/traced real call through the full marked chain
        out = tg.tgemm.mm(A, B, bias=bias)
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(A.float(), B.float(), bias.float())
        rel = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        sys.stderr.write(f"[driver] M={M} out={tuple(out.shape)} max_rel={rel:.4g}\n")

    sys.stderr.write("[driver] done\n")


if __name__ == "__main__":
    main()
