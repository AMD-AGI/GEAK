"""Regression tests for resolve_seam.resolve_gemm_seam (arch/dtype-aware GEMM seam).

Locks the h0 no-engagement fix: on gfx942/bf16 the seam MUST be the live unquantized ROCm leaf
(rocm_unquantized_gemm_impl), NEVER the fp8/gfx950-gated aiter.tuned_gemm. Uses EXPLICIT --gpu-arch so
the tests are deterministic on any box (no torch/vllm/GPU required). The ROCm leaf is returned in both
the vLLM-importable (high) and static-fallback (medium) cases, so we assert on the target regardless of
whether vLLM is installed in the test env.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import resolve_seam  # noqa: E402

ROCM_SEAM = "vllm.model_executor.layers.utils:rocm_unquantized_gemm_impl"
CUDA_SEAM = "torch.nn.functional:linear"


def _seam(quant="none", backend="vllm", arch="gfx942"):
    return resolve_seam.resolve_gemm_seam({"quant": {"method": quant}, "backend": backend},
                                          gpu_arch=arch, backend=backend)


def test_bf16_gfx942_is_rocm_unquant_leaf_not_aiter():
    r = _seam(quant="none", backend="vllm", arch="gfx942")
    assert r["target_callable"] == ROCM_SEAM, r
    assert r["baseline_callable"] == ROCM_SEAM, r
    assert "aiter" not in r["target_callable"], "must not hardcode the gated aiter seam"
    assert r["transpose_b"] is True
    assert r["seam_confidence"] in ("high", "medium")


def test_bf16_cuda_is_f_linear():
    r = _seam(quant="none", backend="vllm", arch="sm90")
    assert r["target_callable"] == CUDA_SEAM, r


def test_fp8_returns_empty_not_a_guessed_seam():
    r = _seam(quant="fp8", backend="vllm", arch="gfx942")
    assert r["target_callable"] == "", r
    assert r["seam_confidence"] == "low"


def test_non_vllm_backend_returns_empty():
    r = _seam(quant="none", backend="sglang", arch="gfx942")
    assert r["target_callable"] == "", r


def test_gfx950_bf16_still_uses_the_live_unquant_leaf():
    # The seam LEAF is the same on any gfx (it internally picks aiter/triton/hipBLASLt); rebinding the
    # leaf engages regardless, so we standardize on it rather than the arch-specific aiter symbol.
    r = _seam(quant="none", backend="vllm", arch="gfx950")
    assert r["target_callable"] == ROCM_SEAM, r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} resolve_seam tests passed.")
