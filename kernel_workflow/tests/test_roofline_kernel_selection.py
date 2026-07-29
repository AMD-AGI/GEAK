"""Regression tests for roofline kernel selection, helper exclusion, and fallback.

Guards two behaviors:

1. Helper/runtime kernels (elementwise casts, buffer fill/copy, RNG, generic
   at::native launches) must NEVER be selected as the roofline result, even if a
   supplied pattern would otherwise match them.
2. When no supplied kernel_pattern matches any target kernel -- e.g. the config
   swapped the GEMM backend so the executed kernel name (CK `Cijk_*` /
   `ck::kernel_gemm_*`) differs from the pattern (Triton `gemm_a8w8_blockscale`)
   -- selection must fall back to the dominant target kernel that carries valid
   roofline metrics, rather than reporting the case as failed.

These stay backend- and layout-agnostic: exclusion is by case-insensitive
substring, fallback ranks purely by observed metrics.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_RK_PATH = os.path.join(_HERE, "..", "scripts", "roofline_kernel.py")
_spec = importlib.util.spec_from_file_location("roofline_kernel_under_test", _RK_PATH)
rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rk)


def _kernel(name, gflops=None, hbm=None, duration=None):
    metrics = {}
    if gflops is not None:
        metrics["performance_gflops"] = gflops
    if hbm is not None:
        metrics["hbm_actual_gbps"] = hbm
    if duration is not None:
        metrics["total_duration_ns"] = duration
    return {"kernel_name": name, "metrics": metrics}


def test_excluded_helper_kernels_are_not_targets():
    assert not rk._is_target_kernel("void at::native::vectorized_elementwise_kernel<4")
    assert not rk._is_target_kernel("__amd_rocclr_copyBuffer")
    assert not rk._is_target_kernel("SomeFillBuffer")
    assert not rk._is_target_kernel("distribution_elementwise_grid_stride")
    assert not rk._is_target_kernel("void at::native::elementwise_kernel_manual_unroll")
    # A real GEMM kernel is a legitimate target.
    assert rk._is_target_kernel("Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV")
    assert rk._is_target_kernel("void ck::kernel_gemm_xdl_cshuffle_v3<...>")


def test_pattern_match_selects_named_kernel():
    kernels = [
        _kernel("_gemm_a8w8_blockscale_kernel", gflops=1000.0),
        _kernel("Cijk_Alik_Bljk", gflops=50000.0),
    ]
    selected = rk._select_kernel(kernels, ["_gemm_a8w8_blockscale_kernel"])
    assert rk._kernel_matches_patterns(selected, ["_gemm_a8w8_blockscale_kernel"])
    assert selected["kernel_name"] == "_gemm_a8w8_blockscale_kernel"


def test_pattern_never_selects_an_excluded_helper():
    # Even if the pattern would textually match a helper kernel, it is excluded.
    kernels = [
        _kernel("void at::native::vectorized_elementwise_kernel<4", gflops=10.0),
        _kernel("Cijk_Alik_Bljk", gflops=50000.0),
    ]
    selected = rk._select_kernel(kernels, [".*elementwise.*"])
    # The only pattern matches an excluded kernel -> no pattern hit -> fallback
    # to the dominant real GEMM kernel.
    assert not rk._kernel_matches_patterns(selected, [".*elementwise.*"])
    assert selected["kernel_name"] == "Cijk_Alik_Bljk"


def test_fallback_selects_dominant_gemm_when_no_pattern_matches():
    # Reproduces the FP8 backend-swap case: patterns target the Triton kernel
    # name but the runtime dispatched to CK kernels.
    kernels = [
        _kernel("void at::native::vectorized_elementwise_kernel<4", gflops=136.0),
        _kernel("__amd_rocclr_copyBuffer"),
        _kernel("void ck::kernel_gemm_xdl_cshuffle_v3<...>", gflops=166783.0, hbm=1446.0),
        _kernel("Cijk_Alik_Bljk", gflops=51858.0, hbm=1705.0),
    ]
    patterns = ["_gemm_a8w8_blockscale_kernel", ".*gemm_a8w8_blockscale.*"]
    selected = rk._select_kernel(kernels, patterns)
    assert not rk._kernel_matches_patterns(selected, patterns)
    # Highest performance_gflops target kernel wins when durations are absent.
    assert selected["kernel_name"] == "void ck::kernel_gemm_xdl_cshuffle_v3<...>"


def test_fallback_ranks_by_duration_before_gflops():
    kernels = [
        _kernel("gemm_fast", gflops=200000.0, duration=100),
        _kernel("gemm_dominant", gflops=50000.0, duration=9000),
    ]
    selected = rk._select_kernel(kernels, ["no_such_pattern"])
    assert not rk._kernel_matches_patterns(selected, ["no_such_pattern"])
    assert selected["kernel_name"] == "gemm_dominant"


def test_no_target_kernel_with_metrics_returns_none():
    kernels = [
        _kernel("void at::native::vectorized_elementwise_kernel<4", gflops=136.0),
        _kernel("__amd_rocclr_copyBuffer"),
        _kernel("Cijk_no_metrics"),  # target but no valid metrics
    ]
    selected = rk._select_kernel(kernels, ["no_such_pattern"])
    assert selected is None


def test_empty_kernels_returns_none():
    selected = rk._select_kernel([], ["anything"])
    assert selected is None


def _full_kernel():
    return {
        "kernel_index": 0,
        "kernel_name": "Cijk_Alik_Bljk",
        "metrics": {"performance_gflops": 51858.0, "hbm_actual_gbps": 1705.0},
        "rows": [{"metric_id": "1.1.0", "value": 1.0}] * 116,
        "compute_rates": [{"metric": "VALU FLOPs (F16)", "value": 0.0}],
        "classification": {"theoretical_bound": "memory_side"},
        "warnings": [],
    }


def test_slim_kernels_drops_raw_fields(monkeypatch):
    monkeypatch.delenv("ROOFLINE_KEEP_RAW_KERNELS", raising=False)
    slim = rk._slim_kernels([_full_kernel()])
    assert list(slim[0].keys()) == ["kernel_index", "kernel_name", "metrics"]
    # Raw analyze rows and the compute-rate table are the bulk; they are dropped.
    assert "rows" not in slim[0]
    assert "compute_rates" not in slim[0]
    # Derived metrics are preserved verbatim.
    assert slim[0]["metrics"]["performance_gflops"] == 51858.0


def test_slim_kernels_env_var_keeps_raw(monkeypatch):
    monkeypatch.setenv("ROOFLINE_KEEP_RAW_KERNELS", "1")
    kept = rk._slim_kernels([_full_kernel()])
    assert "rows" in kept[0]
    assert "compute_rates" in kept[0]


def test_report_kernels_filters_helper_kernels(monkeypatch):
    monkeypatch.delenv("ROOFLINE_KEEP_RAW_KERNELS", raising=False)
    kernels = [
        _kernel("void at::native::vectorized_elementwise_kernel<4", gflops=136.0),
        _kernel("__amd_rocclr_copyBuffer"),
        _kernel("__amd_rocclr_fillBufferAligned"),
        _kernel("distribution_elementwise_grid_stride", gflops=1.0),
        _kernel("reduce_kernel", gflops=2.0),
        _kernel("void ck::kernel_gemm_xdl_cshuffle_v3<...>", gflops=166783.0),
        _kernel("Cijk_Alik_Bljk", gflops=51858.0),
    ]
    reported = rk._report_kernels(kernels)
    names = [k["kernel_name"] for k in reported]
    # Every helper/runtime kernel is filtered out of the persisted report.
    assert names == [
        "void ck::kernel_gemm_xdl_cshuffle_v3<...>",
        "Cijk_Alik_Bljk",  # a real Tensile GEMM is retained (not skipped)
    ]
    # Surviving kernels are slimmed.
    assert all(set(k.keys()) == {"kernel_index", "kernel_name", "metrics"} for k in reported)
