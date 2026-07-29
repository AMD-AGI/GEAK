"""Regression tests for roofline_kernel.py baseline-kernel exclusion.

A config/backend swap (e.g. Triton->CK a8w8_blockscale) deploys a DIFFERENT callable
than the frozen baseline, but the task's `_build_correctness` still runs the baseline
(Triton) kernel during setup, so it leaks into the profiled trace. Being one big kernel
vs the swapped backend's many smaller kernels, it would win the duration-ranked dominant
fallback -- yielding a Triton-vs-Triton "post" reading. `_select_kernel` now drops
`exclude_patterns` from the fallback (while an explicit positive pattern still wins, and
the exclusion is ignored rather than stranding a case).

Imported by file path so the test is location-agnostic.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(os.path.join(HERE, "..", "scripts", "roofline_kernel.py"))
_spec = importlib.util.spec_from_file_location("roofline_kernel_under_test", _MODULE_PATH)
rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rk)


def _k(name, duration_ns, gflops=100.0):
    # Shape a kernel dict that _has_valid_kernel_metrics accepts and _kernel_rank_key ranks.
    return {
        "kernel_name": name,
        "metrics": {
            "total_duration_ns": duration_ns,
            "performance_gflops": gflops,
            "arithmetic_intensity_flops_per_byte": 100.0,
            "achieved_occupancy_pct": 50.0,
        },
    }


# The Triton baseline kernel dominates by raw duration; the deployed CK work is split
# across several shorter kernels. Without exclusion the fallback picks Triton.
TRITON = "_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128"
CK_A = "void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGemm...>"
CK_B = "Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT208x256x32"


def test_fallback_excludes_baseline_kernel():
    kernels = [_k(TRITON, 9000), _k(CK_A, 4000), _k(CK_B, 3500)]
    # no exclusion -> Triton wins by duration (the bug)
    assert rk._select_kernel(kernels, [])["kernel_name"] == TRITON
    # exclude the baseline symbol -> the dominant DEPLOYED (CK) kernel is picked
    picked = rk._select_kernel(kernels, [], exclude_patterns=[TRITON])
    assert picked["kernel_name"] == CK_A  # CK_A (4000) > CK_B (3500)


def test_explicit_pattern_beats_exclusion():
    kernels = [_k(TRITON, 9000), _k(CK_A, 4000)]
    # a positive pattern pins the winner even if it also matches an exclude
    picked = rk._select_kernel(kernels, [TRITON], exclude_patterns=[TRITON])
    assert picked["kernel_name"] == TRITON


def test_exclusion_never_strands_a_case():
    # if excluding empties the candidate set, exclusion is ignored (a reading beats None)
    kernels = [_k(TRITON, 9000)]
    picked = rk._select_kernel(kernels, [], exclude_patterns=[TRITON])
    assert picked is not None and picked["kernel_name"] == TRITON


def test_exclude_patterns_helper_reads_manifest_and_case():
    # case-level overrides target-level; str is normalized to a list; blanks dropped
    manifest = {"target": {"exclude_patterns": ["target_pat"]}}
    assert rk._exclude_patterns(manifest, {}) == ["target_pat"]
    assert rk._exclude_patterns(manifest, {"exclude_patterns": "case_pat"}) == ["case_pat"]
    assert rk._exclude_patterns({"target": {}}, {}) == []
