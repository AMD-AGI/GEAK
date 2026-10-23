"""Observed extraction declaration shapes, with unchanged source-boundary guards."""
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import source_tree


# Compact copies of the eight observed schema shapes, without campaign paths/data.
OBSERVED = [
    ([{"device_kernel": "cross_device_reduce_2stage", "source_file": "kernel_src/csrc/include/custom_all_reduce.cuh"},
      {"device_kernel": "cross_device_reduce_1stage", "source_file": "kernel_src/csrc/include/custom_all_reduce.cuh"}],
     ["csrc/include/custom_all_reduce.cuh"]),
    ([{"device_kernel": name, "source_file": "kernel_src/unified_attention_kernels.py"}
      for name in ("kernel_unified_attention_2d", "kernel_unified_attention_3d", "reduce_segments")],
     ["unified_attention_kernels.py"]),
    ([{"device_kernel": "_fwd_kernel", "file": "kernel_src/extend_attention.py", "logical_boundary": "whole attention"}],
     ["extend_attention.py"]),
    ([{"device_kernel": "_fused_norm_residual_triton_kernel", "file": "kernel_src/fused_norm_residual.py", "what": "whole fused operation"}],
     ["fused_norm_residual.py"]),
    ([{"device_kernel": "create_flashinfer_kv_indices_triton", "file": "kernel_src/_musespark289_kv_impl.py", "scope": "copied computation"}],
     ["_musespark289_kv_impl.py"]),
    ([{"device_kernel": name, "source_file": "kernel_src/aiter_flydsl/" + file}
      for name, file in (("gate_up", "muse_moe_gemm.py"), ("down", "muse_moe_gemm.py"), ("finalize", "muse_moe_finalize.py"))],
     ["aiter_flydsl/muse_moe_gemm.py", "aiter_flydsl/muse_moe_finalize.py"]),
    ([{"device_kernel": "fused_muse_spark_routing_kernel_topk8", "file": "kernel_src/fused_muse_spark_routing.py", "what": "routing leaf"}],
     ["fused_muse_spark_routing.py"]),
    ([{"device_kernel": "quickreduce::allreduce_prototype_twoshot", "file": "kernel_src/csrc/quick_all_reduce.cuh", "what": "native arithmetic"}],
     ["csrc/quick_all_reduce.cuh"]),
]


def task_at(tmp_path, leaves, files=("a.py", "b.py"), section="kernel_extraction"):
    task = tmp_path / "task"
    for name in (*files, "wrapper.py", "cand_bind.py"):
        path = task / "kernel_src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU schema fixture\n")
    meta = {"candidate_bind": {"file": "kernel_src/wrapper.py"}}
    meta.setdefault(section, {})["editable_leaves"] = leaves
    (task / "meta.json").write_text(json.dumps(meta))
    return task


@pytest.mark.parametrize("leaves,expected", OBSERVED)
def test_all_observed_declared_shapes_select_computation_not_wrapper(tmp_path, leaves, expected):
    task = task_at(tmp_path, leaves, expected)
    assert source_tree.editable_sources(task) == expected


@pytest.mark.parametrize("section", ["kernel_extraction", "candidate_bind"])
@pytest.mark.parametrize("leaf", ["a.py", "kernel_src/a.py", {"source_file": "a.py"},
                                  {"file": "kernel_src/a.py"},
                                  {"source_file": "a.py", "file": "kernel_src/a.py"}])
def test_existing_strings_and_both_unambiguous_field_spellings(tmp_path, section, leaf):
    task = task_at(tmp_path, [leaf, "kernel_src/a.py"], section=section)
    assert source_tree.editable_sources(task) == ["a.py"]


@pytest.mark.parametrize("field", ["file", "source_file"])
@pytest.mark.parametrize("value", ["../a.py", "/tmp/a.py", "kernel_src/../a.py", "kernel_src/x/../../a.py",
                                   "kernel_src//a.py", "./a.py", "kernel_src/./a.py", "kernel_src/",
                                   "a\\b.py", "a b.py", "a\nb.py", "missing.py", "cand_bind.py",
                                   "", None, True, [], {}])
def test_malformed_or_unsafe_declared_paths_never_fall_back(tmp_path, field, value):
    task = task_at(tmp_path, [{field: value}])
    with pytest.raises(ValueError):
        source_tree.editable_sources(task)


@pytest.mark.parametrize("leaf", [{"source_file": "a.py", "file": "b.py"},
                                  {"source_file": None, "file": "a.py"},
                                  {"source_file": "a.py", "file": None},
                                  {"device_kernel": "missing-path"}])
def test_ambiguous_or_pathless_objects_are_rejected(tmp_path, leaf):
    task = task_at(tmp_path, [leaf])
    with pytest.raises(ValueError):
        source_tree.editable_sources(task)


@pytest.mark.parametrize("directory", [False, True])
def test_file_field_cannot_cross_source_symlink(tmp_path, directory):
    task = task_at(tmp_path, [{"file": "kernel_src/link/a.py" if directory else "kernel_src/link.py"}])
    external = tmp_path / "external"
    external.mkdir()
    (external / "a.py").write_text("# outside source tree\n")
    link = task / "kernel_src" / ("link" if directory else "link.py")
    link.symlink_to(external if directory else external / "a.py", target_is_directory=directory)
    with pytest.raises(ValueError, match="symlink"):
        source_tree.editable_sources(task)


def test_kernel_src_prefix_is_removed_exactly_once(tmp_path):
    task = task_at(tmp_path, [{"file": "kernel_src/kernel_src/a.py"}], ("kernel_src/a.py",))
    assert source_tree.editable_sources(task) == ["kernel_src/a.py"]
