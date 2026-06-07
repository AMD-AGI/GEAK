"""Unit tests for the JIT/compile-cache artifact denylist + cache relocation.

These pin the GEAK Layer-1/Layer-2 patch-capture fix:

* ``is_jit_cache_artifact`` must recognise JIT runtime caches by exact dir
  name, dir *prefix* (suffixed cache dirs like ``.triton_cache_geak``), dir
  *substring* (any ``*cache*`` dir), nested ``jit/{build,__pycache__}`` trees,
  and unambiguous compile-output extensions (``*.hsaco`` ... ``*.so``) -- while
  NEVER misclassifying genuine source files (including aiter cpp_itfs ``.cuh``
  source and files whose *name* merely contains "cache").
* ``jit_cache_diff_basename_excludes`` must surface every one of those as a
  ``diff``/``git diff`` exclude glob.
* ``jit_cache_env`` must relocate Triton/JIT caches to a stable, per-worktree
  dir OUTSIDE the worktree (the root cause of the 1.0x A/B self-score).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.run.utils.generated_artifacts import (
    is_jit_cache_artifact,
    jit_cache_diff_basename_excludes,
    jit_cache_env,
)


# ---------------------------------------------------------------------------
# is_jit_cache_artifact — POSITIVE (must be flagged as cache / compile output)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        # Exact cache-dir names.
        "flydsl_cache/abc.pkl",
        "__pycache__/mod.cpython-312.pyc",
        # Suffixed Triton cache dir (the literal regression: TRITON_CACHE_DIR
        # pointed at a worktree-local .triton_cache_geak).
        ".triton_cache_geak/HASH/fused_moe_kernel.hsaco",
        ".triton_cache_geak/HASH/fused_moe_kernel.json",
        "triton_cache_slot0/HASH/x.ttir",
        # aiter caches by prefix.
        ".aiter_jit/build/module_x/lib.so",
        ".aiter/build/pa_ragged_HASH/lib.so",
        # torch compile / inductor caches.
        "torch_compile_cache/x/y.py",
        "torchinductor_root/aaa/bbb.py",
        # Any *cache* dir via substring.
        "some_weird_cache/x.json",
        "pkg/sub_cache_dir/payload.bin",
        ".cache/triton/HASH/k.hsaco",
        # Nested jit/build tree.
        "aiter/jit/build/module_a/lib.so",
        # Compile-output extensions anywhere (leaf-suffix rule).
        "deep/dir/kernel.hsaco",
        "deep/dir/kernel.amdgcn",
        "deep/dir/kernel.llir",
        "deep/dir/kernel.ttir",
        "deep/dir/kernel.ttgir",
        "deep/dir/kernel.ptx",
        "deep/dir/kernel.cubin",
        "deep/dir/kernel.spv",
        "deep/dir/kernel.o",
        "deep/dir/kernel.so",
        # Leading ./ and / are tolerated.
        "./.triton_cache_geak/HASH/k.hsaco",
        "/abs/.triton/HASH/k.hsaco",
    ],
)
def test_is_jit_cache_artifact_positive(rel_path: str) -> None:
    assert is_jit_cache_artifact(rel_path) is True


# ---------------------------------------------------------------------------
# is_jit_cache_artifact — NEGATIVE (genuine source must survive capture)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        "fused_moe.py",
        "python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py",
        # aiter cpp_itfs SOURCE — the actual editable kernel; must NOT be stripped.
        "aiter/csrc/cpp_itfs/pa/pa_kernels.cuh",
        "aiter/csrc/cpp_itfs/pa/pa_ragged.cpp.jinja",
        # Files whose NAME merely contains a cache prefix/substring (leaf only).
        "triton_cache_config.py",
        "utils/cache_utils.py",
        "moe/caching.py",
        # Ordinary kernel source extensions.
        "kernel.cuh",
        "kernel.cpp",
        "kernel.cu",
        "kernel.h",
        "kernel.hip",
        "",
        None,
    ],
)
def test_is_jit_cache_artifact_negative(rel_path) -> None:
    assert is_jit_cache_artifact(rel_path) is False


# ---------------------------------------------------------------------------
# jit_cache_diff_basename_excludes
# ---------------------------------------------------------------------------
def test_diff_basename_excludes_cover_plan_patterns() -> None:
    patterns = set(jit_cache_diff_basename_excludes())
    # Prefix globs.
    for expected in (".triton*", ".aiter*", "torch_compile*", "torchinductor*"):
        assert expected in patterns, f"missing prefix glob {expected!r}"
    # Substring glob.
    assert "*cache*" in patterns
    # Exact dir names.
    assert "__pycache__" in patterns
    assert "flydsl_cache" in patterns
    # Compile-output extension globs (incl. the newly-added *.o / *.so).
    for ext in ("*.hsaco", "*.amdgcn", "*.llir", "*.ttir", "*.ttgir",
                "*.ptx", "*.cubin", "*.spv", "*.o", "*.so"):
        assert ext in patterns, f"missing suffix glob {ext!r}"


# ---------------------------------------------------------------------------
# jit_cache_env — relocate caches OUTSIDE the worktree
# ---------------------------------------------------------------------------
def test_jit_cache_env_relocates_outside_worktree(tmp_path: Path) -> None:
    wt = tmp_path / "worktrees" / "slot_0"
    wt.mkdir(parents=True)
    env = jit_cache_env(wt)

    assert set(env) == {"GEAK_JIT_CACHE_DIR", "TRITON_CACHE_DIR"}
    wt_resolved = str(wt.resolve())
    # The crux: neither cache dir may live inside the diffed worktree.
    assert not env["GEAK_JIT_CACHE_DIR"].startswith(wt_resolved)
    assert not env["TRITON_CACHE_DIR"].startswith(wt_resolved)
    # Triton cache nests under the generic root.
    assert env["TRITON_CACHE_DIR"].startswith(env["GEAK_JIT_CACHE_DIR"])
    assert env["TRITON_CACHE_DIR"].endswith("/triton")


def test_jit_cache_env_is_deterministic_and_unique(tmp_path: Path) -> None:
    wt_a = tmp_path / "a"
    wt_b = tmp_path / "b"
    wt_a.mkdir()
    wt_b.mkdir()
    # Stable per worktree (slot reuses its compile cache across rounds).
    assert jit_cache_env(wt_a) == jit_cache_env(wt_a)
    # Unique per worktree (parallel slots never collide).
    assert jit_cache_env(wt_a)["TRITON_CACHE_DIR"] != jit_cache_env(wt_b)["TRITON_CACHE_DIR"]


def test_jit_cache_env_base_override(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    base = tmp_path / "out_of_tree_root"
    env = jit_cache_env(wt, base=base)
    assert env["GEAK_JIT_CACHE_DIR"].startswith(str(base))


@pytest.mark.parametrize("falsy", [None, "", 0])
def test_jit_cache_env_empty_for_falsy_workdir(falsy) -> None:
    assert jit_cache_env(falsy) == {}
