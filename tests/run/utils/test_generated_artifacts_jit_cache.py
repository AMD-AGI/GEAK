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
    baseline_jit_cache_env,
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
    for ext in ("*.hsaco", "*.amdgcn", "*.llir", "*.ttir", "*.ttgir", "*.ptx", "*.cubin", "*.spv", "*.o", "*.so"):
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


# ---------------------------------------------------------------------------
# baseline_jit_cache_env — persistent, repo-keyed cache for the baseline path
# ---------------------------------------------------------------------------
def test_baseline_jit_cache_env_keys_and_nesting(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = baseline_jit_cache_env(repo)
    # Adds AITER_JIT_DIR on top of the two jit_cache_env keys.
    assert set(env) == {"GEAK_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "AITER_JIT_DIR"}
    # Lives under a dedicated "baseline" subtree so it never collides with the
    # per-worktree slot caches.
    assert "/baseline/" in env["GEAK_JIT_CACHE_DIR"]
    assert env["TRITON_CACHE_DIR"].startswith(env["GEAK_JIT_CACHE_DIR"])
    assert env["TRITON_CACHE_DIR"].endswith("/triton")
    assert env["AITER_JIT_DIR"].endswith("/aiter")


def test_baseline_jit_cache_env_keyed_by_repo_not_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    # Stable per repo (warm-up, verifier, collect_baseline all share it).
    assert baseline_jit_cache_env(repo) == baseline_jit_cache_env(repo)
    # Distinct from the per-worktree keying so a baseline cache is never confused
    # with a patched per-slot cache.
    assert baseline_jit_cache_env(repo)["GEAK_JIT_CACHE_DIR"] != jit_cache_env(repo)["GEAK_JIT_CACHE_DIR"]


def test_baseline_jit_cache_env_base_override_and_falsy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = tmp_path / "root"
    env = baseline_jit_cache_env(repo, base=base)
    assert env["GEAK_JIT_CACHE_DIR"].startswith(str(base))
    assert baseline_jit_cache_env(None) == {}


# ---------------------------------------------------------------------------
# strip_excluded_sections — post-render replacement for --no-index pathspecs
# ---------------------------------------------------------------------------
def _section(path: str, mode: str = "100644") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"new file mode {mode}\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n+x\n"
    )


def test_strip_excluded_sections_drops_basename_and_keeps_source() -> None:
    from minisweagent.run.utils.generated_artifacts import strip_excluded_sections

    patch = _section("kernel.py") + _section("__pycache__/m.pyc")
    out, removed = strip_excluded_sections(patch, ["__pycache__"])
    assert "kernel.py" in out
    assert "__pycache__" not in out
    assert removed == ["__pycache__/m.pyc"]


def test_strip_excluded_sections_matches_glob_and_nested_segment() -> None:
    from minisweagent.run.utils.generated_artifacts import strip_excluded_sections

    patch = _section("kernel.py") + _section("a/b/libfoo.so") + _section("nested/.git/HEAD")
    out, removed = strip_excluded_sections(patch, ["*.so", ".git"])
    assert "kernel.py" in out
    assert "libfoo.so" not in out
    assert ".git" not in out
    assert set(removed) == {"a/b/libfoo.so", "nested/.git/HEAD"}


def test_strip_excluded_sections_empty_excludes_is_noop() -> None:
    from minisweagent.run.utils.generated_artifacts import strip_excluded_sections

    patch = _section("kernel.py")
    out, removed = strip_excluded_sections(patch, [])
    assert out == patch
    assert removed == []


# ---------------------------------------------------------------------------
# _section_is_binary / strip_generated_helper_sections — binary sections must
# be dropped REGARDLESS of path, for BOTH git binary-diff markers. The default
# ``git diff`` (no --binary) emits "Binary files a/.. and b/.. differ", which
# ``git apply`` cannot apply; failing to strip it aborts the whole patch. This
# is the name-agnostic backstop so new build/cache dirs don't each need adding
# to the denylist.
# ---------------------------------------------------------------------------
def _binary_section(path: str, marker: str) -> str:
    head = f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n"
    if marker == "default":
        return head + f"Binary files a/{path} and b/{path} differ\n"
    return head + "GIT binary patch\nliteral 4\nzcmZ\n\n"


@pytest.mark.parametrize("marker", ["default", "git_binary"])
def test_section_is_binary_detects_both_markers(marker: str) -> None:
    from minisweagent.run.utils.generated_artifacts import _section_is_binary

    section = _binary_section("anything/at/all.bin", marker).splitlines(keepends=True)
    assert _section_is_binary(section) is True


def test_section_is_binary_false_for_source() -> None:
    from minisweagent.run.utils.generated_artifacts import _section_is_binary

    assert _section_is_binary(_section("kernel.py").splitlines(keepends=True)) is False


@pytest.mark.parametrize("build_dir", ["_geak_build", "some_future_build_dir", ".new_cache"])
def test_strip_generated_helper_drops_default_binary_any_path(build_dir: str) -> None:
    """The actual regression: default-format binary artifacts in an unknown dir
    are stripped while the real kernel source survives."""
    from minisweagent.run.utils.generated_artifacts import strip_generated_helper_sections

    patch = (
        _section("src/ball_query.hip")
        + _binary_section(f"{build_dir}/.ninja_deps", "default")
        + _binary_section(f"{build_dir}/out.so", "default")
    )
    out, removed = strip_generated_helper_sections(patch)
    assert "src/ball_query.hip" in out
    assert build_dir not in out
    assert set(removed) == {f"{build_dir}/.ninja_deps", f"{build_dir}/out.so"}
