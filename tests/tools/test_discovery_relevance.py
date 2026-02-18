"""Tests for discovery relevance scoring, directory handling, and per-kernel recommendations.

Covers scenarios discovered during development:

1. Single-file mode:
   - Triton kernel finds rope-related tests (not random tests)
   - Generic kernel.py uses parent dir name (not "kernel")
   - kernel_function param boosts tests containing that function
   - Non-existent path returns error

2. Directory mode:
   - Directory with .git is used as workspace (not parent)
   - Directory without .git expands upward correctly
   - Per-kernel recommendations: each kernel gets its own best test
   - Generic kernel.py names resolved per-kernel in directory mode

3. Relevance scoring:
   - Name match: kernel name in filename scores highest
   - Path match: kernel name in file path scores second
   - Path proximity: tests near the kernel score higher
   - Content match: tests referencing kernel function names score higher
   - No confidence cap: relevant tests visibly outrank generic ones
   - Irrelevant tests don't appear in top results for specific kernels

4. Edge cases:
   - Empty directory returns zero results
   - Large repo (many test files) returns relevant ones first
   - Directory mode with single kernel collapses to single dict
   - Multiple tests with same name but different paths are distinguishable
"""

import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers to build fake filesystem trees
# ---------------------------------------------------------------------------


def _write_triton_kernel(path: Path, func_name: str = "_my_kernel"):
    """Write a minimal Triton kernel file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(f"""\
        import triton
        import triton.language as tl

        @triton.jit
        def {func_name}(x_ptr, output_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offsets)
            tl.store(output_ptr + offsets, x * 2)
    """)
    )


def _write_test_file(path: Path, *, imports: str = "", has_pytest: bool = True, kernel_ref: str = "", extra: str = ""):
    """Write a test file that scores as a test by content-based detection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if has_pytest:
        lines.append("import pytest")
    lines.append("import torch")
    if imports:
        lines.append(imports)
    lines.append("")
    lines.append("def test_correctness():")
    lines.append("    x = torch.randn(128, device='cuda')")
    if kernel_ref:
        lines.append(f"    out = {kernel_ref}(x)")
    lines.append("    assert torch.allclose(x, x)")
    if extra:
        lines.append(extra)
    path.write_text("\n".join(lines))


def _write_bench_file(path: Path):
    """Write a benchmark file that scores as a benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent("""\
        import torch
        import time

        def bench():
            x = torch.randn(1024, device='cuda')
            warmup = 10
            for _ in range(warmup):
                y = x * 2
            elapsed = 0.0
            for _ in range(100):
                start = time.perf_counter()
                y = x * 2
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
            latency = elapsed / 100 * 1e6
            print(f"latency: {latency:.1f} us")
            print(f"throughput: {1024 / elapsed:.1f} elem/s")

        if __name__ == "__main__":
            bench()
    """)
    )


# ---------------------------------------------------------------------------
# MCP server tests (automated-test-discovery)
# ---------------------------------------------------------------------------


class TestMCPDiscoverSingleFile:
    """Single-file mode: kernel_path is a file."""

    def test_triton_kernel_finds_matching_test(self, tmp_path):
        """When a kernel named 'rope' exists, test_rope.py should rank #1."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "kernels" / "rope.py", "_rope_fwd")
        _write_test_file(tmp_path / "tests" / "test_rope.py", kernel_ref="rope_fwd")
        _write_test_file(tmp_path / "tests" / "test_gemm.py")
        _write_test_file(tmp_path / "tests" / "test_moe.py")

        result = discover(str(tmp_path / "kernels" / "rope.py"), use_llm=False)

        assert result["kernel"]["name"] == "rope"
        assert len(result["tests"]) > 0
        assert "rope" in result["tests"][0]["name"].lower()

    def test_generic_kernel_py_uses_parent_dir_name(self, tmp_path):
        """kernel.py should use parent dir name: gemm/kernel.py -> name='gemm'."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "gemm" / "kernel.py")

        result = discover(str(tmp_path / "gemm" / "kernel.py"), use_llm=False)

        assert result["kernel"]["name"] == "gemm", f"Expected kernel name 'gemm', got '{result['kernel']['name']}'"

    def test_kernel_function_param_boosts_matching_tests(self, tmp_path):
        """Tests containing the kernel_function name should rank higher."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py", "_rope_fwd_kernel")
        # test_rope.py references the function
        _write_test_file(
            tmp_path / "tests" / "test_rope.py",
            extra="    # uses _rope_fwd_kernel internally",
        )
        # test_other.py doesn't reference it
        _write_test_file(tmp_path / "tests" / "test_other.py")

        result = discover(
            str(tmp_path / "rope.py"),
            kernel_function="_rope_fwd_kernel",
            use_llm=False,
        )

        assert len(result["tests"]) >= 2
        assert "rope" in result["tests"][0]["name"].lower()

    def test_kernel_functions_extracted_from_source(self, tmp_path):
        """The result should include extracted function names from the kernel."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py", "_rope_fwd")

        result = discover(str(tmp_path / "rope.py"), use_llm=False)

        assert "_rope_fwd" in result["kernel"]["functions"]

    def test_nonexistent_path_returns_error(self):
        """Non-existent paths should return an error dict."""
        from automated_test_discovery.server import discover

        result = discover("/nonexistent/path.py", use_llm=False)

        assert "error" in result
        assert result["kernel"] is None

    def test_no_confidence_cap(self, tmp_path):
        """Relevant tests can score above 1.0 (no artificial cap)."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py", "_rope_fwd")
        _write_test_file(tmp_path / "tests" / "test_rope.py")

        result = discover(str(tmp_path / "rope.py"), use_llm=False)

        if result["tests"]:
            assert result["tests"][0]["confidence"] > 1.0, (
                "Relevant tests should score above 1.0 to visibly outrank generic ones"
            )

    def test_irrelevant_tests_not_in_top(self, tmp_path):
        """Unrelated tests should not appear above related ones."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")
        _write_test_file(tmp_path / "test_rope.py")
        _write_test_file(tmp_path / "test_moe_expert.py")
        _write_test_file(tmp_path / "test_gemm_a8w8.py")

        result = discover(str(tmp_path / "rope.py"), use_llm=False)

        if result["tests"]:
            assert "rope" in result["tests"][0]["name"].lower(), (
                f"Top test should be rope-related, got {result['tests'][0]['name']}"
            )

    def test_full_path_in_summary(self, tmp_path):
        """Summary should use full file path, not just filename."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")
        _write_test_file(tmp_path / "tests" / "test_rope.py")

        result = discover(str(tmp_path / "rope.py"), use_llm=False)

        if "Recommended test:" in result["summary"]:
            rec = result["summary"].split("Recommended test:")[1].strip()
            assert "/" in rec, "Recommended test should be a full path, not just a filename"


class TestMCPDiscoverDirectory:
    """Directory mode: kernel_path is a directory."""

    def test_directory_with_git_used_as_workspace(self, tmp_path):
        """A directory with .git should be the workspace itself."""
        from automated_test_discovery.server import _expand_workspace

        repo = tmp_path / "my_repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        ws = _expand_workspace(repo)
        assert ws == repo

    def test_directory_without_git_expands_upward(self, tmp_path):
        """A directory without markers should expand upward to find the project root."""
        from automated_test_discovery.server import _expand_workspace

        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "kernels"
        subdir.mkdir(parents=True)

        ws = _expand_workspace(subdir)
        assert ws == tmp_path

    def test_per_kernel_recommendations(self, tmp_path):
        """Each kernel should get its own recommended_test."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "kernels" / "rope.py", "_rope_fwd")
        _write_triton_kernel(tmp_path / "kernels" / "gemm.py", "_gemm_kernel")
        _write_test_file(tmp_path / "tests" / "test_rope.py")
        _write_test_file(tmp_path / "tests" / "test_gemm.py")

        result = discover(str(tmp_path / "kernels"), use_llm=False)

        kernels = result["kernel"] if isinstance(result["kernel"], list) else [result["kernel"]]
        assert len(kernels) == 2

        rope_kernel = next((k for k in kernels if k["name"] == "rope"), None)
        gemm_kernel = next((k for k in kernels if k["name"] == "gemm"), None)

        assert rope_kernel is not None, "Should find rope kernel"
        assert gemm_kernel is not None, "Should find gemm kernel"

        assert rope_kernel["recommended_test"] is not None
        assert gemm_kernel["recommended_test"] is not None
        assert "rope" in rope_kernel["recommended_test"]["name"].lower()
        assert "gemm" in gemm_kernel["recommended_test"]["name"].lower()

    def test_generic_kernel_py_names_resolved_in_dir_mode(self, tmp_path):
        """Generic kernel.py names should use parent dir in directory mode too."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope" / "kernel.py")
        _write_triton_kernel(tmp_path / "gemm" / "kernel.py")

        result = discover(str(tmp_path), use_llm=False)

        kernels = result["kernel"] if isinstance(result["kernel"], list) else [result["kernel"]]
        names = {k["name"] for k in kernels}

        assert "rope" in names, f"Expected 'rope' in kernel names, got {names}"
        assert "gemm" in names, f"Expected 'gemm' in kernel names, got {names}"
        assert "kernel" not in names, f"'kernel' should not be a kernel name, got {names}"

    def test_empty_directory_returns_zero(self, tmp_path):
        """Empty directory should return zero kernels and tests."""
        from automated_test_discovery.server import discover

        result = discover(str(tmp_path), use_llm=False)

        kernels = result.get("kernel", [])
        if isinstance(kernels, dict):
            kernels = [kernels]
        k_count = result.get("total_kernels_found", len(kernels))
        assert k_count == 0

    def test_single_kernel_collapses_to_dict(self, tmp_path):
        """Directory with single kernel should return kernel as dict, not list."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")

        result = discover(str(tmp_path), use_llm=False)

        assert isinstance(result["kernel"], dict), f"Single kernel should be a dict, got {type(result['kernel'])}"

    def test_per_kernel_summary(self, tmp_path):
        """Summary should list per-kernel recommendations."""
        from automated_test_discovery.server import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "kernels" / "rope.py")
        _write_triton_kernel(tmp_path / "kernels" / "gemm.py")
        _write_test_file(tmp_path / "tests" / "test_rope.py")
        _write_test_file(tmp_path / "tests" / "test_gemm.py")

        result = discover(str(tmp_path / "kernels"), use_llm=False)

        assert "Per-kernel recommendations" in result["summary"]


class TestRelevanceScoring:
    """Test the _relevance_score helper directly."""

    def test_name_in_filename_scores_highest(self, tmp_path):
        from automated_test_discovery.server import _relevance_score

        kernel = tmp_path / "rope.py"
        kernel.touch()
        test_match = tmp_path / "test_rope.py"
        test_match.touch()
        test_no_match = tmp_path / "test_gemm.py"
        test_no_match.touch()

        score_match = _relevance_score(test_match, kernel, "rope", ["rope"])
        score_no = _relevance_score(test_no_match, kernel, "rope", ["rope"])

        assert score_match > score_no, (
            f"Name-matched test ({score_match}) should score higher than non-matched ({score_no})"
        )

    def test_name_in_path_scores_second(self, tmp_path):
        from automated_test_discovery.server import _relevance_score

        kernel = tmp_path / "rope.py"
        kernel.touch()
        # "rope" is in the path but not the filename
        test_path_match = tmp_path / "rope_tests" / "test_fwd.py"
        test_path_match.parent.mkdir(parents=True)
        test_path_match.touch()
        test_no_match = tmp_path / "other" / "test_fwd.py"
        test_no_match.parent.mkdir(parents=True)
        test_no_match.touch()

        score_path = _relevance_score(test_path_match, kernel, "rope", ["rope"])
        score_no = _relevance_score(test_no_match, kernel, "rope", ["rope"])

        assert score_path > score_no

    def test_exact_stem_beats_substring(self, tmp_path):
        """test_gemm_a8w8.py should score higher than test_gemm_a8w8_blockscale.py
        for kernel gemm_a8w8 (exact stem match vs substring containment)."""
        from automated_test_discovery.server import _relevance_score

        kernel = tmp_path / "gemm_a8w8.py"
        kernel.touch()
        test_exact = tmp_path / "test_gemm_a8w8.py"
        test_exact.touch()
        test_substring = tmp_path / "test_gemm_a8w8_blockscale.py"
        test_substring.touch()

        score_exact = _relevance_score(test_exact, kernel, "gemm_a8w8", ["gemm", "a8w8"])
        score_sub = _relevance_score(test_substring, kernel, "gemm_a8w8", ["gemm", "a8w8"])

        assert score_exact > score_sub, (
            f"Exact match ({score_exact}) should beat substring ({score_sub}). "
            f"test_gemm_a8w8.py must rank above test_gemm_a8w8_blockscale.py for kernel gemm_a8w8"
        )

    def test_path_proximity_boosts_nearby_tests(self, tmp_path):
        from automated_test_discovery.server import _relevance_score

        # Kernel deep in the tree
        kernel = tmp_path / "project" / "src" / "ops" / "triton" / "rope.py"
        kernel.parent.mkdir(parents=True)
        kernel.touch()
        # Nearby test (same subtree, within 2 hops of shared parent)
        test_near = tmp_path / "project" / "src" / "ops" / "tests" / "test_something.py"
        test_near.parent.mkdir(parents=True)
        test_near.touch()
        # Far test (completely different subtree, > 4 hops from shared parent)
        test_far = tmp_path / "other_project" / "deep" / "nested" / "dir" / "subdir" / "test_something.py"
        test_far.parent.mkdir(parents=True)
        test_far.touch()

        score_near = _relevance_score(test_near, kernel, "something", [])
        score_far = _relevance_score(test_far, kernel, "something", [])

        assert score_near > score_far, f"Nearby test ({score_near}) should score higher than far test ({score_far})"


class TestDiscoveryPipelineDirectory:
    """Test the main discovery pipeline's directory handling."""

    def test_expand_workspace_for_dir_uses_dir_itself(self, tmp_path):
        """_expand_workspace_for_dir should use the given directory as workspace."""
        from minisweagent.tools.discovery import DiscoveryPipeline

        target = tmp_path / "my_project"
        target.mkdir()
        (target / ".git").mkdir()

        pipeline = DiscoveryPipeline(workspace_path=target)
        pipeline._expand_workspace_for_dir(target)

        assert pipeline.workspace == target

    def test_expand_workspace_for_dir_fallback(self, tmp_path):
        """Dir without markers should still use the dir itself (not parent)."""
        from minisweagent.tools.discovery import DiscoveryPipeline

        target = tmp_path / "some_subdir"
        target.mkdir()

        pipeline = DiscoveryPipeline(workspace_path=target)
        pipeline._expand_workspace_for_dir(target)

        assert pipeline.workspace == target, f"Should use dir itself as workspace, got {pipeline.workspace}"

    def test_kernel_name_uses_parent_for_generic(self, tmp_path):
        """Kernel files named kernel.py should use parent dir as name."""
        from minisweagent.tools.discovery import DiscoveryPipeline

        (tmp_path / ".git").mkdir()
        rope_dir = tmp_path / "rope"
        rope_dir.mkdir()
        kernel = rope_dir / "kernel.py"
        _write_triton_kernel(kernel, "_rope_fwd")

        pipeline = DiscoveryPipeline(workspace_path=tmp_path)
        result = pipeline.run(kernel_path=kernel, interactive=False)

        assert len(result.kernels) == 1
        assert result.kernels[0].kernel_name == "rope", f"Expected 'rope', got '{result.kernels[0].kernel_name}'"

    def test_directory_discovery_finds_nested_kernels(self, tmp_path):
        """Directory mode should recursively find kernels in subdirectories."""
        from minisweagent.tools.discovery import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "ops" / "rope" / "kernel.py", "_rope_fwd")
        _write_triton_kernel(tmp_path / "ops" / "gemm" / "kernel.py", "_gemm_kernel")
        _write_triton_kernel(tmp_path / "ops" / "norm" / "kernel.py", "_norm_kernel")

        result = discover(kernel_path=tmp_path, interactive=False)

        names = {k.kernel_name for k in result.kernels}
        assert "rope" in names
        assert "gemm" in names
        assert "norm" in names


class TestDiscoveryPatternExtraction:
    """Test that _extract_test_patterns works on discovered test files."""

    def test_extracts_tolerances(self, tmp_path):
        """Should extract atol/rtol values from test files."""
        from minisweagent.tools.discovery import DiscoveryPipeline

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")
        test = tmp_path / "test_rope.py"
        test.write_text(
            textwrap.dedent("""\
            import pytest
            import torch

            def test_correctness():
                x = torch.randn(128)
                torch.testing.assert_close(x, x, atol=1e-3, rtol=1e-4)
        """)
        )

        pipeline = DiscoveryPipeline(workspace_path=tmp_path)
        result = pipeline.run(kernel_path=tmp_path / "rope.py", interactive=False)

        matched = [t for t in result.tests if t.patterns is not None]
        assert len(matched) > 0, "Should extract patterns from at least one test"
        patterns = matched[0].patterns
        assert len(patterns.tolerances) > 0, "Should extract tolerance values"

    def test_extracts_dtypes(self, tmp_path):
        """Should extract torch dtype references from test files."""
        from minisweagent.tools.discovery import DiscoveryPipeline

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")
        test = tmp_path / "test_rope.py"
        test.write_text(
            textwrap.dedent("""\
            import pytest
            import torch

            def test_fp16():
                x = torch.randn(128, dtype=torch.float16)
                y = torch.randn(128, dtype=torch.bfloat16)
                assert True
        """)
        )

        pipeline = DiscoveryPipeline(workspace_path=tmp_path)
        result = pipeline.run(kernel_path=tmp_path / "rope.py", interactive=False)

        matched = [t for t in result.tests if t.patterns is not None]
        assert len(matched) > 0
        dtypes = matched[0].patterns.dtypes
        assert "torch.float16" in dtypes
        assert "torch.bfloat16" in dtypes


class TestFormatDiscoveryForAgent:
    """Test the enriched discovery context formatter."""

    def test_includes_kernel_analysis(self, tmp_path):
        """Formatted context should include kernel type and language."""
        from minisweagent.agents.unit_test_agent import format_discovery_for_agent
        from minisweagent.tools.discovery import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py", "_rope_fwd")

        result = discover(kernel_path=tmp_path / "rope.py", interactive=False)
        context = format_discovery_for_agent(result)

        assert "Kernel Analysis" in context
        assert "triton" in context.lower()

    def test_includes_language_guidance(self, tmp_path):
        """Formatted context should include language-specific testing guidance."""
        from minisweagent.agents.unit_test_agent import format_discovery_for_agent
        from minisweagent.tools.discovery import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")

        result = discover(kernel_path=tmp_path / "rope.py", interactive=False)
        context = format_discovery_for_agent(result)

        assert "Language-Specific Testing Guidance" in context
        assert "torch.testing.assert_close" in context

    def test_none_result_returns_empty(self):
        """None result should return empty string."""
        from minisweagent.agents.unit_test_agent import format_discovery_for_agent

        assert format_discovery_for_agent(None) == ""

    def test_includes_extracted_patterns(self, tmp_path):
        """Formatted context should include extracted test patterns when available."""
        from minisweagent.agents.unit_test_agent import format_discovery_for_agent
        from minisweagent.tools.discovery import discover

        (tmp_path / ".git").mkdir()
        _write_triton_kernel(tmp_path / "rope.py")
        test = tmp_path / "test_rope.py"
        test.write_text(
            textwrap.dedent("""\
            import pytest
            import torch

            def test_correctness():
                x = torch.randn(128, dtype=torch.float16)
                torch.testing.assert_close(x, x, atol=1e-3, rtol=1e-4)
        """)
        )

        result = discover(kernel_path=tmp_path / "rope.py", interactive=False)
        context = format_discovery_for_agent(result)

        assert "Extracted Test Patterns" in context or "torch.float16" in context
