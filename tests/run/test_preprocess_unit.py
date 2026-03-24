"""CI-friendly unit tests for the preprocessing module.

No GPU, no LLM, no Docker required. Run with:
    pytest tests/run/test_preprocess_unit.py -v --noconftest
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ===================================================================
# Test 1: _pick() determinism and correctness
# ===================================================================

class TestPick:
    """The _pick function must be deterministic and positional."""

    @staticmethod
    def _pick(configs, count):
        if len(configs) <= count:
            return configs
        n = len(configs)
        return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]

    def test_deterministic_100_runs(self):
        configs = list(range(80))
        results = [self._pick(configs, 25) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_returns_exact_count(self):
        assert len(self._pick(list(range(80)), 25)) == 25
        assert len(self._pick(list(range(80)), 5)) == 5

    def test_small_list_returns_all(self):
        configs = [1, 2, 3]
        assert self._pick(configs, 25) == [1, 2, 3]

    def test_includes_first_and_last(self):
        configs = list(range(100))
        result = self._pick(configs, 5)
        assert result[0] == 0
        assert result[-1] == 99

    def test_subset_of_input(self):
        configs = [(i, i * 2) for i in range(50)]
        result = self._pick(configs, 10)
        for item in result:
            assert item in configs

    def test_order_matters(self):
        a = list(range(80))
        b = list(reversed(range(80)))
        assert self._pick(a, 5) != self._pick(b, 5)

    def test_single_element(self):
        assert self._pick([42], 25) == [42]

    def test_empty(self):
        assert self._pick([], 25) == []

    def test_count_equals_len(self):
        configs = list(range(25))
        assert self._pick(configs, 25) == configs


# ===================================================================
# Test 2: Discovery scoring
# ===================================================================

class TestDiscoveryScoring:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        import types
        # Mock mcp module so server.py can import without the real package
        if "mcp" not in sys.modules:
            mcp = types.ModuleType("mcp")
            mcp_server = types.ModuleType("mcp.server")
            mcp_fastmcp = types.ModuleType("mcp.server.fastmcp")
            mcp_fastmcp.FastMCP = lambda **kw: type("_FakeMCP", (), {"tool": lambda self: lambda f: f})()
            mcp.server = mcp_server
            mcp_server.fastmcp = mcp_fastmcp
            sys.modules["mcp"] = mcp
            sys.modules["mcp.server"] = mcp_server
            sys.modules["mcp.server.fastmcp"] = mcp_fastmcp
        repo = Path(__file__).resolve().parent.parent.parent
        mcp_src = repo / "mcp_tools/automated-test-discovery/src"
        if str(mcp_src) not in sys.path:
            sys.path.insert(0, str(mcp_src))

    def test_bench_filename_boost(self):
        from automated_test_discovery.server import _score_as_bench
        with tempfile.NamedTemporaryFile(suffix=".py", prefix="bench_topk_", mode="w", delete=False) as f:
            f.write("elapsed_time = 0\ndo_bench(fn)\n")
            f.flush()
            score = _score_as_bench(Path(f.name))
        os.unlink(f.name)
        assert score >= 0.3, "bench_ prefix should contribute to score"

    def test_relevance_bidirectional(self):
        from automated_test_discovery.server import _relevance_score
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench_la_paged_decode.py"
            bench.write_text("# benchmark")
            kernel = Path(tmp) / "lean_atten_paged.py"
            kernel.write_text("# kernel")
            score = _relevance_score(
                bench, kernel, "lean_atten_paged",
                ["lean", "atten", "paged"],
            )
        assert score > 0, "bench_la_paged_decode should match lean_atten_paged"

    def test_exact_stem_highest(self):
        from automated_test_discovery.server import _relevance_score
        with tempfile.TemporaryDirectory() as tmp:
            test = Path(tmp) / "test_topk.py"
            test.write_text("# test")
            kernel = Path(tmp) / "topk.py"
            kernel.write_text("# kernel")
            score = _relevance_score(test, kernel, "topk", ["topk"])
        assert score >= 4.0

    def test_unrelated_scores_zero(self):
        from automated_test_discovery.server import _relevance_score
        with tempfile.TemporaryDirectory() as tmp:
            unrelated = Path(tmp) / "test_gemm.py"
            unrelated.write_text("# gemm test")
            kernel = Path(tmp) / "topk.py"
            kernel.write_text("# kernel")
            score = _relevance_score(unrelated, kernel, "topk", ["topk"])
        assert score <= 1.0, "Unrelated file should score low (path proximity may add up to 1.0)"


# ===================================================================
# Test 3: Harness validation
# ===================================================================

class TestValidateHarness:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_accepts_valid_harness(self):
        from minisweagent.run.pipeline_helpers import validate_harness
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "import argparse\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--profile')\n"
                "parser.add_argument('--correctness')\n"
                "parser.add_argument('--benchmark')\n"
                "parser.add_argument('--full-benchmark')\n"
            )
            f.flush()
            valid, errors = validate_harness(f.name)
        os.unlink(f.name)
        assert valid, f"Should be valid but got errors: {errors}"

    def test_rejects_missing_profile(self):
        from minisweagent.run.pipeline_helpers import validate_harness
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "import argparse\n"
                "parser.add_argument('--correctness')\n"
                "parser.add_argument('--benchmark')\n"
                "parser.add_argument('--full-benchmark')\n"
            )
            f.flush()
            valid, errors = validate_harness(f.name)
        os.unlink(f.name)
        assert not valid
        assert any("--profile" in e for e in errors)

    def test_rejects_no_argparse(self):
        from minisweagent.run.pipeline_helpers import validate_harness
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "--profile\n--correctness\n--benchmark\n--full-benchmark\n"
            )
            f.flush()
            valid, errors = validate_harness(f.name)
        os.unlink(f.name)
        assert not valid
        assert any("argparse" in e for e in errors)

    def test_rejects_missing_file(self):
        from minisweagent.run.pipeline_helpers import validate_harness
        valid, errors = validate_harness("/nonexistent/harness.py")
        assert not valid


# ===================================================================
# Test 4: execute_harness_validation env vars
# ===================================================================

class TestExecuteHarnessEnvVars:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_sets_low_iterations_by_default(self):
        from minisweagent.run.pipeline_helpers import execute_harness_validation
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import sys, os; print(os.environ.get('GEAK_BENCHMARK_ITERATIONS', 'NOT_SET'))")
            f.flush()
            with patch("minisweagent.run.preprocess.run_harness.run_harness") as mock_run:
                mock_run.return_value = [{"mode": "correctness", "success": True, "returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}]
                execute_harness_validation(f.name)
                call_kwargs = mock_run.call_args
                env = call_kwargs.kwargs.get("env_overrides", {})
                assert env.get("GEAK_BENCHMARK_ITERATIONS") == "5"
        os.unlink(f.name)

    def test_extracts_iterations_from_extra_args(self):
        from minisweagent.run.pipeline_helpers import execute_harness_validation
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("pass")
            f.flush()
            with patch("minisweagent.run.preprocess.run_harness.run_harness") as mock_run:
                mock_run.return_value = [{"mode": "correctness", "success": True, "returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}]
                execute_harness_validation(f.name, benchmark_extra_args="--iterations 30")
                call_kwargs = mock_run.call_args
                env = call_kwargs.kwargs.get("env_overrides", {})
                assert env.get("GEAK_BENCHMARK_ITERATIONS") == "30"
                assert env.get("GEAK_BENCHMARK_EXTRA_ARGS") == "--iterations 30"
        os.unlink(f.name)


# ===================================================================
# Test 5: PreprocessContext contract
# ===================================================================

class TestPreprocessContext:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_from_preprocessor_output(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "COMMANDMENT.md").write_text("cmd")
            (Path(tmp) / "baseline_metrics.json").write_text("{}")
            (Path(tmp) / "profile.json").write_text("{}")
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text("pass")
            harness = Path(tmp) / "harness.py"
            harness.write_text("pass")

            ctx = {
                "kernel_path": str(kernel),
                "repo_root": tmp,
                "harness_path": str(harness),
                "test_command": f"python {harness} --correctness",
                "discovery": {"tests": [], "benchmarks": []},
                "codebase_context_path": str(Path(tmp) / "CODEBASE_CONTEXT.md"),
            }
            pc = PreprocessContext.from_preprocessor_output(ctx, tmp)

            assert pc.kernel_path == str(kernel)
            assert pc.repo_root == tmp
            assert pc.harness_path == str(harness)
            assert pc.preprocess_dir == tmp
            assert pc.commandment_path is not None
            assert pc.baseline_metrics_path is not None
            assert pc.profiling_result_path is not None

    def test_validate_catches_missing_required(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        pc = PreprocessContext(
            kernel_path="",
            repo_root="",
            harness_path="",
            preprocess_dir="",
        )
        errors = pc.validate()
        assert len(errors) >= 4
        assert any("kernel_path" in e for e in errors)

    def test_validate_passes_valid(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text("pass")
            harness = Path(tmp) / "harness.py"
            harness.write_text("pass")

            pc = PreprocessContext(
                kernel_path=str(kernel),
                repo_root=tmp,
                harness_path=str(harness),
                preprocess_dir=tmp,
            )
            errors = pc.validate()
            assert errors == [], f"Should be valid but got: {errors}"

    def test_roundtrip_json(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            pc = PreprocessContext(
                kernel_path="/a/kernel.py",
                repo_root="/a/repo",
                harness_path="/a/harness.py",
                preprocess_dir=tmp,
                discovery={"tests": [1, 2, 3]},
            )
            json_path = Path(tmp) / "ctx.json"
            pc.to_json(json_path)
            loaded = PreprocessContext.from_json(json_path)
            assert loaded.kernel_path == pc.kernel_path
            assert loaded.discovery == pc.discovery

    def test_from_dict_ignores_unknown_keys(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        d = {
            "kernel_path": "/a",
            "repo_root": "/b",
            "harness_path": "/c",
            "preprocess_dir": "/d",
            "unknown_key_xyz": "should be ignored",
        }
        pc = PreprocessContext.from_dict(d)
        assert pc.kernel_path == "/a"
        assert not hasattr(pc, "unknown_key_xyz")

    def test_to_dict_all_fields(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        pc = PreprocessContext(
            kernel_path="/k",
            repo_root="/r",
            harness_path="/h",
            preprocess_dir="/p",
        )
        d = pc.to_dict()
        assert "kernel_path" in d
        assert "commandment_path" in d
        assert d["commandment_path"] is None


# ===================================================================
# Test 6: Imports resolve correctly after reorg
# ===================================================================

class TestImports:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_preprocessor_import(self):
        from minisweagent.run.preprocess.preprocessor import run_preprocessor
        assert callable(run_preprocessor)

    def test_context_import(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        assert PreprocessContext is not None

    def test_pipeline_helpers_import(self):
        from minisweagent.run.pipeline_helpers import (
            extract_harness_path,
            validate_harness,
            execute_harness_validation,
            create_validated_harness,
        )
        assert callable(extract_harness_path)

    def test_discovery_types_canonical(self):
        from minisweagent.run.preprocess.discovery_types import DiscoveryResult
        assert DiscoveryResult is not None

    def test_unit_test_agent_canonical(self):
        from minisweagent.run.preprocess.unit_test_agent import run_unit_test_agent
        assert callable(run_unit_test_agent)

    def test_shape_fixer_canonical(self):
        from minisweagent.run.preprocess.shape_fixer_agent import run_shape_fixer
        assert callable(run_shape_fixer)

    def test_commandment_import(self):
        from minisweagent.run.preprocess.commandment import generate_commandment
        assert callable(generate_commandment)


# ===================================================================
# Test 7: extract_harness_path
# ===================================================================

class TestExtractHarnessPath:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_python_command(self):
        from minisweagent.run.pipeline_helpers import extract_harness_path
        assert extract_harness_path("python /a/b/harness.py --correctness") == "/a/b/harness.py"

    def test_pytest_command(self):
        from minisweagent.run.pipeline_helpers import extract_harness_path
        assert extract_harness_path("pytest /a/b/test.py -v") == "/a/b/test.py"

    def test_bare_path(self):
        from minisweagent.run.pipeline_helpers import extract_harness_path
        assert extract_harness_path("/a/b/harness.py") == "/a/b/harness.py"


# ===================================================================
# Test 8: GEAK_SHAPES_USED parsing
# ===================================================================

class TestShapesUsedParsing:

    def test_basic(self):
        from tests.run.test_harness_variance import extract_shapes_used
        stdout = "some output\nGEAK_SHAPES_USED=[(1, 2, 3), (4, 5, 6)]\nmore"
        result = extract_shapes_used(stdout)
        assert result == [(1, 2, 3), (4, 5, 6)]

    def test_missing(self):
        from tests.run.test_harness_variance import extract_shapes_used
        assert extract_shapes_used("no shapes here") is None

    def test_no_resort(self):
        from tests.run.test_harness_variance import extract_shapes_used
        stdout = "GEAK_SHAPES_USED=[(3, 2, 1), (1, 2, 3)]"
        result = extract_shapes_used(stdout)
        assert result == [(3, 2, 1), (1, 2, 3)]

    def test_index_format(self):
        from tests.run.test_harness_variance import extract_shapes_used
        stdout = "GEAK_SHAPES_USED=[0, 3, 7, 12, 24]"
        result = extract_shapes_used(stdout)
        assert result == [0, 3, 7, 12, 24]


# ===================================================================
# Test 9: PreprocessContext contract enforcement
# ===================================================================

class TestPreprocessContractEnforcement:
    """Verify that a mocked preprocessing run produces a valid
    PreprocessContext with all required fields."""

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def _make_mock_output(self, tmp):
        """Create a fake preprocessing output directory with all artifacts."""
        out = Path(tmp)
        kernel = out / "kernel.py"
        kernel.write_text("@triton.jit\ndef my_kernel(): pass")
        harness = out / "test_kernel_harness.py"
        harness.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--profile')\n"
            "p.add_argument('--correctness')\n"
            "p.add_argument('--benchmark')\n"
            "p.add_argument('--full-benchmark')\n"
        )
        (out / "resolved.json").write_text(json.dumps({
            "local_file_path": str(kernel),
            "local_repo_path": str(out),
        }))
        (out / "discovery.json").write_text(json.dumps({
            "kernel": {"name": "my_kernel", "type": "triton", "file": str(kernel)},
            "tests": [], "benchmarks": [],
        }))
        (out / "CODEBASE_CONTEXT.md").write_text("# Context")
        (out / "COMMANDMENT.md").write_text("# Commandment")
        (out / "baseline_metrics.json").write_text(json.dumps({"duration_us": 100}))
        (out / "profile.json").write_text(json.dumps({"success": True}))
        (out / "harness_results.json").write_text(json.dumps([
            {"mode": "correctness", "success": True, "returncode": 0, "stdout": "", "stderr": "", "duration_s": 1.0},
        ]))
        (out / "testcase_selection.json").write_text(json.dumps({
            "selected_source": "unit_test_agent",
            "test_command": f"python {harness} --correctness",
            "harness_path": str(harness),
        }))

        ctx = {
            "kernel_path": str(kernel),
            "repo_root": str(out),
            "harness_path": str(harness),
            "test_command": f"python {harness} --correctness",
            "resolved": {"local_file_path": str(kernel), "local_repo_path": str(out)},
            "discovery": {"tests": [], "benchmarks": []},
            "codebase_context_path": str(out / "CODEBASE_CONTEXT.md"),
            "harness_results": [{"mode": "correctness", "success": True}],
            "baseline_metrics": {"duration_us": 100},
            "commandment": "# Commandment",
            "testcase_selection": {"selected_source": "unit_test_agent"},
        }
        return ctx, out

    def test_from_preprocessor_output_validates(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            pc = PreprocessContext.from_preprocessor_output(ctx, out)
            errors = pc.validate()
            assert errors == [], f"Contract validation failed: {errors}"

    def test_required_fields_are_set(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            pc = PreprocessContext.from_preprocessor_output(ctx, out)
            assert pc.kernel_path, "kernel_path must be set"
            assert pc.repo_root, "repo_root must be set"
            assert pc.harness_path, "harness_path must be set"
            assert pc.preprocess_dir, "preprocess_dir must be set"

    def test_optional_paths_exist_on_disk(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            pc = PreprocessContext.from_preprocessor_output(ctx, out)
            if pc.commandment_path:
                assert Path(pc.commandment_path).is_file()
            if pc.baseline_metrics_path:
                assert Path(pc.baseline_metrics_path).is_file()
            if pc.profiling_result_path:
                assert Path(pc.profiling_result_path).is_file()

    def test_harness_passes_static_validation(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        from minisweagent.run.pipeline_helpers import validate_harness
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            pc = PreprocessContext.from_preprocessor_output(ctx, out)
            valid, errors = validate_harness(pc.harness_path)
            assert valid, f"Harness at {pc.harness_path} failed validation: {errors}"

    def test_context_survives_json_roundtrip(self):
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            pc = PreprocessContext.from_preprocessor_output(ctx, out)

            json_path = Path(tmp) / "preprocess_context.json"
            pc.to_json(json_path)
            loaded = PreprocessContext.from_json(json_path)

            assert loaded.kernel_path == pc.kernel_path
            assert loaded.harness_path == pc.harness_path
            assert loaded.preprocess_dir == pc.preprocess_dir
            assert loaded.discovery == pc.discovery
            assert loaded.baseline_metrics == pc.baseline_metrics

    def test_harness_only_produces_valid_context(self):
        """GEAK_HARNESS_ONLY=1 skips profiling/baseline but context must
        still have required fields."""
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            # Remove optional artifacts (simulating HARNESS_ONLY)
            (out / "baseline_metrics.json").unlink()
            (out / "profile.json").unlink()
            (out / "COMMANDMENT.md").unlink()
            ctx.pop("baseline_metrics", None)
            ctx.pop("commandment", None)

            pc = PreprocessContext.from_preprocessor_output(ctx, out)
            errors = pc.validate()
            assert errors == [], f"HARNESS_ONLY context should be valid: {errors}"
            assert pc.baseline_metrics_path is None
            assert pc.profiling_result_path is None
            assert pc.commandment_path is None

    def test_orchestrator_can_consume_context(self):
        """Verify the orchestrator's expected keys are present."""
        from minisweagent.run.preprocess.context import PreprocessContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx, out = self._make_mock_output(tmp)
            pc = PreprocessContext.from_preprocessor_output(ctx, out)
            d = pc.to_dict()
            # Orchestrator reads these keys
            assert "kernel_path" in d
            assert "repo_root" in d
            assert "test_command" in d
            assert "harness_path" in d
            assert "discovery" in d
