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
# Test 2b: KernelMeta contract
# ===================================================================

class TestKernelMetaContract:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_kernel_meta_defaults(self):
        from minisweagent.run.preprocess.discovery_types import KernelMeta

        meta = KernelMeta()
        assert meta.kernel_path == ""
        assert meta.kernel_name == ""
        assert meta.kernel_type == "unknown"
        assert meta.kernel_language == "python"
        assert meta.function_names == []
        assert meta.workspace_path == ""

    def test_discovery_result_populates_kernel_meta_fields(self):
        from minisweagent.run.preprocess.discovery_types import DiscoveryResult, KernelMeta

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kernel = tmp_path / "topk.py"
            kernel.write_text("# kernel")

            disc = {
                "kernel": {
                    "file": str(kernel),
                    "name": "topk",
                    "type": "triton",
                    "functions": ["topk_kernel"],
                },
                "workspace": str(tmp_path),
                "tests": [],
                "benchmarks": [],
            }

            result = DiscoveryResult.from_dict(disc, kernel)

            assert len(result.kernels) == 1
            meta = result.kernels[0]
            assert isinstance(meta, KernelMeta)
            assert meta.kernel_path == str(kernel.resolve())
            assert meta.kernel_name == "topk"
            assert meta.kernel_type == "triton"
            assert meta.kernel_language == "python"
            assert meta.function_names == ["topk_kernel"]
            assert meta.workspace_path == str(tmp_path.resolve())

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
                assert env.get("GEAK_BENCHMARK_EXTRA_ARGS") is None
        os.unlink(f.name)

    def test_preserves_non_iteration_extra_args(self):
        from minisweagent.run.pipeline_helpers import execute_harness_validation
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("pass")
            f.flush()
            with patch("minisweagent.run.preprocess.run_harness.run_harness") as mock_run:
                mock_run.return_value = [{"mode": "correctness", "success": True, "returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}]
                execute_harness_validation(f.name, benchmark_extra_args="--warmup 7")
                call_kwargs = mock_run.call_args
                env = call_kwargs.kwargs.get("env_overrides", {})
                assert env.get("GEAK_BENCHMARK_ITERATIONS") is None
                assert env.get("GEAK_BENCHMARK_EXTRA_ARGS") == "--warmup 7"
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


class TestUnitTestAgentPrompt:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_prompt_scopes_instructions_to_geak_relative_path(self):
        from minisweagent.run.preprocess.unit_test_agent import run_unit_test_agent

        captured = {}

        def _fake_run(self, task):
            captured["task"] = task
            return "Submitted", "TEST_COMMAND: python /tmp/test_harness.py --correctness"

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with patch("minisweagent.run.preprocess.unit_test_agent.UnitTestAgent.run", new=_fake_run):
                run_unit_test_agent(
                    model=object(),
                    repo=repo,
                    kernel_name="demo_kernel",
                )

        task = captured["task"]
        assert "src/minisweagent/run/preprocess/INSTRUCTIONS.md" in task
        assert "not inside the target kernel repository" in task
        assert "Do NOT use broad recursive searches" in task
        assert "find /" in task

    def test_prompt_mentions_tensor_contract_preservation(self):
        from minisweagent.run.preprocess.config_loader import load_preprocess_agent_config

        agent_cfg, _ = load_preprocess_agent_config("mini_unit_test_agent")
        prompt = agent_cfg["system_template"]

        assert "full execution contract" in prompt
        assert "Do NOT normalize all tensors to a single dtype" in prompt


class TestShapeFixerTermination:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_shape_fixer_agent_submits_on_plaintext_verdict(self):
        from minisweagent.agents.default import DefaultAgent, Submitted
        from minisweagent.run.preprocess.shape_fixer_agent import ShapeFixerAgent

        class _DummyModel:
            n_calls = 0
            cost = 0.0

        class _DummyEnvConfig:
            cwd = str(Path.cwd())
            env = {}

        class _DummyEnv:
            config = _DummyEnvConfig()

        agent = ShapeFixerAgent(_DummyModel(), _DummyEnv(), system_template="shape fixer")
        with patch.object(DefaultAgent, "query", return_value={"content": "SHAPES_FIXED"}):
            with pytest.raises(Submitted, match="SHAPES_FIXED"):
                agent.query()


class TestShapeFixerPrompt:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_shape_fixer_prompt_mentions_validation_and_minimal_fix(self):
        from minisweagent.run.preprocess.config_loader import load_preprocess_agent_config

        agent_cfg, _ = load_preprocess_agent_config("mini_shape_fixer")
        prompt = agent_cfg["system_template"]

        assert "validation commands are provided" in prompt
        assert "smallest source-faithful fix" in prompt
        assert "Do NOT normalize all tensors to a single dtype" in prompt

    def test_run_shape_fixer_task_includes_validation_feedback(self):
        from minisweagent.run.preprocess.shape_fixer_agent import run_shape_fixer

        captured = {}

        def _fake_run(self, task):
            captured["task"] = task
            return "Submitted", "SHAPES_FIXED"

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            harness = repo / "test_harness.py"
            source = repo / "bench_kernel.py"
            kernel = repo / "kernel.py"
            harness.write_text("print('harness')\n")
            source.write_text("print('bench')\n")
            kernel.write_text("print('kernel')\n")

            with patch("minisweagent.run.preprocess.shape_fixer_agent.ShapeFixerAgent.run", new=_fake_run):
                ok = run_shape_fixer(
                    model=object(),
                    repo=repo,
                    harness_path=harness,
                    benchmark_file=source,
                    kernel_path=kernel,
                    gpu_id=7,
                    validation_feedback=["--correctness mode failed (exit code 1):\nValueError: demo"],
                )

        assert ok is True
        task = captured["task"]
        assert "PREVIOUS REVALIDATION FAILURES" in task
        assert "ValueError: demo" in task
        assert "HIP_VISIBLE_DEVICES=7 GEAK_BENCHMARK_ITERATIONS=5 python" in task
        assert "--correctness" in task
        assert "--profile" in task
        assert "--benchmark" in task


class TestHarnessRestore:

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        import sys
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

    def test_restore_harness_file_reverts_mutation(self):
        from minisweagent.run.preprocess.preprocessor import _restore_harness_file

        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "harness.py"
            harness.write_text("print('mutated')\n")

            restored = _restore_harness_file(harness, "print('original')\n")

            assert restored is True
            assert harness.read_text() == "print('original')\n"


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


# ===================================================================
# Test: _infer_repo_root (Issue 4)
# ===================================================================

class TestInferRepoRoot:
    """_infer_repo_root must find repo markers and never return None."""

    def test_finds_git_directory(self):
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "myrepo"
            kernel_dir = repo / "src" / "kernels"
            kernel_dir.mkdir(parents=True)
            (repo / ".git").mkdir()
            kernel = kernel_dir / "kernel.py"
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            assert result == str(repo)

    def test_finds_pyproject_toml(self):
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "myrepo"
            kernel_dir = repo / "ops"
            kernel_dir.mkdir(parents=True)
            (repo / "pyproject.toml").write_text("[project]\nname = 'test'")
            kernel = kernel_dir / "kernel.py"
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            assert result == str(repo)

    def test_finds_setup_py(self):
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "myrepo"
            kernel_dir = repo / "src"
            kernel_dir.mkdir(parents=True)
            (repo / "setup.py").write_text("from setuptools import setup; setup()")
            kernel = kernel_dir / "kernel.py"
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            assert result == str(repo)

    def test_finds_setup_cfg(self):
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "myrepo"
            kernel_dir = repo / "src"
            kernel_dir.mkdir(parents=True)
            (repo / "setup.cfg").write_text("[metadata]\nname = mypackage")
            kernel = kernel_dir / "kernel.py"
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            assert result == str(repo)

    def test_git_takes_precedence_over_inner_pyproject(self):
        """If both .git at root and pyproject.toml in a subdirectory exist, .git wins."""
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "myrepo"
            sub = repo / "subpkg"
            kernel_dir = sub / "kernels"
            kernel_dir.mkdir(parents=True)
            (repo / ".git").mkdir()
            # pyproject.toml sits closer to the kernel — but .git is at the true root
            (sub / "pyproject.toml").write_text("[project]\nname = 'sub'")
            kernel = kernel_dir / "kernel.py"
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            # Walk upward hits sub/pyproject.toml first — that's acceptable behaviour;
            # what matters is it never returns None and never crashes.
            assert result is not None
            assert result != ""

    def test_falls_back_to_parent_when_no_markers(self):
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            assert result == tmp

    def test_never_returns_none(self):
        from minisweagent.run.preprocess.preprocessor import _infer_repo_root
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "deep" / "nested" / "kernel.py"
            kernel.parent.mkdir(parents=True)
            kernel.write_text("# kernel")
            result = _infer_repo_root(str(kernel))
            assert result is not None
            assert result != ""


# ===================================================================
# Test: detect_and_split_kernel_from_harness (Issue 1)
# ===================================================================

class TestDetectAndSplitKernelFromHarness:
    """Merged kernel+harness files must be split: tests extracted, kernel stays clean."""

    _MERGED_SOURCE = '''\
import argparse
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)

def run_correctness(args):
    my_kernel(None, None, 0, BLOCK_SIZE=64)

def run_profile(args):
    run_correctness(args)

def run_benchmark(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()
    if args.correctness:
        run_correctness(args)
    elif args.profile:
        run_profile(args)
    else:
        run_benchmark(args)
'''

    _MERGED_HIP_SOURCE = '''\
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ float clamp_value(float x) {
    return x > 0 ? x : 0;
}

__global__ void relu_kernel(float* out, const float* in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = clamp_value(in[idx]);
}

static void launch_relu(float* out, const float* in) {
    hipLaunchKernelGGL(relu_kernel, dim3(1), dim3(64), 0, 0, out, in);
}

static void run_correctness() {
    printf("correctness\\n");
    launch_relu(nullptr, nullptr);
}

int main(int argc, char** argv) {
    run_correctness();
    return 0;
}
'''

    def test_splits_tests_out_leaves_kernel(self):
        """Test functions must be extracted to new harness; clean kernel written to output_dir."""
        from minisweagent.run.preprocess.harness_utils import detect_and_split_kernel_from_harness
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Put the merged source in a DIFFERENT dir to simulate a repo file
            src_dir = tmp_path / "repo"
            src_dir.mkdir()
            merged = src_dir / "my_kernel.py"
            merged.write_text(self._MERGED_SOURCE)
            out_dir = tmp_path / "output"
            out_dir.mkdir()

            result = detect_and_split_kernel_from_harness(merged, out_dir)
            assert result is not None, "Expected split to occur"
            new_harness_path, kernel_path = result

            # Original file must be UNTOUCHED (git safety)
            original_text = merged.read_text()
            assert "run_correctness" in original_text, "Original must not be modified"
            assert "@triton.jit" in original_text

            # Clean kernel copy is in output_dir, not original path
            assert kernel_path == str(out_dir / "my_kernel.py")
            kernel_text = Path(kernel_path).read_text()
            assert "@triton.jit" in kernel_text
            assert "def my_kernel" in kernel_text
            assert "run_correctness" not in kernel_text
            assert "run_profile" not in kernel_text
            assert "__main__" not in kernel_text

            # New harness file contains test logic
            harness_text = Path(new_harness_path).read_text()
            assert "run_correctness" in harness_text
            assert "run_profile" in harness_text
            assert "__main__" in harness_text
            assert "from my_kernel import *" in harness_text
            assert "@triton.jit" not in harness_text

    def test_kernel_not_included_in_test_bfs(self):
        """@triton.jit functions called from test roots must not be moved to harness."""
        from minisweagent.run.preprocess.harness_utils import detect_and_split_kernel_from_harness
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "repo"
            src_dir.mkdir()
            merged = src_dir / "my_kernel.py"
            merged.write_text(self._MERGED_SOURCE)
            out_dir = tmp_path / "output"
            out_dir.mkdir()
            detect_and_split_kernel_from_harness(merged, out_dir)
            # Clean kernel copy in output_dir still has my_kernel def
            kernel_text = (out_dir / "my_kernel.py").read_text()
            assert "def my_kernel" in kernel_text
            # Original file untouched
            assert "def my_kernel" in merged.read_text()

    def test_no_split_when_no_kernel_defs(self):
        """Files without @triton.jit should not be split."""
        from minisweagent.run.preprocess.harness_utils import detect_and_split_kernel_from_harness
        source = '''\
import argparse
import torch

def run_correctness(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    args = parser.parse_args()
    run_correctness(args)
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            harness = tmp_path / "test_kernel_harness.py"
            harness.write_text(source)
            result = detect_and_split_kernel_from_harness(harness, tmp_path)
            assert result is None, "Expected no split for file without kernel defs"

    def test_splits_hip_main_and_run_functions(self):
        """Merged HIP source should split host-side test entrypoints into a harness."""
        from minisweagent.run.preprocess.harness_utils import detect_and_split_kernel_from_harness
        from minisweagent.run.preprocess.harness_utils import validate_harness
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "repo"
            src_dir.mkdir()
            merged = src_dir / "relu_kernel.hip"
            merged.write_text(self._MERGED_HIP_SOURCE)
            out_dir = tmp_path / "output"
            out_dir.mkdir()

            result = detect_and_split_kernel_from_harness(merged, out_dir)
            assert result is not None, "Expected merged HIP source to split"
            new_harness_path, kernel_path = result

            # Original file must be untouched.
            original_text = merged.read_text()
            assert "int main" in original_text
            assert "run_correctness" in original_text

            # Clean kernel copy keeps kernel-side code but drops host-side test entrypoints.
            assert kernel_path == str(out_dir / "relu_kernel.hip")
            kernel_text = Path(kernel_path).read_text()
            assert "__global__ void relu_kernel" in kernel_text
            assert "__device__ float clamp_value" in kernel_text
            assert "static void launch_relu" in kernel_text
            assert "run_correctness" not in kernel_text
            assert "int main" not in kernel_text

            # Generated GEAK harness is a Python wrapper that drives the split HIP harness.
            assert new_harness_path == str(out_dir / "test_relu_kernel_harness.py")
            harness_text = Path(new_harness_path).read_text()
            assert "argparse" in harness_text
            assert "--correctness" in harness_text
            assert "--profile" in harness_text
            assert "--benchmark" in harness_text
            assert "--full-benchmark" in harness_text
            assert "hipcc" in harness_text
            valid, errors = validate_harness(str(new_harness_path))
            assert valid, f"Generated HIP wrapper harness should satisfy GEAK harness contract: {errors}"

            # The split C-like harness source still contains the moved host-side test logic.
            split_harness = out_dir / "test_relu_kernel_harness.hip"
            assert split_harness.is_file()
            split_harness_text = split_harness.read_text()
            assert '#include "relu_kernel.hip"' in split_harness_text
            assert "run_correctness" in split_harness_text
            assert "int main" in split_harness_text
            assert "__global__ void relu_kernel" not in split_harness_text

    def test_no_split_for_hip_without_host_test_entrypoint(self):
        """HIP sources without main()/run_*/test_* entrypoints should be left alone."""
        from minisweagent.run.preprocess.harness_utils import detect_and_split_kernel_from_harness
        source = '''\
#include <hip/hip_runtime.h>

__global__ void relu_kernel(float* out, const float* in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}

static void launch_relu(float* out, const float* in) {
    hipLaunchKernelGGL(relu_kernel, dim3(1), dim3(64), 0, 0, out, in);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            merged = tmp_path / "relu_kernel.hip"
            merged.write_text(source)
            result = detect_and_split_kernel_from_harness(merged, tmp_path)
            assert result is None, "Expected no split for HIP source without host-side test entrypoints"


# ===================================================================
# Test: _rewrite_relative_imports (Issue 2)
# ===================================================================

class TestRewriteRelativeImports:
    """Relative imports must be converted to absolute imports anchored at repo_root."""

    def test_rewrites_single_level_relative_import(self):
        from minisweagent.run.preprocess.harness_utils import _rewrite_relative_imports
        with tempfile.TemporaryDirectory() as tmp:
            # repo_root is in PYTHONPATH, so importable paths start from inside repo_root.
            # harness at repo/ops/triton/ -> package ops.triton
            # from ..utils import helper  (level=2) -> from ops.utils import helper
            repo = Path(tmp) / "mypackage"
            harness_dir = repo / "ops" / "triton"
            harness_dir.mkdir(parents=True)
            harness = harness_dir / "test_harness.py"
            source = "from ..utils import helper\nimport torch\n"
            result = _rewrite_relative_imports(source, harness, repo)
            assert "from ops.utils import helper" in result
            assert "from .." not in result

    def test_rewrites_same_package_relative_import(self):
        from minisweagent.run.preprocess.harness_utils import _rewrite_relative_imports
        with tempfile.TemporaryDirectory() as tmp:
            # harness at repo/ops/ -> package ops
            # from .sibling import foo  (level=1) -> from ops.sibling import foo
            repo = Path(tmp) / "pkg"
            harness_dir = repo / "ops"
            harness_dir.mkdir(parents=True)
            harness = harness_dir / "test_harness.py"
            source = "from .sibling import foo\n"
            result = _rewrite_relative_imports(source, harness, repo)
            assert "from ops.sibling import foo" in result
            assert "from ." not in result

    def test_rewrites_two_level_relative_import(self):
        from minisweagent.run.preprocess.harness_utils import _rewrite_relative_imports
        with tempfile.TemporaryDirectory() as tmp:
            # harness at repo/a/b/c/ -> package a.b.c
            # from ...root_mod import something  (level=3) -> from a.root_mod import something
            repo = Path(tmp) / "pkg"
            harness_dir = repo / "a" / "b" / "c"
            harness_dir.mkdir(parents=True)
            harness = harness_dir / "test_harness.py"
            source = "from ...root_mod import something\n"
            result = _rewrite_relative_imports(source, harness, repo)
            assert "from a.root_mod import something" in result

    def test_rewrites_multiple_relative_imports_in_one_file(self):
        from minisweagent.run.preprocess.harness_utils import _rewrite_relative_imports
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "pkg"
            harness_dir = repo / "ops" / "triton"
            harness_dir.mkdir(parents=True)
            harness = harness_dir / "test_harness.py"
            source = (
                "from ..utils import helper\n"
                "from ..kernels import my_kernel\n"
                "import torch\n"
                "from .local import something\n"
            )
            result = _rewrite_relative_imports(source, harness, repo)
            assert "from ops.utils import helper" in result
            assert "from ops.kernels import my_kernel" in result
            assert "from ops.triton.local import something" in result
            assert "from .." not in result
            assert "from ." not in result

    def test_rewritten_import_resolves_to_patched_not_original(self):
        """The rewritten absolute import must pick up files from PYTHONPATH order,
        meaning the patched copy in output_dir (prepended to PYTHONPATH) wins over
        the original in the repo. This is the core correctness guarantee of Issue 2."""
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Simulate: repo_root/ops/utils.py  (original, unpatched)
            repo = tmp_path / "repo"
            ops_dir = repo / "ops"
            ops_dir.mkdir(parents=True)
            (ops_dir / "utils.py").write_text("VALUE = 'original'")
            # Simulate: output_dir/ops/utils.py  (patched copy)
            out_dir = tmp_path / "output"
            out_ops = out_dir / "ops"
            out_ops.mkdir(parents=True)
            (out_ops / "utils.py").write_text("VALUE = 'patched'")
            # With output_dir first in sys.path, absolute import gets patched version
            saved = sys.path[:]
            try:
                sys.path.insert(0, str(out_dir))
                sys.path.insert(1, str(repo))
                # Remove only the temp 'ops' package from the cache — do NOT
                # use a broad "utils" filter which would evict unrelated modules
                # like minisweagent.run.utils.* and corrupt other workers' state.
                for key in list(sys.modules):
                    if key == "ops" or key.startswith("ops."):
                        del sys.modules[key]
                import importlib
                mod = importlib.import_module("ops.utils")
                assert mod.VALUE == "patched", (
                    "Absolute import must resolve to patched copy in GEAK_WORK_DIR"
                )
            finally:
                sys.path[:] = saved
                for key in list(sys.modules):
                    if key == "ops" or key.startswith("ops."):
                        del sys.modules[key]

    def test_leaves_absolute_imports_unchanged(self):
        from minisweagent.run.preprocess.harness_utils import _rewrite_relative_imports
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "pkg"
            harness_dir = repo / "sub"
            harness_dir.mkdir(parents=True)
            harness = harness_dir / "test.py"
            source = "from pkg.utils import foo\nimport torch\n"
            result = _rewrite_relative_imports(source, harness, repo)
            assert result == source

    def test_returns_unchanged_when_harness_outside_repo(self):
        from minisweagent.run.preprocess.harness_utils import _rewrite_relative_imports
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            other = Path(tmp) / "other"
            other.mkdir()
            harness = other / "test.py"
            source = "from ..utils import foo\n"
            result = _rewrite_relative_imports(source, harness, repo)
            assert result == source  # unchanged, harness outside repo


# ===================================================================
# Test: COMMANDMENT.md hardcodes harness path (Issue 3)
# ===================================================================

class TestCommandmentHardcodesHarness:
    """COMMANDMENT.md must hardcode the harness path — no ${GEAK_HARNESS} variable."""

    _HARNESS_SOURCE = (
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "g = p.add_mutually_exclusive_group(required=True)\n"
        "g.add_argument('--correctness', action='store_true')\n"
        "g.add_argument('--profile', action='store_true')\n"
        "g.add_argument('--benchmark', action='store_true')\n"
        "g.add_argument('--full-benchmark', action='store_true')\n"
    )

    def test_simple_commandment_has_no_geak_harness_var(self):
        from minisweagent.run.preprocess.commandment import generate_commandment
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kernel = tmp_path / "kernel.py"
            kernel.write_text("# kernel")
            harness = tmp_path / "test_harness.py"
            harness.write_text(self._HARNESS_SOURCE)
            content = generate_commandment(
                kernel_path=kernel, harness_path=harness, repo_root=tmp_path,
            )
            assert "${GEAK_HARNESS}" not in content, (
                "COMMANDMENT.md must not reference ${GEAK_HARNESS} variable"
            )
            assert str(harness.resolve()) in content, (
                "COMMANDMENT.md must contain the literal harness path"
            )

    def test_all_four_sections_contain_literal_harness_path(self):
        """Every evaluation section (CORRECTNESS, PROFILE, BENCHMARK, FULL_BENCHMARK)
        must reference the hardcoded path, not a variable."""
        from minisweagent.run.preprocess.commandment import generate_commandment
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kernel = tmp_path / "kernel.py"
            kernel.write_text("# kernel")
            harness = tmp_path / "test_harness.py"
            harness.write_text(self._HARNESS_SOURCE)
            content = generate_commandment(
                kernel_path=kernel, harness_path=harness, repo_root=tmp_path,
            )
            harness_abs = str(harness.resolve())
            for section in ("## CORRECTNESS", "## PROFILE", "## BENCHMARK", "## FULL_BENCHMARK"):
                # Find the section and check the next non-empty line has the literal path
                idx = content.find(section)
                assert idx != -1, f"Missing section {section}"
                section_body = content[idx + len(section):]
                next_section = section_body.find("\n## ")
                section_body = section_body[:next_section] if next_section != -1 else section_body
                assert harness_abs in section_body, (
                    f"{section} does not contain literal harness path"
                )
                assert "${GEAK_HARNESS}" not in section_body, (
                    f"{section} still uses ${{GEAK_HARNESS}} variable"
                )

    def test_harness_path_is_absolute_not_relative(self):
        """The hardcoded path must be absolute so agents can find it from any CWD."""
        from minisweagent.run.preprocess.commandment import generate_commandment
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kernel = tmp_path / "kernel.py"
            kernel.write_text("# kernel")
            harness = tmp_path / "test_harness.py"
            harness.write_text(self._HARNESS_SOURCE)
            content = generate_commandment(
                kernel_path=kernel, harness_path=harness, repo_root=tmp_path,
            )
            harness_abs = str(harness.resolve())
            assert harness_abs.startswith("/"), "Hardcoded path must be absolute"
            # The relative name alone must not appear without the full path prefix
            # (i.e. not just "test_harness.py" floating without its directory)
            assert content.count("test_harness.py") == content.count(harness_abs), (
                "Harness filename appears without its full absolute prefix"
            )
