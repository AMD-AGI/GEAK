# Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive tests for multi-file support in OpenEvolve.

Tests cover:
- Multi-file diff parsing and application (code_utils)
- Multi-file Program model (database)
- Directory loading (database)
- Prompt generation with multi-file and baseline profiling (sampler)
- COMMANDMENT evaluator (commandment_evaluator)
- Backward compatibility with single-file workflows
"""

import json
import os
import shutil
import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase, load_program_from_directory
from openevolve.utils.code_utils import (
    apply_diff,
    apply_multifile_diff,
    extract_diffs,
    extract_multifile_diffs,
    format_multifile_diff,
)
from openevolve.prompt.sampler import PromptSampler
from openevolve.commandment_evaluator import (
    CommandmentEvaluator,
    CommandmentGenerator,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

KERNEL_CODE = """\
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

TEST_CODE = """\
import torch
import pytest

def test_add_correctness():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    expected = x + y
    # placeholder - real test would call add_kernel
    assert True
"""

UTILS_CODE = """\
def get_grid(n_elements, block_size):
    return (n_elements + block_size - 1) // block_size
"""


# ---------------------------------------------------------------------------
# Multi-file diff tests
# ---------------------------------------------------------------------------

class TestMultifileDiffs(unittest.TestCase):
    """Tests for extract_multifile_diffs, apply_multifile_diff, format_multifile_diff."""

    def test_extract_multifile_diffs_basic(self):
        """Parse diffs with file: prefix."""
        diff_text = """
Let me optimize the kernel:

<<<<<<< SEARCH file:kernel.py
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
=======
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_last")
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy="evict_last")
>>>>>>> REPLACE

And update the test:

<<<<<<< SEARCH file:tests/test_add.py
    assert True
=======
    assert torch.allclose(expected, result, atol=1e-5)
>>>>>>> REPLACE
"""
        result = extract_multifile_diffs(diff_text)

        self.assertIn("kernel.py", result)
        self.assertIn("tests/test_add.py", result)
        self.assertEqual(len(result["kernel.py"]), 1)
        self.assertEqual(len(result["tests/test_add.py"]), 1)

        # Check content
        search, replace = result["kernel.py"][0]
        self.assertIn("tl.load(x_ptr + offsets, mask=mask)", search)
        self.assertIn("eviction_policy", replace)

    def test_extract_multifile_diffs_empty_for_single_file(self):
        """Single-file diffs (no file: prefix) return empty dict."""
        diff_text = """
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
"""
        result = extract_multifile_diffs(diff_text)
        self.assertEqual(result, {})

    def test_extract_multifile_diffs_multiple_per_file(self):
        """Multiple diffs targeting the same file."""
        diff_text = """
<<<<<<< SEARCH file:kernel.py
line1
=======
line1_new
>>>>>>> REPLACE

<<<<<<< SEARCH file:kernel.py
line2
=======
line2_new
>>>>>>> REPLACE
"""
        result = extract_multifile_diffs(diff_text)
        self.assertEqual(len(result["kernel.py"]), 2)

    def test_apply_multifile_diff_basic(self):
        """Apply multi-file diffs to a files dict."""
        files = {
            "kernel.py": "def foo():\n    x = 1\n    return x\n",
            "utils.py": "def bar():\n    y = 2\n    return y\n",
        }
        diff_text = """
<<<<<<< SEARCH file:kernel.py
    x = 1
=======
    x = 42
>>>>>>> REPLACE
"""
        result = apply_multifile_diff(files, diff_text)

        self.assertIn("x = 42", result["kernel.py"])
        # utils.py should be unchanged
        self.assertEqual(result["utils.py"], files["utils.py"])

    def test_apply_multifile_diff_fallback_to_single(self):
        """When no file: prefix found, falls back to single-file on default_file."""
        files = {
            "kernel.py": "def foo():\n    x = 1\n    return x\n",
        }
        diff_text = """
<<<<<<< SEARCH
    x = 1
=======
    x = 99
>>>>>>> REPLACE
"""
        result = apply_multifile_diff(files, diff_text, default_file="kernel.py")
        self.assertIn("x = 99", result["kernel.py"])

    def test_apply_multifile_diff_missing_file_warns(self):
        """Diff targeting non-existent file should warn and skip."""
        files = {"kernel.py": "code here"}
        diff_text = """
<<<<<<< SEARCH file:nonexistent.py
old
=======
new
>>>>>>> REPLACE
"""
        result = apply_multifile_diff(files, diff_text)
        # kernel.py should be unchanged
        self.assertEqual(result["kernel.py"], "code here")

    def test_format_multifile_diff(self):
        """format_multifile_diff produces valid parseable diff."""
        diff_str = format_multifile_diff("kernel.py", "old code", "new code")
        parsed = extract_multifile_diffs(diff_str)
        self.assertIn("kernel.py", parsed)
        self.assertEqual(parsed["kernel.py"][0][0], "old code")
        self.assertEqual(parsed["kernel.py"][0][1], "new code")

    def test_single_file_backward_compat(self):
        """Original extract_diffs and apply_diff still work."""
        diff_text = """
<<<<<<< SEARCH
    x = 1
=======
    x = 2
>>>>>>> REPLACE
"""
        diffs = extract_diffs(diff_text)
        self.assertEqual(len(diffs), 1)

        original = "def foo():\n    x = 1\n    return x"
        result = apply_diff(original, diff_text)
        self.assertIn("x = 2", result)


# ---------------------------------------------------------------------------
# Multi-file Program model tests
# ---------------------------------------------------------------------------

class TestMultifileProgram(unittest.TestCase):
    """Tests for multi-file extensions to the Program dataclass."""

    def test_is_multifile_false_for_single(self):
        """Single-file program returns False."""
        p = Program(id="test1", code="x = 1")
        self.assertFalse(p.is_multifile())

    def test_is_multifile_true(self):
        """Program with files dict returns True."""
        p = Program(
            id="test2",
            code="",
            files={"kernel.py": KERNEL_CODE, "utils.py": UTILS_CODE},
            main_file="kernel.py",
        )
        self.assertTrue(p.is_multifile())

    def test_get_file(self):
        """Get file content by path."""
        p = Program(
            id="test3",
            code="",
            files={"kernel.py": KERNEL_CODE},
            main_file="kernel.py",
        )
        self.assertEqual(p.get_file("kernel.py"), KERNEL_CODE)

    def test_get_file_missing_raises(self):
        """Getting non-existent file raises KeyError."""
        p = Program(id="test4", code="", files={"a.py": "code"})
        with self.assertRaises(KeyError):
            p.get_file("nonexistent.py")

    def test_set_file(self):
        """Set file content."""
        p = Program(id="test5", code="", files={})
        p.set_file("new.py", "print('hello')")
        self.assertEqual(p.files["new.py"], "print('hello')")

    def test_get_all_code_single_file(self):
        """get_all_code returns code for single-file programs."""
        p = Program(id="test6", code="x = 1")
        self.assertEqual(p.get_all_code(), "x = 1")

    def test_get_all_code_multifile(self):
        """get_all_code concatenates all files with headers."""
        p = Program(
            id="test7",
            code="",
            files={"kernel.py": "kernel_code", "utils.py": "utils_code"},
            main_file="kernel.py",
        )
        all_code = p.get_all_code()
        self.assertIn("# === FILE: kernel.py ===", all_code)
        self.assertIn("# === FILE: utils.py ===", all_code)
        self.assertIn("kernel_code", all_code)
        self.assertIn("utils_code", all_code)
        # Main file should come first
        kernel_pos = all_code.index("kernel.py")
        utils_pos = all_code.index("utils.py")
        self.assertLess(kernel_pos, utils_pos)

    def test_to_dict_includes_files(self):
        """to_dict serializes files and main_file."""
        p = Program(
            id="test8",
            code="",
            files={"a.py": "code_a"},
            main_file="a.py",
        )
        d = p.to_dict()
        self.assertEqual(d["files"], {"a.py": "code_a"})
        self.assertEqual(d["main_file"], "a.py")

    def test_from_dict_with_files(self):
        """from_dict reconstructs files and main_file."""
        data = {
            "id": "test9",
            "code": "",
            "files": {"b.py": "code_b"},
            "main_file": "b.py",
        }
        p = Program.from_dict(data)
        self.assertTrue(p.is_multifile())
        self.assertEqual(p.get_file("b.py"), "code_b")

    def test_database_add_multifile(self):
        """Multi-file program can be added and retrieved from database."""
        config = Config()
        config.database.in_memory = True
        db = ProgramDatabase(config.database)

        program = Program(
            id="mf1",
            code="",
            files={"kernel.py": KERNEL_CODE, "tests/test.py": TEST_CODE},
            main_file="kernel.py",
            metrics={"score": 0.8},
        )
        db.add(program)

        retrieved = db.get("mf1")
        self.assertIsNotNone(retrieved)
        self.assertTrue(retrieved.is_multifile())
        self.assertEqual(len(retrieved.files), 2)
        self.assertEqual(retrieved.main_file, "kernel.py")

    def test_migrate_copies_files(self):
        """migrate_programs copies files and main_file correctly."""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 2
        config.database.population_size = 100  # Prevent eviction
        db = ProgramDatabase(config.database)

        program = Program(
            id="mig1",
            code="",
            files={"k.py": "kernel", "t.py": "test"},
            main_file="k.py",
            metrics={"combined_score": 0.9},
        )
        db.add(program, target_island=0)
        db.migrate_programs()

        # Find migrated programs
        migrated = [
            p for p in db.programs.values()
            if "migrant" in p.id
        ]
        self.assertGreater(len(migrated), 0)
        for m in migrated:
            self.assertEqual(m.files, {"k.py": "kernel", "t.py": "test"})
            self.assertEqual(m.main_file, "k.py")


# ---------------------------------------------------------------------------
# Directory loading tests
# ---------------------------------------------------------------------------

class TestDirectoryLoading(unittest.TestCase):
    """Tests for load_program_from_directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write(self, rel_path, content):
        path = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def test_load_simple_directory(self):
        """Load a directory with kernel.py and tests/test.py."""
        self._write("kernel.py", KERNEL_CODE)
        self._write("tests/test_add.py", TEST_CODE)

        files, main_file = load_program_from_directory(self.tmpdir)

        self.assertEqual(len(files), 2)
        self.assertEqual(main_file, "kernel.py")
        self.assertIn("kernel.py", files)
        self.assertIn(os.path.join("tests", "test_add.py"), files)

    def test_main_file_detection_kernel(self):
        """kernel.py is preferred as main_file."""
        self._write("kernel.py", "kernel code")
        self._write("utils.py", "utils code")

        _, main_file = load_program_from_directory(self.tmpdir)
        self.assertEqual(main_file, "kernel.py")

    def test_main_file_detection_main(self):
        """main.py is detected as main_file when kernel.py is absent."""
        self._write("main.py", "main code")
        self._write("helper.py", "helper code")

        _, main_file = load_program_from_directory(self.tmpdir)
        self.assertEqual(main_file, "main.py")

    def test_main_file_detection_by_name(self):
        """Files with 'kernel' in the name are preferred."""
        self._write("my_kernel.py", "code")
        self._write("utils.py", "code")

        _, main_file = load_program_from_directory(self.tmpdir)
        self.assertEqual(main_file, "my_kernel.py")

    def test_ignores_non_source_files(self):
        """Non-source files (.txt, .json, etc.) are ignored."""
        self._write("kernel.py", "code")
        self._write("README.md", "# readme")
        self._write("data.json", "{}")
        self._write("config.yaml", "key: val")

        files, _ = load_program_from_directory(self.tmpdir)
        self.assertEqual(len(files), 1)
        self.assertIn("kernel.py", files)

    def test_ignores_pycache(self):
        """__pycache__ directories are skipped."""
        self._write("kernel.py", "code")
        self._write("__pycache__/kernel.cpython-310.pyc", "bytecode")

        files, _ = load_program_from_directory(self.tmpdir)
        self.assertEqual(len(files), 1)

    def test_empty_directory_raises(self):
        """Empty directory raises ValueError."""
        with self.assertRaises(ValueError):
            load_program_from_directory(self.tmpdir)

    def test_hip_files_included(self):
        """HIP (.hip, .h) files are loaded."""
        self._write("kernel.hip", "hip code")
        self._write("kernel.h", "header code")

        files, _ = load_program_from_directory(self.tmpdir)
        self.assertEqual(len(files), 2)

    def test_load_from_geak_msa_add_kernel(self):
        """Integration: load from GEAK-msa examples/add_kernel if available."""
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        add_kernel_dir = os.environ.get("GEAK_ADD_KERNEL_DIR", "")
        if not add_kernel_dir:
            for candidate in [
                os.path.join(_project_root, "..", "GEAK-msa", "examples", "add_kernel"),
                "/workspace/examples/add_kernel",
                os.path.join(os.path.expanduser("~"), "GEAK-msa", "examples", "add_kernel"),
            ]:
                if os.path.isdir(candidate):
                    add_kernel_dir = os.path.abspath(candidate)
                    break
        if not os.path.isdir(add_kernel_dir):
            self.skipTest(f"GEAK-msa add_kernel not available at {add_kernel_dir}")

        files, main_file = load_program_from_directory(add_kernel_dir)
        self.assertGreater(len(files), 0)
        self.assertEqual(main_file, "kernel.py")
        self.assertIn("kernel.py", files)
        self.assertIn("triton", files["kernel.py"])


# ---------------------------------------------------------------------------
# Prompt sampler tests (multi-file + baseline profiling)
# ---------------------------------------------------------------------------

class TestMultifilePromptSampler(unittest.TestCase):
    """Tests for multi-file prompt building and baseline profiling formatting."""

    def setUp(self):
        config = Config()
        self.sampler = PromptSampler(config.prompt)

    def test_multifile_prompt_uses_multifile_template(self):
        """When is_multifile=True, the diff_user_multifile template is selected."""
        files = {"kernel.py": KERNEL_CODE, "utils.py": UTILS_CODE}
        prompt = self.sampler.build_prompt(
            current_program=KERNEL_CODE,
            program_metrics={"score": 0.5},
            is_multifile=True,
            program_files=files,
            main_file="kernel.py",
        )
        user_msg = prompt["user"]
        # Should contain multi-file specific content
        self.assertIn("kernel.py", user_msg)
        self.assertIn("utils.py", user_msg)
        self.assertIn("file:", user_msg)  # diff format example

    def test_multifile_prompt_contains_file_listing(self):
        """Prompt contains a listing of all files."""
        files = {"kernel.py": KERNEL_CODE, "tests/test.py": TEST_CODE}
        prompt = self.sampler.build_prompt(
            current_program=KERNEL_CODE,
            is_multifile=True,
            program_files=files,
            main_file="kernel.py",
        )
        user_msg = prompt["user"]
        self.assertIn("`kernel.py`", user_msg)
        self.assertIn("(main)", user_msg)

    def test_multifile_prompt_contains_file_contents(self):
        """Prompt contains the actual code of each file."""
        files = {"kernel.py": KERNEL_CODE}
        prompt = self.sampler.build_prompt(
            current_program=KERNEL_CODE,
            is_multifile=True,
            program_files=files,
            main_file="kernel.py",
        )
        user_msg = prompt["user"]
        self.assertIn("add_kernel", user_msg)
        self.assertIn("tl.load", user_msg)

    def test_baseline_profiling_in_prompt(self):
        """Baseline profiling data is formatted and included in the prompt."""
        profiling = {
            "latency_us": 42.5,
            "bandwidth_gb_s": 1200.0,
            "hbm_utilization": 78.3,
            "compute_busy": 22.1,
            "l2_hit_rate": 45.0,
            "tflops": 15.2,
        }
        prompt = self.sampler.build_prompt(
            current_program=KERNEL_CODE,
            is_multifile=True,
            program_files={"kernel.py": KERNEL_CODE},
            main_file="kernel.py",
            baseline_profiling=profiling,
        )
        user_msg = prompt["user"]
        self.assertIn("42.5", user_msg)  # latency
        self.assertIn("1200.0", user_msg)  # bandwidth
        self.assertIn("Memory-bound", user_msg)  # bottleneck analysis

    def test_single_file_prompt_still_works(self):
        """Single-file prompt (is_multifile=False) uses original template."""
        prompt = self.sampler.build_prompt(
            current_program="x = 1",
            program_metrics={"score": 0.5},
            is_multifile=False,
        )
        user_msg = prompt["user"]
        # Should NOT contain multi-file format
        self.assertNotIn("File Contents", user_msg)
        # Should contain the hardcoded matrix sizes from original template
        self.assertIn("64x128", user_msg)

    def test_format_baseline_profiling_empty(self):
        """Empty profiling dict returns informative message."""
        result = self.sampler._format_baseline_profiling({})
        self.assertIn("No detailed profiling", result)

    def test_format_baseline_profiling_compute_bound(self):
        """Compute-bound bottleneck is detected."""
        profiling = {
            "compute_busy": 85.0,
            "hbm_utilization": 15.0,
        }
        result = self.sampler._format_baseline_profiling(profiling)
        self.assertIn("Compute-bound", result)

    def test_format_baseline_profiling_latency_bound(self):
        """Latency-bound bottleneck is detected."""
        profiling = {
            "compute_busy": 10.0,
            "hbm_utilization": 10.0,
        }
        result = self.sampler._format_baseline_profiling(profiling)
        self.assertIn("Latency-bound", result)


# ---------------------------------------------------------------------------
# COMMANDMENT evaluator tests
# ---------------------------------------------------------------------------

class TestCommandmentGenerator(unittest.TestCase):
    """Tests for CommandmentGenerator."""

    def test_generate_basic(self):
        """Generate a basic COMMANDMENT.md."""
        gen = CommandmentGenerator()
        content = gen.generate(
            kernel_dir="/tmp/test_kernel",
            setup_commands=["mkdir -p results", "export FOO=bar"],
            correctness_commands=["python -m pytest tests/ -v"],
            profile_commands=["python profile.py"],
            profiling_results={"latency_us": 100.0, "tflops": 10.0},
        )
        self.assertIn("## SETUP", content)
        self.assertIn("## CORRECTNESS", content)
        self.assertIn("## PROFILE", content)
        self.assertIn("VALIDATED", content)
        self.assertIn("Config Hash:", content)
        self.assertIn("latency_us", content)

    def test_generate_default_commands(self):
        """Generator creates default commands when none provided."""
        gen = CommandmentGenerator()
        content = gen.generate(kernel_dir="/tmp/test")
        self.assertIn("## SETUP", content)
        self.assertIn("## CORRECTNESS", content)
        self.assertIn("## PROFILE", content)

    def test_save_creates_file(self):
        """save() writes COMMANDMENT.md to disk."""
        gen = CommandmentGenerator()
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "COMMANDMENT.md")
            gen.save(
                output_path=path,
                kernel_dir=tmpdir,
                setup_commands=["echo setup"],
            )
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("## SETUP", content)
        finally:
            shutil.rmtree(tmpdir)


class TestCommandmentEvaluator(unittest.TestCase):
    """Tests for CommandmentEvaluator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_commandment(self, setup_cmds=None, correctness_cmds=None, profile_cmds=None):
        """Helper to write a COMMANDMENT.md for testing."""
        gen = CommandmentGenerator()
        path = os.path.join(self.tmpdir, "COMMANDMENT.md")
        gen.save(
            output_path=path,
            kernel_dir=self.tmpdir,
            setup_commands=setup_cmds or ["echo 'setup done'"],
            correctness_commands=correctness_cmds or ["echo 'correct'"],
            profile_commands=profile_cmds or ["echo '{\"latency_us\": 50.0}'"],
        )
        return path

    def test_parse_commandment(self):
        """Evaluator correctly parses COMMANDMENT.md sections."""
        path = self._write_commandment()
        evaluator = CommandmentEvaluator(commandment_path=path)
        self.assertIn("SETUP", evaluator.sections)
        self.assertIn("CORRECTNESS", evaluator.sections)
        self.assertIn("PROFILE", evaluator.sections)

    def test_evaluate_success(self):
        """Successful evaluation with simple echo commands."""
        path = self._write_commandment(
            profile_cmds=["echo '{\"latency_us\": 50.0, \"tflops\": 5.0}'"],
        )
        baseline_path = os.path.join(self.tmpdir, "baseline_metrics.json")
        with open(baseline_path, "w") as f:
            json.dump({"latency_us": 100.0}, f)

        evaluator = CommandmentEvaluator(
            commandment_path=path,
            baseline_metrics_path=baseline_path,
        )
        result = evaluator.evaluate(
            program_files={"kernel.py": "print('hello')"},
            program_id="test_eval_1",
        )
        self.assertTrue(result.success)
        self.assertTrue(result.correctness_passed)
        # Speedup = baseline_latency / target_latency = 100 / 50 = 2.0
        self.assertAlmostEqual(result.speedup, 2.0)

    def test_evaluate_correctness_failure(self):
        """Evaluation fails when correctness commands fail."""
        path = self._write_commandment(
            correctness_cmds=["exit 1"],
        )
        evaluator = CommandmentEvaluator(commandment_path=path)
        result = evaluator.evaluate(
            program_files={"kernel.py": "code"},
            program_id="test_fail_1",
        )
        self.assertFalse(result.success)
        self.assertFalse(result.correctness_passed)

    def test_evaluate_timeout(self):
        """Commands that exceed timeout are handled gracefully."""
        path = self._write_commandment(
            correctness_cmds=["sleep 60"],
        )
        evaluator = CommandmentEvaluator(commandment_path=path, timeout=1)
        result = evaluator.evaluate(
            program_files={"kernel.py": "code"},
            program_id="test_timeout_1",
        )
        self.assertFalse(result.success)

    def test_speedup_no_baseline(self):
        """Without baseline metrics, speedup defaults to 1.0."""
        path = self._write_commandment()
        evaluator = CommandmentEvaluator(commandment_path=path)
        result = evaluator.evaluate(
            program_files={"kernel.py": "code"},
            program_id="test_no_baseline",
        )
        self.assertEqual(result.speedup, 1.0)


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    """Ensure single-file workflows are completely unaffected."""

    def test_single_file_program_works(self):
        """Single-file Program behaves identically to before."""
        p = Program(
            id="compat1",
            code="def test(): pass",
            language="python",
            metrics={"score": 0.5},
        )
        self.assertFalse(p.is_multifile())
        self.assertEqual(p.get_all_code(), "def test(): pass")
        self.assertEqual(p.code, "def test(): pass")

        d = p.to_dict()
        self.assertEqual(d["files"], {})
        self.assertIsNone(d["main_file"])

    def test_single_file_database_roundtrip(self):
        """Single-file programs can be stored and retrieved from DB."""
        config = Config()
        config.database.in_memory = True
        db = ProgramDatabase(config.database)

        p = Program(id="compat2", code="x = 1", metrics={"score": 0.3})
        db.add(p)
        retrieved = db.get("compat2")
        self.assertEqual(retrieved.code, "x = 1")
        self.assertFalse(retrieved.is_multifile())

    def test_single_file_diff_still_works(self):
        """Original extract_diffs + apply_diff work unchanged."""
        original = "x = 1\ny = 2"
        diff = """
<<<<<<< SEARCH
x = 1
=======
x = 42
>>>>>>> REPLACE
"""
        result = apply_diff(original, diff)
        self.assertIn("x = 42", result)
        self.assertIn("y = 2", result)

    def test_load_single_file_with_separator(self):
        """
        Loading a file with 146# separator through the single-file path
        works identically (we don't touch it).
        """
        separator = "#" * 146
        code = f"kernel_code\n\n{separator}\n\ntest_code"

        p = Program(id="compat3", code=code)
        self.assertFalse(p.is_multifile())
        self.assertIn(separator, p.code)
        self.assertIn("kernel_code", p.code)
        self.assertIn("test_code", p.code)


if __name__ == "__main__":
    unittest.main()
