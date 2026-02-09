#!/usr/bin/env python3
"""
Test that the OpenEvolve controller correctly handles directory-based
multi-file programs. This tests the __init__ path (no LLM calls needed).
"""

import os
import sys
import tempfile
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from openevolve.config import Config


def _get_test_config():
    """Build a Config with a dummy API key and sampling config for testing."""
    config = Config()
    # Set sampling config required by LLMEnsemble
    config.llm.sampling = {"fn": "random"}
    # Set db_path to a temp directory 
    config.database.db_path = "programs_db"
    # Set an API key for the default model
    for model_cfg in config.llm.models:
        model_cfg.api_key = os.environ.get("OPENAI_API_KEY", "test-key-dummy")
    for model_cfg in config.llm.evaluator_models:
        model_cfg.api_key = os.environ.get("OPENAI_API_KEY", "test-key-dummy")
    return config


def test_controller_directory_init():
    """Test OpenEvolve controller initialization with a directory."""
    from openevolve.controller import OpenEvolve

    # Use the GEAK-msa add_kernel as a multi-file directory
    kernel_dir = os.environ.get("GEAK_ADD_KERNEL_DIR", "")
    if not kernel_dir:
        for candidate in [
            os.path.join(_PROJECT_ROOT, "..", "GEAK-msa", "examples", "add_kernel"),
            "/workspace/examples/add_kernel",
            os.path.join(os.path.expanduser("~"), "GEAK-msa", "examples", "add_kernel"),
        ]:
            if os.path.isdir(candidate):
                kernel_dir = os.path.abspath(candidate)
                break
    if not kernel_dir or not os.path.isdir(kernel_dir):
        print("SKIPPED: GEAK-msa add_kernel not available")
        return

    # Create a minimal evaluation file
    eval_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    eval_file.write("""
def evaluate(program_path):
    return {"score": 0.5}
""")
    eval_file.close()

    output_dir = tempfile.mkdtemp(prefix="oe_test_")

    try:
        config = _get_test_config()
        controller = OpenEvolve(
            initial_program_path=kernel_dir,
            evaluation_file=eval_file.name,
            config=config,
            output_dir=output_dir,
        )

        print(f"  is_multifile: {controller.is_multifile}")
        assert controller.is_multifile, "Should be multifile for directory input"

        print(f"  initial_files count: {len(controller.initial_files)}")
        assert len(controller.initial_files) > 0, "Should have loaded files"

        print(f"  initial_main_file: {controller.initial_main_file}")
        assert controller.initial_main_file == "kernel.py", \
            f"Expected kernel.py, got {controller.initial_main_file}"

        print(f"  initial_program_code length: {len(controller.initial_program_code)}")
        assert len(controller.initial_program_code) > 0, "Should have code"

        print(f"  language: {controller.language}")

        print("  PASSED")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        os.unlink(eval_file.name)

    return True


def test_controller_single_file_init():
    """Test that single-file init still works."""
    from openevolve.controller import OpenEvolve

    # Use any existing single file
    single_file = os.path.join(_PROJECT_ROOT, "examples", "tb", "initial_programs", "rocm", "test_add_kernel.py")
    if not os.path.isfile(single_file):
        print("SKIPPED: single file not available")
        return True

    eval_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    eval_file.write("def evaluate(p): return {'score': 0.5}\n")
    eval_file.close()

    output_dir = tempfile.mkdtemp(prefix="oe_test_sf_")

    try:
        config = _get_test_config()
        controller = OpenEvolve(
            initial_program_path=single_file,
            evaluation_file=eval_file.name,
            config=config,
            output_dir=output_dir,
        )

        print(f"  is_multifile: {controller.is_multifile}")
        assert not controller.is_multifile, "Should NOT be multifile for single file input"

        print(f"  initial_files count: {len(controller.initial_files)}")
        assert len(controller.initial_files) == 0, "Should have no files dict"

        print(f"  initial_program_code length: {len(controller.initial_program_code)}")
        assert len(controller.initial_program_code) > 0, "Should have code"

        print("  PASSED")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        os.unlink(eval_file.name)

    return True


def test_controller_with_baseline_metrics():
    """Test that controller loads baseline metrics when available."""
    from openevolve.controller import OpenEvolve

    kernel_dir = os.environ.get("GEAK_ADD_KERNEL_DIR", "")
    if not kernel_dir:
        for candidate in [
            os.path.join(_PROJECT_ROOT, "..", "GEAK-msa", "examples", "add_kernel"),
            "/workspace/examples/add_kernel",
            os.path.join(os.path.expanduser("~"), "GEAK-msa", "examples", "add_kernel"),
        ]:
            if os.path.isdir(candidate):
                kernel_dir = os.path.abspath(candidate)
                break
    if not kernel_dir or not os.path.isdir(kernel_dir):
        print("SKIPPED: GEAK-msa add_kernel not available")
        return True

    # Create a copy with baseline metrics
    import shutil
    tmpdir = tempfile.mkdtemp(prefix="oe_test_bl_")
    kernel_copy = os.path.join(tmpdir, "kernel_dir")
    shutil.copytree(kernel_dir, kernel_copy)

    # Write baseline metrics file
    metrics_dir = os.path.join(kernel_copy, "benchmark", "baseline")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump({
            "latency_us": 42.0,
            "bandwidth_gb_s": 900.0,
            "hbm_utilization": 70.0,
            "compute_busy": 20.0,
        }, f)

    eval_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    eval_file.write("def evaluate(p): return {'score': 0.5}\n")
    eval_file.close()

    output_dir = os.path.join(tmpdir, "output")

    try:
        config = _get_test_config()
        controller = OpenEvolve(
            initial_program_path=kernel_copy,
            evaluation_file=eval_file.name,
            config=config,
            output_dir=output_dir,
        )

        print(f"  baseline_profiling: {controller.baseline_profiling}")
        assert controller.baseline_profiling is not None, "Should have loaded baseline profiling"
        assert controller.baseline_profiling["latency_us"] == 42.0
        print("  PASSED")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        os.unlink(eval_file.name)
        shutil.rmtree(tmpdir)

    return True


if __name__ == "__main__":
    tests = [
        ("Controller directory init (multi-file)", test_controller_directory_init),
        ("Controller single-file init (backward compat)", test_controller_single_file_init),
        ("Controller with baseline metrics", test_controller_with_baseline_metrics),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\nTest: {name}")
        try:
            result = func()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    print("All controller tests PASSED!")
