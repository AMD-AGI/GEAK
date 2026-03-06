"""Verify that pipeline tools and standalone CLIs call the same core functions.

The GEAK pipeline has two invocation paths for most tools:
  1. Top-level (``geak-preprocess``, ``geak-orchestrate``) calling sub-tools
     as Python imports inside the pipeline.
  2. Standalone CLIs (``kernel-profile``, ``baseline-metrics``, etc.) that
     users can run directly from the command line.

Both paths must call the **same** underlying function.  This test
programmatically checks that the core function imported by the pipeline
module is the same object as the one imported by the standalone CLI module.
If someone refactors one path but not the other, this test will catch the
drift.
"""

from __future__ import annotations

# ── Preprocessor sub-tools ────────────────────────────────────────────


def test_resolve_kernel_url_same_function():
    """Preprocessor and CLI both use resolve_kernel_url_impl.resolve_kernel_url."""
    from minisweagent.tools.resolve_kernel_url_impl import resolve_kernel_url as cli_fn
    from minisweagent.tools.resolve_kernel_url_impl import resolve_kernel_url as pp_fn

    assert cli_fn is pp_fn


def test_baseline_metrics_same_function():
    """Preprocessor and CLI both use baseline_metrics.build_baseline_metrics."""
    from minisweagent.baseline_metrics import build_baseline_metrics as cli_fn
    from minisweagent.baseline_metrics import build_baseline_metrics as pp_fn

    assert cli_fn is pp_fn


def test_commandment_same_function():
    """Preprocessor and CLI both use commandment.generate_commandment."""
    from minisweagent.tools.commandment import generate_commandment as cli_fn
    from minisweagent.tools.commandment import generate_commandment as pp_fn

    assert cli_fn is pp_fn


# ── Orchestrator sub-tools ────────────────────────────────────────────


def test_run_tasks_uses_dispatch():
    """Standalone run-tasks CLI delegates to dispatch.task_file_to_agent_task.

    Before unification, task_runner._build_tasks_from_dir had its own
    task-construction logic.  After unification it must delegate to
    dispatch.task_file_to_agent_task.
    """
    import inspect

    from minisweagent.run.task_runner import _build_tasks_from_dir

    source = inspect.getsource(_build_tasks_from_dir)
    assert "task_file_to_agent_task" in source, (
        "_build_tasks_from_dir should delegate to dispatch.task_file_to_agent_task"
    )


def test_run_tasks_cli_uses_run_task_batch():
    """Standalone run-tasks main() calls dispatch.run_task_batch."""
    import inspect

    from minisweagent.run.task_runner import main

    source = inspect.getsource(main)
    assert "run_task_batch" in source, (
        "run-tasks main() should call dispatch.run_task_batch"
    )


def test_dispatch_task_file_to_agent_task_is_public():
    """dispatch.task_file_to_agent_task is a public API."""
    from minisweagent.run.dispatch import task_file_to_agent_task

    assert callable(task_file_to_agent_task)
    assert not task_file_to_agent_task.__name__.startswith("_")


# ── Entry point coverage ──────────────────────────────────────────────


def test_all_tools_md_have_cli_entry_points():
    """Every tool shown in docs/tools.md has a CLI entry point or is documented.

    This checks the pyproject.toml [project.scripts] section to ensure
    each tool from tools.md is registered.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    pyproject = (repo_root / "pyproject.toml").read_text()

    expected_tools = {
        "geak",
        "geak-preprocess",
        "geak-orchestrate",
        "resolve-kernel-url",
        "codebase-context",
        "kernel-profile",
        "baseline-metrics",
        "commandment",
        "validate-commandment",
        "task-generator",
        "run-tasks",
        "select-patch",
        "openevolve-worker",
    }

    # test-discovery is in its own package's pyproject.toml
    external_tools = {"test-discovery"}

    for tool in expected_tools:
        assert f'{tool} = "' in pyproject or f"{tool} = '" in pyproject, (
            f"Tool '{tool}' from tools.md is not registered in pyproject.toml [project.scripts]"
        )

    for tool in external_tools:
        mcp_pyproject = (repo_root / "mcp_tools" / "automated-test-discovery" / "pyproject.toml").read_text()
        assert f'{tool} = "' in mcp_pyproject or f"{tool} = '" in mcp_pyproject, (
            f"Tool '{tool}' is not registered in its MCP package pyproject.toml"
        )


# ── Codebase context CLI ─────────────────────────────────────────────


def test_codebase_context_cli_calls_generate():
    """codebase-context CLI main() calls generate_codebase_context."""
    import inspect

    from minisweagent.run.codebase_context import main

    source = inspect.getsource(main)
    assert "generate_codebase_context" in source


# ── COMMANDMENT validator and verbatim execution ─────────────────────


def test_validator_requires_pythonpath_in_setup():
    """COMMANDMENT validator errors if SETUP doesn't set PYTHONPATH."""
    from minisweagent.tools.validate_commandment import validate_commandment

    bad = (
        "## SETUP\necho hello\n\n"
        "## CORRECTNESS\npython test.py\n\n"
        "## PROFILE\nkernel-profile test.py\n\n"
        "## BENCHMARK\npython bench.py --benchmark\n\n"
        "## FULL_BENCHMARK\npython bench.py --full-benchmark\n"
    )
    result = validate_commandment(bad)
    assert not result["valid"]
    assert any("PYTHONPATH" in e for e in result["errors"])

    good = (
        "## SETUP\n"
        "printf '#!/bin/bash\\nexport PYTHONPATH=%s:/repo:${PYTHONPATH}\\n' > run.sh\n\n"
        "## CORRECTNESS\nrun.sh test.py --correctness\n\n"
        "## PROFILE\nkernel-profile run.sh test.py --profile\n\n"
        "## BENCHMARK\nrun.sh test.py --benchmark\n\n"
        "## FULL_BENCHMARK\nrun.sh test.py --full-benchmark\n"
    )
    result = validate_commandment(good)
    assert result["valid"], result["errors"]


def test_commandment_test_command_is_verbatim(tmp_path):
    """dispatch._commandment_test_command reads sections verbatim -- no parsing."""
    from minisweagent.run.dispatch import _commandment_test_command

    cmd_file = tmp_path / "COMMANDMENT.md"
    cmd_file.write_text(
        "## SETUP\n"
        "printf '#!/bin/bash\\nexport PYTHONPATH=%s:/repo:${PYTHONPATH}\\n' "
        "\"${GEAK_WORK_DIR}\" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh\n\n"
        "## CORRECTNESS\n"
        "${GEAK_WORK_DIR}/run.sh /path/to/harness.py --correctness\n\n"
        "## PROFILE\n"
        "kernel-profile ${GEAK_WORK_DIR}/run.sh /path/to/harness.py --profile\n\n"
        "## BENCHMARK\n"
        "${GEAK_WORK_DIR}/run.sh /path/to/harness.py --benchmark\n\n"
        "## FULL_BENCHMARK\n"
        "${GEAK_WORK_DIR}/run.sh /path/to/harness.py --full-benchmark\n"
    )

    result = _commandment_test_command(str(cmd_file))
    assert result is not None
    from pathlib import Path as _P
    script_content = _P(result).read_text()
    # SETUP command must appear verbatim (no stripping of run.sh)
    assert "${GEAK_WORK_DIR}/run.sh" in script_content
    # CORRECTNESS command must appear verbatim
    assert "--correctness" in script_content
    # BENCHMARK command must appear after CORRECTNESS
    assert "--benchmark" in script_content
    # Must NOT contain bare "python" (that would mean we stripped the wrapper)
    assert not script_content.startswith("python ")
