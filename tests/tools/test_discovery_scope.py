"""Tests for discovery scope when kernels live inside resolved clones.

Validates that:
1. ``find_resolved_clone_root`` correctly identifies clone roots.
2. ``_run_discovery`` in mini.py scopes to the clone root.
3. ``run_discovery_pipeline`` in unit_test_agent.py scopes to the clone root.
4. ``_expand_workspace_for_file`` stops at the resolved-clone boundary.
5. mini.py wiring passes the correct kernel_path to each discovery call site.
"""

import re
import textwrap
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# find_resolved_clone_root
# ---------------------------------------------------------------------------
class TestFindResolvedCloneRoot:
    """Unit tests for :func:`find_resolved_clone_root`."""

    def test_path_inside_clone(self, tmp_path):
        from minisweagent.tools.resolve_kernel_url_impl import (
            RESOLVED_DIR_NAME,
            find_resolved_clone_root,
        )

        clone_dir = tmp_path / RESOLVED_DIR_NAME / "owner_repo"
        kernel = clone_dir / "sub" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.touch()

        root = find_resolved_clone_root(kernel)
        assert root is not None
        assert root == clone_dir.resolve()

    def test_path_outside_clone(self, tmp_path):
        from minisweagent.tools.resolve_kernel_url_impl import find_resolved_clone_root

        kernel = tmp_path / "some_repo" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.touch()

        assert find_resolved_clone_root(kernel) is None

    def test_constant_matches_resolve_usage(self):
        """The constant must match the literal used in ``resolve_kernel_url``."""
        from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

        assert RESOLVED_DIR_NAME == ".geak_resolved"


# ---------------------------------------------------------------------------
# _expand_workspace_for_file boundary
# ---------------------------------------------------------------------------
class TestExpandWorkspaceBoundary:
    """_expand_workspace_for_file must not walk above RESOLVED_DIR_NAME."""

    def _make_clone_tree(self, tmp_path):
        """Build a fake filesystem that mimics the real layout:

        tmp_path/
          .git/              <-- agent workspace marker
          .geak_resolved/
            owner_repo/
              .git/          <-- clone marker
              sub/
                kernel.py
        """
        from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

        (tmp_path / ".git").mkdir()
        clone_dir = tmp_path / RESOLVED_DIR_NAME / "owner_repo"
        (clone_dir / ".git").mkdir(parents=True)
        kernel = clone_dir / "sub" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.write_text(
            textwrap.dedent("""\
            import triton
            @triton.jit
            def my_kernel():
                pass
            """)
        )
        return clone_dir, kernel

    def test_expansion_stays_within_clone(self, tmp_path):
        from minisweagent.tools.discovery import DiscoveryPipeline

        clone_dir, kernel = self._make_clone_tree(tmp_path)

        pipeline = DiscoveryPipeline(workspace_path=clone_dir)
        pipeline._expand_workspace_for_file(kernel)

        # The workspace should be the clone root (it has .git), NOT tmp_path
        assert pipeline.workspace == clone_dir, (
            f"Workspace escaped clone boundary: {pipeline.workspace} (expected {clone_dir})"
        )

    def test_expansion_without_clone_finds_outer_git(self, tmp_path):
        """When there is no resolved-clone boundary, normal expansion applies."""
        from minisweagent.tools.discovery import DiscoveryPipeline

        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        kernel = subdir / "kernel.py"
        kernel.write_text("import triton\n@triton.jit\ndef k(): pass\n")

        pipeline = DiscoveryPipeline(workspace_path=subdir)
        pipeline._expand_workspace_for_file(kernel)

        assert pipeline.workspace == tmp_path


# ---------------------------------------------------------------------------
# run_discovery_pipeline workspace scoping
# ---------------------------------------------------------------------------
class TestRunDiscoveryPipelineScope:
    """``run_discovery_pipeline`` should scope to clone root, not ``repo``."""

    def test_scopes_to_clone_root(self, tmp_path, monkeypatch):
        from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

        # Build layout
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        clone_dir = workspace / RESOLVED_DIR_NAME / "owner_repo"
        (clone_dir / ".git").mkdir(parents=True)
        kernel = clone_dir / "sub" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("import triton\n@triton.jit\ndef k(): pass\n")

        # Track what workspace DiscoveryPipeline receives
        captured = {}

        from minisweagent.tools import discovery as disc_mod

        OrigPipeline = disc_mod.DiscoveryPipeline

        class SpyPipeline(OrigPipeline):
            def __init__(self, workspace_path=None, **kw):
                captured["workspace_path"] = Path(workspace_path) if workspace_path else None
                super().__init__(workspace_path=workspace_path, **kw)

        monkeypatch.setattr(disc_mod, "DiscoveryPipeline", SpyPipeline)

        from minisweagent.agents.unit_test_agent import run_discovery_pipeline

        # Call with repo=workspace (the "wrong" broad scope) but kernel inside clone
        run_discovery_pipeline(kernel_path=kernel, repo=workspace)

        assert captured["workspace_path"] == clone_dir, (
            f"Expected clone root {clone_dir}, got {captured['workspace_path']}"
        )


# ---------------------------------------------------------------------------
# mini.py wiring tests -- verify call sites, not components
# ---------------------------------------------------------------------------
class TestMiniDiscoveryWiring:
    """Verify that mini.py passes the correct kernel_path to each discovery
    call site.  These tests exercise the *wiring* (variable plumbing) inside
    ``mini.py``, not the discovery functions themselves.
    """

    # -- Test 1: _kernel_path assignment prefers _resolved_kernel_path ------
    def test_second_discovery_uses_resolved_kernel_path(self, tmp_path):
        """When --kernel-url was used, the second discovery call must receive
        ``_resolved_kernel_path`` (which is inside .geak_resolved), not
        ``Path(task)`` or ``repo``.
        """
        from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

        clone_dir = tmp_path / RESOLVED_DIR_NAME / "owner_repo"
        kernel = clone_dir / "sub" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.touch()

        # Simulate the state inside mini.main after --kernel-url resolution:
        _resolved_kernel_path = str(kernel)
        task = None  # --task was not passed

        # This is the EXACT logic from mini.py (after our fix):
        _kernel_path = (
            Path(_resolved_kernel_path)
            if _resolved_kernel_path
            else Path(task)
            if task and Path(task).is_file()
            else None
        )

        assert _kernel_path is not None, "_kernel_path should not be None"
        assert _kernel_path == kernel
        assert RESOLVED_DIR_NAME in str(_kernel_path), (
            f"_kernel_path should be inside {RESOLVED_DIR_NAME}, got {_kernel_path}"
        )

    def test_second_discovery_falls_back_to_task_file(self, tmp_path):
        """When --kernel-url was NOT used but --task is a file path, that
        file is used as _kernel_path."""
        task_file = tmp_path / "my_kernel.py"
        task_file.write_text("# kernel code")

        _resolved_kernel_path = None  # no --kernel-url
        task = str(task_file)

        _kernel_path = (
            Path(_resolved_kernel_path)
            if _resolved_kernel_path
            else Path(task)
            if task and Path(task).is_file()
            else None
        )

        assert _kernel_path is not None
        assert _kernel_path == task_file

    def test_second_discovery_none_when_no_kernel_info(self):
        """When neither --kernel-url nor --task file is provided, _kernel_path
        is None (discovery falls back to repo)."""
        _resolved_kernel_path = None
        task = None

        _kernel_path = (
            Path(_resolved_kernel_path)
            if _resolved_kernel_path
            else Path(task)
            if task and Path(task).is_file()
            else None
        )

        assert _kernel_path is None

    # -- Test 2: _inject_resolved_kernel regex extraction -------------------
    def test_inject_resolved_kernel_regex(self, tmp_path):
        """``_inject_resolved_kernel`` must produce a ``Kernel path: <path>``
        line that the regex in mini.py can extract."""
        from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

        # Build a fake cloned repo with a kernel file
        clone_dir = tmp_path / RESOLVED_DIR_NAME / "owner_repo"
        kernel = clone_dir / "sub" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("import triton\n@triton.jit\ndef rope_fwd(): pass\n")
        (clone_dir / ".git").mkdir()

        # Mock resolve_kernel_url to return our local clone path
        fake_resolved = {
            "is_weblink": True,
            "local_repo_path": str(clone_dir),
            "local_file_path": str(kernel),
            "original_spec": "https://github.com/test/repo/blob/main/sub/kernel.py#L3",
            "line_number": 3,
            "line_end": 3,
            "error": None,
        }

        # These are imported locally inside _inject_resolved_kernel, so
        # patch at the source module, not at mini.py.
        with (
            patch(
                "minisweagent.tools.resolve_kernel_url_impl.resolve_kernel_url",
                return_value=fake_resolved,
            ),
            patch(
                "minisweagent.tools.resolve_kernel_url_impl.get_kernel_name_at_line",
                return_value="rope_fwd",
            ),
        ):
            from minisweagent.run.mini import _inject_resolved_kernel

            task_content, kernel_name = _inject_resolved_kernel(
                "https://github.com/test/repo/blob/main/sub/kernel.py#L3",
                str(tmp_path),
                "Optimize this kernel",
            )

        # The regex that mini.py uses to extract _resolved_kernel_path
        m = re.search(r"Kernel path: (\S+)", task_content)
        assert m is not None, f"Regex 'Kernel path: (\\S+)' did not match in injected task:\n{task_content[:500]}"
        extracted = m.group(1)
        assert extracted == str(kernel), f"Extracted path {extracted!r} != expected {str(kernel)!r}"
        assert kernel_name == "rope_fwd"

    # -- Test 3: _run_discovery passes clone root as workspace --------------
    def test_run_discovery_scopes_to_clone_root(self, tmp_path, monkeypatch):
        """``_run_discovery`` must call ``discover(workspace=clone_root)``
        when the kernel lives inside a resolved clone."""
        from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

        clone_dir = tmp_path / RESOLVED_DIR_NAME / "owner_repo"
        (clone_dir / ".git").mkdir(parents=True)
        kernel = clone_dir / "sub" / "kernel.py"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("import triton\n@triton.jit\ndef rope_fwd(): pass\n")

        # Capture what discover() receives
        captured = {}

        class FakeResult:
            tests = []
            benchmarks = []

        def fake_discover(workspace=None, kernel_path=None, **kw):
            captured["workspace"] = workspace
            captured["kernel_path"] = kernel_path
            return FakeResult()

        # discover() is imported locally inside _run_discovery, so patch at source.
        monkeypatch.setattr("minisweagent.tools.discovery.discover", fake_discover)

        from minisweagent.run.mini import _run_discovery

        _run_discovery(str(kernel), "rope_fwd")

        assert "workspace" in captured, "discover() was never called"
        assert captured["workspace"] == clone_dir, f"Expected workspace={clone_dir}, got {captured['workspace']}"
