"""Unit tests for resolve_kernel_url (script/module)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from geak_agent.resolve_kernel_url import (
    _parse_github_blob,
    cleanup_resolved_path,
    is_weblink,
    resolve_kernel_url,
)


class TestIsWeblink:
    def test_https_is_weblink(self):
        assert is_weblink("https://github.com/foo/bar") is True

    def test_http_is_weblink(self):
        assert is_weblink("http://example.com/file.py") is True

    def test_local_path_not_weblink(self):
        assert is_weblink("/home/user/kernel.py") is False
        assert is_weblink("relative/path.py") is False

    def test_empty_or_none_not_weblink(self):
        assert is_weblink("") is False
        assert is_weblink("   ") is False
        assert is_weblink(None) is False

    def test_whitespace_stripped(self):
        assert is_weblink("  https://github.com/foo/bar  ") is True


class TestParseGithubBlob:
    def test_github_blob_url(self):
        url = "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_op_gelu.py"
        got = _parse_github_blob(url)
        assert got == ("ROCm", "aiter", "main", "aiter/ops/triton/moe/moe_op_gelu.py")

    def test_raw_github_url(self):
        url = "https://raw.githubusercontent.com/ROCm/aiter/main/aiter/ops/triton/moe/moe_op_gelu.py"
        got = _parse_github_blob(url)
        assert got == ("ROCm", "aiter", "main", "aiter/ops/triton/moe/moe_op_gelu.py")

    def test_non_github_returns_none(self):
        assert _parse_github_blob("https://gitlab.com/owner/repo/-/blob/main/file.py") is None
        assert _parse_github_blob("https://example.com/file.py") is None

    def test_short_raw_url_returns_none(self):
        assert _parse_github_blob("https://raw.githubusercontent.com/owner/repo") is None


class TestResolveKernelUrl:
    def test_empty_spec_returns_error(self):
        out = resolve_kernel_url("")
        assert out["is_weblink"] is False
        assert out["error"] == "Empty spec"
        assert out["local_file_path"] == ""

    def test_none_spec_returns_error(self):
        out = resolve_kernel_url(None)
        assert out["error"] == "Empty spec"

    def test_local_path_returned_unchanged(self):
        out = resolve_kernel_url("/path/to/local/kernel.py")
        assert out["is_weblink"] is False
        assert out["local_file_path"] == "/path/to/local/kernel.py"
        assert out["local_repo_path"] is None
        assert out["error"] is None

    def test_relative_path_returned_unchanged(self):
        out = resolve_kernel_url("aiter/ops/triton/moe/moe_op_gelu.py")
        assert out["is_weblink"] is False
        assert out["local_file_path"] == "aiter/ops/triton/moe/moe_op_gelu.py"

    def test_unsupported_url_returns_error(self):
        out = resolve_kernel_url("https://gitlab.com/owner/repo/file.py")
        assert out["is_weblink"] is True
        assert out["error"] == "Only GitHub blob or raw URLs are supported"
        assert out["local_repo_path"] is None

    @pytest.mark.parametrize("url", [
        "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_op_gelu.py",
        "https://raw.githubusercontent.com/ROCm/aiter/main/aiter/ops/triton/moe/moe_op_gelu.py",
    ])
    def test_github_url_clone_success(self, url):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = "aiter/ops/triton/moe/moe_op_gelu.py"
            full_path = Path(tmpdir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("# mock kernel")

            with (
                patch("geak_agent.resolve_kernel_url.tempfile.mkdtemp", return_value=tmpdir),
                patch("geak_agent.resolve_kernel_url.subprocess.run") as mock_run,
            ):
                mock_run.return_value = type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()

                out = resolve_kernel_url(url)

            assert out["is_weblink"] is True
            assert out["error"] is None
            assert out["local_repo_path"] == tmpdir
            assert out["local_file_path"] == str(full_path.resolve())
            assert Path(out["local_file_path"]).exists()

    def test_github_url_clone_failure_returns_error(self):
        url = "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_op_gelu.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("geak_agent.resolve_kernel_url.tempfile.mkdtemp", return_value=tmpdir),
                patch("geak_agent.resolve_kernel_url.subprocess.run") as mock_run,
            ):
                mock_run.return_value = type("R", (), {
                    "returncode": 1,
                    "stderr": "fatal: repository not found",
                    "stdout": "",
                })()

                out = resolve_kernel_url(url)

            assert out["is_weblink"] is True
            assert out["error"] is not None
            assert "not found" in out["error"] or "failed" in out["error"].lower()
            assert out["local_repo_path"] is None

    def test_github_url_file_missing_in_clone_returns_error(self):
        url = "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_op_gelu.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("geak_agent.resolve_kernel_url.tempfile.mkdtemp", return_value=tmpdir),
                patch("geak_agent.resolve_kernel_url.subprocess.run") as mock_run,
            ):
                mock_run.return_value = type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()

                out = resolve_kernel_url(url)

            assert out["is_weblink"] is True
            assert out["error"] is not None
            assert "File not found" in out["error"] or "not found" in out["error"]


class TestCleanupResolvedPath:
    def test_none_no_op(self):
        cleanup_resolved_path(None)

    def test_non_geak_dir_no_op(self, tmp_path):
        cleanup_resolved_path(str(tmp_path))
        assert tmp_path.exists()

    def test_geak_kernel_dir_removed(self, tmp_path):
        geak_dir = tmp_path / "geak_kernel_aiter_abc123"
        geak_dir.mkdir()
        (geak_dir / "file.txt").write_text("x")
        cleanup_resolved_path(str(geak_dir))
        assert not geak_dir.exists()
