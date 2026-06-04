"""Regression tests for ``SaveAndTestTool._strip_binary_files_from_patch``.

A ``git diff`` / ``diff -ruN`` renders a changed binary file as a
``Binary files ... differ`` stub with no full-index delta. Such a stub cannot
be reapplied (``git apply`` errors "cannot apply binary patch ... without full
index line"), and because ``git apply`` is **atomic** it rejects the WHOLE
patch — including the real source edits (e.g. ``kernel.hip: patch does not
apply``). This previously degraded AVO's per-step verification to the agent's
self-reported speedup and forced the lineage tag to the dirty-worktree fallback.

These tests pin that binary sections are stripped while source hunks survive.
"""

from __future__ import annotations

from minisweagent.tools.save_and_test import SaveAndTestTool


def test_strip_binary_section_keeps_source_hunk():
    patch = (
        "diff --git a/silu.hip b/silu.hip\n"
        "index 1111111..2222222 100644\n"
        "--- a/silu.hip\n"
        "+++ b/silu.hip\n"
        "@@ -17,3 +17,3 @@\n"
        "-old line\n"
        "+new line\n"
        " ctx\n"
        "diff --git a/applications_silu b/applications_silu\n"
        "new file mode 100755\n"
        "index 0000000..abcdef0\n"
        "Binary files /dev/null and b/applications_silu differ\n"
    )
    out = SaveAndTestTool._strip_binary_files_from_patch(patch)
    # binary build artifact dropped...
    assert "applications_silu" not in out
    assert "Binary files" not in out
    # ...source hunk preserved.
    assert "diff --git a/silu.hip b/silu.hip" in out
    assert "+new line" in out


def test_strip_binary_section_when_binary_comes_first():
    # Binary section precedes the source section — ordering must not drop source.
    patch = (
        "diff --git a/bin_artifact b/bin_artifact\n"
        "index 0000000..1111111 100755\n"
        "Binary files a/bin_artifact and b/bin_artifact differ\n"
        "diff --git a/kernel.hip b/kernel.hip\n"
        "index 3333333..4444444 100644\n"
        "--- a/kernel.hip\n"
        "+++ b/kernel.hip\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    out = SaveAndTestTool._strip_binary_files_from_patch(patch)
    assert "bin_artifact" not in out
    assert "Binary files" not in out
    assert "diff --git a/kernel.hip b/kernel.hip" in out
    assert "+y" in out


def test_strip_binary_section_noop_for_text_only_patch():
    patch = (
        "diff --git a/k.py b/k.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/k.py\n"
        "+++ b/k.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    # No binary content → returned unchanged.
    assert SaveAndTestTool._strip_binary_files_from_patch(patch) == patch
