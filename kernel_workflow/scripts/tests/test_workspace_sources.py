# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for #480, including the generated skip-existing builder."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
MATERIALIZE = SCRIPTS / "materialize_workspace.sh"
GUARD = SCRIPTS / "workspace_sources.py"
OVERLAY = Path(".overlay/candidate/aiter_meta/csrc/kernels/quant_kernels.cu")
MARKER = "CANDIDATE_SOURCE_MUST_BE_COMPILED"


def run(*args, **kwargs):
    return subprocess.run(
        [str(a) for a in args], text=True, capture_output=True, check=False, **kwargs
    )


def materialize(src, dst, *extra):
    result = run("bash", MATERIALIZE, "--src", src, "--dst", dst, *extra)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"]
    return dst


def mirror(source, link):
    """The faulty generated builder: leave an existing link untouched."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(source)


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / "parent workspace"
    root.mkdir()
    (root / "quant_kernels.cu").write_text("int main() { return 0; }\n")
    mirror(root / "quant_kernels.cu", root / OVERLAY)
    return root


@pytest.fixture
def compiler():
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler required for the #error regression")
    return compiler


def compile_source(compiler, source):
    return run(compiler, "-x", "c++", "-fsyntax-only", source)


def test_old_copy_misses_candidate_error(parent, tmp_path, compiler):
    child = tmp_path / "broken"
    shutil.copytree(parent, child, symlinks=True)
    candidate = child / "quant_kernels.cu"
    candidate.write_text(f"#error {MARKER}\n")
    mirror(candidate, child / OVERLAY)
    assert compile_source(compiler, child / OVERLAY).returncode == 0
    assert MARKER in compile_source(compiler, candidate).stderr


def test_materialized_builder_compiles_candidate_across_generations(
    parent, tmp_path, compiler
):
    original = (parent / "quant_kernels.cu").read_bytes()
    src = parent
    for generation in ("engineer", "verify", "next_wave"):
        child = materialize(src, tmp_path / generation / "workspace")
        candidate = child / "quant_kernels.cu"
        candidate.write_text(f"#error {MARKER}\n")
        mirror(candidate, child / OVERLAY)
        assert (child / OVERLAY).resolve() == candidate
        result = compile_source(compiler, child / OVERLAY)
        assert result.returncode != 0
        assert MARKER in result.stderr
        src = child
    assert (parent / "quant_kernels.cu").read_bytes() == original


@pytest.mark.parametrize(
    "kind", ["relative", "directory", "chain", "dangling", "prefix"]
)
def test_link_shapes(parent, tmp_path, kind):
    link = parent / "source_alias"
    if kind == "relative":
        link.symlink_to("quant_kernels.cu")
    elif kind == "directory":
        link.symlink_to(parent / OVERLAY.parent, target_is_directory=True)
    elif kind == "chain":
        link.symlink_to(parent / OVERLAY)
    elif kind == "dangling":
        link.symlink_to(parent / "later.cu")
    else:
        vendor = tmp_path / (parent.name + "-vendor")
        vendor.mkdir()
        (vendor / "header.h").write_text("// external\n")
        link.symlink_to(vendor / "header.h")
    child = materialize(parent, tmp_path / "child")
    if kind == "directory":
        actual = child / "source_alias" / "quant_kernels.cu"
    else:
        actual = child / "source_alias"
    expected = child / ("later.cu" if kind == "dangling" else "quant_kernels.cu")
    if kind == "prefix":
        expected = vendor / "header.h"
    assert actual.resolve() == expected
    assert actual.is_symlink() or kind == "directory"


def test_shared_references_vendor_and_relative_external_links(parent, tmp_path):
    external = tmp_path / "vendor"
    external.mkdir()
    (external / "header.h").write_text("// vendor\n")
    (parent / "vendor").symlink_to(external, target_is_directory=True)
    (parent / "relative_vendor").symlink_to("../vendor", target_is_directory=True)
    (parent / "reference_io.pt").write_bytes(b"golden")
    (parent / "baseline_src").mkdir()
    (parent / "baseline_src/original.cu").symlink_to(parent / "quant_kernels.cu")
    child = materialize(parent, tmp_path / "deeper/child")
    assert (child / "vendor/header.h").resolve() == external / "header.h"
    assert (child / "relative_vendor/header.h").resolve() == external / "header.h"
    assert (child / "reference_io.pt").resolve() == parent / "reference_io.pt"
    assert (child / "baseline_src/original.cu").resolve() == parent / "quant_kernels.cu"
    assert run(sys.executable, GUARD, "check", "--workspace", child).returncode == 0


def test_source_root_alias(parent, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    (parent / "aliased.cu").symlink_to(alias / "quant_kernels.cu")
    child = materialize(alias, tmp_path / "child")
    assert (child / "aliased.cu").resolve() == child / "quant_kernels.cu"


def gpu_run(workspace, code):
    # Exercise the real wrapper without using a GPU or running ROCm discovery.
    env = dict(
        os.environ,
        GEAK_GPU_REQUIRE_IDLE="0",
        KERNEL_ENV_SKIP_ENUM_REAP="1",
        KERNEL_ENV_KEEP_ARCH="1",
        GEAK_GPU_ALLOWED="9876",
        TORCH_EXTENSIONS_DIR=str(workspace / ".torch_ext"),
    )
    return run(
        "bash",
        SCRIPTS / "gpu_lock.sh",
        "9876",
        sys.executable,
        "-c",
        code,
        cwd=workspace,
        env=env,
    )


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("kind", ["file", "directory", "relative"])
def test_gpu_wrapper_rejects_stale_sources(parent, tmp_path, when, kind):
    child = materialize(parent, tmp_path / "child")
    stale = child / "stale"
    target = parent if kind == "directory" else parent / "quant_kernels.cu"
    target = os.path.relpath(target, child) if kind == "relative" else str(target)
    create = f"from pathlib import Path; Path('stale').symlink_to({target!r}); "
    if when == "before":
        stale.symlink_to(target)
    result = gpu_run(child, (create if when == "during" else "") + "print('FAKE_PASS')")
    assert result.returncode == 86
    assert "GEAK_SOURCE_INVALID" in result.stderr
    assert ("FAKE_PASS" in result.stdout) == (when == "during")
    audit = json.loads((child / ".geak/invalid_measurements.jsonl").read_text())
    assert audit["status"] == "invalid_measurement"
    assert str(parent) in audit["reason"]
    # Fixing the link allows remeasurement, but does not erase the failed experiment.
    stale.unlink()
    assert gpu_run(child, "print('PASS')").returncode == 0
    assert (child / ".geak/invalid_measurements.jsonl").exists()


def test_wrapper_preserves_command_failure(parent, tmp_path):
    child = materialize(parent, tmp_path / "child")
    assert gpu_run(child, "raise SystemExit(7)").returncode == 7


def test_wrapper_rejects_swallowed_input_validation_failure(parent, tmp_path):
    child = materialize(parent, tmp_path / "child")
    command = [
        sys.executable,
        str(GUARD),
        "check-input",
        "--workspace",
        str(child),
        "--source",
        "quant_kernels.cu",
        "--input",
        str(parent / "quant_kernels.cu"),
    ]
    result = gpu_run(
        child, f"import subprocess; subprocess.run({command!r}); print('FAKE_PASS')"
    )
    assert result.returncode == 86
    assert "FAKE_PASS" in result.stdout
    assert "GEAK_SOURCE_INVALID" in result.stderr


def test_corrupt_provenance_fails_closed(parent, tmp_path):
    child = materialize(parent, tmp_path / "child")
    (child / ".geak/workspace.json").write_text("[]\n")
    result = gpu_run(child, "print('FAKE_PASS')")
    assert result.returncode == 86
    assert "FAKE_PASS" not in result.stdout
    assert "invalid workspace source state" in result.stderr


def test_ancestor_detection_survives_next_generation(parent, tmp_path):
    child = materialize(parent, tmp_path / "child")
    grandchild = materialize(child, tmp_path / "grandchild")
    (grandchild / "stale.cu").symlink_to(parent / "quant_kernels.cu")
    assert (
        run(sys.executable, GUARD, "check", "--workspace", grandchild).returncode == 86
    )


def test_bind_refreshes_and_checks_real_compiler_input(parent, tmp_path, compiler):
    child = materialize(parent, tmp_path / "child")
    link = child / OVERLAY
    link.unlink()
    link.symlink_to(parent / "quant_kernels.cu")
    candidate = child / "quant_kernels.cu"
    candidate.write_text(f"#error {MARKER}\n")
    result = run(
        sys.executable,
        GUARD,
        "bind",
        "--workspace",
        child,
        "--source",
        "quant_kernels.cu",
        "--link",
        OVERLAY,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert evidence["resolved_input"] == str(candidate)
    assert MARKER in compile_source(compiler, link).stderr


@pytest.mark.parametrize("input_kind", ["parent", "external", "missing"])
def test_check_input_rejects_wrong_path_even_with_identical_bytes(
    parent, tmp_path, input_kind
):
    child = materialize(parent, tmp_path / "child")
    actual = parent / "quant_kernels.cu"
    if input_kind != "parent":
        actual = tmp_path / "other.cu"
        if input_kind == "external":
            actual.write_bytes((child / "quant_kernels.cu").read_bytes())
    result = run(
        sys.executable,
        GUARD,
        "check-input",
        "--workspace",
        child,
        "--source",
        "quant_kernels.cu",
        "--input",
        actual,
    )
    assert result.returncode == 86
    assert "GEAK_SOURCE_INVALID" in result.stderr


def test_bind_refuses_to_mutate_external_tree_or_regular_file(parent, tmp_path):
    child = materialize(parent, tmp_path / "child")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "kernel.cu").write_text("// immutable\n")
    (child / "vendor").symlink_to(vendor, target_is_directory=True)
    for link in ("vendor/kernel.cu", "quant_kernels.cu"):
        result = run(
            sys.executable,
            GUARD,
            "bind",
            "--workspace",
            child,
            "--source",
            "quant_kernels.cu",
            "--link",
            link,
        )
        assert result.returncode == 86
    assert (vendor / "kernel.cu").read_text() == "// immutable\n"
    assert (child / "quant_kernels.cu").read_text() == "int main() { return 0; }\n"


def test_metadata_excluded_from_patch_and_invalid_history_not_copied(parent, tmp_path):
    child = materialize(parent, tmp_path / "child")
    (child / ".geak/invalid_measurements.jsonl").write_text("{}\n")
    grandchild = materialize(child, tmp_path / "grandchild")
    assert not (grandchild / ".geak/invalid_measurements.jsonl").exists()
    assert run("git", "init", "-q", cwd=child).returncode == 0
    ignored = run(
        "git",
        "check-ignore",
        ".geak/workspace.json",
        ".geak/invalid_measurements.jsonl",
        cwd=child,
    )
    assert ignored.returncode == 0
    assert len(ignored.stdout.splitlines()) == 2
