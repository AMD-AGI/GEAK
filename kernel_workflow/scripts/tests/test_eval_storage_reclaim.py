"""Issue #429: materialize excludes nested aiter JIT *.so; reclaim bounds round growth.

These scripts are bash (not under coverage.sources), but CI must still execute them so a
regression in exclude/reclaim is caught without a GPU. Synthetic trees use small filled
files — production uses ~0.3–0.6GiB module_aiter_operator.so copies; the path-amplification
geometry is identical.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
MATERIALIZE = SCRIPTS / "materialize_workspace.sh"
RECLAIM = SCRIPTS / "reclaim_eval_artifacts.sh"


def _du(path: Path) -> int:
    out = subprocess.check_output(["du", "-sb", str(path)], text=True)
    return int(out.split()[0])


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _seed_canonical(root: Path, so_bytes: int = 2 * 1024 * 1024) -> Path:
    """Canonical workspace shaped like an MXFP4 MoE task with nested aiter JIT .so."""
    can = root / "canonical"
    (can / "kernel_src").mkdir(parents=True)
    (can / "kernel_src" / "kernel.py").write_text("BLOCK = 64\n")
    (can / "aiter" / "jit").mkdir(parents=True)
    (can / "aiter" / "__init__.py").write_text("# vendor stub\n")
    (can / "aiter" / "ops.py").write_text("def noop():\n    return 1\n")
    so = can / "aiter" / "jit" / "module_aiter_operator.so"
    with open(so, "wb") as f:
        f.write(b"\0" * so_bytes)
    # Nested build path (also seen in production)
    nested = can / "aiter" / "jit" / "build" / "module_aiter_operator" / "build"
    nested.mkdir(parents=True)
    with open(nested / "module_aiter_operator.so", "wb") as f:
        f.write(b"\0" * so_bytes)
    # Immutable golden as absolute symlink target
    golden = root / "oracle" / "reference_io.pt"
    golden.parent.mkdir(parents=True)
    golden.write_bytes(b"GOLDEN" * 1000)
    os.symlink(golden, can / "reference_io.pt")
    return can


def test_materialize_excludes_nested_so_and_preserves_symlink(tmp_path: Path):
    can = _seed_canonical(tmp_path)
    dst = tmp_path / "eng0" / "workspace"
    dst.mkdir(parents=True)
    proc = _run(
        [
            "bash",
            str(MATERIALIZE),
            "--src",
            str(can),
            "--dst",
            str(dst),
            "--shared-root",
            str(tmp_path / "eval" / "_shared"),
            "--link-aiter",
        ]
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["ok"] is True
    assert summary["n_so"] == 0
    assert summary["so_bytes"] == 0
    assert not list(dst.rglob("*.so"))
    assert not (dst / "aiter" / "jit").exists() or not any((dst / "aiter" / "jit").rglob("*"))
    # Editable kernel_src remains a real directory (not a share symlink)
    assert (dst / "kernel_src" / "kernel.py").is_file()
    assert not (dst / "kernel_src").is_symlink()
    # reference_io stays a symlink (never -h dereference)
    assert (dst / "reference_io.pt").is_symlink()
    # aiter shared via symlink when --link-aiter
    assert (dst / "aiter").is_symlink()
    shared = (tmp_path / "eval" / "_shared" / "aiter").resolve()
    assert shared.is_dir()
    assert not list(shared.rglob("*.so"))


def test_broken_tar_without_so_exclude_amplifies(tmp_path: Path):
    """Control: agent that omits *.so exclude copies every nested .so (issue root cause)."""
    can = _seed_canonical(tmp_path, so_bytes=1024 * 1024)
    broken = tmp_path / "broken"
    broken.mkdir()
    # Mimic a verify agent that forgot --exclude='*.so' (and lacked wildcards-match-slash)
    _run(
        [
            "bash",
            "-c",
            f"( cd {can} && tar --exclude='./.git' --exclude='./.torch_ext' -cf - . )"
            f" | ( cd {broken} && tar -xf - )",
        ]
    )
    sos = list(broken.rglob("*.so"))
    assert len(sos) >= 2
    assert sum(p.stat().st_size for p in sos) >= 2 * 1024 * 1024


def test_reclaim_drops_old_round_workspaces_and_keeps_canonical(tmp_path: Path):
    can = _seed_canonical(tmp_path, so_bytes=512 * 1024)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Canonical
    _run(
        [
            "bash",
            str(MATERIALIZE),
            "--src",
            str(can),
            "--dst",
            str(eval_dir / "workspace"),
            "--shared-root",
            str(eval_dir / "_shared"),
            "--link-aiter",
        ]
    )
    (eval_dir / "workspace" / "kernel_src" / "kernel.py").write_text("BLOCK = 128\n")

    # Simulate 2 rounds × 2 engineers with BROKEN copies that include .so
    for rnd in (1, 2):
        for eng in (0, 1):
            ws = eval_dir / f"round_{rnd}" / f"engineer_{eng}" / "workspace"
            ws.mkdir(parents=True)
            _run(
                [
                    "bash",
                    "-c",
                    f"( cd {can} && tar -cf - . ) | ( cd {ws} && tar -xf - )",
                ]
            )
            # verify clone with .so
            vws = eval_dir / f"round_{rnd}" / f"engineer_{eng}" / "verify" / f"ws2_{rnd}{eng}"
            vws.mkdir(parents=True)
            _run(
                [
                    "bash",
                    "-c",
                    f"( cd {can} && tar -cf - . ) | ( cd {vws} && tar -xf - )",
                ]
            )

    # Wave archive with .so (AKA/GEAK peak amplifier)
    arch = eval_dir / "wave1_archive_testrun"
    arch.mkdir()
    _run(
        [
            "bash",
            "-c",
            f"( cd {can} && tar -cf - . ) | ( cd {arch} && tar -xf - )",
        ]
    )

    before = _du(eval_dir)
    assert before > 4 * 512 * 1024  # multiple .so copies

    proc = _run(
        ["bash", str(RECLAIM), "--eval-dir", str(eval_dir), "--keep-round", "2"]
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["ok"] is True
    assert summary["bytes_reclaimed"] > 0
    # Old round engineer workspaces gone
    assert not (eval_dir / "round_1" / "engineer_0" / "workspace").exists()
    # Canonical kept + edit preserved
    assert (eval_dir / "workspace" / "kernel_src" / "kernel.py").read_text() == "BLOCK = 128\n"
    # Soft reclaim lightens archive .so but may keep dir
    soft_so = list(arch.rglob("*.so"))
    assert soft_so == []

    # Hard pressure: force-heavy removes engineer workspaces in kept round + archives
    _run(
        [
            "bash",
            str(RECLAIM),
            "--eval-dir",
            str(eval_dir),
            "--keep-round",
            "2",
            "--force-heavy",
        ]
    )
    assert not (eval_dir / "round_2" / "engineer_0" / "workspace").exists()
    assert not arch.exists()
    assert (eval_dir / "workspace").is_dir()


def test_disk_pressure_reclaim_does_not_abort(tmp_path: Path):
    """Policy: soft/hard pressure triggers reclaim exit 0 — never stops optimize."""
    can = _seed_canonical(tmp_path, so_bytes=256 * 1024)
    eval_dir = tmp_path / "eval"
    (eval_dir / "round_1" / "engineer_0" / "workspace").mkdir(parents=True)
    _run(
        [
            "bash",
            "-c",
            f"( cd {can} && tar -cf - . ) | ( cd {eval_dir}/round_1/engineer_0/workspace && tar -xf - )",
        ]
    )
    # Soft budget exceeded path is advisory in materialize; reclaim always exits 0
    proc = _run(
        [
            "bash",
            str(MATERIALIZE),
            "--src",
            str(can),
            "--dst",
            str(eval_dir / "workspace"),
            "--soft-budget-bytes",
            "1",
        ]
    )
    assert proc.returncode == 0
    assert "soft budget exceeded" in (proc.stderr or "")
    proc2 = _run(
        ["bash", str(RECLAIM), "--eval-dir", str(eval_dir), "--keep-round", "1", "--force-heavy"]
    )
    assert proc2.returncode == 0
    summary = json.loads(proc2.stdout.strip().splitlines()[-1])
    assert summary["force_heavy"] == 1


@pytest.mark.parametrize("link", [False, True])
def test_materialize_telemetry_jsonl(tmp_path: Path, link: bool):
    can = _seed_canonical(tmp_path, so_bytes=64 * 1024)
    parent = tmp_path / "round_1" / "engineer_0"
    dst = parent / "workspace"
    dst.mkdir(parents=True)
    cmd = ["bash", str(MATERIALIZE), "--src", str(can), "--dst", str(dst)]
    if link:
        cmd += ["--shared-root", str(tmp_path / "_shared"), "--link-aiter"]
    _run(cmd)
    tel = parent / "materialize_telemetry.jsonl"
    assert tel.is_file()
    line = json.loads(tel.read_text().strip().splitlines()[-1])
    assert line["n_so"] == 0
