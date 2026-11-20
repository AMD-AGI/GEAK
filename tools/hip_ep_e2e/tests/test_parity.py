#!/usr/bin/env python3
"""Pure-logic coverage for tools/hip_ep_e2e/parity.py: the comparison of two
logits dumps and the verdict. Producing the dumps needs the GPU and is verified
on the gfx1151 box.

    python3 -m pytest tools/hip_ep_e2e/tests/test_parity.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import parity  # noqa: E402

V = 512


def _dump(path, seed=0, flip=0, noise=0.0, kernel_lib="/pkg/a/lib/libk.so", ids_shift=0):
    """A fake two-text dump: 200 prefill positions (8 sampled) + 16 decode steps each."""
    rng = np.random.default_rng(seed)
    arrays = {"kernel_lib": np.array(kernel_lib)}
    for i in range(2):
        ids = np.arange(200, dtype=np.int32) + ids_shift
        logits = rng.standard_normal((200, V)).astype(np.float32)
        dlogits = rng.standard_normal((16, V)).astype(np.float32)
        if noise:
            nrng = np.random.default_rng(100 + i)
            logits += noise * nrng.standard_normal(logits.shape).astype(np.float32) \
                * np.linalg.norm(logits) / np.sqrt(logits.size)
            dlogits += noise * nrng.standard_normal(dlogits.shape).astype(np.float32) \
                * np.linalg.norm(dlogits) / np.sqrt(dlogits.size)
        argmax = logits.argmax(-1).astype(np.int32)
        if i == 0 and flip:
            argmax[:flip] = (argmax[:flip] + 1) % V
        pos = np.linspace(0, 199, 8).round().astype(np.int64)
        arrays.update({f"ids_{i}": ids, f"argmax_{i}": argmax, f"pos_{i}": pos,
                       f"logits_{i}": logits[pos].astype(np.float16),
                       f"dargmax_{i}": dlogits.argmax(-1).astype(np.int32),
                       f"dlogits_{i}": dlogits.astype(np.float16)})
    np.savez(path, **arrays)
    return str(path)


def test_identical_is_exact(tmp_path):
    a = _dump(tmp_path / "a.npz")
    b = _dump(tmp_path / "b.npz", kernel_lib="/pkg/b/lib/libk.so")
    c = parity.compare(a, b)
    assert c["top1"] == 1.0 and c["relL2"] == 0.0 and c["exact"]
    assert parity.verdict(c)


def test_one_flip_in_200_passes(tmp_path):
    a = _dump(tmp_path / "a.npz")
    b = _dump(tmp_path / "b.npz", flip=2)
    c = parity.compare(a, b)
    # 2 flips over 2 x (200 prefill + 16 decode) positions
    assert c["top1"] == pytest.approx(1 - 2 / 432)
    assert c["prefill_top1"] == pytest.approx(1 - 2 / 400)
    assert c["decode_top1"] == 1.0
    assert not c["exact"]
    assert parity.verdict(c)


def test_rel_l2_measured(tmp_path):
    a = _dump(tmp_path / "a.npz")
    c = parity.compare(a, _dump(tmp_path / "b.npz", noise=0.02))
    assert c["relL2"] == pytest.approx(0.02, rel=0.1)


def test_verdict_bounds():
    assert parity.verdict({"top1": 1.0, "relL2": 0.025})
    assert not parity.verdict({"top1": 1.0, "relL2": 0.035})
    assert not parity.verdict({"top1": 0.985, "relL2": 0.0})


def test_different_token_ids_refuse(tmp_path):
    a = _dump(tmp_path / "a.npz")
    b = _dump(tmp_path / "b.npz", ids_shift=1)
    with pytest.raises(ValueError, match="token ids"):
        parity.compare(a, b)


def test_kernel_lib_must_come_from_package():
    parity.check_kernel_lib("/pkg/a/lib/libcustom_kernels_gfx11-generic.so", "/pkg/a")
    with pytest.raises(RuntimeError, match="not from"):
        parity.check_kernel_lib("/pkg/a/lib/libcustom_kernels_gfx11-generic.so", "/pkg/a-cand")
    with pytest.raises(RuntimeError, match="no kernel library"):
        parity.check_kernel_lib("", "/pkg/a")


def test_failed_parity_leaves_no_verdict(tmp_path, monkeypatch):
    out = tmp_path / "parity"
    out.mkdir()
    (out / "parity.json").write_text('{"pass": true}')

    def broken(*args, **kwargs):
        raise RuntimeError("parity_dump exited 1")

    monkeypatch.setattr(parity.hipep_env.Env, "from_environ", staticmethod(lambda pkg=None: pkg))
    monkeypatch.setattr(parity, "dump", broken)
    monkeypatch.setattr(sys, "argv", ["parity.py", "--model", "m", "--base", "a", "--cand", "b",
                                      "--out", str(out)])
    with pytest.raises(RuntimeError):
        parity.main()
    assert not (out / "parity.json").exists(), "an earlier run's verdict survived a failed rerun"


def test_kernel_lib_check_sees_through_a_symlinked_package(tmp_path):
    real = tmp_path / "hip-ep-0.5.3"
    (real / "lib").mkdir(parents=True)
    link = tmp_path / "pkg"
    link.symlink_to(real)
    mapped = str(real / "lib" / "libcustom_kernels_gfx11-generic.so")  # what /proc/self/maps shows
    parity.check_kernel_lib(mapped, str(link))
    with pytest.raises(RuntimeError, match="not from"):
        parity.check_kernel_lib(mapped, str(tmp_path / "other"))


def test_texts_follow_the_timed_lengths():
    assert [t["len"] for t in parity.texts([1024, 2048, 4096])] == [1024, 2048, 4096]
    assert all(t["file"].endswith("parity_text.txt") for t in parity.texts([512]))


def test_a_dump_from_another_packages_library_is_refused(tmp_path, monkeypatch):
    pkg = tmp_path / "cand"
    (pkg / "lib").mkdir(parents=True)
    out = tmp_path / "cand.npz"

    def run(cmd, **kwargs):
        np.savez(out, kernel_lib=str(tmp_path / "base" / "lib" / "libcustom_kernels_gfx1151.so"))
        return parity.subprocess.CompletedProcess(cmd, 0, "", "")

    env = parity.hipep_env.Env(pkg=str(pkg), therock="/tr", sysroot="/s",
                               lock_path=str(tmp_path / "lock"), lock_wait=1)
    monkeypatch.setattr(parity.subprocess, "run", run)
    monkeypatch.setenv("HIPEP_PY", "/py")
    with pytest.raises(RuntimeError, match="is not from package"):
        parity.dump(env, "m", "texts.json", str(out))
