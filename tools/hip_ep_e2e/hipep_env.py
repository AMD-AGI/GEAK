#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Shared paths and process plumbing for running a hip-ep package from the GEAK
container.

The release package needs glibc 2.43 and the GEAK container has 2.35, so every
package binary is started through an Ubuntu 26.04 loader copied into a sysroot:

    <sysroot>/ld-linux-x86-64.so.2 --library-path <sysroot>:<pkg>/lib:<therock>/lib <exe> ...

The EP library is looked up relative to the CWD, so runs start in <pkg>/lib.
GPU runs hold the same serving-GPU lock as e2e_workflow/scripts/bench_e2e.sh,
so a model-level run never overlaps a GEAK serving benchmark.
"""
import contextlib
import dataclasses
import fcntl
import glob
import hashlib
import json
import os
import subprocess
import time

LOADER = "ld-linux-x86-64.so.2"
K = "lib/Runtime/Kernels"  # the kernel library's sources in the hip-ep tree

# The machine paths the tools read. None has a default: they differ per machine.
PATHS = {
    "HIPEP_PKG": "the hip-ep release package, unzipped",
    "HIPEP_THEROCK": "the TheRock dist the package pins",
    "HIPEP_SYSROOT": "a sysroot holding the Ubuntu 26.04 loader and runtime (glibc 2.43)",
    "HIPEP_PY": "a Python 3.14 with the package's OGA and ORT wheels",
    "HIPEP_SRC": "a hip-ep checkout at the release tag (build_kernels.sh clones it when missing)",
    "HIPEP_BUILD": "a hip-ep build dir (build_kernels.sh configures it when missing)",
}


def serving_lock_path():
    """The serving-GPU lock file: SERVING_GPU_LOCK, else bench_e2e.sh's default."""
    return os.environ.get("SERVING_GPU_LOCK", "/tmp/geak_serving_gpu_0.lock")


def lock_wait():
    """Seconds a tool waits for the serving-GPU lock before it gives up."""
    return int(os.environ.get("SERVING_LOCK_WAIT", "7200"))


def require(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set: {PATHS[name]}")
    return value


@dataclasses.dataclass
class Env:
    pkg: str
    therock: str
    sysroot: str

    @classmethod
    def from_environ(cls, pkg=None):
        return cls(
            pkg=os.path.abspath(pkg or require("HIPEP_PKG")),
            therock=require("HIPEP_THEROCK"),
            sysroot=require("HIPEP_SYSROOT"),
        )

    @property
    def lib(self):
        return os.path.join(self.pkg, "lib")

    @property
    def python(self):
        """The Python that runs OGA in-process (its wheel is cp314-only)."""
        return require("HIPEP_PY")


def pkg_argv(env, exe, args=()):
    """Command line that runs `exe` (absolute path) against `env.pkg`'s libraries."""
    libpath = ":".join((env.sysroot, env.lib, os.path.join(env.therock, "lib")))
    return [os.path.join(env.sysroot, LOADER), "--library-path", libpath, exe, *args]


def pkg_home(out_dir, pkg):
    """The HOME for runs of `pkg` under out_dir: one per package path, so two
    candidates that are both named `pkg` never share one."""
    pkg = os.path.abspath(pkg).rstrip("/")
    key = hashlib.sha1(pkg.encode()).hexdigest()[:8]
    return os.path.join(out_dir, f"home-{os.path.basename(pkg)}-{key}")


def pkg_env(env, home, extra=None):
    """Process env for a package run. HOME and TMPDIR are the package's own: hip-ep
    keeps compiled models in the temp dir, keyed by the model alone, and kernel
    changes can leave them stale."""
    tmp = os.path.join(home, "tmp")
    os.makedirs(tmp, exist_ok=True)
    out = {k: v for k, v in os.environ.items()
           if not k.startswith(("HIPDNN_", "LD_LIBRARY_PATH", "LD_PRELOAD"))}
    out.update(HOME=home, TMPDIR=tmp, THEROCK_DIST=env.therock)
    out.update(extra or {})
    return out


@contextlib.contextmanager
def hold_lock(path, wait_s):
    """Hold an exclusive flock on `path` for the duration of the block, waiting up to wait_s."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"lock {path} busy for {wait_s}s")
                time.sleep(5)
        yield
    finally:
        os.close(fd)


def serving_lock():
    """Hold the serving-GPU lock for the duration of the block."""
    return hold_lock(serving_lock_path(), lock_wait())


def run_logged(name, env, argv, home, log, extra=None, timeout=3600):
    """Run a package command under the serving lock, in <pkg>/lib, with its output in `log`.
    Raises, naming it `name`, if it exits non-zero."""
    with serving_lock():
        p = subprocess.run(argv, cwd=env.lib, env=pkg_env(env, home, extra), capture_output=True,
                           text=True, timeout=timeout, check=False)
    with open(log, "w") as f:
        f.write(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{name} exited {p.returncode}; log: {log}")
    return p


def fresh(path):
    """`path` with any earlier verdict removed, so a failed rerun never leaves one behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    return path


def write_json(path, obj):
    """Write `obj` to `path` atomically, so a reader never sees half a verdict."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def kernel_lib(pkg):
    """The package's kernel library, lib/libcustom_kernels_<arch>.so (one per release build)."""
    libs = glob.glob(os.path.join(pkg, "lib", "libcustom_kernels_*.so"))
    if len(libs) != 1:
        raise RuntimeError(f"{pkg}/lib holds {len(libs)} libcustom_kernels_<arch>.so; expected one")
    return libs[0]


def pkg_info(pkg):
    return {"pkg": pkg, "kernel_lib_md5": md5(kernel_lib(pkg))}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
