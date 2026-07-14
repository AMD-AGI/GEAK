"""Build-time hook for GEAK — all metadata + deps live in pyproject.toml.

The only reason setup.py still exists: a PEP 621 [project] table can't express a
build-time hook, and GEAK needs one so that

    pip install git+https://github.com/AMD-AGI/GEAK

runs a best-effort bootstrap during the wheel build (clone the full repo to
$GEAK_HOME + install the Claude Code CLI). The logic lives in geak/bootstrap.py
so the same code also backs the `geak-setup` command — the fallback to re-run if
this install-time hook does not fire (e.g. pip served a cached wheel).

Set GEAK_SKIP_BOOTSTRAP=1 to build/install the package without side effects
(CI, docker image builds, or using geak purely as a library dependency).
"""
import os
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.abspath(os.path.dirname(__file__))


def run_bootstrap():
    # Best-effort: never fail the install because a network step hiccupped.
    if os.environ.get("GEAK_SKIP_BOOTSTRAP"):
        return
    try:
        sys.path.insert(0, HERE)
        from geak.bootstrap import main as bootstrap_main

        bootstrap_main()
    except Exception as exc:
        sys.stderr.write("[geak-setup WARN] bootstrap step failed: %r\n" % (exc,))
        sys.stderr.write("[geak-setup WARN] finish it manually by running: geak-setup\n")


class BuildPyWithBootstrap(build_py):
    """Runs at wheel-build time — which for `pip install git+...` is the user's
    machine — so the clone + Claude Code install happen during `pip install`."""

    def run(self):
        super().run()
        run_bootstrap()


setup(cmdclass={"build_py": BuildPyWithBootstrap})
