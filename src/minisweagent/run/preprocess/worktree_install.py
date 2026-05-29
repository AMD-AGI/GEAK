"""Editable-install the worktree before each harness invocation.

Why this exists
---------------
GEAK evaluates candidate patches by copying the target repo into a
worktree, applying the patch, then running ``python3 harness.py
--correctness / --benchmark / --profile``.  For repos whose Python
package wraps compiled C++/HIP kernels (``aiter``, ``sglang``,
``sgl_kernel``), the ``.so`` artifacts live inside the source tree and
are produced by ``setup.py develop`` / ``pip install -e``.  Without
re-running editable install on the worktree, ``import aiter`` would
either:

  * resolve to the wheel-installed ``aiter`` in site-packages (i.e.
    GEAK silently evaluates the BASELINE, not the candidate patch), or
  * resolve to the worktree's Python files but use stale ``.so`` files
    built before the patch landed.

Either way the measurement is meaningless.  This module fixes that by
running the appropriate editable-install command BEFORE every harness
subprocess.  Setuptools / pip are naturally incremental: when nothing
has changed they no-op in a couple of seconds; when a ``.hip`` file
changed they only rebuild that file.

Multi-package repos
-------------------
Modern ML-kernel repos commonly ship as a *monorepo* with several
separately-installable Python distributions side-by-side. Concrete
example: sglang has ``python/pyproject.toml`` (the ``sglang`` Python
wrapper) and ``sgl-kernel/pyproject.toml`` (the C++/HIP kernel
extension that exports the ``sgl_kernel`` import name). The agent
edits live in ``sgl-kernel/`` but the legacy ``_INSTALL_LAYOUTS`` only
matched the FIRST marker it saw, so ``sgl-kernel/`` was never
editable-installed and ``import sgl_kernel`` always resolved to the
wheel in site-packages — silently turning every benchmark into a
baseline measurement.

To stay generic across repos with arbitrary depth/branching of
installable sub-projects (sglang, vllm, flashinfer, multi-package
research repos, ...), this module now performs a bounded recursive
walk and editable-installs *every* installable sub-project it finds.
A sub-project is anything with a ``setup.py`` or ``pyproject.toml``;
once found, its sub-tree is NOT recursed further (sub-packages of a
sub-project are installed transitively by their parent's install).

Two-tier install strategy
-------------------------
Tier 1 (preferred): ``pip install -e <subdir> --no-build-isolation``.
This is the standard PEP-517 path; works for clean Python wrappers
(sglang's ``python/``) and for kernel repos whose ``pyproject.toml``
correctly declares scikit-build-core as the build backend.

Tier 2 (fallback): when Tier 1 fails (typically because the upstream's
``pyproject.toml`` is broken — e.g. sgl-kernel's CUDA variant declares
``setuptools.build_meta`` for a flat-C++-tree layout, which trips
"Multiple top-level packages discovered" auto-discovery), we look for
an ARCH-specific ``setup_<arch>.py`` (``setup_rocm.py``, ``setup_cuda.py``,
``setup_cpu.py``, ``setup_musa.py``) or a plain ``setup.py`` next to
the failed marker, and run ``python3 <setup_file> develop --no-deps -q``
directly. This is exactly what sgl-kernel's upstream CI does
(``scripts/ci/amd/amd_ci_install_dependency.sh`` runs
``python3 setup_rocm.py install`` after swapping pyproject.toml). Tier
2 must first ``pip uninstall -y <distribution>`` so the worktree's
egg-link wins over the existing wheel install (otherwise the physical
``site-packages/<pkg>/`` directory shadows the egg-link). The uninstall
is one-shot per process per distribution.

Per-run snapshot + restore
--------------------------
The editable install mutates site-packages globally (pip uninstalls the
original wheel and installs an egg-link pointing at the worktree).  To
let the user keep using the wheel-installed package after GEAK exits,
we snapshot the original ``pip show`` output the very first time we
touch a package, and at GEAK shutdown we run ``pip uninstall -y`` +
``pip install --force-reinstall --no-deps <pkg>==<version>`` to put the
wheel back.

The snapshot lives at ``~/.geak-env-backup/<pkg>/info.json`` and is
preserved across crashes: a second GEAK run that finds the snapshot
already populated will NOT overwrite it (otherwise it would record the
editable install from the previous crashed run, and restore would be a
no-op).  The restore step deletes the snapshot on success.

Design constraints (per user request)
-------------------------------------
* Only TWO public functions: :func:`ensure_worktree_installed` and
  :func:`restore_original_packages`.
* Install happens on EVERY harness invocation (best-effort, never
  blocks the harness if it fails — the harness call will surface the
  real error).
* No worktree-creation-time install; only at harness-run time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)

# Marker filenames that indicate "this directory is an editable-installable
# Python distribution". Order = priority when both are present in the same
# directory (we use the FIRST match to derive the install command — the
# command itself is the same modulo ``setup.py`` vs ``pyproject.toml``
# preference at install time, which pip handles internally).
_INSTALLABLE_MARKERS: tuple[str, ...] = ("setup.py", "pyproject.toml")

# Directories whose names should NEVER be descended into during sub-project
# discovery. These are either VCS metadata, build artifacts, vendored deps,
# or test scaffolding which would either explode time-to-install or pull in
# wrong/unwanted distributions.
_DISCOVERY_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        # Build / packaging artifacts.
        "build",
        "dist",
        "wheel",
        "wheels",
        "_build",
        # Caches.
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "__pycache__",
        # Virtual envs.
        ".venv",
        "venv",
        "env",
        # Vendored dependencies / submodules.
        "3rdparty",
        "third_party",
        "third-party",
        "thirdparty",
        "vendor",
        "vendored",
        "submodules",
        "external",
        "deps",
        # JS toolchains that occasionally appear in ML repos.
        "node_modules",
        # GEAK-generated helpers.
        "_eval_worktree",
        ".geak_resolved",
        # Test scaffolding (we install deliverable packages, not their tests).
        "tests",
        "test",
        "testing",
    }
)

# Glob-form skip rules layered on top of the literal name set.
_DISCOVERY_SKIP_DIR_GLOBS: tuple[str, ...] = (
    "*.egg-info",
    "*.egg-link",
    "build_*",
    ".*",  # any dotted name (covers .geak_resolved, .pixi, etc. uniformly)
)

# Heuristic: avoid sub-projects that are clearly NOT shippable distributions
# (test harnesses, e2e scaffolding, examples). These often have heavy or
# platform-specific build deps and installing them is wasteful at best,
# breaking at worst.
_DISCOVERY_SKIP_NAME_RE: re.Pattern[str] = re.compile(
    r"(?ix)^(e2e[_-]?tests?|examples?|samples?|docs?|benchmarks?|"
    r"scripts?|tools?|tutorials?|playground|sandbox)$"
)

# Bounds on the recursive walk. ``max_depth`` is measured from the worktree
# root: depth 0 is the root itself, 1 = direct children, etc. ``max_targets``
# caps total installs so a pathological repo can't lock the loop for hours.
_DEFAULT_DISCOVERY_MAX_DEPTH = 2
_DEFAULT_DISCOVERY_MAX_TARGETS = 6


def _build_install_argv(target_dir: Path) -> list[str]:
    """Return the ``pip install -e <target_dir>`` invocation we use for
    every discovered sub-project. Unified for ``setup.py`` and
    ``pyproject.toml`` layouts: pip handles both internally, so we don't
    need separate code paths.
    """
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(target_dir),
        "--no-deps",
        "--no-build-isolation",
        "-q",
    ]

# Snapshot directory for original pip-show metadata.  Lives under
# $HOME so it survives output-dir cleanup at GEAK end.
_SNAPSHOT_ROOT = Path(os.environ.get("GEAK_ENV_BACKUP_DIR", str(Path.home() / ".geak-env-backup")))

# Per-process serialization: two threads (or two _run_single calls
# from parallel agents) hitting the same worktree at once would
# corrupt the build.  A reentrant per-worktree lock is overkill;
# a single global lock is enough since installs are short.
_INSTALL_LOCK = threading.Lock()

# Track which (worktree, layout) pairs we've already installed in
# THIS process — same worktree being hit by multiple harness calls
# in quick succession can skip the redundant invocation.  Keyed by
# the resolved worktree path string.
_INSTALLED_IN_PROCESS: set[str] = set()

# Track which distribution names we've already ``pip uninstall``-ed as
# part of Tier 2 fallback in this process. Once a wheel is gone, we
# don't need to uninstall it again — subsequent ``setup.py develop``
# calls just overwrite the egg-link.
_TIER2_UNINSTALLED_PKGS: set[str] = set()

# Map torch build flavor → ordered list of ``setup_<arch>.py`` filenames
# we'll look for as the Tier 2 fallback. ``setup.py`` is always tried
# last as a generic fallback. Adding a new platform is one line here.
_ARCH_TO_SETUP_PY_NAMES: dict[str, tuple[str, ...]] = {
    "rocm": ("setup_rocm.py", "setup.py"),
    "cuda": ("setup_cuda.py", "setup.py"),
    "musa": ("setup_musa.py", "setup.py"),
    "cpu":  ("setup_cpu.py",  "setup.py"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_worktree_installed(
    worktree: str | os.PathLike[str] | None,
    *,
    force: bool = False,
) -> dict:
    """Editable-install every installable sub-project under ``worktree``.

    Performs a bounded recursive scan (see :func:`_discover_install_targets`)
    and runs ``pip install -e <subdir>`` for each discovered installable
    sub-project. This is what makes monorepo-style layouts (sglang's
    ``python/`` + ``sgl-kernel/``, vllm-style ``python`` + ``csrc``, ...)
    work correctly — the legacy single-marker behavior would silently
    skip every sub-project after the first match.

    Best-effort: any single sub-project's failure is logged but never
    raised. Other targets still get installed; the downstream harness
    call surfaces the real underlying error if the kernel build itself
    failed.

    Parameters
    ----------
    worktree : path-like | None
        The worktree to install. ``None`` or a non-directory path is a
        no-op.
    force : bool, default False
        Re-run install even when we've already installed a particular
        ``(worktree, target)`` pair in this process. Set True from
        places that know source files were just modified (e.g.
        ``save_and_test`` after the agent edited a kernel header).

    Returns
    -------
    dict with keys (kept backwards-compatible with the legacy one-target
    return shape, plus a new ``targets`` list):
        ``installed`` (bool): True iff at least one install was actually
            invoked AND every invocation succeeded. False on no-op,
            partial failure, or full failure.
        ``layout`` (str | None): rel-path of the FIRST discovered marker
            (e.g. ``"sgl-kernel/pyproject.toml"``), preserved for legacy
            callers; ``None`` when nothing was discovered.
        ``package`` (str | None): best-guess package name for the FIRST
            target (typically the directory basename).
        ``returncode`` (int): the FIRST non-zero subprocess returncode,
            or 0 if every install succeeded.
        ``stderr_tail`` (str): last ~800 chars of stderr from the first
            failure, empty when all succeeded.
        ``duration_s`` (float): wall time of the FIRST install call
            (legacy callers only inspect this for one number; total
            install time is the sum of per-target ``duration_s``).
        ``targets`` (list[dict]): per-sub-project results, each with
            keys ``layout``, ``package``, ``returncode``, ``installed``,
            ``stderr_tail``, ``duration_s``, ``argv``. Empty list when
            nothing was discovered.
    """
    result: dict = {
        "installed": False,
        "layout": None,
        "package": None,
        "returncode": 0,
        "stderr_tail": "",
        "duration_s": 0.0,
        "targets": [],
    }
    if not worktree:
        return result
    wt = Path(os.fspath(worktree))
    if not wt.is_dir():
        return result

    # Profile-aware short-circuit: wheel-only packages (e.g. vLLM via
    # ``shadow_worktree``) have no setup.py / pyproject.toml to
    # ``pip install -e`` against — running the discovery+install loop
    # would either no-op or surface confusing errors.  Skip cleanly.
    try:
        from minisweagent.kernel_packages import detect_packages

        for profile in detect_packages(wt):
            if profile.skip_install:
                logger.info(
                    "ensure_worktree_installed: skip_install profile %s active for %s; "
                    "no editable-install needed (shadow worktree)",
                    profile.name,
                    wt,
                )
                result["skipped_for_profile"] = profile.name
                return result
    except Exception as _exc:  # noqa: BLE001 — defensive
        logger.debug("ensure_worktree_installed: profile detection raised: %s", _exc)

    discovered = _discover_install_targets(wt)
    if not discovered:
        return result

    # Populate legacy single-target fields from the FIRST discovered target
    # so callers that only inspect ``layout`` / ``package`` still get a
    # sensible value.
    first_marker_rel, _first_argv, first_target_dir = discovered[0]
    result["layout"] = first_marker_rel
    result["package"] = first_target_dir.name

    any_install_ran = False
    all_succeeded = True
    first_failure_rc = 0
    first_failure_stderr = ""

    for marker_rel, argv, target_dir in discovered:
        per_result = _run_one_install(
            worktree=wt,
            marker_rel=marker_rel,
            argv=argv,
            target_dir=target_dir,
            force=force,
        )
        result["targets"].append(per_result)

        if per_result["installed"]:
            any_install_ran = True
        if per_result["returncode"] not in (0,):
            all_succeeded = False
            if first_failure_rc == 0:
                first_failure_rc = per_result["returncode"]
                first_failure_stderr = per_result["stderr_tail"]

        # Surface the first target's duration via the legacy field so
        # existing callers (``save_and_test`` log line) keep printing
        # something meaningful.
        if marker_rel == first_marker_rel:
            result["duration_s"] = per_result["duration_s"]

    result["installed"] = any_install_ran and all_succeeded
    result["returncode"] = first_failure_rc
    result["stderr_tail"] = first_failure_stderr
    return result


def _run_one_install(
    *,
    worktree: Path,
    marker_rel: str,
    argv: list[str],
    target_dir: Path,
    force: bool,
) -> dict:
    """Install a single sub-project. Returns a structured result dict.

    Dedup is keyed by ``(worktree_resolved, target_resolved)`` so adding
    a new sub-project to a worktree that was previously installed does
    NOT skip the new sub-project, while idempotent re-calls to the same
    sub-project remain cheap.
    """
    wt_key = str(worktree.resolve())
    target_key = str(target_dir.resolve())
    dedup_key = (wt_key, target_key)

    per: dict = {
        "layout": marker_rel,
        "package": target_dir.name,
        "argv": list(argv),
        "installed": False,
        "returncode": 0,
        "stderr_tail": "",
        "duration_s": 0.0,
        # Which install strategy ultimately succeeded (or last attempted):
        # "tier1:pip" / "tier2:setup_<arch>.py" / "" when not attempted.
        "tier_used": "",
    }

    with _INSTALL_LOCK:
        if not force and dedup_key in _INSTALLED_IN_PROCESS:
            return per

        # Snapshot the wheel-installed version BEFORE we replace it with
        # the worktree egg-link, but only the first time we ever touch
        # this package (across all GEAK runs). The package name is a
        # best-effort guess from the target dir's basename.
        guessed_pkg = _guess_package_name(target_dir, marker_rel)
        _snapshot_original_if_needed(guessed_pkg)

        import time as _time
        t0 = _time.monotonic()
        tier1_rc = 0
        tier1_stderr = ""
        try:
            proc = subprocess.run(
                argv,
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=600,
            )
            tier1_rc = proc.returncode
            tier1_stderr = (proc.stderr or "")
            per["returncode"] = proc.returncode
            per["duration_s"] = round(_time.monotonic() - t0, 2)
            per["tier_used"] = "tier1:pip"
            if proc.returncode == 0:
                _INSTALLED_IN_PROCESS.add(dedup_key)
                per["installed"] = True
                logger.info(
                    "worktree_install: editable install OK for %s (%s, %.1fs)",
                    target_dir, " ".join(argv), per["duration_s"],
                )
            else:
                tail = tier1_stderr[-800:]
                per["stderr_tail"] = tail
                logger.warning(
                    "worktree_install: tier1 (pip install -e) FAILED for %s (rc=%s)\n"
                    "stderr tail:\n%s",
                    target_dir, proc.returncode, tail,
                )
        except subprocess.TimeoutExpired:
            tier1_rc = -1
            tier1_stderr = "TIMEOUT after 600s"
            per["returncode"] = -1
            per["stderr_tail"] = tier1_stderr
            per["duration_s"] = round(_time.monotonic() - t0, 2)
            per["tier_used"] = "tier1:pip"
            logger.warning("worktree_install: tier1 timed out for %s after 600s", target_dir)
        except Exception as exc:  # pragma: no cover — defensive
            tier1_rc = -1
            tier1_stderr = str(exc)
            per["returncode"] = -1
            per["stderr_tail"] = tier1_stderr
            per["duration_s"] = round(_time.monotonic() - t0, 2)
            per["tier_used"] = "tier1:pip"
            logger.warning("worktree_install: tier1 raised for %s: %s", target_dir, exc)

        # Tier 2 fallback: when tier1 failed AND the sub-project ships an
        # arch-specific or plain ``setup.py`` we can use directly, run
        # ``python3 <setup_file> develop --no-deps -q``. This is the
        # only path that works for kernel repos whose pyproject.toml is
        # broken / mismatched for the active platform (sgl-kernel
        # ROCm being the canonical example).
        if tier1_rc != 0:
            tier2_result = _try_tier2_setup_py_develop(
                worktree=worktree,
                target_dir=target_dir,
                marker_rel=marker_rel,
                guessed_pkg=guessed_pkg,
            )
            if tier2_result is not None:
                # Tier 2 was attempted — its outcome supersedes tier1.
                per["tier_used"] = tier2_result["tier_used"]
                per["argv"] = tier2_result["argv"]
                per["returncode"] = tier2_result["returncode"]
                per["stderr_tail"] = tier2_result["stderr_tail"]
                # duration_s aggregates both tiers so callers get the
                # real wall-time spent on this sub-project.
                per["duration_s"] = round(per["duration_s"] + tier2_result["duration_s"], 2)
                if tier2_result["returncode"] == 0:
                    _INSTALLED_IN_PROCESS.add(dedup_key)
                    per["installed"] = True

        # Post-install verification (C-fix): when EITHER tier reported
        # success, double-check that the editable install really points
        # at the worktree. A success rc with a wheel-shadowed import
        # path is the silent failure mode that sent the round_1
        # sgl-kernel run measuring baselines. Fail-loud unless the user
        # explicitly opts out.
        if per["installed"] and not os.environ.get("GEAK_ALLOW_WHEEL_FALLBACK"):
            ok, diag = _verify_install_resolves_to_target(target_dir, guessed_pkg)
            if not ok:
                logger.warning("worktree_install: %s", diag)
                # Demote success: caller (save_and_test) treats this
                # as a failed install and surfaces the diagnostic in
                # its stderr-tail log.
                per["installed"] = False
                per["returncode"] = per["returncode"] or 100  # synthetic non-zero
                per["stderr_tail"] = diag
                # Drop dedup so a future call doesn't skip-and-stay-broken.
                _INSTALLED_IN_PROCESS.discard(dedup_key)

    return per


def _try_tier2_setup_py_develop(
    *,
    worktree: Path,
    target_dir: Path,
    marker_rel: str,
    guessed_pkg: str,
) -> dict | None:
    """Tier 2 fallback: ``pip uninstall -y <pkg>`` + ``python3 setup_<arch>.py develop``.

    Returns ``None`` when no arch-appropriate setup.py file exists in
    ``target_dir`` (signal: caller should keep tier1's failure as the
    final result). Otherwise returns a result dict with the fields the
    caller merges into its ``per`` summary:
    ``{"tier_used", "argv", "returncode", "stderr_tail", "duration_s"}``.

    The ``pip uninstall`` is one-shot per distribution per process: a
    second sub-project that uses the same distribution name (rare) won't
    repeat the uninstall.
    """
    arch = _detect_torch_arch()
    setup_path = _find_arch_setup_py(target_dir, arch)
    if setup_path is None:
        return None

    import time as _time

    setup_filename = setup_path.name
    tier_label = f"tier2:{setup_filename}"
    logger.info(
        "worktree_install: tier1 failed for %s; attempting tier2 fallback "
        "via %s (arch=%s)",
        target_dir, setup_filename, arch,
    )

    t0 = _time.monotonic()

    # Step 1: uninstall the wheel (only once per dist per process). The
    # editable egg-link cannot win over a physical site-packages dir
    # otherwise. Failures here are not fatal — pip uninstall is a
    # no-op when the package isn't installed.
    if guessed_pkg and guessed_pkg not in _TIER2_UNINSTALLED_PKGS:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", guessed_pkg],
                capture_output=True, text=True, timeout=120,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(
                "worktree_install: tier2 uninstall of %s raised (continuing): %s",
                guessed_pkg, exc,
            )
        _TIER2_UNINSTALLED_PKGS.add(guessed_pkg)

    # Step 2: python3 <setup_file> develop --no-deps -q
    argv = [sys.executable, setup_filename, "develop", "--no-deps", "-q"]
    result: dict = {
        "tier_used": tier_label,
        "argv": list(argv),
        "returncode": 0,
        "stderr_tail": "",
        "duration_s": 0.0,
    }
    try:
        proc = subprocess.run(
            argv,
            cwd=str(target_dir),  # MUST be the sub-project dir, not worktree root
            capture_output=True,
            text=True,
            timeout=900,  # incremental builds ~30-60s; clean builds ~5-10min
        )
        result["returncode"] = proc.returncode
        result["duration_s"] = round(_time.monotonic() - t0, 2)
        if proc.returncode == 0:
            logger.info(
                "worktree_install: tier2 install OK for %s (%s %s, %.1fs)",
                target_dir, sys.executable, " ".join(argv[1:]), result["duration_s"],
            )
        else:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            result["stderr_tail"] = tail
            logger.warning(
                "worktree_install: tier2 install FAILED for %s (rc=%s) via %s\n"
                "stderr/stdout tail:\n%s",
                target_dir, proc.returncode, " ".join(argv), tail,
            )
    except subprocess.TimeoutExpired:
        result["returncode"] = -1
        result["stderr_tail"] = "TIMEOUT after 900s"
        result["duration_s"] = round(_time.monotonic() - t0, 2)
        logger.warning("worktree_install: tier2 timed out for %s after 900s", target_dir)
    except Exception as exc:  # pragma: no cover — defensive
        result["returncode"] = -1
        result["stderr_tail"] = str(exc)
        result["duration_s"] = round(_time.monotonic() - t0, 2)
        logger.warning("worktree_install: tier2 raised for %s: %s", target_dir, exc)

    return result


def _verify_install_resolves_to_target(
    target_dir: Path, guessed_pkg: str
) -> tuple[bool, str]:
    """Post-install sanity check: confirm the editable install really
    points at the worktree, not at the wheel in site-packages.

    Why this exists: ``pip install -e <dir>`` and
    ``setup_<arch>.py develop`` both happily report success even when
    a physical ``site-packages/<pkg>/`` directory shadows the egg-link
    (this is exactly the failure mode that sent every benchmark in
    the round_1 sgl-kernel run measuring the BASELINE wheel — see
    "Per-run snapshot + restore" docstring above for context).

    We use ``pip show <pkg>`` rather than spawning a real ``import
    <pkg>`` to avoid the cost / side-effects of actually loading
    torch + the kernel binary just to validate a path.

    Returns ``(ok, diagnostic)``. ``ok`` is False when the metadata
    shows a location outside ``target_dir``; ``diagnostic`` has a
    one-line explanation safe to log.
    """
    if not guessed_pkg:
        # No package name to query — skip the check rather than
        # false-alarm. Caller logs nothing in this branch.
        return True, ""
    info = _pip_show(guessed_pkg)
    if not info:
        # pip show could not find the distribution. This means our
        # ``_guess_package_name`` heuristic missed (rare but possible
        # for repos with non-standard naming). Skip the check rather
        # than false-positive.
        return True, ""

    editable_loc = info.get("editable project location") or ""
    location = info.get("location") or ""
    try:
        target_resolved = str(target_dir.resolve())
    except OSError:
        return True, ""

    candidates: list[str] = []
    for candidate in (editable_loc, location):
        if not candidate:
            continue
        try:
            cand_resolved = str(Path(candidate).resolve())
        except (OSError, RuntimeError):
            continue
        candidates.append(cand_resolved)
        # The reported location can be the target itself OR a sub-dir
        # of it (e.g. for sgl-kernel the egg-link points at
        # ``<target>/python``, not ``<target>``).
        if cand_resolved == target_resolved or cand_resolved.startswith(target_resolved + os.sep):
            return True, ""

    diag = (
        f"install verification failed for {guessed_pkg!r}: "
        f"pip show reports location={candidates!r} but worktree target is "
        f"{target_resolved!r} — the harness will silently run against the "
        f"wheel-installed binary in site-packages, not your worktree edits. "
        f"Set env GEAK_ALLOW_WHEEL_FALLBACK=1 to disable this check."
    )
    return False, diag


def _detect_torch_arch() -> str:
    """Return one of ``"rocm" | "cuda" | "musa" | "cpu"`` based on the
    installed torch build. Used to pick which ``setup_<arch>.py`` file
    to try in Tier 2 fallback.

    Detection order:
      * torch.version.hip is set & truthy → "rocm"
      * torch.version.cuda is set & truthy → "cuda"
      * torch built against MUSA (Moore Threads) → "musa"
      * default fallback → "cpu"

    Imports torch lazily and swallows any ImportError so test
    environments without torch don't break (they fall back to "cpu",
    which is conservative — generic ``setup.py`` is then tried).
    """
    try:
        import torch  # noqa: PLC0415 — lazy on purpose
    except Exception:  # pragma: no cover — torch missing
        return "cpu"
    try:
        if getattr(torch.version, "hip", None):
            return "rocm"
        if getattr(torch.version, "cuda", None):
            return "cuda"
        # Moore Threads MUSA torch build exposes torch.version.musa.
        if getattr(torch.version, "musa", None):
            return "musa"
    except Exception:  # pragma: no cover — defensive against forks
        pass
    return "cpu"


def _find_arch_setup_py(target_dir: Path, arch: str) -> Path | None:
    """Return the first ``setup_<arch>.py`` (or ``setup.py``) that exists
    in ``target_dir`` for ``arch``, or ``None`` when nothing usable.

    Order is "arch-specific first, generic last" so a repo that ships
    BOTH ``setup_rocm.py`` and ``setup.py`` uses the rocm-specific one
    on a ROCm host.
    """
    candidates = _ARCH_TO_SETUP_PY_NAMES.get(arch, ("setup.py",))
    for name in candidates:
        candidate = target_dir / name
        if candidate.is_file():
            return candidate
    return None


def restore_original_packages() -> dict:
    """Undo every editable install made during this GEAK process.

    For each package we snapshotted: ``pip uninstall -y <pkg>`` to drop
    the egg-link / dist-info, then ``pip install --force-reinstall
    --no-deps <pkg>==<version>`` to restore the original wheel.  On
    success the snapshot is deleted.

    Best-effort: failures are logged but never raised — the GEAK
    finally block must continue running.

    Returns a dict ``{"restored": [pkg, ...], "failed": [pkg, ...]}``
    describing what happened.
    """
    out: dict[str, list[str]] = {"restored": [], "failed": []}
    if not _SNAPSHOT_ROOT.is_dir():
        return out
    for snap_dir in sorted(_SNAPSHOT_ROOT.iterdir()):
        info_path = snap_dir / "info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            pkg = info["name"]
            version = info.get("version", "")
        except Exception as exc:
            logger.warning("worktree_install: bad snapshot at %s: %s", info_path, exc)
            continue

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                capture_output=True, text=True, timeout=120,
            )
            if version:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", f"{pkg}=={version}"],
                    capture_output=True, text=True, timeout=300,
                )
                if proc.returncode != 0:
                    logger.warning(
                        "worktree_install: could not reinstall original %s==%s (rc=%s); "
                        "the editable install was removed but you may need to manually "
                        "`pip install %s`. stderr: %s",
                        pkg, version, proc.returncode, pkg, (proc.stderr or "")[-400:],
                    )
                    out["failed"].append(pkg)
                    continue
            out["restored"].append(pkg)
            logger.info("worktree_install: restored %s==%s", pkg, version or "<unknown>")
            shutil.rmtree(snap_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("worktree_install: restore failed for %s: %s", pkg, exc)
            out["failed"].append(pkg)
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _discover_install_targets(
    worktree: Path,
    *,
    max_depth: int = _DEFAULT_DISCOVERY_MAX_DEPTH,
    max_targets: int = _DEFAULT_DISCOVERY_MAX_TARGETS,
) -> list[tuple[str, list[str], Path]]:
    """Bounded recursive walk that returns every installable sub-project.

    A directory is a "target" if it contains :data:`_INSTALLABLE_MARKERS`
    (``setup.py`` or ``pyproject.toml``). When a target is found, its
    sub-tree is NOT descended further: any nested installable directory
    is a sub-package of the parent and gets installed transitively when
    the parent's ``pip install -e`` runs.

    Returns a list of ``(marker_rel_posix, install_argv, target_abs_dir)``
    tuples in deterministic order: shallower first, then alphabetical
    within the same depth. The first entry is therefore identical (in
    spirit) to what the legacy ``_detect_layout`` used to return.

    Skip rules avoid descending into VCS/build/cache/vendored/test
    directories (see :data:`_DISCOVERY_SKIP_DIR_NAMES` /
    :data:`_DISCOVERY_SKIP_DIR_GLOBS` /
    :data:`_DISCOVERY_SKIP_NAME_RE`). The walk is bounded by
    ``max_depth`` (rooted at ``worktree``) and ``max_targets`` to keep
    pathological monorepos from locking the loop for hours.
    """
    targets: list[tuple[str, list[str], Path]] = []
    seen: set[Path] = set()

    def _emit(target_dir: Path) -> None:
        marker_name = next(
            (m for m in _INSTALLABLE_MARKERS if (target_dir / m).is_file()),
            None,
        )
        if marker_name is None:
            return
        try:
            rel = (target_dir / marker_name).relative_to(worktree).as_posix()
        except ValueError:
            rel = marker_name
        argv = _build_install_argv(target_dir)
        targets.append((rel, argv, target_dir))

    def _is_target(d: Path) -> bool:
        return any((d / m).is_file() for m in _INSTALLABLE_MARKERS)

    def _walk(d: Path, depth: int) -> None:
        if len(targets) >= max_targets:
            return
        try:
            d_resolved = d.resolve()
        except OSError:
            return
        if d_resolved in seen:
            return
        seen.add(d_resolved)

        if _is_target(d):
            _emit(d)
            # Don't descend into an installable sub-tree: nested
            # installables are sub-packages of THIS one.
            return

        if depth >= max_depth:
            return

        try:
            children = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for child in children:
            if not child.is_dir():
                continue
            if _should_skip_discovery_dir(child.name):
                continue
            _walk(child, depth + 1)

    _walk(worktree, 0)
    return targets


def _should_skip_discovery_dir(name: str) -> bool:
    """Return True when a directory ``name`` must NOT be descended into
    during sub-project discovery. Combines the literal name set, glob
    patterns, and the test/example/etc heuristic.
    """
    if name in _DISCOVERY_SKIP_DIR_NAMES:
        return True
    if any(fnmatch(name, g) for g in _DISCOVERY_SKIP_DIR_GLOBS):
        return True
    if _DISCOVERY_SKIP_NAME_RE.match(name):
        return True
    return False


def _detect_layout(worktree: Path) -> tuple[str, list[str], str] | None:
    """Legacy single-marker detection. Kept for backward compatibility
    with callers / tests that expect the historical 3-tuple shape.

    Returns ``(marker_rel, install_argv, install_cwd_rel)`` for the FIRST
    installable sub-project found, or ``None`` if there is none. Modern
    callers should use :func:`_discover_install_targets` directly to
    handle multi-package monorepos correctly.
    """
    targets = _discover_install_targets(worktree, max_targets=1)
    if not targets:
        return None
    marker_rel, argv, _target_dir = targets[0]
    # ``cwd_rel`` is left as ``"."`` because the install argv now embeds
    # the absolute path of the target directory, so the cwd of the
    # subprocess is irrelevant to behavior.
    return marker_rel, argv, "."


def _guess_package_name(target_dir: Path, marker: str) -> str:
    """Best-effort: derive the distribution name (what ``pip show`` /
    ``pip install`` takes) for a sub-project rooted at ``target_dir``.

    Used only for snapshotting the wheel-installed original so we can
    restore it on GEAK shutdown. Heuristic: target directory basename,
    lowercased, with ``_`` → ``-``. Works correctly for sglang's
    ``python/`` (distribution: sglang? – overridden) and ``sgl-kernel/``
    (distribution: ``sglang-kernel``? – overridden), aiter, vllm-style
    repos, etc.
    """
    name = target_dir.name.lower().replace("_", "-")
    # Well-known directory→distribution overrides. Empirical, additive:
    # if a future repo uses a different naming convention we just add
    # an entry. This list intentionally stays small; everything else
    # falls back to the directory basename which is right ~90% of the
    # time for ML kernel repos.
    _DIR_TO_DIST = {
        # sglang's python wrapper lives at <repo>/python/.
        "python": "sglang",
        # sgl-kernel's distribution name on PyPI is sglang-kernel.
        "sgl-kernel": "sglang-kernel",
        # aiter / sglang root identities (when called with worktree root).
        "sglang": "sglang",
        "aiter": "aiter",
    }
    return _DIR_TO_DIST.get(name, name)


def _snapshot_original_if_needed(pkg: str) -> Path | None:
    """Save ``pip show <pkg>`` metadata if we haven't already.

    Returns the snapshot directory on success, ``None`` when the
    package isn't installed (nothing to restore) or the snapshot
    already exists (preserving the genuine original from a prior run).
    """
    try:
        _SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("worktree_install: cannot create snapshot root: %s", exc)
        return None

    snap_dir = _SNAPSHOT_ROOT / pkg
    info_path = snap_dir / "info.json"
    if info_path.is_file():
        # Preserve the existing snapshot — it may be from a prior
        # crashed run and overwriting it now would lose the genuine
        # original (we'd record the editable install instead).
        return snap_dir

    info = _pip_show(pkg)
    if not info or "name" not in info:
        # Package wasn't installed before we touched it; nothing to
        # restore.  Drop a marker so we don't keep probing.
        return None

    # Refuse to snapshot an already-editable install — that would
    # capture the worktree path instead of the wheel version.
    if _looks_like_editable_install(info):
        logger.info(
            "worktree_install: %s is already an editable install at %s; "
            "skipping snapshot (assuming a previous GEAK run left it that way).",
            pkg, info.get("location", "<unknown>"),
        )
        return None

    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        logger.info(
            "worktree_install: snapshotted original %s==%s (location=%s) to %s",
            info["name"], info.get("version", "?"), info.get("location", "?"), snap_dir,
        )
        return snap_dir
    except OSError as exc:
        logger.debug("worktree_install: snapshot write failed for %s: %s", pkg, exc)
        return None


def _pip_show(pkg: str) -> dict:
    """Return parsed ``pip show <pkg>`` fields, or {} when not installed."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    info: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            info[k.strip().lower()] = v.strip()
    return info


def _looks_like_editable_install(info: dict) -> bool:
    """Heuristic: pip show prints ``Editable project location:`` for editable installs."""
    return any(k.startswith("editable") for k in info)


__all__ = ["ensure_worktree_installed", "restore_original_packages"]
