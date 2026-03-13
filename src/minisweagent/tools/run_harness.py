# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Execute a test harness in subprocess and return structured results.

The harness is a Python script created by UnitTestAgent that supports four
CLI modes: ``--correctness``, ``--profile``, ``--benchmark``, and
``--full-benchmark``.  Each mode exercises a different code path (shape
subsets, output formatting, profiler-friendly execution, etc.).

This module can:
  - Run a single mode and return ``{mode, success, returncode, stdout, stderr, duration_s}``
  - Run **all** modes in sequence (correctness -> profile -> benchmark ->
    full-benchmark), short-circuiting on first failure
  - Be used as a Python API from the preprocessor/pipeline_helpers
  - Be used as a standalone CLI: ``run-harness <path> --mode all``

Environment setup mirrors what COMMANDMENT's ``## SETUP`` section does:
PYTHONPATH is set to ``repo_root`` and HIP_VISIBLE_DEVICES to the target GPU.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODES = ("correctness", "profile", "benchmark", "full-benchmark")

MODE_TO_FLAG: dict[str, str] = {
    "correctness": "--correctness",
    "profile": "--profile",
    "benchmark": "--benchmark",
    "full-benchmark": "--full-benchmark",
}

MODE_TIMEOUTS: dict[str, int] = {
    "correctness": 300,
    "profile": 120,
    "benchmark": 600,
    "full-benchmark": 900,
}

_STDERR_TAIL_LINES = 60


def _build_env(
    repo_root: str | None,
    gpu_id: int,
    env_overrides: dict[str, str] | None,
) -> dict[str, str]:
    """Build subprocess environment matching COMMANDMENT SETUP conventions."""
    env = os.environ.copy()

    if repo_root:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else repo_root
        # FlyDSL: when build-fly exists, set LD_LIBRARY_PATH and PYTHONPATH so harness can import kernels + flydsl (no-op for other repos)
        for candidate in (Path(repo_root), Path(repo_root).parent):
            mlir_libs = candidate / "build-fly" / "python_packages" / "flydsl" / "_mlir" / "_mlir_libs"
            build_pkg = candidate / "build-fly" / "python_packages"
            if mlir_libs.is_dir():
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = f"{mlir_libs}:{existing_ld}" if existing_ld else str(mlir_libs)
                if build_pkg.is_dir():
                    env["PYTHONPATH"] = f"{build_pkg}:{candidate}:{env.get('PYTHONPATH', '')}"
                break

    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    if env_overrides:
        env.update(env_overrides)

    return env


def _run_single(
    harness_path: str,
    mode: str,
    *,
    env: dict[str, str],
    timeout: int,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run a single harness mode and return a structured result dict."""
    flag = MODE_TO_FLAG[mode]
    cmd = [sys.executable, harness_path, flag]

    if mode in ("benchmark", "full-benchmark"):
        extra = env.get("GEAK_BENCHMARK_EXTRA_ARGS", "").strip()
        if extra:
            cmd.extend(extra.split())

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
        )
        duration_s = round(time.monotonic() - t0, 2)
        return {
            "mode": mode,
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": duration_s,
        }
    except subprocess.TimeoutExpired as exc:
        duration_s = round(time.monotonic() - t0, 2)
        return {
            "mode": mode,
            "success": False,
            "returncode": -1,
            "stdout": (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": f"TIMEOUT after {timeout}s",
            "duration_s": duration_s,
        }
    except Exception as exc:
        duration_s = round(time.monotonic() - t0, 2)
        return {
            "mode": mode,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "duration_s": duration_s,
        }


def run_harness(
    harness_path: str,
    mode: str = "correctness",
    *,
    repo_root: str | None = None,
    gpu_id: int = 0,
    timeout: int | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Execute a test harness and return structured results.

    Parameters
    ----------
    harness_path:
        Absolute path to the harness Python script.
    mode:
        One of ``"correctness"``, ``"profile"``, ``"benchmark"``,
        ``"full-benchmark"``, or ``"all"``.  ``"all"`` runs every mode
        in sequence and short-circuits on first failure.
    repo_root:
        Repository root added to PYTHONPATH.
    gpu_id:
        GPU device for HIP_VISIBLE_DEVICES.
    timeout:
        Per-mode timeout in seconds.  ``None`` uses the per-mode defaults
        from :data:`MODE_TIMEOUTS`.
    env_overrides:
        Extra environment variables merged into the subprocess env.

    Returns
    -------
    dict for a single mode, list[dict] for ``"all"``.
    Each result dict has keys: ``mode``, ``success``, ``returncode``,
    ``stdout``, ``stderr``, ``duration_s``.
    """
    harness = Path(harness_path)
    if not harness.is_file():
        err = {
            "mode": mode,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Harness file not found: {harness}",
            "duration_s": 0.0,
        }
        return [err] if mode == "all" else err

    env = _build_env(repo_root, gpu_id, env_overrides)
    cwd = repo_root

    if mode == "all":
        results: list[dict[str, Any]] = []
        for m in MODES:
            t = timeout if timeout is not None else MODE_TIMEOUTS[m]
            logger.info("run_harness: running --%s (timeout=%ds)", m, t)
            result = _run_single(harness_path, m, env=env, timeout=t, cwd=cwd)
            results.append(result)

            if result["success"]:
                logger.info(
                    "run_harness: --%s passed (%.1fs)", m, result["duration_s"]
                )
            else:
                logger.warning(
                    "run_harness: --%s FAILED (rc=%d, %.1fs)",
                    m,
                    result["returncode"],
                    result["duration_s"],
                )
                break  # short-circuit
        return results

    if mode not in MODE_TO_FLAG:
        return {
            "mode": mode,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Unknown mode: {mode!r}. Valid: {', '.join(MODES)} or 'all'",
            "duration_s": 0.0,
        }

    t = timeout if timeout is not None else MODE_TIMEOUTS[mode]
    return _run_single(harness_path, mode, env=env, timeout=t, cwd=cwd)


def format_results(results: dict | list[dict]) -> str:
    """Format run_harness results as a human-readable summary."""
    if isinstance(results, dict):
        results = [results]

    lines: list[str] = []
    all_passed = True
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        if not r["success"]:
            all_passed = False
        lines.append(
            f"  --{r['mode']}: {status}  (rc={r['returncode']}, {r['duration_s']}s)"
        )
        if not r["success"] and r["stderr"]:
            tail = r["stderr"].strip().splitlines()[-_STDERR_TAIL_LINES:]
            for line in tail:
                lines.append(f"    | {line}")

    header = "Harness execution: ALL PASSED" if all_passed else "Harness execution: FAILED"
    return header + "\n" + "\n".join(lines)


def results_errors(results: list[dict]) -> list[str]:
    """Extract error descriptions from failed modes for retry feedback."""
    errors: list[str] = []
    for r in results:
        if r["success"]:
            continue
        stderr_tail = r["stderr"].strip().splitlines()[-_STDERR_TAIL_LINES:]
        stderr_summary = "\n".join(stderr_tail)
        errors.append(
            f"--{r['mode']} mode failed (exit code {r['returncode']}):\n{stderr_summary}"
        )
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: ``run-harness <harness> --mode all [--gpu 0] [--repo-root ...]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute a test harness across one or all modes",
    )
    parser.add_argument("harness", help="Path to the test harness Python script")
    parser.add_argument(
        "--mode",
        choices=[*MODES, "all"],
        default="all",
        help="Harness mode to run (default: all)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root for PYTHONPATH (default: harness parent dir)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID for HIP_VISIBLE_DEVICES (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-mode timeout in seconds (default: mode-specific)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON instead of human-readable summary",
    )
    args = parser.parse_args()

    harness = str(Path(args.harness).resolve())
    repo_root = args.repo_root
    if repo_root is None:
        repo_root = str(Path(harness).parent)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_harness(
        harness,
        mode=args.mode,
        repo_root=repo_root,
        gpu_id=args.gpu,
        timeout=args.timeout,
    )

    if args.output_json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))

    if isinstance(results, list):
        ok = all(r["success"] for r in results)
    else:
        ok = results["success"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
